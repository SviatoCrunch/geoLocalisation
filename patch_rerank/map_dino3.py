"""DINOv3 (satellite SAT-493M) token-grid extractor for map crops, via HuggingFace transformers.

Parallel to ``map_dino`` (which builds DINOv2 via torch.hub) but for ``facebook/dinov3-*`` models,
which load through ``transformers.AutoModel`` and prepend ``1 CLS + num_register_tokens`` prefix
tokens (4 registers for ViT-L/16) before the spatial patch tokens. This drops that prefix and returns
L2-normalised ``(h, w, D)`` token grids in exactly the convention the patch matcher expects.

The satellite checkpoint (``...-sat493m``) is trained on Maxar 0.6 m/px ortho imagery — the same domain
and GSD as the ESRI z18 basemap — so it is the map-side backbone. The QUERY frames must be embedded with
the SAME model for mutual-NN to be meaningful. Needs ``HF_TOKEN`` (the repo is gated).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"


def build_dinov3_extractor(model_id: str = DEFAULT_MODEL, device: str = "cuda"):
    """→ dict with the loaded model + the layout facts read from its config/processor (no guessing:
    prefix length = 1 CLS + config.num_register_tokens; patch/D from config; mean/std from processor)."""
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    cfg = model.config
    n_reg = int(getattr(cfg, "num_register_tokens", 4) or 0)
    patch = int(getattr(cfg, "patch_size", 16))
    D = int(getattr(cfg, "hidden_size", 1024))
    mean = [float(x) for x in getattr(proc, "image_mean", [0.485, 0.456, 0.406])]
    std = [float(x) for x in getattr(proc, "image_std", [0.229, 0.224, 0.225])]
    return {"model": model, "n_prefix": 1 + n_reg, "patch": patch, "D": D,
            "mean": mean, "std": std, "model_id": model_id, "backbone": model_id}


@torch.no_grad()
def extract_grids_v3(images, ext, output_px: int, device: str = "cuda", amp: bool = True):
    """List of RGB ``(H,W,3)`` uint8 → list of ``(h,w,D)`` float32 L2-normalised token grids (on CPU).

    Each image is resized to ``output_px`` (a multiple of the patch size), normalised with the model's
    own mean/std, forwarded in one batch; the ``n_prefix`` CLS/register tokens are dropped and the
    remaining ``(output_px/patch)²`` patch tokens are reshaped row-major to ``(h,w,D)`` and L2-normed."""
    model, patch, n_prefix = ext["model"], ext["patch"], ext["n_prefix"]
    if output_px % patch:
        raise ValueError(f"output_px {output_px} must be a multiple of patch {patch}")
    h = w = output_px // patch
    mean = torch.tensor(ext["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(ext["std"]).view(1, 3, 1, 1)
    tens = []
    for im in images:
        t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0   # (3,H,W)
        if t.shape[1] != output_px or t.shape[2] != output_px:
            t = F.interpolate(t.unsqueeze(0), size=(output_px, output_px), mode="bilinear",
                              align_corners=False)[0]
        tens.append(t)
    pv = ((torch.stack(tens) - mean) / std).to(device)                                    # (B,3,P,P)
    ac = (torch.autocast(device_type="cuda", dtype=torch.float16)
          if (amp and str(device).startswith("cuda")) else contextlib.nullcontext())
    with ac:
        out = model(pixel_values=pv)
    hs = out.last_hidden_state[:, n_prefix:, :].float()                                    # (B, h*w, D)
    B, N, D = hs.shape
    if N != h * w:
        raise ValueError(f"got {N} patch tokens, expected {h * w} (output_px {output_px}, patch {patch}, "
                         f"n_prefix {n_prefix}) — check num_register_tokens")
    grids = F.normalize(hs.reshape(B, h, w, D), dim=-1)
    return [g.cpu() for g in grids]
