"""Dense optical flow backends for calibration — lazy heavy imports.

For each adjacent frame pair (a, b) we return BOTH directions:
    fab = flow a->b (displacement of a's pixels),  used to march labels b->a
    fba = flow b->a,                               used to march labels a->b
so ``propagate.chain_warp`` (which warps a label map into the *destination* grid
via the destination->source flow) can propagate a keyframe either forward or
backward through the sequence from a single per-pair cache.

Backends:
    "farneback" — cv2.calcOpticalFlowFarneback, CPU, always available (default)
    "raft"      — torchvision RAFT-Large, GPU-accelerated, higher quality

Both return float32 (H,W,2) = (dx, dy) in pixels, sampled on the full grid.
"""
from __future__ import annotations

import numpy as np


# ------------------------------- Farneback (cv2, CPU) -------------------------------

def _farneback(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    ).astype(np.float32)


def pair_flows_farneback(gray_a: np.ndarray, gray_b: np.ndarray):
    """(fab, fba) for one adjacent pair, both (H,W,2) float32."""
    return _farneback(gray_a, gray_b), _farneback(gray_b, gray_a)


# ------------------------------- RAFT (torchvision, GPU) -------------------------------

_RAFT = {"model": None, "device": None}


def _raft_model(device: str):
    """Cache the RAFT-Large model + weights on first use."""
    if _RAFT["model"] is None or _RAFT["device"] != device:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).eval().to(device)
        _RAFT["model"] = (model, weights.transforms())
        _RAFT["device"] = device
    return _RAFT["model"]


def _raft_prep(gray_a, gray_b, transforms, device):
    """Two grayscale (H,W) uint8 frames -> the RAFT-normalised (1,3,H8,W8) batch
    pair, resized so H,W are multiples of 8 (a RAFT requirement)."""
    import torch
    h, w = gray_a.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8

    def to_batch(g):
        t = torch.from_numpy(g).float()[None, None].repeat(1, 3, 1, 1) / 255.0  # (1,3,H,W)
        return torch.nn.functional.interpolate(t, size=(h8, w8), mode="bilinear",
                                               align_corners=False)

    a, b = transforms(to_batch(gray_a), to_batch(gray_b))
    return a.to(device), b.to(device), (h, w), (h8, w8)


def _raft_one(gray_a, gray_b, model, transforms, device):
    import torch
    a, b, (h, w), (h8, w8) = _raft_prep(gray_a, gray_b, transforms, device)
    with torch.no_grad():
        flow = model(a, b)[-1]                       # (1,2,H8,W8) forward a->b
    flow = torch.nn.functional.interpolate(flow, size=(h, w), mode="bilinear",
                                           align_corners=False)
    flow[:, 0] *= w / w8                             # rescale dx,dy to full-res pixels
    flow[:, 1] *= h / h8
    return flow[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H,W,2)


def pair_flows_raft(gray_a: np.ndarray, gray_b: np.ndarray, device: str | None = None):
    """(fab, fba) via RAFT-Large. ``device`` defaults to cuda if available."""
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transforms = _raft_model(device)
    return (_raft_one(gray_a, gray_b, model, transforms, device),
            _raft_one(gray_b, gray_a, model, transforms, device))


def pair_flows(gray_a: np.ndarray, gray_b: np.ndarray, backend: str = "farneback",
               device: str | None = None):
    """Dispatch to the chosen backend. Returns (fab, fba)."""
    if backend == "farneback":
        return pair_flows_farneback(gray_a, gray_b)
    if backend == "raft":
        return pair_flows_raft(gray_a, gray_b, device)
    raise ValueError(f"unknown flow backend: {backend!r}")
