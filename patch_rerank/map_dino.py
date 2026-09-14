"""Fresh DINO extraction for map crops, in the SAME feature space as the query/gallery H5.

Reads the query H5's ``backbone``/``patch_size`` attrs and the stored descriptor dim D, rebuilds the
same DINOv2 extractor via ``map_extract`` and applies a RandomProjector ONLY if the backbone's native
dim != D (i.e. the H5 was projected). If D == native dim (e.g. dinov2_vitl14 → 1024), NO projector is
used — so crops match the H5 raw tokens exactly.
"""
from __future__ import annotations

import numpy as np
import torch

_BACKBONE_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024,
                 "dinov2_vitg14": 1536}


def build_matched_extractor(query_h5: str, device: str = "cuda", proj_seed: int = 0):
    """→ (extractor, projector_or_None, patch_size, backbone, D). projector present iff H5 was projected."""
    import h5py
    from map_extract.dino import build_dino_extractor, RandomProjector
    with h5py.File(query_h5, "r") as f:
        backbone = str(f.attrs.get("backbone", "dinov2_vitg14"))
        patch = int(f.attrs.get("patch_size", 14))
        D = None
        for k in f.keys():
            if hasattr(f[k], "keys") and "ift_dino" in f[k]:
                D = int(np.asarray(f[k]["ift_dino"]).shape[0]); break
    ext = build_dino_extractor(backbone)
    ext.to(device) if hasattr(ext, "to") else None
    native = _BACKBONE_DIM.get(backbone)
    proj = None
    if native is not None and D is not None and native != D:      # H5 was projected → match it
        proj = RandomProjector(D, seed=proj_seed)
    return ext, proj, patch, backbone, D


@torch.no_grad()
def extract_grids(images, extractor, projector, patch_size, device, amp=True):
    """List of RGB (H,W,3) uint8 → list of token grids (h,w,D) torch (L2-normalised, same as gallery)."""
    from map_extract.extract import _tile_to_batch, _extract_level_features
    batch, h_r, w_r = _tile_to_batch(images, device, patch_size)
    feats = _extract_level_features(extractor, batch, h_r, w_r, projector, amp=amp)  # list of (1,D,h,w)
    out = []
    for a in feats:
        D = a.shape[1]
        g = torch.from_numpy(a[0].reshape(D, h_r * w_r).T.reshape(h_r, w_r, D).astype(np.float32))
        out.append(g)
    return out
