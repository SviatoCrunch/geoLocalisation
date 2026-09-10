"""Frozen DINOv2 feature extractor (SERVER-ONLY, heavy).

Implements :class:`FeatureExtractor` using a frozen DINOv2 backbone (torch.hub). Returns
patch tokens as an (h,w,D) grid + a global descriptor (L2-normalised mean-pool — a simple
stand-in for AnyLoc VLAD; swap in a VLAD aggregator later). Not exercised by the unit
tests (they use a fake extractor); this is the real adapter for the server.

NOTE: this is the ONLY heavy/external dependency in the package and it is optional —
importing ``footprint_assoc`` does not import torch.
"""
from __future__ import annotations

import numpy as np


class DinoV2Extractor:
    def __init__(self, model_name: str = "dinov2_vitg14", device: str = "cuda",
                 image_size: int = 518):
        import torch
        self._torch = torch
        self.device = device
        self.image_size = int(image_size)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(device)
        self.patch = 14
        for p in self.model.parameters():
            p.requires_grad_(False)

    def extract(self, image: np.ndarray):
        torch = self._torch
        import torch.nn.functional as F
        x = torch.from_numpy(np.ascontiguousarray(image)).float()
        if x.ndim == 3:
            x = x.permute(2, 0, 1)
        x = x.unsqueeze(0) / 255.0
        s = self.image_size
        x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False).to(self.device)
        with torch.no_grad():
            out = self.model.forward_features(x)
            tok = out["x_norm_patchtokens"][0]                 # (N, D)
        g = s // self.patch
        D = tok.shape[-1]
        grid = tok.reshape(g, g, D).float().cpu().numpy()
        glob = tok.mean(0)
        glob = (glob / glob.norm().clamp_min(1e-12)).float().cpu().numpy()
        return glob, grid
