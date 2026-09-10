"""Synthetic fixtures for sky_filter tests (no segmentation model needed)."""
from __future__ import annotations

import numpy as np


def keep_top_sky(H=8, W=8, sky_rows=4):
    """Pixel keep-mask: top ``sky_rows`` rows are sky (False), rest ground (True)."""
    keep = np.ones((H, W), bool)
    keep[:sky_rows, :] = False
    return keep


class FakeNeuralMasker:
    """Deterministic stand-in for the neural backend: top half = sky."""
    name = "neural"
    version = "fake"

    def mask(self, frame_id, image):
        h = np.asarray(image).shape[0] if image is not None else 8
        w = np.asarray(image).shape[1] if image is not None else 8
        return keep_top_sky(h, w, h // 2)
