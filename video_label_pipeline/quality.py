"""Quality + embedding backend.

The pipeline needs two things per frame: a quality label (good/usable/bad) and a
DINO embedding (for diversity selection). Both come from the same frozen DINOv2
backbone, so the default backend loads ONE ``quality_cls.QualityPredictor`` and
exposes ``score`` and ``embed``.

``quality_cls`` lives in the RevisitAnything repo (``quality-cls/src``), not in
geoLocalisation. It is imported lazily and must be importable at runtime — put
``quality-cls/src`` on ``PYTHONPATH`` (or install the package) and have its
``checkpoints/head.pt`` available. The pipeline itself only depends on the small
``Backend`` protocol below, so tests inject a fake and need neither torch nor the
checkpoint.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Backend(Protocol):
    """What the pipeline requires of a quality/embedding backend."""

    def score(self, frames_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
        """-> [(label, confidence), ...] aligned with ``frames_bgr``."""

    def embed(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """-> float array [N, D] of DINO embeddings aligned with ``frames_bgr``."""


class QualityClsBackend:
    """Default backend backed by ``quality_cls.QualityPredictor`` (one model load
    used for both scoring and embedding). Heavy imports happen in ``__init__``."""

    def __init__(self, checkpoint=None, config=None, device=None):
        from quality_cls import QualityPredictor  # lazy: torch + DINOv2
        self._p = QualityPredictor(checkpoint=checkpoint, config=config, device=device)

    @staticmethod
    def _to_rgb(frames_bgr):
        import cv2
        return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    def score(self, frames_bgr):
        if not frames_bgr:
            return []
        preds = self._p.predict_batch(self._to_rgb(frames_bgr))
        return [(pr.label, float(pr.confidence)) for pr in preds]

    def embed(self, frames_bgr):
        import torch
        if not frames_bgr:
            return np.zeros((0, 1), dtype=np.float32)
        tensors = [self._p._to_tensor(x) for x in self._to_rgb(frames_bgr)]
        bs = self._p.batch_size
        out = []
        for i in range(0, len(tensors), bs):
            batch = torch.stack(tensors[i:i + bs]).to(self._p.device)
            with torch.no_grad():
                out.append(self._p.model.embed(batch).cpu().numpy())
        return np.concatenate(out, axis=0)
