"""neural — segmentation-based sky masker (SERVER-ONLY, heavy).

Runs a frozen semantic-segmentation model (SegFormer / UAVid family) and marks the sky
class as NOT-kept. Lazy torch/transformers import — importing sky_filter does not pull
them. Not exercised by unit tests (they inject a fake); this is the real adapter.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class NeuralSkyMasker:
    name = "neural"
    version = "1.0"

    def __init__(self, cfg, sky_class_id: int = 0):
        self.cfg = cfg
        self.sky_class_id = int(sky_class_id)
        self._model = None
        self._proc = None

    def _lazy(self):
        if self._model is None:
            import torch
            from transformers import (SegformerForSemanticSegmentation,
                                       SegformerImageProcessor)
            self._torch = torch
            self._proc = SegformerImageProcessor.from_pretrained(self.cfg.neural_model)
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self.cfg.neural_model).eval().to(self.cfg.device)

    def mask(self, frame_id: str, image) -> Optional[np.ndarray]:
        if image is None:
            return None
        self._lazy()
        torch = self._torch
        import torch.nn.functional as F
        inp = self._proc(images=image, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            logits = self._model(**inp).logits            # (1,C,h,w)
        H, W = np.asarray(image).shape[:2]
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        seg = up.argmax(1)[0].cpu().numpy()
        return seg != self.sky_class_id                   # True = keep ground
