"""Sky segmentation for query drone frames (ONNX skyseg) — vendored, optional.

Only used by the ``--images`` (query) path with ``--segment_sky``. onnxruntime is a lazy
import; the model (skyseg.onnx) auto-downloads on first use to ``~/.cache/skyseg/``
(override via ``$SKYSEG_ONNX``). Map (``--tif``) extraction never imports this.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

MODEL_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
_INPUT_SIZE = 320


def _default_model_path() -> Path:
    env = os.environ.get("SKYSEG_ONNX")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "skyseg" / "skyseg.onnx"


class SkySegmenter:
    """RGB frame → per-pixel sky mask using the skyseg ONNX model."""

    def __init__(self, model_path=None, providers=None) -> None:
        import onnxruntime as ort  # lazy: only when sky filtering is used

        self.model_path = Path(model_path) if model_path else _default_model_path()
        if not self.model_path.is_file():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[sky] downloading skyseg.onnx → {self.model_path} …")
            urlretrieve(MODEL_URL, self.model_path)

        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def segment(self, image_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """(H, W) uint8 mask where 255 marks sky. ``image_rgb`` is (H, W, 3) RGB."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {image_rgb.shape}")

        h, w = image_rgb.shape[:2]
        inp = cv2.resize(image_rgb, (_INPUT_SIZE, _INPUT_SIZE)).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]

        out = self.session.run(None, {self.input_name: inp})[0]
        result = np.squeeze(out).astype(np.float32)
        result = cv2.resize(result, (w, h))
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        return np.where(result > threshold, 255, 0).astype(np.uint8)


def sky_keep_mask(sky_mask: np.ndarray, patch_grid_hw, sky_patch_threshold: float = 0.5):
    """Boolean keep-mask over the patch grid (row-major); True = keep (ground)."""
    patch_h, patch_w = patch_grid_hw
    sky = (sky_mask > 0).astype(np.float32)
    ratio = cv2.resize(sky, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    return (ratio <= sky_patch_threshold).reshape(-1)
