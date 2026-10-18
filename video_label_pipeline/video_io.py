"""Thin OpenCV video decoding helpers."""
from __future__ import annotations

from pathlib import Path

import cv2


def probe(path) -> tuple[float, int]:
    """Return ``(fps, n_frames)`` without decoding pixels."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def read_all_frames(path, every: int = 1):
    """Decode a video.

    Returns ``(fps, n_frames_total, kept)`` where ``kept`` is a list of
    ``(frame_idx, timestamp_s, bgr)``. ``frame_idx`` is the original (full-rate)
    index even when ``every`` > 1, and ``n_frames_total`` counts every decoded
    frame. For faithful content-adaptive segmentation keep ``every == 1`` (the
    algorithm assumes adjacent frames).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    idx = 0
    kept = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if every <= 1 or idx % every == 0:
            ts = idx / fps if fps > 0 else 0.0
            kept.append((idx, ts, frame))
        idx += 1
    cap.release()
    return fps, idx, kept
