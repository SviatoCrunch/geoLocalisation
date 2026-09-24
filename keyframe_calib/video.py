"""Video frame ingestion for calibration — lazy cv2, sequential decode.

Lets the calibrator take a VIDEO FILE directly instead of a pre-extracted frames
folder: the dense GT masks define which frame indices belong to the calibration
segment, and these helpers pull exactly those frames (as grayscale) from the video.
``extract_segment`` dumps a contiguous run of frames to PNG so you can label them.
"""
from __future__ import annotations

from pathlib import Path


def read_frames_gray(video_path, indices, downscale: int = 1):
    """Sequentially decode ``video_path`` and return {index: grayscale (H,W) uint8}
    for the requested frame ``indices`` (0-based). One pass; stops after the largest
    wanted index. Raises if any requested index is missing from the stream."""
    import cv2

    wanted = set(int(i) for i in indices)
    if not wanted:
        return {}
    maxw = max(wanted)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    out: dict[int, "any"] = {}
    idx = -1
    try:
        while idx < maxw:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx not in wanted:
                continue
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if downscale > 1:
                g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale),
                               interpolation=cv2.INTER_AREA)
            out[idx] = g
    finally:
        cap.release()
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"video ended before frames {sorted(missing)} (has 0..{idx})")
    return out


def video_fps(video_path) -> float:
    """Frames-per-second of the video (0.0 if unknown) — for reporting delta in seconds."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def extract_segment(video_path, start: int, count: int, out_dir, downscale: int = 1,
                    stride: int = 1, log=print) -> list[int]:
    """Dump a contiguous run of frames to ``out_dir`` as ``{index:06d}.png`` so you can
    densely label them for calibration. Returns the written frame indices."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(start, start + count, stride))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    wanted = set(indices)
    maxw = max(indices) if indices else -1
    written = []
    idx = -1
    try:
        while idx < maxw:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx not in wanted:
                continue
            if downscale > 1:
                frame = cv2.resize(frame, (frame.shape[1] // downscale, frame.shape[0] // downscale),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{idx:06d}.png"), frame)
            written.append(idx)
    finally:
        cap.release()
    log(f"extracted {len(written)} frames [{written[0] if written else '-'}.."
        f"{written[-1] if written else '-'}] -> {out_dir}")
    return written
