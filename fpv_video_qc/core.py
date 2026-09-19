"""Per-frame corruption metrics + segment grouping for analog FPV video.

The heavy signal (`line_noise_from_gray`) and the segment grouping are pure numpy / stdlib so they are
unit-testable without cv2; the full `frame_score` uses cv2 for gray/HSV conversion in the video loop.
"""
from __future__ import annotations

import numpy as np

FRAME_KEYS = ("tdiff", "line_noise", "chroma_impulse", "lap_var", "score")


def line_noise_from_gray(gray: np.ndarray) -> float:
    """Fraction of rows that look like RF-breakup / torn-or-garbage scanlines.

    A scanline is flagged when its mean absolute horizontal-neighbour difference (row "texture") is a
    robust outlier vs the frame's median row AND exceeds an absolute floor — so a clean smooth frame
    (uniformly low row texture) scores ~0, while rows of random RF garbage / tearing score high.
    """
    g = gray.astype(np.float32)
    if g.ndim != 2 or g.shape[0] < 3:
        return 0.0
    row_tex = np.abs(np.diff(g, axis=1)).mean(axis=1)        # (H,) horizontal texture per row
    med = float(np.median(row_tex))
    mad = float(np.median(np.abs(row_tex - med))) + 1e-6
    bad = (row_tex > med + 6.0 * mad) & (row_tex > 15.0)     # robust outlier AND absolute garbage
    return float(bad.mean())


def combine_score(tdiff: float, line_noise: float, chroma_impulse: float) -> float:
    """Single 0..~1 corruption score. line_noise (breakup rows) dominates; a temporal spike or heavy
    impulse chroma also lifts it. tdiff is normalised by 255 and capped."""
    return float(line_noise + 0.5 * min(tdiff / 255.0, 1.0) + 0.3 * chroma_impulse)


def frame_score(bgr: np.ndarray, prev_gray: np.ndarray | None):
    """(metrics dict, gray) for one BGR frame. `prev_gray` is the previous frame's small-gray (or None).

    Metrics: tdiff (mean|Δ| vs prev, 0..255), line_noise (0..1), chroma_impulse (0..1 frac of
    over-saturated specks), lap_var (Laplacian variance = detail/noise), score (combined).
    """
    import cv2
    small = cv2.resize(bgr, (256, 144), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    tdiff = 0.0 if prev_gray is None else float(np.abs(gray.astype(np.float32)
                                                       - prev_gray.astype(np.float32)).mean())
    line_noise = line_noise_from_gray(gray)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    chroma_impulse = float(((s > 230) & (v > 40)).mean())    # over-saturated impulse specks
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    score = combine_score(tdiff, line_noise, chroma_impulse)
    return {"tdiff": round(tdiff, 3), "line_noise": round(line_noise, 4),
            "chroma_impulse": round(chroma_impulse, 4), "lap_var": round(lap_var, 1),
            "score": round(score, 4)}, gray


def bad_segments(flags, min_len: int = 2, merge_gap: int = 3):
    """Group a per-frame bad/good boolean list into corrupted segments.

    Consecutive bad frames form a run; runs separated by <= `merge_gap` good frames are merged; runs
    shorter than `min_len` frames are dropped (isolated single-frame blips are ignored). Returns
    ``[(start, end), ...]`` inclusive frame indices.
    """
    runs = []
    start = None
    for i, b in enumerate(list(flags) + [False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            runs.append([start, i - 1])
            start = None
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= min_len]
