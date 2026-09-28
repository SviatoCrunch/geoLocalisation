"""Class-balance weights from the train pixel histogram (build_report.json).

Severe imbalance here (background 76%, sky 19%, tower/bridge <0.07%), so the default is
**median-frequency balancing** (Eigen & Fergus 2015): ``w_c = median(freq)/freq_c``. That
keeps common classes near <1 and lifts rare ones without the explosive values plain
inverse-frequency gives. Weights are optionally clipped (rare thin classes like tower can
hit ~33x, which destabilises training) and the sampler (see dataset) handles the rest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import BACKGROUND_ID, CLASSES, NUM_CLASSES


def load_counts(report_path: str | Path) -> list[int]:
    """Pull the per-class train pixel counts out of build_report.json."""
    d = json.loads(Path(report_path).read_text(encoding="utf-8"))
    h = d["train_pixel_hist"]
    assert h["classes"] == CLASSES, f"class order mismatch: {h['classes']} != {CLASSES}"
    return list(h["counts"])


def compute_weights(counts, scheme: str = "median", clip: float | None = 15.0,
                    ignore_background: bool = False) -> list[float]:
    """Per-class loss weights.

    scheme: ``median`` (median-freq balancing, default) | ``inverse`` (1/freq, normalised to
    mean 1) | ``none`` (all ones). ``clip`` caps the max weight (None = no cap).
    ``ignore_background`` sets the background weight to 0 (use with background-as-class when
    you don't want to reward predicting background).
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    freq = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    present = counts > 0
    if scheme == "median":
        med = float(np.median(freq[present])) if present.any() else 0.0
        w = np.divide(med, freq, out=np.zeros_like(freq), where=present)
    elif scheme == "inverse":
        w = np.divide(1.0, freq, out=np.zeros_like(freq), where=present)
        w = w / (w[present].mean() if present.any() else 1.0)  # normalise to mean 1
    elif scheme == "none":
        w = np.ones(NUM_CLASSES)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    if clip is not None:
        w = np.minimum(w, clip)
    if ignore_background:
        w[BACKGROUND_ID] = 0.0
    return [float(x) for x in w]
