"""§11 split / domain-shift audit — spatial split by city + extent (photometric = not_available)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .common import write_json


def _spatial(sr):
    xy = sr.q_xy
    cities = Counter(q.split(":", 1)[0] for q in sr.query_ids)
    return {"n": len(sr.query_ids), "per_city": dict(cities),
            "centroid_xy": (xy.mean(0).tolist() if len(xy) else None),
            "std_xy": (xy.std(0).tolist() if len(xy) else None),
            "bbox_xy": ([float(xy[:, 0].min()), float(xy[:, 0].max()),
                         float(xy[:, 1].min()), float(xy[:, 1].max())] if len(xy) else None)}


def run(ctx) -> dict:
    out = {"train": _spatial(ctx.sr["train"]), "val": _spatial(ctx.sr["val"]),
           "features": {k: "not_available" for k in
                        ("brightness", "blur", "sky_fraction", "scene_type", "fov_height",
                         "camera_source", "time_season")},
           "note": "query H5 stores DINO tokens (not images) → photometric/scene features are "
                   "not_available here; spatial split (per-city counts + extent) is reported. "
                   "No new heavy model added just for this audit."}
    write_json(Path(ctx.args.output_dir) / "split_shift_audit.json", out)
    return out
