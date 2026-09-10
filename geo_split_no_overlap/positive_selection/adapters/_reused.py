"""Vendored geometry/frame helpers — byte-faithful copies of the four functions this
subsystem reused from ``make_multicity_split`` (``_merc``, ``_parse_stem``,
``_collect_frames``, ``_city_and_dir``). Copied here so ``geo_split_no_overlap`` has NO
external dependency on the RevisitAnything repo (self-isolated subfolder). Depends only
on numpy + stdlib. If the source rule ever changes, re-sync these verbatim.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_R = 6378137.0  # EPSG:3857 spherical Web-Mercator radius (m) — matches the tile grid


def _merc(lat, lon):
    x = _R * np.radians(lon)
    y = _R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def _parse_stem(stem: str):
    toks = re.split(r"[_,]", stem)
    nums = []
    for t in toks[1:]:
        try:
            nums.append(float(t))
        except ValueError:
            pass
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def _city_and_dir(arg: str):
    if "=" in arg:
        city, d = arg.split("=", 1)
        return city, Path(d).expanduser()
    d = Path(arg).expanduser()
    return re.sub(r"^gt_", "", d.name), d


def _collect_frames(root: Path):
    frames, seen = [], set()
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        ll = _parse_stem(p.stem)
        if ll is None:
            continue
        key = (round(ll[0], 7), round(ll[1], 7))
        if key in seen:
            continue
        seen.add(key)
        frames.append({"stem": p.stem, "lat": ll[0], "lon": ll[1], "path": str(p)})
    return frames
