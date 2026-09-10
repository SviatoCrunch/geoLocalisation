"""Stable fingerprint of a materialized positive-set (rule + params + inputs).

Same strategy + params + inputs -> same digest; any change to the rule, a parameter,
or the resulting associations changes it. Never uses Python ``hash()`` (salted).
Geometry is rounded so float noise does not flip the digest across machines.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

from .models import GalleryIndex


def compute_fingerprint(strategy_name: str, strategy_version: str,
                        resolved_params: Mapping, point_to_tile_ids: Mapping,
                        gallery: GalleryIndex) -> str:
    referenced = sorted({t for tiles in point_to_tile_ids.values() for t in tiles})
    payload = {
        "strategy": {"name": strategy_name, "version": strategy_version,
                     "params": _jsonable(resolved_params)},
        "crs": gallery.crs,
        "assoc": {pid: sorted(point_to_tile_ids[pid]) for pid in sorted(point_to_tile_ids)},
        "tiles": {tid: _tile_sig(gallery.get(tid)) for tid in referenced},
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode())
    return h.hexdigest()


def _tile_sig(t) -> list:
    return [t.city, round(t.center_x, 3), round(t.center_y, 3),
            round(t.lat, 7), round(t.size_m, 3) if t.size_m == t.size_m else None]


def _jsonable(obj):
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
