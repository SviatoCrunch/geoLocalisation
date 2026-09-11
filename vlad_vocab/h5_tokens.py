"""Token reading + sampling from group-per-tile DINO galleries (map_extract output).

Ported from RevisitAnything's ``build_multicity_vlad`` / ``fit_vlad_vocab`` token readers so
the normalisation + group ordering are byte-identical to how the VLAD gallery is built. numpy
at import; torch/h5py used inside functions (cheap import for tests).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CITY_PREFIX = re.compile(r"^map_dinov2_(?:\d+_)?")   # old "..._1000_" or new "..._<city>_s250m_d1024"
_CITY_SUFFIX = re.compile(r"_fp16$")
_STRIP = re.compile(r"_s\d+m(_d\d+)?$")           # strip the new "_s250m_d1024" tail too


def city_from_stem(stem: str) -> str:
    s = _CITY_SUFFIX.sub("", _CITY_PREFIX.sub("", stem))
    return _STRIP.sub("", s) or stem


def tile_groups(f):
    """Group keys holding ift_dino, ordered by ``tile_index`` attr when present (else key)."""
    keys = [k for k in f.keys() if hasattr(f[k], "keys") and "ift_dino" in f[k]]

    def _order(k):
        ti = f[k].attrs.get("tile_index")
        return (0, int(ti)) if ti is not None else (1, k)
    return sorted(keys, key=_order)


def tokens_from_ift(ds):
    """``ift_dino`` (1, D, H, W) or (D, H, W) -> ((H*W, D) float32, D)."""
    arr = np.asarray(ds)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    D, H, W = arr.shape
    return arr.reshape(D, H * W).T.astype(np.float32), int(D)


def sample_tokens(h5_paths, per_tile: int, max_tokens: int, seed: int):
    """Up to ``max_tokens`` L2-normalised tokens, ~``per_tile`` per tile (one read pass).

    Returns ``(X (n, D) float tensor, n_tiles, D)``. Normalisation matches the VLAD build.
    """
    import h5py
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(int(seed))
    bufs, n_tiles, D = [], 0, None
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            for k in tile_groups(f):
                toks_np, d = tokens_from_ift(f[k]["ift_dino"])
                D = d if D is None else D
                if d != D:
                    raise SystemExit(f"{Path(p).name}/{k}: token dim {d} != {D} (mixed backbones?)")
                t = torch.from_numpy(toks_np).float()
                idx = torch.randperm(t.shape[0], generator=g)[:min(per_tile, t.shape[0])]
                bufs.append(F.normalize(t[idx], dim=1))
                n_tiles += 1
    if not bufs:
        raise SystemExit("no tiles with ift_dino found in the input galleries")
    X = torch.cat(bufs, dim=0)
    if X.shape[0] > max_tokens:
        sel = torch.randperm(X.shape[0], generator=g)[:max_tokens]
        X = X[sel]
    return X, n_tiles, int(D)
