"""Build an OFFLINE compact prefilter index from an existing rerank store. No DINO rerun; the 65 GB
store is opened read-only and never modified.

For every stored (position, level) grid it pools the (h,w,D) token grid to compact descriptors:
  mean  -> (D,)         L2-poolable global descriptor
  grid4 -> (4, 4, D)    coarse spatial (adaptive avg pool)
  grid8 -> (8, 8, D)    finer spatial
Descriptors are stored CONTIGUOUSLY (one dataset per pooling) keyed by the store's (crop_i, level),
so all crops of a query load in a single slice read. fp16. A fingerprint of the source px/py + levels
ties the index to its store so the survival evaluator can refuse a mismatched pair.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
      python -m patch_rerank.build_compact_index \
      --store $OUT/kup/rerank_store_kup_hybrid.h5 --pools mean grid4 grid8 \
      --out $OUT/kup/compact_index_kup.h5
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
_GRID = {"mean": None, "grid4": 4, "grid8": 8}          # None = global mean; int g = g x g adaptive pool


def _fingerprint(px, py, levels):
    h = hashlib.sha1()
    h.update(np.asarray(px, np.float64).tobytes())
    h.update(np.asarray(py, np.float64).tobytes())
    h.update(np.asarray(sorted(levels), np.float64).tobytes())
    return h.hexdigest()


def _pool(grid_thwd, spec):
    """grid_thwd: torch (h,w,D). spec=None -> (D,) mean; spec=g -> (g,g,D) adaptive-avg."""
    import torch
    import torch.nn.functional as F
    if spec is None:
        return grid_thwd.reshape(-1, grid_thwd.shape[-1]).mean(0)          # (D,)
    x = grid_thwd.permute(2, 0, 1).unsqueeze(0).float()                    # (1,D,h,w)
    p = F.adaptive_avg_pool2d(x, (spec, spec))[0]                          # (D,g,g)
    return p.permute(1, 2, 0).contiguous()                                # (g,g,D)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", nargs="+", default=["mean", "grid4", "grid8"],
                    help=f"any of {list(_GRID)}")
    ap.add_argument("--levels-m", type=float, nargs="+", default=None, help="subset of store levels (default all)")
    ap.add_argument("--device", default="cpu", help="pooling device (cpu is fine; grids are small)")
    args = ap.parse_args(argv)

    import h5py
    import torch
    from tqdm import tqdm

    bad = [p for p in args.pools if p not in _GRID]
    if bad:
        raise SystemExit(f"unknown pools {bad}; choose from {list(_GRID)}")

    sf = h5py.File(Path(args.store).expanduser(), "r")
    px = np.asarray(sf["px"][:], float); py = np.asarray(sf["py"][:], float)
    levels_all = [float(x) for x in sf.attrs["levels_m"]]
    levels = [L for L in (args.levels_m or levels_all) if L in levels_all]
    n_pos = len(px)
    D = int(sf["p0/l1000"].shape[-1]) if "p0/l1000" in sf else int(sf[f"p0/l{int(levels[0])}"].shape[-1])
    crops = [(i, int(L)) for i in range(n_pos) for L in levels if f"p{i}/l{int(L)}" in sf]
    M = len(crops)
    print(f"[build] store={args.store} positions={n_pos} levels={[int(x) for x in levels]} "
          f"crops={M} D={D} pools={args.pools}", flush=True)

    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    fout = h5py.File(out, "w")
    fout.create_dataset("crop_i", data=np.array([c[0] for c in crops], np.int32))
    fout.create_dataset("crop_level", data=np.array([c[1] for c in crops], np.int32))
    fout.create_dataset("px", data=px); fout.create_dataset("py", data=py)      # for store<->index mapping
    dsets = {}
    for p in args.pools:
        shape = (M, D) if _GRID[p] is None else (M, _GRID[p], _GRID[p], D)
        dsets[p] = fout.create_dataset(p, shape=shape, dtype=np.float16, chunks=True)
    fout.attrs.update({"schema_version": SCHEMA_VERSION, "source_store": str(args.store),
                       "source_fingerprint": _fingerprint(px, py, levels_all),
                       "pools": list(args.pools), "dtype": "float16", "D": D,
                       "levels_m": [float(x) for x in levels], "n_positions": n_pos, "n_crops": M})

    dev = args.device
    for r, (i, L) in enumerate(tqdm(crops, desc="compact", unit="crop")):
        g = torch.from_numpy(np.asarray(sf[f"p{i}/l{int(L)}"]).astype(np.float32)).to(dev)
        for p in args.pools:
            dsets[p][r] = _pool(g, _GRID[p]).half().cpu().numpy()

    fout.close(); sf.close()
    print(f"[ok] compact index -> {out}  ({M} crops x {args.pools})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
