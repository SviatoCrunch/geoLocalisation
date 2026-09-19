"""Coarse→fine bridge: per-query top-K city cells → the dense stride-250 tiles they cover.

Stage 1 (coarse) = this e2c/supervlad checkpoint ranks the NON-OVERLAPPING 1000 m checkerboard
cells of the query's OWN city; we keep the top-K (default 70, the measured recall knee). Stage 2
(fine) needs finer candidates, so each kept cell is expanded to the DENSE stride-250 tiles whose
centre falls inside it (± an optional border to cover cell boundaries). The union of those dense
tiles is the per-query candidate set a weighted-pyramid / patch-RANSAC reranker consumes.

Output JSON: ``{query_id: {"cells": [cell_id...], "dense": [dense_tile_id...]}}`` + a summary
(coarse recall@K = fraction of queries whose GT cell is in the top-K, and dense-candidate counts).

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with shapely \
           --with pyyaml --with tqdm python -m geo_e2c_train.export_shortlist \
      --ckpt /…/run_supervlad_cell_w1/best.pt \
      --split-config /…/split_checker1000.yaml --split-json /…/split_checker1000/split.json \
      --galleries kramatorsc=/…/map_dinov2_kramatorsc_s250m_d1024.h5 kup=/…/… liman_day=/…/… \
      --queries   kramatorsc=/…/query_kramatorsc_d1024.h5 kup=/…/… liman_day=/…/… \
      --dense-index /…/dict/tiles_index_dense.h5 \
      --assign /…/dict/k32.pt --k 70 --which val test --out /…/shortlist_k70.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

_COS = 0.657                                   # EPSG:3857 → true metres at ~48.9°


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def dense_within_cell(cell_cx, cell_cy, dense_cx, dense_cy, reach_grid) -> np.ndarray:
    """Boolean mask of dense tiles whose centre lies within a cell (± reach), axis-aligned.

    ``reach_grid`` = (cell_half + border) in EPSG:3857 grid units. Pure geometry (no city
    filtering here — the caller pre-restricts to the cell's city)."""
    return (np.abs(dense_cx - cell_cx) <= reach_grid) & (np.abs(dense_cy - cell_cy) <= reach_grid)


def gather_dense(cell_rows, cell_xy, dense_xy, dense_city, cell_city_of_row, reach_grid) -> list:
    """Union (sorted) of dense-tile indices covered by the given cell rows (same-city per cell)."""
    keep = np.zeros(dense_xy.shape[0], bool)
    for r in cell_rows:
        cc = cell_city_of_row[r]
        same = dense_city == cc
        m = dense_within_cell(cell_xy[r, 0], cell_xy[r, 1], dense_xy[:, 0], dense_xy[:, 1], reach_grid)
        keep |= (m & same)
    return sorted(int(i) for i in np.nonzero(keep)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-config", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--galleries", nargs="+", required=True, help="city=dense d1024 H5 (coarse V source)")
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--dense-index", default=None,
                    help="tiles_index H5 of the FULL dense stride-250 gallery (build_index, no "
                         "--grid-snap-m). OMIT to emit a cells-only shortlist (dense=[]): the fine "
                         "reranker then slides fresh DINO within the top-K cells straight off the "
                         "GeoTIFF, so no dense (overlapping) gallery is needed at all.")
    ap.add_argument("--assign", default=None)
    ap.add_argument("--k", type=int, default=70, help="shortlist size (top-K coarse cells)")
    ap.add_argument("--border-m", type=float, default=0.0,
                    help="expand each cell by this many TRUE metres (cover boundary dense tiles)")
    ap.add_argument("--which", nargs="+", default=["val", "test"])
    ap.add_argument("--query-size-m", type=float, default=1000.0)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from geo_train_batching.adapters.split_relevance import build_split_relevance
    from geo_split_no_overlap.positive_selection import build_gallery_index
    from .data import TileGridLoader, QueryTokenStore
    from .eval import build_gallery_V

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    rc = ck.get("resolved_config", {})
    keep = {f.name for f in fields(E2cModelConfig)}
    cfg_kwargs = {k: rc[k] for k in rc if k in keep}
    if args.assign:
        cfg_kwargs["assign_path"] = args.assign
    cfg = E2cModelConfig(**cfg_kwargs)
    model = build_e2c_model(cfg, device=args.device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[ckpt] epoch {ck.get('epoch')} | agg={cfg.agg} pyramid={cfg.pyramid_mode}", flush=True)

    galleries = dict(_kv(a) for a in args.galleries)
    queries = dict(_kv(a) for a in args.queries)
    tiles = TileGridLoader(galleries, grid_size=None, cache=False)
    store = QueryTokenStore(queries)

    # dense geometry (full stride-250 gallery) — centres in the same EPSG:3857 grid as the cells.
    # Optional: without it we emit a cells-only shortlist and the fine reranker slides fresh DINO
    # within the top-K cells off the GeoTIFF (no dense/overlapping gallery needed).
    if args.dense_index:
        dense = build_gallery_index(args.dense_index, "EPSG:3857", args.tile_size_m)
        dtiles = dense.all_tiles()
        dense_ids = [t.tile_id for t in dtiles]
        dense_xy = np.array([[t.center_x, t.center_y] for t in dtiles], float).reshape(-1, 2)
        dense_city = np.array([t.city for t in dtiles])
        reach_grid = (args.tile_size_m / 2.0 + args.border_m) / _COS  # true metres → grid units
    else:
        dense_ids, dense_xy, dense_city, reach_grid = [], np.zeros((0, 2)), np.array([]), 0.0
        print("[shortlist] no --dense-index -> cells-only shortlist (dense=[]); "
              "fine rerank = fresh DINO within top-K cells", flush=True)

    galV = None
    out = {"meta": {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "k": args.k,
                    "border_m": args.border_m, "n_dense": len(dense_ids)},
           "shortlist": {}}
    split_data = json.loads(Path(args.split_json).expanduser().read_text(encoding="utf-8"))
    for w in args.which:
        sr = build_split_relevance(args.split_config, args.split_json, w,
                                   query_size_m=args.query_size_m)
        if galV is None:
            galV = build_gallery_V(model, tiles, sr.tile_ids, args.eval_chunk, args.device, progress=True)
        cell_city_of_row = np.array([str(t).split(":", 1)[0] for t in sr.tile_ids])
        pos_row = {q: i for i, q in enumerate(sr.query_ids)}          # only queries WITH a positive cell
        # EVERY split query we have tokens for gets a shortlist — a query without a positive checkerboard
        # cell (strict rule / near a gap) is still scored and localized; positives feed only the recall
        # diagnostic below, so a geometry-rule mismatch no longer silently drops frames.
        qids = [q for q in split_data.get(w, []) if store.has(q)]
        no_token = [q for q in split_data.get(w, []) if not store.has(q)]
        Q = torch.stack([model.encode_query(store.tokens(q).to(args.device)) for q in qids])
        S = model.score(Q, galV, tile_chunk=args.eval_chunk)         # (B, cells)
        qcity = np.array([str(q).split(":", 1)[0] for q in qids])
        allowed = torch.from_numpy(qcity[:, None] == cell_city_of_row[None, :]).to(S.device)
        S = S.masked_fill(~allowed, float("-inf"))
        order = S.argsort(dim=1, descending=True).cpu().numpy()

        hit, npos, counts = 0, 0, []
        for bi, q in enumerate(qids):
            topk = [int(r) for r in order[bi][:args.k]]
            pos = set(int(x) for x in sr.relevance.pos_of(pos_row[q])) if q in pos_row else set()
            if pos:                                               # recall only over queries with a GT cell
                npos += 1
                hit += int(any(r in pos for r in topk))          # coarse recall@K (GT cell in shortlist)
            dense_rows = (gather_dense(topk, sr.tile_xy, dense_xy, dense_city, cell_city_of_row,
                                       reach_grid) if dense_ids else [])
            counts.append(len(dense_rows))
            out["shortlist"][q] = {"cells": [sr.tile_ids[r] for r in topk],
                                   "dense": [dense_ids[r] for r in dense_rows]}
        cnt = np.array(counts) if counts else np.array([0])
        n = len(qids)
        print(f"[{w}] n={n} (with_pos={npos}, no_token={len(no_token)}) "
              f"coarse_recall@{args.k}={hit/npos if npos else 0:.3f} "
              f"dense/query median={int(np.median(cnt))} min={int(cnt.min())} max={int(cnt.max())}",
              flush=True)
        if no_token:
            print(f"[{w}] no tokens in query store for {len(no_token)}: {no_token[:5]}"
                  f"{' …' if len(no_token) > 5 else ''}", flush=True)
        out["meta"].setdefault("per_split", {})[w] = {
            "n": n, "with_pos": npos, "no_token": len(no_token),
            "coarse_recall_at_k": (hit / npos if npos else 0.0),
            "dense_per_query_median": int(np.median(cnt))}

    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] shortlist -> {args.out}", flush=True)
    tiles.close(); store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
