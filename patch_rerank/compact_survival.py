"""EVALUATION-ONLY compact-prefilter survival sweep. NO MAGSAC, NO full-grid reads, no production
integration: it only measures whether a cheap compact score would KEEP the crops the full baseline
picked, so we can choose a budget before wiring a prefilter into map_rerank.

Inputs:
  --compact-index  (build_compact_index.py output: mean/grid4/grid8 pooled descriptors per crop)
  --baseline       (map_rerank --dump-scores: {query: [[px,py,level,geom_score],...]}  = exact per-crop)
  --queries        city=query H5 (QueryGridStore) for the query token grid (pooled the same way)

For each query it computes a cheap score per baseline crop:
  mean   -> cosine(mean(query tokens), crop mean)
  grid4/8-> chamfer (mean over query tokens of max cosine to the crop's g*g tokens); or mnn count
ranks crops by the cheap score, selects a budget (global top-N, or diversity: cap levels/position +
positions/cell + min distinct cells via an optional planner), and reports SURVIVAL vs the baseline:
  - winner_pair   : baseline's single highest-geom-score crop kept?
  - top5_pair     : fraction of baseline's 5 highest crops kept
  - winner_pos    : baseline's best position (max Sum-over-levels) kept (any level)?
  - distinct cells/positions/levels retained after selection.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy python -m patch_rerank.compact_survival \
      --compact-index $OUT/kup/compact_index_kup.h5 --baseline $OUT/kup/baseline_scores.json \
      --queries kup=$OUT/kup/query_kup_d1024.h5 --representations mean grid4 grid8 \
      --grid-score chamfer --budgets 128 256 512 1024 --top-levels-per-position 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cheap_scores(rep, qtok, crop_desc, grid_score, chunk=256):
    """Cheap similarity of one query to n crops. qtok (Nq,D) L2-normed. crop_desc: (n,D) for mean, or
    (n,g*g,D) for grid. -> (n,) numpy scores. mean=cosine; grid=chamfer(mean-of-max) or mnn count."""
    import torch
    import torch.nn.functional as F
    dev = qtok.device
    if rep == "mean":
        c = F.normalize(crop_desc.float(), dim=1)                       # (n,D)
        qm = F.normalize(qtok.mean(0, keepdim=True), dim=1)             # (1,D)
        return (c @ qm.t()).squeeze(1).cpu().numpy()
    if crop_desc.dim() == 4:                                            # (n,g,g,D) -> (n,g*g,D)
        crop_desc = crop_desc.reshape(crop_desc.shape[0], -1, crop_desc.shape[-1])
    out = np.empty(crop_desc.shape[0], np.float32)
    for s in range(0, crop_desc.shape[0], chunk):
        c = F.normalize(crop_desc[s:s + chunk].float(), dim=2).to(dev)  # (b,g2,D)
        sims = torch.einsum("nd,bgd->bng", qtok, c)                     # (b,Nq,g2)
        if grid_score == "mnn":                                        # mutual-NN count / Nq
            q2r = sims.argmax(2); r2q = sims.argmax(1)                 # (b,Nq),(b,g2)
            back = torch.gather(r2q, 1, q2r)
            mutual = (back == torch.arange(qtok.shape[0], device=dev)).float().mean(1)
            out[s:s + chunk] = mutual.cpu().numpy()
        else:                                                          # chamfer: mean_i max_j cos
            out[s:s + chunk] = sims.max(2).values.mean(1).cpu().numpy()
    return out


def _select(order, crops, budget, top_levels_per_pos, pos_per_cell, min_cells, cell_of):
    """order: crop indices sorted best-first by cheap score. Returns selected crop-index list<=budget,
    honouring caps. cell_of[cropidx] = set of cells (for diversity), or None to skip cell caps."""
    per_pos, per_cell, cells_used, chosen = {}, {}, set(), []
    for ci in order:
        if len(chosen) >= budget:
            break
        px, py, L = crops[ci][0], crops[ci][1], crops[ci][2]
        pos = (px, py)
        if top_levels_per_pos and per_pos.get(pos, 0) >= top_levels_per_pos:
            continue
        cs = cell_of[ci] if cell_of is not None else None
        if cs is not None and pos_per_cell:
            # skip if EVERY cell of this position already hit its position cap AND we've met min_cells
            if all(per_cell.get(c, 0) >= pos_per_cell for c in cs) and len(cells_used) >= (min_cells or 0):
                continue
        chosen.append(ci)
        per_pos[pos] = per_pos.get(pos, 0) + 1
        if cs is not None:
            for c in cs:
                per_cell[c] = per_cell.get(c, 0) + 1
                cells_used.add(c)
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact-index", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--representations", nargs="+", default=["mean", "grid4", "grid8"])
    ap.add_argument("--grid-score", choices=["chamfer", "mnn"], default="chamfer")
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--top-levels-per-position", type=int, default=0, help="0 = no cap (global top-N)")
    ap.add_argument("--positions-per-cell", type=int, default=0, help="cell-diversity (needs --shortlist)")
    ap.add_argument("--min-cells", type=int, default=0)
    ap.add_argument("--shortlist", default=None, help="for cell diversity: map positions->cells")
    ap.add_argument("--dense-index", default=None)
    ap.add_argument("--k", type=int, default=70)
    ap.add_argument("--step-m", type=float, default=250.0)
    ap.add_argument("--search-radius-m", type=float, default=0.0)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import h5py
    import torch
    import torch.nn.functional as F
    from .query_io import QueryGridStore

    ix = h5py.File(Path(args.compact_index).expanduser(), "r")
    ci_pos = np.asarray(ix["crop_i"][:]); ci_lvl = np.asarray(ix["crop_level"][:])
    ipx = np.asarray(ix["px"][:], float); ipy = np.asarray(ix["py"][:], float)
    pos_i = {(round(float(ipx[i]), 2), round(float(ipy[i]), 2)): i for i in range(len(ipx))}
    row_of = {(int(ci_pos[r]), int(ci_lvl[r])): r for r in range(len(ci_pos))}
    reps_avail = [p for p in args.representations if p in ix]
    print(f"[survival] index={args.compact_index} crops={len(ci_pos)} reps={reps_avail} "
          f"grid_score={args.grid_score} budgets={args.budgets}", flush=True)
    # load each representation fully into RAM ONCE — per-query h5py fancy-indexing (list of rows) is
    # pathologically slow on the big grid8 dataset; numpy gather from a resident array is instant.
    rep_arr = {}
    for rep in reps_avail:
        rep_arr[rep] = np.asarray(ix[rep][:])
        print(f"    loaded {rep} {rep_arr[rep].shape} {rep_arr[rep].nbytes / 1e9:.2f} GB into RAM", flush=True)

    baseline = json.loads(Path(args.baseline).expanduser().read_text())
    qstore = QueryGridStore(dict(a.split("=", 1) for a in args.queries))
    dev = args.device

    # optional planner for cell diversity / distinct-cell metrics
    cellmap = None
    if args.shortlist and args.dense_index:
        from .map_rerank import _plan_query
        with h5py.File(Path(args.dense_index).expanduser(), "r") as f:
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]]
            lat = np.asarray(f["lat"][:], float); lon = np.asarray(f["lon"][:], float)
        drow = {t: i for i, t in enumerate(ids)}
        radius = args.tile_size_m / 2.0
        if args.search_radius_m:
            radius = min(args.search_radius_m, args.tile_size_m / 2.0)
        sj = json.loads(Path(args.shortlist).expanduser().read_text())["shortlist"]
        cellmap = {}                                     # query -> {(px,py): set(cell_ids)}
        for q, entry in sj.items():
            pl = _plan_query(entry, args.k, drow, lat, lon, args.step_m, radius)
            if pl is None:
                continue
            uniq, cell_pos, _ = pl
            m = {}
            for c in uniq:
                for (px, py) in cell_pos[c]:
                    m.setdefault((px, py), set()).add(c)
            cellmap[q] = m

    # accumulate survival per (rep, budget)
    combos = [(rep, b) for rep in reps_avail for b in args.budgets]
    agg = {c: {"winner": 0, "top5": 0.0, "winpos": 0, "dcells": [], "dpos": [], "dlev": [], "n": 0}
           for c in combos}
    for q, crops in baseline.items():
        if not crops or not qstore.has(q):
            continue
        qfeat, _, _, _ = qstore.get(q)
        qtok = F.normalize(qfeat.float().to(dev), dim=1)
        base = np.array([c[3] for c in crops], float)             # exact geom scores
        winner = int(base.argmax())                               # baseline's best single crop
        top5 = set(np.argsort(-base)[:5].tolist())
        possum = {}                                               # position -> Sum-levels geom
        for j, c in enumerate(crops):
            possum[(c[0], c[1])] = possum.get((c[0], c[1]), 0.0) + c[3]
        win_pos = max(possum, key=possum.get)
        cell_of = None
        if cellmap is not None and q in cellmap:
            cm = cellmap[q]
            cell_of = [cm.get((c[0], c[1]), set()) for c in crops]

        # map each baseline crop -> compact-index row (skip crops missing from the index)
        rows, valid = [], []
        for j, c in enumerate(crops):
            i = pos_i.get((round(float(c[0]), 2), round(float(c[1]), 2)))
            r = row_of.get((i, int(c[2]))) if i is not None else None
            rows.append(r); valid.append(r is not None)
        valid = np.array(valid)

        vr = np.array([rows[j] for j in range(len(crops)) if valid[j]], dtype=np.int64)
        idx_valid = np.nonzero(valid)[0]                          # baseline-crop indices with a desc
        for rep in reps_avail:
            arr = torch.from_numpy(rep_arr[rep][vr].astype(np.float32)).to(dev)   # RAM gather (fast)
            cheap_v = _cheap_scores(rep, qtok, arr, args.grid_score)
            cheap = np.full(len(crops), -1e9)
            cheap[idx_valid] = cheap_v
            order = list(np.argsort(-cheap))                      # best cheap first (invalids sink)
            for b in args.budgets:
                chosen = set(_select(order, crops, b, args.top_levels_per_position,
                                     args.positions_per_cell, args.min_cells, cell_of))
                a = agg[(rep, b)]
                a["n"] += 1
                a["winner"] += int(winner in chosen)
                a["top5"] += len(top5 & chosen) / len(top5)
                a["winpos"] += int(any((crops[j][0], crops[j][1]) == win_pos for j in chosen))
                a["dpos"].append(len({(crops[j][0], crops[j][1]) for j in chosen}))
                a["dlev"].append(len({crops[j][2] for j in chosen}))
                if cell_of is not None:
                    a["dcells"].append(len(set().union(*[cell_of[j] for j in chosen]) if chosen else set()))

    print(f"\n{'Rep':<7}{'Score':<9}{'Budget':>7}{'WinnerSurv':>12}{'Top5Surv':>10}"
          f"{'WinPosSurv':>12}{'dPos':>7}{'dLev':>6}{'dCells':>8}")
    table = []
    for rep in reps_avail:
        for b in args.budgets:
            a = agg[(rep, b)]; n = max(1, a["n"])
            row = {"rep": rep, "grid_score": args.grid_score if rep != "mean" else "cosine", "budget": b,
                   "winner_survival": a["winner"] / n, "top5_survival": a["top5"] / n,
                   "winpos_survival": a["winpos"] / n,
                   "mean_distinct_pos": float(np.mean(a["dpos"])) if a["dpos"] else 0.0,
                   "mean_distinct_lev": float(np.mean(a["dlev"])) if a["dlev"] else 0.0,
                   "mean_distinct_cells": float(np.mean(a["dcells"])) if a["dcells"] else None}
            table.append(row)
            dc = f"{row['mean_distinct_cells']:.1f}" if row["mean_distinct_cells"] is not None else "-"
            print(f"{rep:<7}{row['grid_score']:<9}{b:>7}{a['winner']/n:>12.3f}{a['top5']/n:>10.3f}"
                  f"{a['winpos']/n:>12.3f}{np.mean(a['dpos']) if a['dpos'] else 0:>7.0f}"
                  f"{np.mean(a['dlev']) if a['dlev'] else 0:>6.1f}{dc:>8}")
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(
            {"meta": {"index": args.compact_index, "grid_score": args.grid_score,
                      "top_levels_per_position": args.top_levels_per_position,
                      "positions_per_cell": args.positions_per_cell, "min_cells": args.min_cells},
             "table": table}), encoding="utf-8")
        print(f"[ok] -> {args.out}", flush=True)
    ix.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
