"""Limited-map coarse retrieval eval (NO rerank): restrict each query's candidate gallery to the tiles
inside a square of area A km2 centred on that query, and sweep A over --areas-km2 50 60 70 80 90 100 ...

Shows how the trained e2c retriever alone (no fine/rerank stage) degrades as the operating map grows:
a small A has few distractors (easy), a large A many. Reuses the model + gallery + split machinery of
geo_e2c_train unchanged; the only addition is the per-query spatial mask.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with shapely --with pyyaml \
           --with tqdm python -m geo_e2c_augtrain.eval_limited_map \
      --ckpt /.../run_supervlad_cell_w1_aug/best.pt \
      --split-config /.../split_checker1000/config.yaml --split-json /.../split_checker1000/split.json \
      --galleries kramatorsc=/.../map_dinov2_kramatorsc_s250m_d1024.h5 kup=/.../... liman_day=/.../... \
      --queries kramatorsc=/.../query_kramatorsc_d1024.h5 kup=/.../... liman_day=/.../... \
      --assign /.../dict/k32.pt --which val test --areas-km2 50 60 70 80 90 100 \
      --out /.../limited_map_eval.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

_COS = 0.657                                    # EPSG:3857 -> true metres at ~48.9 deg
_KS = (1, 5, 10, 20, 50, 100)
_DTHR = (250.0, 500.0, 1000.0)


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-config", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--galleries", nargs="+", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--assign", default=None)
    ap.add_argument("--which", nargs="+", default=["val", "test"])
    ap.add_argument("--areas-km2", type=float, nargs="+", default=[50, 60, 70, 80, 90, 100])
    ap.add_argument("--query-size-m", type=float, default=1000.0)
    ap.add_argument("--same-city", action="store_true", default=True,
                    help="restrict to the query's own city (default on; other cities are far anyway)")
    ap.add_argument("--all-cities", dest="same_city", action="store_false")
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from geo_train_batching.adapters.split_relevance import build_split_relevance
    from geo_e2c_train.data import TileGridLoader, QueryTokenStore
    from geo_e2c_train.eval import build_gallery_V

    ck = torch.load(Path(args.ckpt).expanduser(), map_location="cpu", weights_only=False)
    rc = ck.get("resolved_config", {})
    keep = {f.name for f in fields(E2cModelConfig)}
    cfg_kwargs = {k: rc[k] for k in rc if k in keep}
    if args.assign:
        cfg_kwargs["assign_path"] = args.assign
    model = build_e2c_model(E2cModelConfig(**cfg_kwargs), device=args.device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[ckpt] {args.ckpt} epoch {ck.get('epoch')} | agg={rc.get('agg')} pyramid={rc.get('pyramid_mode')}",
          flush=True)

    galleries = dict(_kv(a) for a in args.galleries)
    tiles = TileGridLoader(galleries, cache=False)
    store = QueryTokenStore(dict(_kv(a) for a in args.queries))
    dev = args.device

    out = {"meta": {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "areas_km2": list(args.areas_km2),
                    "same_city": bool(args.same_city)}, "per_split": {}}
    galV = None
    for w in args.which:
        sr = build_split_relevance(args.split_config, args.split_json, w, query_size_m=args.query_size_m)
        if galV is None:
            galV = build_gallery_V(model, tiles, sr.tile_ids, args.eval_chunk, dev, progress=True)
        tile_city = np.array([str(t).split(":", 1)[0] for t in sr.tile_ids])
        qids = [q for q in sr.query_ids if store.has(q)]
        qid_row = {q: i for i, q in enumerate(sr.query_ids)}
        with torch.no_grad():
            Q = torch.stack([model.encode_query(store.tokens(q).to(dev)) for q in qids])
            S = model.score(Q, galV, tile_chunk=args.eval_chunk).cpu().numpy()      # (B, M)
        tile_xy = sr.tile_xy                                                        # (M, 2) EPSG:3857
        print(f"\n[{w}] n={len(qids)} tiles={len(sr.tile_ids)}")
        print(f"    {'A(km2)':>7}{'cand':>7}" + "".join(f"{'R@'+str(k):>7}" for k in _KS)
              + f"{'mdRank':>8}" + "".join(f"{'d@'+str(int(t)):>7}" for t in _DTHR))
        rows = []
        for A in args.areas_km2:
            half_crs = (np.sqrt(A) * 1000.0 / 2.0) / _COS                           # true m -> EPSG:3857
            ranks, hit, dists, cands = [], {k: 0 for k in _KS}, [], []
            for bi, q in enumerate(qids):
                qi = qid_row[q]
                pos = set(int(x) for x in sr.relevance.pos_of(qi))
                if not pos:
                    continue
                cx, cy = sr.q_xy[qi]
                allowed = (np.abs(tile_xy[:, 0] - cx) <= half_crs) & (np.abs(tile_xy[:, 1] - cy) <= half_crs)
                if args.same_city:
                    allowed &= (tile_city == q.split(":", 1)[0])
                cands.append(int(allowed.sum()))
                s = np.where(allowed, S[bi], -np.inf)
                order = np.argsort(-s)
                r = next((j for j, t in enumerate(order) if int(t) in pos), len(order))
                ranks.append(r + 1)
                for k in _KS:
                    if r < k:
                        hit[k] += 1
                top1 = int(order[0])
                dists.append(float(np.linalg.norm(tile_xy[top1] - sr.q_xy[qi])) * _COS)
            n = len(ranks)
            row = {"area_km2": A, "n": n, "mean_candidates": float(np.mean(cands)) if cands else 0.0,
                   "median_rank": float(np.median(ranks)) if ranks else float("nan"),
                   **{f"R@{k}": (hit[k] / n if n else 0.0) for k in _KS},
                   **{f"distR@{int(t)}m": (float((np.array(dists) <= t).mean()) if dists else 0.0)
                      for t in _DTHR}}
            rows.append(row)
            print(f"    {A:>7.0f}{row['mean_candidates']:>7.0f}"
                  + "".join(f"{row[f'R@{k}']:>7.3f}" for k in _KS)
                  + f"{row['median_rank']:>8.0f}"
                  + "".join(f"{row[f'distR@{int(t)}m']:>7.3f}" for t in _DTHR))
        out["per_split"][w] = rows

    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
        print(f"\n[ok] -> {args.out}", flush=True)
    tiles.close(); store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
