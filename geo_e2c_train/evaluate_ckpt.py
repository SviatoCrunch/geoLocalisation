"""Evaluate a saved best.pt (no retrain): exact-tile Recall@K + median rank + distance-based recall.

Rebuilds the model from the checkpoint's ``resolved_config`` (agg / pyramid_mode / dims), loads the
weights, and ranks val/test queries against the full gallery. Use it to check whether an exact-tile
R@1=0 is a near-miss on the dense stride-250 gallery (see ``distR@Xm``) or a true failure.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with shapely \
           --with pyyaml --with tqdm python -m geo_e2c_train.evaluate_ckpt \
      --ckpt /…/run_residual_cell/best.pt \
      --split-config /home/ubuntu/work/split_pyr250.yaml \
      --split-json   /…/split_pyr250/split.json \
      --galleries kramatorsc=/…/map_dinov2_kramatorsc_s250m_d1024.h5 kup=/…/… liman_day=/…/… \
      --queries   kramatorsc=/…/query_kramatorsc_d1024.h5 kup=/…/… liman_day=/…/… \
      --assign /home/ubuntu/work/out/gallery_h5/dict/k32.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    ap.add_argument("--assign", default=None, help="override assign_path (else use ckpt's)")
    ap.add_argument("--which", nargs="+", default=["val", "test"])
    ap.add_argument("--query-size-m", type=float, default=1000.0)
    ap.add_argument("--safe-eps-area", type=float, default=0.0)
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--ks", default="1,5,10,20,50,60,70,100,200",
                    help="comma recall cutoffs (shortlist sizing for two-stage rerank)")
    ap.add_argument("--same-city", action="store_true",
                    help="rank each query only against tiles of its own city (known-AO ceiling)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    import torch
    from dataclasses import fields
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from geo_train_batching.adapters.split_relevance import build_split_relevance
    from .data import TileGridLoader, QueryTokenStore
    from .eval import build_gallery_V, score_against_gallery

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    rc = ck.get("resolved_config", {})
    keep = {f.name for f in fields(E2cModelConfig)}
    cfg_kwargs = {k: rc[k] for k in rc if k in keep}
    if args.assign:
        cfg_kwargs["assign_path"] = args.assign
    cfg = E2cModelConfig(**cfg_kwargs)
    print(f"[ckpt] epoch {ck.get('epoch')} | agg={cfg.agg} pyramid={cfg.pyramid_mode} "
          f"k={cfg.k} d_token={cfg.d_token}", flush=True)

    model = build_e2c_model(cfg, device=args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    galleries = dict(_kv(a) for a in args.galleries)
    queries = dict(_kv(a) for a in args.queries)
    eval_tiles = TileGridLoader(galleries, grid_size=None, cache=False)
    store = QueryTokenStore(queries)

    results = {}
    galV = None
    for w in args.which:
        sr = build_split_relevance(args.split_config, args.split_json, w,
                                   query_size_m=args.query_size_m, safe_eps_area=args.safe_eps_area)
        if galV is None:                              # gallery identical across splits → build once
            galV = build_gallery_V(model, eval_tiles, sr.tile_ids, args.eval_chunk, args.device,
                                   progress=True)
        results[w] = score_against_gallery(model, store, sr, galV, args.device, args.eval_chunk,
                                           ks=ks, same_city=args.same_city)
        print(f"[{w}] {json.dumps(results[w])}", flush=True)

    out = Path(args.ckpt).with_suffix(".eval.json")
    out.write_text(json.dumps({"epoch": ck.get("epoch"), **results}, indent=2), encoding="utf-8")
    print(f"[ok] -> {out}", flush=True)
    eval_tiles.close(); store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
