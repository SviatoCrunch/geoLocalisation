"""Standalone coarse ranker: score every query against ONE gallery H5 → top-K shortlist.

A thin, split-free wrapper around the same pieces ``export_shortlist`` uses (``build_e2c_model`` +
``TileGridLoader`` + ``QueryTokenStore`` + ``build_gallery_V`` + ``model.encode_query`` /
``model.score``), but it ranks over ALL tiles of a single gallery H5 (e.g. the 221-tile
``kup_prodaction.h5``) with no split-config / positive-relevance machinery. Cell ids are taken
straight from the gallery groups (``<city>:<groupkey>`` = ``kup:{tile_index}_lvl0``), so the shortlist
matches the DINOv3 pyramid store built with the same ids. Output JSON = the ``export_shortlist`` schema
(``{"shortlist": {qid: {"cells": [...], "dense": []}}}``) that ``search_pyramid_s3`` consumes.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
      python -m geo_e2c_train.coarse_rank_gallery \
      --ckpt /…/run_supervlad_cell_w1_best.pt --assign /…/dict/k32.pt \
      --gallery kup=/…/product_emb/kup_prodaction.h5 --queries kup=/…/kup/query_kup_d1024.h5 \
      --city kup --k 100 --device cuda --out /…/kup/shortlist_kup_prod_k100.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def _gallery_tile_ids(gallery_h5: str, city: str):
    """Every group with an ift_dino → tile_id ``<city>:<groupkey>`` (matches TileGridLoader keys)."""
    import h5py
    with h5py.File(Path(gallery_h5).expanduser(), "r") as f:
        keys = [k for k in f.keys() if hasattr(f[k], "keys") and "ift_dino" in f[k]]
    return [f"{city}:{k}" for k in keys]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gallery", required=True, help="city=gallery_h5 (group-per-tile ift_dino)")
    ap.add_argument("--queries", required=True, help="city=query_d1024.h5 (DINOv2 tokens)")
    ap.add_argument("--city", required=True)
    ap.add_argument("--assign", default=None, help="override resolved_config.assign_path (k32.pt)")
    ap.add_argument("--k", type=int, default=100, help="shortlist size (top-K cells)")
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap queries (smoke)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from tqdm import tqdm
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from .data import QueryTokenStore, TileGridLoader
    from .eval import build_gallery_V

    ck = torch.load(Path(args.ckpt).expanduser(), map_location="cpu", weights_only=False)
    rc = ck.get("resolved_config", {})
    keep = {f.name for f in fields(E2cModelConfig)}
    cfg_kwargs = {k: rc[k] for k in rc if k in keep}
    if args.assign:
        cfg_kwargs["assign_path"] = args.assign
    cfg = E2cModelConfig(**cfg_kwargs)
    model = build_e2c_model(cfg, device=args.device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[ckpt] epoch {ck.get('epoch')} agg={cfg.agg} pyramid={cfg.pyramid_mode} "
          f"d_token={cfg.d_token} assign={cfg.assign_path}", flush=True)

    gcity, gpath = _kv(args.gallery)
    tiles = TileGridLoader({gcity: gpath}, grid_size=None, cache=False)
    store = QueryTokenStore(dict([_kv(args.queries)]))
    tile_ids = _gallery_tile_ids(gpath, gcity)
    if not tile_ids:
        raise SystemExit(f"no ift_dino groups in {gpath}")
    print(f"[gallery] {gcity}: {len(tile_ids)} tiles", flush=True)

    galV = build_gallery_V(model, tiles, tile_ids, args.eval_chunk, args.device, progress=True)

    # every query the store has for this city (id <city>:<stem>, mirrors QueryTokenStore construction)
    import h5py
    qpath = _kv(args.queries)[1]
    qids = []
    with h5py.File(Path(qpath).expanduser(), "r") as f:
        for k in f.keys():
            if not (hasattr(f[k], "keys") and "ift_dino" in f[k]):
                continue
            fn = f[k].attrs.get("filename", "")
            stem = Path(str(fn)).stem if fn else k
            qid = f"{args.city}:{stem}"
            if store.has(qid):
                qids.append(qid)
    qids = sorted(set(qids))
    if args.limit:
        qids = qids[:args.limit]
    if not qids:
        raise SystemExit(f"no queries for city {args.city!r} found in the store")
    print(f"[queries] {len(qids)}", flush=True)

    Q = torch.stack([model.encode_query(store.tokens(q).to(args.device)) for q in qids])
    with torch.no_grad():
        S = model.score(Q, galV, tile_chunk=args.eval_chunk)          # (B, n_tiles)
    order = S.argsort(dim=1, descending=True).cpu().numpy()

    out = {"meta": {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "k": args.k,
                    "n_tiles": len(tile_ids), "n_queries": len(qids), "gallery": gpath},
           "shortlist": {}}
    for bi, q in enumerate(qids):
        topk = [int(r) for r in order[bi][:args.k]]
        out["shortlist"][q] = {"cells": [tile_ids[r] for r in topk], "dense": []}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] shortlist ({len(qids)} q × top-{args.k}) -> {args.out}", flush=True)
    tiles.close(); store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
