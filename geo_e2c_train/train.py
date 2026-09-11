"""DSS training loop for the e2c model on a geo_split split (isolated glue).

Wires: build_e2c_model (agg × pyramid_mode) + geo_train_batching (split→relevance→pairs, neighbour
cache, logical-batch planner, symmetric multi-positive CE) + map_extract H5s (tile grids + query
tokens). Tile grids are cached in RAM (the expensive read happens once) and a tqdm bar tracks epochs.
The objective is the SAME proven e2c/DSS loss — this file only orchestrates.

Run (residual + classic cell apex, d1024)::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with shapely \
           --with pyyaml --with tqdm python -m geo_e2c_train.train \
      --split-config /home/ubuntu/work/split_pyr250.yaml \
      --split-json   /home/ubuntu/work/out/gallery_h5/split_pyr250/split.json \
      --galleries kramatorsc=/…/map_dinov2_kramatorsc_s250m_d1024.h5 \
                  kup=/…/map_dinov2_kup_s250m_d1024.h5 \
                  liman_day=/…/map_dinov2_liman_day_s250m_d1024.h5 \
      --queries   kramatorsc=/…/query_kramatorsc_d1024.h5 kup=/…/query_kup_d1024.h5 \
                  liman_day=/…/query_liman_day_d1024.h5 \
      --assign /home/ubuntu/work/out/gallery_h5/dict/k32.pt \
      --agg residual --pyramid-mode cell --k 32 --d-token 1024 --scales 8 4 2 1 \
      --epochs 40 --b-log 32 --out /home/ubuntu/work/out/gallery_h5/split_pyr250/run_residual_cell
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# reduce CUDA fragmentation for the eval gallery-V pass (residual cdist spikes); set before torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _kv(arg: str):
    if "=" not in arg:
        raise SystemExit(f"expected city=path, got {arg!r}")
    c, p = arg.split("=", 1)
    return c, p


def train(args) -> dict:
    import numpy as np
    import torch
    from tqdm import tqdm

    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from geo_train_batching import (build_neighbour_cache, plan_logical_batch,
                                    build_cross_relevance, symmetric_multipositive_ce)
    from geo_train_batching.adapters.split_relevance import build_split_relevance, build_pairs_from_split
    from .data import TileGridLoader, QueryTokenStore
    from .eval import build_gallery_V, score_against_gallery

    dev = args.device
    torch.manual_seed(args.seed)
    galleries = dict(_kv(a) for a in args.galleries)
    queries = dict(_kv(a) for a in args.queries)

    # split → relevance (positives IDENTICAL to the split's) for each subset
    sr = {w: build_split_relevance(args.split_config, args.split_json, w,
                                   query_size_m=args.query_size_m, safe_eps_area=args.safe_eps_area)
          for w in ("train", "val", "test")}
    tiles = TileGridLoader(galleries, grid_size=args.grid_size)              # cached: train pair tiles
    eval_tiles = TileGridLoader(galleries, grid_size=args.grid_size, cache=False)  # streamed: full gallery
    store = QueryTokenStore(queries)
    for w in ("train", "val", "test"):
        n = sum(1 for q in sr[w].query_ids if store.has(q))
        print(f"[data] {w}: {len(sr[w].query_ids)} split queries, {n} with tokens", flush=True)
    print(f"[data] tiles {len(sr['train'].tile_ids)} | agg={args.agg} pyramid={args.pyramid_mode} "
          f"K={args.k} d_token={args.d_token}", flush=True)

    cfg = E2cModelConfig(agg=args.agg, k=args.k, assign_path=args.assign, d_token=args.d_token,
                         pyramid_mode=args.pyramid_mode, scales_cells=tuple(args.scales),
                         concentric_sizes_m=tuple(args.concentric_sizes) if args.concentric_sizes
                         else (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0),
                         tile_size_m=args.tile_size_m, d_group=args.d_group, d_out=args.d_out,
                         d_hidden=args.d_hidden)
    model = build_e2c_model(cfg, device=dev)
    params = model.trainable_parameters()
    print(f"[model] trainable {sum(p.numel() for p in params)/1e6:.2f}M | resolved={model.resolved_config().get('pyramid_mode')}",
          flush=True)

    # canonical train pairs + a preloaded RAM cache of their tile grids
    pairs = build_pairs_from_split(sr["train"])
    if not pairs:
        raise SystemExit("no canonical train pairs (no train query has a positive with tokens?)")
    pair_qid = [sr["train"].query_ids[p.query_index] for p in pairs]
    pair_trow = [int(p.canonical_tile_row) for p in pairs]
    tid = sr["train"].tile_ids
    print(f"[pairs] {len(pairs)} canonical train pairs | preloading {len(set(pair_trow))} tile grids …",
          flush=True)
    for r in tqdm(sorted(set(pair_trow)), desc="preload grids", unit="tile"):
        tiles.grid(tid[r])                                          # warm the RAM cache

    def q_embed(qid):
        return model.encode_query(store.tokens(qid).to(dev))

    with torch.no_grad():
        model.eval()
        pv = torch.stack([q_embed(q) for q in pair_qid]).cpu().numpy()
    nbr = build_neighbour_cache(pv, epoch=0, top_k=args.top_k)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best = {"median_rank": float("inf"), "epoch": -1}
    logf = (out / "metrics.jsonl").open("w", encoding="utf-8")

    for epoch in tqdm(range(args.epochs), desc=f"{args.agg}/{args.pyramid_mode} epochs", unit="ep"):
        model.train()
        eseed = int(hashlib.sha256(f"dss|{args.seed}|{epoch}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(eseed)
        order = list(range(len(pairs))); rng.shuffle(order); unused = list(order)
        freq, ep_loss, nb = {}, 0.0, 0
        while unused:
            seed_pair = unused.pop(0)
            plan = plan_logical_batch(pairs, nbr, B_log=args.b_log, seed_pair=seed_pair,
                                      rng=rng, freq_counter=freq)
            unused = [u for u in unused if u not in set(plan.pair_indices)]
            if plan.B_log_actual < 2:
                continue
            pib = [pairs[i] for i in plan.pair_indices]
            G = tiles.stack([tid[pair_trow[i]] for i in plan.pair_indices]).float().to(dev)
            V = model.build_V(G)
            Q = torch.stack([q_embed(pair_qid[i]) for i in plan.pair_indices])
            S = model.score(Q, V)
            R_pos, R_cand = build_cross_relevance(pib, sr["train"].relevance)
            loss = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=args.tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1

        row = {"epoch": epoch, "train_loss": ep_loss / max(nb, 1)}
        do_eval = (epoch > 0 and epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
        if do_eval:
            galV = build_gallery_V(model, eval_tiles, sr["val"].tile_ids, args.eval_chunk,
                                   dev, progress=True)                 # built ONCE; val+test share it
            for w in ("val", "test"):
                row[w] = score_against_gallery(model, store, sr[w], galV, dev, args.eval_chunk)
            del galV
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
            vmr = row["val"].get("median_rank", float("inf"))
            if vmr == vmr and vmr < best["median_rank"]:
                best = {"median_rank": vmr, "epoch": epoch, **row}
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "resolved_config": model.resolved_config()}, out / "best.pt")
        logf.write(json.dumps(row) + "\n"); logf.flush()
        tqdm.write(f"[e{epoch}] loss {row['train_loss']:.4f}" + (
            f" | val md-rank {row['val'].get('median_rank','-')} R@10 {row['val'].get('R@10',0):.3f}"
            f" | test md-rank {row['test'].get('median_rank','-')}" if do_eval else " (no eval)"))
        if do_eval and epoch - best["epoch"] >= args.patience:
            tqdm.write(f"[stop] no val improvement since e{best['epoch']}"); break

    logf.close()
    (out / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"[ok] best epoch {best['epoch']} val md-rank {best['median_rank']} -> {out}", flush=True)
    tiles.close(); eval_tiles.close(); store.close()
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-config", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--galleries", nargs="+", required=True, help="city=path (raw d1024 tile galleries)")
    ap.add_argument("--queries", nargs="+", required=True, help="city=path (query token H5, d1024)")
    ap.add_argument("--assign", required=True, help="dict/k32.pt (assign_weight + centroids)")
    ap.add_argument("--agg", choices=["supervlad", "vlad", "residual"], default="residual")
    ap.add_argument("--pyramid-mode", choices=["cell", "concentric"], default="cell")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--d-token", type=int, default=1024)
    ap.add_argument("--scales", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--concentric-sizes", type=float, nargs="+", default=None)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--query-size-m", type=float, default=1000.0)
    ap.add_argument("--safe-eps-area", type=float, default=0.0)
    ap.add_argument("--d-group", type=int, default=32)
    ap.add_argument("--d-out", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--b-log", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="eval every N epochs (full-gallery pass streams all tiles from disk — costly)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--grid-size", type=int, default=0, help="resample token grids to g×g (0=keep)")
    ap.add_argument("--eval-chunk", type=int, default=64,
                    help="tiles per gallery-V chunk at eval (residual cdist spikes — keep small on T4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.grid_size = args.grid_size or None
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
