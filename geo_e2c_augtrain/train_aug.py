"""e2c training with per-epoch AUGMENTED drone-query re-encode. Thin fork of geo_e2c_train.train:
EVERYTHING (split relevance, neighbour cache, logical-batch planner, symmetric multi-positive CE,
gallery, model, checkpointing, eval) is imported UNCHANGED. The only differences:

  * TRAIN queries come from AugmentedQueryStore (raw frame -> uav_augment -> sky -> DINO -> same
    seed-0 projection), re-encoded every epoch. VAL/TEST evaluate with the FROZEN query H5 (clean,
    un-augmented) - augmentation is train-only.
  * `--resume ckpt` continues from run_supervlad_cell_w1_best.pt (fine-tune) instead of from scratch.
  * `--eval-every` defaults to 3.

Run: see geo_e2c_augtrain/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _kv(arg):
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
                                    build_cross_relevance, build_cross_weights,
                                    symmetric_multipositive_ce)
    from geo_train_batching.adapters.split_relevance import build_split_relevance, build_pairs_from_split
    from geo_e2c_train.data import TileGridLoader, QueryTokenStore
    from geo_e2c_train.eval import build_gallery_V, score_against_gallery
    from .augmented_query_store import AugmentedQueryStore

    dev = args.device
    torch.manual_seed(args.seed)
    galleries = dict(_kv(a) for a in args.galleries)
    queries = dict(_kv(a) for a in args.queries)          # FROZEN query H5 (val/test eval + proj ref)
    gt = dict(_kv(a) for a in args.gt)                    # raw drone image dirs (train augmentation)

    sr = {w: build_split_relevance(args.split_config, args.split_json, w,
                                   query_size_m=args.query_size_m, safe_eps_area=args.safe_eps_area)
          for w in ("train", "val", "test")}
    if args.pos_weighted_loss and not sr["train"].relevance.has_weights:
        raise SystemExit("--pos-weighted-loss needs a weighting positive selector")

    tiles = TileGridLoader(galleries, grid_size=args.grid_size)
    eval_tiles = TileGridLoader(galleries, grid_size=args.grid_size, cache=False)
    frozen = QueryTokenStore(queries)                     # clean tokens for val/test
    ref_h5 = queries[next(iter(queries))]
    aug = AugmentedQueryStore(gt, ref_h5, device=dev, preset=args.aug_preset,
                              segment_sky=not args.no_sky, base_seed=args.seed, batch=args.aug_batch,
                              amp=args.amp)

    # canonical train pairs (positives IDENTICAL to the split's); train queries must have a raw frame
    pairs = build_pairs_from_split(sr["train"])
    pairs = [p for p in pairs if aug.has(sr["train"].query_ids[p.query_index])]
    if not pairs:
        raise SystemExit("no train pairs whose query has a raw drone frame under --gt")
    pair_qid = [sr["train"].query_ids[p.query_index] for p in pairs]
    pair_trow = [int(p.canonical_tile_row) for p in pairs]
    tid = sr["train"].tile_ids
    if args.map_pyramid_source == "legacy_token_partition":
        print(f"[pairs] {len(pairs)} train pairs | preloading {len(set(pair_trow))} tile grids ...", flush=True)
        for r in tqdm(sorted(set(pair_trow)), desc="preload grids", unit="tile"):
            tiles.grid(tid[r])
    else:
        print(f"[pairs] {len(pairs)} train pairs | native source (S cache, no grid preload)", flush=True)
    train_qids = sorted(set(pair_qid))

    cfg = E2cModelConfig(agg=args.agg, k=args.k, assign_path=args.assign, d_token=args.d_token,
                         pyramid_mode=args.pyramid_mode, scales_cells=tuple(args.scales),
                         concentric_sizes_m=tuple(args.concentric_sizes) if args.concentric_sizes
                         else (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0),
                         tile_size_m=args.tile_size_m, d_group=args.d_group, d_out=args.d_out,
                         d_hidden=args.d_hidden, map_pyramid_source=args.map_pyramid_source)
    model = build_e2c_model(cfg, device=dev)
    if args.resume:
        ck = torch.load(Path(args.resume).expanduser(), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[resume] loaded {args.resume} (epoch {ck.get('epoch')}) - continue training", flush=True)
    params = model.trainable_parameters()
    print(f"[model] trainable {sum(p.numel() for p in params)/1e6:.2f}M | agg={args.agg} "
          f"pyramid={model.resolved_config().get('pyramid_mode')}", flush=True)

    # ── map-pyramid SOURCE strategy (same 2-point dispatch as geo_e2c_train.train) ─────
    from native_map_pyramid.source import (LegacyTokenPartitionSource, NativeHierarchicalSource,
                                           build_expected_identity, open_native_cache)
    native_ident = None
    if args.map_pyramid_source == "native_hierarchical":
        if not (args.agg == "supervlad" and args.pyramid_mode == "cell"):
            raise SystemExit("native_hierarchical requires --agg supervlad --pyramid-mode cell")
        if not args.native_cache:
            raise SystemExit("--native-cache is required for --map-pyramid-source native_hierarchical")
        expected = build_expected_identity(
            assign_weight=model.core.agg.assign.weight, k=args.k, d_token=args.d_token,
            backbone=args.native_backbone, dino_layer=args.native_dino_layer,
            dino_facet=args.native_dino_facet, output_px=args.native_output_px,
            tile_size_m=args.tile_size_m, proj_seed=args.native_proj_seed,
            proj_in_dim=args.native_proj_in_dim)
        ncache = open_native_cache(args.native_cache, expected)          # STRICT fingerprint check
        native_ident = ncache.identity
        train_source = eval_source = NativeHierarchicalSource(ncache, dev)
        print(f"[map-source] native_hierarchical | cache={args.native_cache} "
              f"fp={ncache.fingerprint[:12]} tiles={len(ncache.tile_ids)}", flush=True)
    else:
        train_source = LegacyTokenPartitionSource(tiles, dev)
        eval_source = LegacyTokenPartitionSource(eval_tiles, dev)
        print("[map-source] legacy_token_partition", flush=True)

    def q_embed(qid):
        return model.encode_query(aug.tokens(qid).to(dev))

    aug.set_epoch(0, qids=train_qids, progress=True)      # epoch-0 tokens for pair/neighbour setup
    with torch.no_grad():
        model.eval()
        pv = torch.stack([q_embed(q) for q in pair_qid]).cpu().numpy()
    nbr = build_neighbour_cache(pv, epoch=0, top_k=args.top_k)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best = {"median_rank": float("inf"), "epoch": -1}
    logf = (out / "metrics.jsonl").open("w", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(
        {"map_pyramid_source": args.map_pyramid_source,
         "resolved_config": model.resolved_config(),
         "native_cache": ({"path": args.native_cache, "fingerprint": native_ident.get("fingerprint"),
                           "identity": native_ident} if native_ident else None),
         "args": {k: v for k, v in vars(args).items()}}, indent=2, default=str), encoding="utf-8")

    for epoch in tqdm(range(args.epochs), desc=f"aug {args.agg}/{args.pyramid_mode}", unit="ep"):
        if epoch > 0:
            aug.set_epoch(epoch, qids=train_qids, progress=False)   # fresh augmentation this epoch
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
            batch_tile_ids = [tid[pair_trow[i]] for i in plan.pair_indices]
            V = train_source.build_V(model, batch_tile_ids)         # legacy grids OR native S cache
            Q = torch.stack([q_embed(pair_qid[i]) for i in plan.pair_indices])
            S = model.score(Q, V)
            R_pos, R_cand = build_cross_relevance(pib, sr["train"].relevance)
            if args.pos_weighted_loss:
                W = build_cross_weights(pib, sr["train"].relevance, R_pos)
                loss = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=args.tau,
                                                  weights=W, weight_power=args.pos_weight_power)
            else:
                loss = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=args.tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1

        row = {"epoch": epoch, "train_loss": ep_loss / max(nb, 1),
               "map_pyramid_source": args.map_pyramid_source}
        do_eval = (epoch > 0 and epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
        if do_eval:
            model.eval()
            _bvf = None if args.map_pyramid_source == "legacy_token_partition" \
                else (lambda ids: eval_source.build_V(model, ids))
            galV = build_gallery_V(model, eval_tiles, sr["val"].tile_ids, args.eval_chunk, dev,
                                   progress=True, build_V_fn=_bvf)
            for w in ("val", "test"):                     # eval on FROZEN (clean) tokens
                row[w] = score_against_gallery(model, frozen, sr[w], galV, dev, args.eval_chunk)
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
    tiles.close(); eval_tiles.close(); frozen.close(); aug.close()
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-config", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--galleries", nargs="+", required=True, help="city=path (frozen d1024 tile galleries)")
    ap.add_argument("--queries", nargs="+", required=True, help="city=path (FROZEN query H5: val/test + proj ref)")
    ap.add_argument("--gt", nargs="+", required=True, help="city=dir (raw drone frames for train augmentation)")
    ap.add_argument("--assign", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (e.g. run_supervlad_cell_w1_best.pt)")
    ap.add_argument("--aug-preset", default="stage3_strong", choices=["stage2_exact", "stage3_strong"])
    ap.add_argument("--no-sky", action="store_true", help="disable sky masking in re-encode")
    ap.add_argument("--aug-batch", type=int, default=8, help="drone frames per DINO re-encode batch")
    ap.add_argument("--agg", choices=["supervlad", "vlad", "residual"], default="supervlad")
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
    ap.add_argument("--pos-weighted-loss", action="store_true")
    ap.add_argument("--pos-weight-power", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=3, help="eval every N epochs (default 3)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--grid-size", type=int, default=0)
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    # ── experimental map-pyramid source (default = legacy = current behaviour) ─────────
    ap.add_argument("--map-pyramid-source", choices=["legacy_token_partition", "native_hierarchical"],
                    default="legacy_token_partition",
                    help="map-cell feature source (native = COG-crop hierarchy; needs --native-cache)")
    ap.add_argument("--native-cache", default=None, help="native_hierarchical: path to the S cache H5")
    ap.add_argument("--native-backbone", default="dinov2_vitg14")
    ap.add_argument("--native-dino-layer", type=int, default=None)
    ap.add_argument("--native-dino-facet", default="value")
    ap.add_argument("--native-output-px", type=int, default=840)
    ap.add_argument("--native-proj-seed", type=int, default=0)
    ap.add_argument("--native-proj-in-dim", type=int, default=1536)
    args = ap.parse_args(argv)
    args.grid_size = args.grid_size or None
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
