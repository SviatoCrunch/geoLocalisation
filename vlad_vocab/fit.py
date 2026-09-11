"""Fit VLAD vocabulary (K-means) + recast → e2c dictionary blob k<K>.pt (CLI).

One read pass over the DINO galleries samples tokens; a self-contained GPU K-means is fit
for each K; each vocabulary is recast into ``<out>/k<K>.pt`` = {assign_weight, centroids, …}
(the file ``build_e2c_model(assign_path=…)`` loads), plus ``<out>/c<K>_centers.pt`` (raw
centroids) and ``<out>/k<K>.meta.json`` for provenance.

Run (server)::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
      python -m vlad_vocab.fit \
      --h5 /home/ubuntu/work/out/gallery_h5/map_dinov2_kramatorsc_s250m_d1024.h5 \
           /home/ubuntu/work/out/gallery_h5/map_dinov2_kup_s250m_d1024.h5 \
           /home/ubuntu/work/out/gallery_h5/map_dinov2_liman_day_s250m_d1024.h5 \
      --k 32 --out /home/ubuntu/work/out/e2c_dict --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", nargs="+", type=Path, required=True,
                    help="group-per-tile DINO galleries (map_extract output)")
    ap.add_argument("--k", nargs="+", type=int, default=[32])
    ap.add_argument("--out", type=Path, required=True, help="dir; writes <out>/k<K>.pt")
    ap.add_argument("--per-tile", type=int, default=48, help="tokens sampled per tile (default 48)")
    ap.add_argument("--max-tokens", type=int, default=600_000, help="global sampled-token cap")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="evenly subsample at most N tiles across galleries to bound I/O "
                         "(0 = read all tiles; e.g. 4000 reads ~28 GiB instead of ~80)")
    ap.add_argument("--iters", type=int, default=50, help="max Lloyd iterations (default 50)")
    ap.add_argument("--init-prob", type=float, default=0.01,
                    help="SuperVLAD assignment γ calibration target (assignment_from_centroids)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    import torch
    from .h5_tokens import sample_tokens
    from .kmeans import kmeans
    from .recast import recast_centroids

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, n_tiles, D = sample_tokens(args.h5, args.per_tile, args.max_tokens, args.seed,
                                  max_tiles=args.max_tiles)
    print(f"[sample] {X.shape[0]} tokens (D={D}) from {n_tiles} tiles over {len(args.h5)} galleries")

    args.out.mkdir(parents=True, exist_ok=True)
    for K in sorted(set(int(k) for k in args.k)):
        C, inertia, n_it = kmeans(X, K, args.iters, args.seed, device)
        blob = recast_centroids(C, X, init_prob=args.init_prob)
        assert tuple(blob["assign_weight"].shape) == (K, D), blob["assign_weight"].shape
        assert tuple(blob["centroids"].shape) == (K, D), blob["centroids"].shape

        torch.save(C.float(), args.out / f"c{K}_centers.pt")
        torch.save(blob, args.out / f"k{K}.pt")
        meta = {"k": K, "token_dim": D, "n_sample": int(X.shape[0]), "n_tiles": n_tiles,
                "per_tile": args.per_tile, "iters_run": n_it, "inertia": inertia,
                "init_prob": args.init_prob, "gamma": float(blob.get("alpha", 0.0)),
                "mean_top1_top2_gap": float(blob.get("mean_top1_top2_gap", 0.0)),
                "seed": args.seed, "device": device,
                "sources": [str(p) for p in args.h5], "git_commit": _git_commit()}
        (args.out / f"k{K}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] K={K:>3} D={D} γ={meta['gamma']:.2f} inertia={inertia:.1f} iters={n_it} "
              f"-> {args.out / f'k{K}.pt'}  (assign_weight+centroids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
