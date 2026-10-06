"""KMZ of the fine-rerank accuracy for a diagnostics archive (``search_pyramid_s3 --diag-dir``).

Reads ``summary.jsonl`` and writes a KMZ: per frame a folder with the GT point (green), the reranked
top-1 (red) and top-2..K (orange), each with its level-footprint square and the top-1 geo error in the
folder name — so you can eyeball where each of the 40 test frames landed. Works on a partial run.

Run::  python -m patch_rerank.diag_kmz --diag-dir /…/diag_kram_test   # -> <dir>/accuracy.kmz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search_pyramid_s3 import _write_kmz


def build_per_query(diag_dir: str | Path, topk: int = 5) -> dict:
    d = Path(diag_dir).expanduser()
    pq = {}
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        ft = (s.get("final_topk") or [])[:topk] if topk else (s.get("final_topk") or [])
        pq[s["query_id"]] = {
            "gt": s.get("gt"), "fine_dist_m": (s.get("metrics") or {}).get("fine_dist_m"),
            "topk": [{"rank": (t.get("rerank_rank") or t.get("rank")), "level_m": t["level_m"],
                      "cell_score": t["cell_score"], "lat": t["lat"], "lon": t["lon"]} for t in ft]}
    return pq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <diag-dir>/accuracy.kmz")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args(argv)
    pq = build_per_query(args.diag_dir, args.topk)
    out = Path(args.out).expanduser() if args.out else Path(args.diag_dir).expanduser() / "accuracy.kmz"
    _write_kmz(out, pq)
    d = sorted(v["fine_dist_m"] for v in pq.values() if v["fine_dist_m"] is not None)
    if d:
        med = d[len(d) // 2]
        print(f"[kmz] frames={len(pq)} with_gt={len(d)} top1 median={med:.0f}m "
              f"distR@250m={sum(x <= 250 for x in d) / len(d):.3f} "
              f"@1000m={sum(x <= 1000 for x in d) / len(d):.3f}", flush=True)
    print(f"[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
