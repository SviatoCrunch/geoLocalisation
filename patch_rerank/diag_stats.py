"""Summary metrics for a cv2.MAGSAC++ diagnostics archive (``search_pyramid_s3 --diag-dir``).

Computes point-9 of the spec from the archive itself (one stream over records.jsonl + summary.jsonl),
so it covers the FULL set across resumes — not just one invocation's ``search.json``:

  * coarse recall@100   — fraction of frames whose GT lies within distR of ANY of the 100 candidate
                          cell centres (the coarse shortlist's reach);
  * fine top-1 / top-5  — distR@250/500/1000m + median / p90 / p95 of the geo error;
  * counts              — frames, candidates, positions, levels, cv2 calls, failed checks (skipped /
                          no-consensus), total verify time.

Run::  python -m patch_rerank.diag_stats --diag-dir /…/diag_kram_cv2_magsac   # writes stats.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

_THR = (250.0, 500.0, 1000.0)


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[k]


def _dist_report(dists):
    d = [x for x in dists if x is not None]
    rep = {"n": len(d), "median_m": _pct(d, 0.5), "p90_m": _pct(d, 0.9), "p95_m": _pct(d, 0.95)}
    rep.update({f"distR@{int(t)}m": (sum(x <= t for x in d) / len(d) if d else None) for t in _THR})
    return rep


def compute(diag_dir: str | Path) -> dict:
    d = Path(diag_dir).expanduser()
    manifest = json.loads((d / "manifest.json").read_text())

    # ---- one pass over records.jsonl: counts + per-query candidate cell centres ----
    status = collections.Counter()
    n_rows = n_cv2 = 0
    verify_s = 0.0
    levels = set()
    max_pos = 0
    cand_centres: dict = collections.defaultdict(dict)        # query_id -> {cell_id: (lat, lon)}
    rec_path = d / "records.jsonl"
    if rec_path.exists():
        with open(rec_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n_rows += 1
                status[r["status"]] += 1
                n_cv2 += int(r["cv2_called"])
                verify_s += float(r.get("verify_s") or 0.0)
                levels.add(int(r["level_id"]))
                max_pos = max(max_pos, int(r["position_id"]) + 1)
                cl, co = r.get("cell_lat"), r.get("cell_lon")
                if cl is not None and co is not None:
                    cand_centres[r["query_id"]][r["cell_id"]] = (cl, co)

    # ---- summary.jsonl: fine distances + GT for coarse recall ----
    top1, top5, coarse_min = [], [], []
    n_frames = 0
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        n_frames += 1
        gt = s.get("gt")
        m = s.get("metrics") or {}
        if gt and None not in gt:
            top1.append(m.get("fine_dist_m"))
            top5.append(m.get("fine_dist_topk_m"))
            centres = cand_centres.get(s["query_id"], {})
            if centres:
                coarse_min.append(min(_haversine_m(gt[0], gt[1], la, lo) for la, lo in centres.values()))

    coarse = {"n": len(coarse_min),
              **{f"recall@100_{int(t)}m": (sum(x <= t for x in coarse_min) / len(coarse_min)
                                           if coarse_min else None) for t in _THR}}

    n_candidates = sum(len(c) for c in cand_centres.values())
    return {
        "diag_dir": str(d), "fingerprint": manifest.get("fingerprint"),
        "coarse_recall@100": coarse,
        "fine_top1": _dist_report(top1), "fine_top5": _dist_report(top5),
        "counts": {"frames": n_frames, "candidates": n_candidates,
                   "positions_per_cell_max": max_pos, "levels": sorted(levels),
                   "cv2_calls": n_cv2, "checks_total": n_rows,
                   "failed_checks": status.get("skipped_insufficient_matches", 0)
                   + status.get("no_consensus", 0),
                   "status_breakdown": dict(status)},
        "time": {"verify_s_total": verify_s},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <diag-dir>/stats.json")
    args = ap.parse_args(argv)
    stats = compute(args.diag_dir)
    out = Path(args.out).expanduser() if args.out else Path(args.diag_dir).expanduser() / "stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    c, f1, f5 = stats["counts"], stats["fine_top1"], stats["fine_top5"]
    print(f"[stats] frames={c['frames']} candidates={c['candidates']} cv2_calls={c['cv2_calls']} "
          f"failed={c['failed_checks']} verify={stats['time']['verify_s_total']:.0f}s", flush=True)
    print(f"[fine]  top1 distR@250m={f1.get('distR@250m')} median={f1.get('median_m')}  | "
          f"top5 distR@250m={f5.get('distR@250m')} median={f5.get('median_m')}", flush=True)
    print(f"[coarse] recall@100 @250m={stats['coarse_recall@100'].get('recall@100_250m')} "
          f"@1000m={stats['coarse_recall@100'].get('recall@100_1000m')}", flush=True)
    print(f"[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
