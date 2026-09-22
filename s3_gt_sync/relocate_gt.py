"""Relocate mis-associated GT stills to their TRUE video+frame (don't drop them).

A GT still is a literal frame of some video. If match_still scores a low NCC
against the chunk the KMZ linked it to, the still actually belongs to a DIFFERENT
chunk. This tool searches the OTHER candidate videos in the same json (same KMZ /
flight by default) and re-assigns the still to the chunk where it matches with
high NCC - keeping its name (N + file). It fixes KMZ<->video mis-association by
image evidence instead of discarding real GT.

    python -m s3_gt_sync.relocate_gt --json /home/ubuntu/work/gt_cramatorsc/kram_3.0.json \
        --gt-flat /home/ubuntu/work/gt_cramatorsc/GT_flat --min-ncc 0.9 \
        --out /home/ubuntu/work/gt_cramatorsc/kram_3.0.relocated.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from kmz_frame_locator.core import match_still

from .subvideo_index import gt_index


# ------------------------- pure helpers (testable) -------------------------

def candidate_videos(rec: dict) -> list[str]:
    """Distinct video URLs referenced in this json (same KMZ / flight)."""
    seen: list[str] = []
    for v in rec.get("videos", []):
        u = v.get("video")
        if u and u not in seen:
            seen.append(u)
    return seen


def misplaced_anchors(rec: dict, min_ncc: float) -> list[dict]:
    """Placed anchors with ncc < min_ncc (wrong video) + any recorded gt_low_ncc."""
    out: list[dict] = []
    for v in rec.get("videos", []):
        vurl = v.get("video")
        for sv in v.get("sub_videos", []):
            for f in sv.get("gt_frames", []):
                if f.get("ncc", 1.0) < min_ncc:
                    out.append({"n": f["n"], "file": f.get("file"),
                                "lat": f.get("lat"), "lon": f.get("lon"),
                                "from_video": vurl,
                                "from_frame": f.get("frame_index_in_chunk"),
                                "from_ncc": f.get("ncc")})
        for e in v.get("gt_low_ncc", []):
            out.append({"n": e["n"], "file": None, "lat": None, "lon": None,
                        "from_video": vurl, "from_frame": e.get("frame_index_in_chunk"),
                        "from_ncc": e.get("ncc")})
    return out


def find_true_home(still_path, candidate_urls, get_video, match_fn, skip_url=None):
    """Return (url, (idx, ncc, fps)) of the best-matching candidate, or None.

    get_video(url) -> local path or None; match_fn(still, video) -> (idx, ncc, fps)
    or None. Both injected so the search logic is unit-testable without real I/O."""
    best = None
    for url in candidate_urls:
        if url == skip_url:
            continue
        vp = get_video(url)
        if not vp:
            continue
        r = match_fn(still_path, vp)
        if not r:
            continue
        if best is None or r[1] > best[1][1]:
            best = (url, r)
    return best


def place_anchor_in_rec(rec: dict, url: str, anchor: dict) -> str:
    """Insert a relocated anchor into url's sub_video that contains its frame.
    Returns a status string. Mutates rec."""
    for v in rec.get("videos", []):
        if v.get("video") != url:
            continue
        idx = anchor["frame_index_in_chunk"]
        for sv in v.get("sub_videos", []):
            if sv["start_frame"] <= idx <= sv["end_frame"]:
                sv.setdefault("gt_frames", []).append({
                    "n": anchor["n"], "file": anchor.get("file"),
                    "lat": anchor.get("lat"), "lon": anchor.get("lon"),
                    "frame_index_in_chunk": idx,
                    "frame_index_in_subvideo": idx - sv["start_frame"],
                    "time_s": anchor.get("time_s"),
                    "ncc": anchor.get("ncc"), "relocated": True})
                return f"placed in sub{sv['sub_index']}"
        v.setdefault("relocated_on_bad_frame", []).append(anchor["n"])
        return "matched but frame is bad/no sub-video"
    return "target video not in json"


def remove_anchor(rec: dict, from_video: str, n: int) -> None:
    """Remove anchor n from from_video's gt_frames / gt_low_ncc."""
    for v in rec.get("videos", []):
        if v.get("video") != from_video:
            continue
        for sv in v.get("sub_videos", []):
            sv["gt_frames"] = [f for f in sv.get("gt_frames", []) if f.get("n") != n]
        if "gt_low_ncc" in v:
            v["gt_low_ncc"] = [e for e in v["gt_low_ncc"] if e.get("n") != n]


# ------------------------------- I/O runner -------------------------------

class _VideoCache:
    def __init__(self, td: Path):
        self.td = td
        self._map: dict[str, Path | None] = {}

    def get(self, url: str):
        if url in self._map:
            return self._map[url]
        local = self.td / Path(url).name
        if not local.exists():
            subprocess.run(["aws", "s3", "cp", url, str(local)], capture_output=True)
        p = local if local.exists() else None
        self._map[url] = p
        return p


def _match_fn(still_path, video_path):
    m = match_still(Path(still_path), Path(video_path))
    if m is None:
        return None
    idx, fps, _t, ncc, _f, _n = m
    return idx, ncc, fps


def relocate(json_path, gt_flat, min_ncc=0.9, out_path=None, dry_run=False, log=print):
    rec = json.loads(Path(json_path).read_text(encoding="utf-8"))
    gt = gt_index(Path(gt_flat).expanduser())
    cands = candidate_videos(rec)
    mp = misplaced_anchors(rec, min_ncc)
    log(f"[relocate] {len(mp)} anchors with ncc<{min_ncc}; {len(cands)} candidate videos")

    report = {"json": str(json_path), "min_ncc": min_ncc,
              "n_candidates": len(cands), "relocations": [], "unresolved": []}

    with tempfile.TemporaryDirectory() as _td:
        cache = _VideoCache(Path(_td))
        for a in mp:
            still = gt.get(a["n"])
            if still is None:
                report["unresolved"].append({**a, "reason": "no GT_flat file for n"})
                continue
            best = find_true_home(still, cands, cache.get, _match_fn, skip_url=a["from_video"])
            if best is None or best[1][1] < min_ncc:
                report["unresolved"].append({
                    **a, "best_ncc": (round(best[1][1], 4) if best else None),
                    "reason": "no candidate matched >= min_ncc"})
                continue
            url, (idx, ncc, fps) = best
            entry = {"n": a["n"], "file": (still.name if still else a.get("file")),
                     "lat": a.get("lat"), "lon": a.get("lon"),
                     "from_video": a["from_video"], "from_ncc": a.get("from_ncc"),
                     "to_video": url, "to_frame": idx, "to_ncc": round(ncc, 4),
                     "time_s": round(idx / fps, 3) if fps else None}
            report["relocations"].append(entry)
            if not dry_run:
                remove_anchor(rec, a["from_video"], a["n"])
                anchor = {"n": a["n"], "file": entry["file"], "lat": a.get("lat"),
                          "lon": a.get("lon"), "frame_index_in_chunk": idx,
                          "time_s": entry["time_s"], "ncc": round(ncc, 4)}
                entry["placement"] = place_anchor_in_rec(rec, url, anchor)
            log(f"  n{a['n']} {a.get('from_ncc')} -> {Path(url).name} @f{idx} ncc {ncc:.4f}")

    if not dry_run and out_path:
        Path(out_path).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[relocate] wrote corrected json -> {out_path}")
    log(f"[relocate] relocated={len(report['relocations'])} unresolved={len(report['unresolved'])}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--gt-flat", required=True)
    ap.add_argument("--min-ncc", type=float, default=0.9)
    ap.add_argument("--out", default=None, help="corrected json (default: <json>.relocated.json)")
    ap.add_argument("--report", default=None, help="write relocation report json here")
    ap.add_argument("--dry-run", action="store_true", help="only report, do not rewrite")
    args = ap.parse_args(argv)
    out = args.out or (str(Path(args.json).with_suffix("")) + ".relocated.json")
    rep = relocate(args.json, args.gt_flat, args.min_ncc,
                   None if args.dry_run else out, args.dry_run)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[relocate] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
