"""Per-KMZ sub-video index: chunk -> quality sub-videos (good/usable runs) -> GT frames inside.

A **sub-video** = a maximal run of quality-cls ``good``/``usable`` frames, from the first good/usable
frame (inclusive) up to the first ``bad`` (exclusive). For every GT still (``GT_flat/<N>_<lat>_<lon>.jpg``)
we find its exact frame index in its source chunk (CV match), then which sub-video it lands in and its
index WITHIN that sub-video. One JSON per KMZ flight.

Orchestrates two things that already exist (nothing re-implemented):
  * quality-cls (its OWN torch/DINOv2 env, via subprocess) grades every frame of a chunk -> sub-videos;
  * kmz_frame_locator.core (parse_placemarks + match_still) for KMZ parsing + still->frame-index match.

Run (from ~/work/geoLocalisation; cv2/numpy/pyyaml here, torch only in the quality env)::

    uv run --with opencv-python --with numpy --with pyyaml python -m s3_gt_sync.subvideo_index \
      --kmz s3://geo-reference/gt/raw/kram/ \
      --gt-flat /home/ubuntu/work/gt_cramatorsc/GT_flat \
      --out-dir /home/ubuntu/work/gt_cramatorsc \
      --quality-python /home/ubuntu/work/RevisitAnything/quality-cls/.venv/bin/python \
      --quality-repo   /home/ubuntu/work/RevisitAnything/quality-cls
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from kmz_frame_locator.core import match_still, parse_placemarks
from kmz_video_audit.core import extract_kml_bytes

_GT_RE = re.compile(r"^(\d+)_")
_IDX_RE = re.compile(r"(\d+)\.jpg$", re.IGNORECASE)
KEEP = {"good", "usable"}


def _aws_bytes(s3_url: str) -> bytes:
    return subprocess.run(["aws", "s3", "cp", s3_url, "-"], capture_output=True).stdout


def list_kmz(prefix_url: str) -> list[str]:
    if prefix_url.lower().endswith(".kmz"):
        return [prefix_url]
    m = re.match(r"s3://([^/]+)/(.*)$", prefix_url)
    if not m:
        raise SystemExit(f"bad --kmz S3 URL: {prefix_url}")
    bucket, prefix = m.group(1), m.group(2)
    keys = subprocess.run(
        ["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix,
         "--query", "Contents[?ends_with(Key, `.kmz`)].Key", "--output", "text"],
        capture_output=True, text=True).stdout.split()
    return [f"s3://{bucket}/{k}" for k in keys]


def gt_index(gt_flat: Path) -> dict:
    idx = {}
    for p in sorted(gt_flat.glob("*.jpg")):
        m = _GT_RE.match(p.name)
        if m:
            idx.setdefault(int(m.group(1)), p)
    return idx


def _full_fps_config(quality_repo: Path, frames_dir: Path, td: Path) -> Path:
    """Copy quality-cls default.yaml but grade EVERY frame (target_fps null) into a temp frames dir."""
    import yaml
    cfg = yaml.safe_load((quality_repo / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["extract"]["target_fps"] = None
    cfg["extract"]["dedup_threshold"] = 0.0
    cfg["paths"]["frames_dir"] = str(frames_dir)
    out = td / "qc_fullfps.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def grade_chunk(chunk: Path, q_python: str, q_repo: Path, config: Path, frames_dir: Path,
                td: Path, log) -> list[str]:
    """Grade every frame of a chunk with quality-cls -> grades index-aligned (0..N-1).

    Cleans up the extracted frames + predictions CSV before returning: only the in-memory grades
    are kept, so disk footprint stays minimal even over a long multi-chunk run (per user: no video/
    frame artifacts must linger)."""
    csv = td / f"pred_{chunk.stem}.csv"
    cmd = [q_python, "-m", "quality_cls.score", str(chunk), "--config", str(config), "--out", str(csv)]
    log(f"  quality-cls grading {chunk.name} ...")
    p = subprocess.run(cmd, cwd=str(q_repo), capture_output=True, text=True)
    try:
        if p.returncode != 0 or not csv.exists():
            raise RuntimeError(f"quality-cls failed on {chunk.name}: {p.stderr[-1500:]}")
        grades = {}
        import csv as _csv
        with csv.open(encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                m = _IDX_RE.search(row["frame_path"])
                if m:
                    grades[int(m.group(1))] = row["pred_label"]
        n = (max(grades) + 1) if grades else 0
        return [grades.get(i, "bad") for i in range(n)]
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)   # drop the extracted frames
        csv.unlink(missing_ok=True)                       # drop the predictions csv


def split_subvideos(grades: list[str]) -> list[tuple[int, int]]:
    """Maximal runs of good/usable frames (inclusive start .. inclusive end).

    QUALITY-ONLY split (legacy): a run breaks only on a `bad` frame. It does NOT
    see scene cuts between two good frames (a spliced chunk) -> see
    ``split_by_quality_and_overlap`` for the continuity-aware version."""
    runs, s = [], None
    for i, g in enumerate(list(grades) + ["__bad__"]):
        good = g in KEEP
        if good and s is None:
            s = i
        elif not good and s is not None:
            runs.append((s, i - 1)); s = None
    return runs


# ----------------------- structure annotation (JSON only) -----------------------

def rle_grades(grades: list[str]) -> list[dict]:
    """Run-length timeline of the raw quality grades: the FULL video structure in
    a compact form, so downstream tools know it by index without touching video."""
    out: list[dict] = []
    for i, g in enumerate(grades):
        if out and out[-1]["label"] == g and out[-1]["end"] == i - 1:
            out[-1]["end"] = i
        else:
            out.append({"label": g, "start": i, "end": i})
    return out


def bad_frame_ranges(grades: list[str]) -> list[list[int]]:
    """[[start, end], ...] inclusive index ranges of frames graded not good/usable."""
    return [[r["start"], r["end"]] for r in rle_grades(grades) if r["label"] not in KEEP]


# ------------------- continuity-aware split (ORB + RANSAC, CPU) -------------------

def overlap_ratio(desc_a, kp_a, desc_b, kp_b, matcher, ransac_thr: float) -> float:
    """RANSAC-homography inlier ratio between two frames' ORB features.

    High = same continuous scene (survives fast UAV ego-motion, which a homography
    explains); low = a cut/splice (no consistent planar transform). This is the
    UAV-appropriate continuity signal (see literature: motion-compensated overlap)."""
    if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0.0
    matches = matcher.match(desc_a, desc_b)
    if len(matches) < 8:
        return 0.0
    pa = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pb = np.float32([kp_b[m.trainIdx].pt for m in matches])
    _H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_thr)
    if mask is None:
        return 0.0
    return float(mask.sum()) / float(len(matches))


def link_runs(keep_indices, ratio_of, *, bridge_max_gap: int, min_inlier_ratio: float,
              min_scene_len: int) -> list[tuple[int, int]]:
    """Group KEEP frames into continuous-scene runs (pure, testable).

    A boundary is inserted between consecutive KEEP frames when the overlap
    (``ratio_of(prev, cur)``) drops below ``min_inlier_ratio`` OR the bad-frame gap
    between them exceeds ``bridge_max_gap`` (a long dropout -> do not bridge). Short
    bad gaps with sufficient overlap are BRIDGED into one sub-video. Runs shorter
    than ``min_scene_len`` frames are dropped."""
    runs: list[tuple[int, int]] = []
    start = prev = None
    for idx in keep_indices:
        if prev is None:
            start = idx
        else:
            bad_between = idx - prev - 1
            cont = (bad_between <= bridge_max_gap) and (ratio_of(prev, idx) >= min_inlier_ratio)
            if not cont:
                runs.append((start, prev)); start = idx
        prev = idx
    if start is not None:
        runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_scene_len]


def split_by_quality_and_overlap(chunk_path, grades: list[str], *, orb_features=1200,
        downscale=2, ransac_thr=4.0, min_inlier_ratio=0.15, bridge_max_gap=8,
        min_scene_len=6, log=None) -> list[tuple[int, int]]:
    """Sub-videos = continuous-scene runs of KEEP frames.

    Cuts a good run at splices (inlier ratio collapses) and bridges short bad-frame
    dropouts (overlap survives). CPU ORB is cheap at this scale. Returns inclusive
    (start_frame, end_frame) runs, indexed in the chunk."""
    keep = [i for i, g in enumerate(grades) if g in KEEP]
    if not keep:
        return []
    keepset = set(keep)
    orb = cv2.ORB_create(nfeatures=orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cap = cv2.VideoCapture(str(chunk_path))
    ratios: dict[tuple[int, int], float] = {}
    prev = None  # (idx, kp, desc)
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx >= len(grades):
            break
        if idx not in keepset:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // downscale, gray.shape[0] // downscale))
        kp, desc = orb.detectAndCompute(gray, None)
        if prev is not None:
            pidx, pkp, pdesc = prev
            if idx - pidx - 1 <= bridge_max_gap:   # only score pairs we might bridge
                ratios[(pidx, idx)] = overlap_ratio(pdesc, pkp, desc, kp, matcher, ransac_thr)
        prev = (idx, kp, desc)
    cap.release()
    runs = link_runs(keep, lambda a, b: ratios.get((a, b), 0.0),
                     bridge_max_gap=bridge_max_gap, min_inlier_ratio=min_inlier_ratio,
                     min_scene_len=min_scene_len)
    if log:
        log(f"  overlap split: {len(keep)} keep frames -> {len(runs)} sub-videos "
            f"(quality-only: {len(split_subvideos(grades))})")
    return runs


def _city_ver(kmz_url: str) -> tuple[str, str]:
    m = re.search(r"gt/raw/([^/]+)/([^/]+)/", kmz_url)
    return (m.group(1), m.group(2)) if m else ("unknown", Path(kmz_url).stem)


def process_kmz(kmz_url: str, gt: dict, q_python: str, q_repo: Path, td: Path, log,
                overlap: dict | None = None, min_ncc: float = 0.0) -> dict:
    city, ver = _city_ver(kmz_url)
    pms = parse_placemarks(extract_kml_bytes(_aws_bytes(kmz_url)))
    by_video: dict = {}
    for pm in pms:
        if pm.video_url and pm.n in gt:
            by_video.setdefault(pm.video_url, []).append(pm)
    rec = {"kmz": kmz_url, "city": city, "version": ver, "videos": []}
    frames_dir = td / "qc_frames"
    config = _full_fps_config(q_repo, frames_dir, td)
    for vurl, pmlist in by_video.items():
        local = td / Path(vurl).name
        if not local.exists():
            subprocess.run(["aws", "s3", "cp", vurl, str(local)], capture_output=True)
        if not local.exists():
            rec["videos"].append({"video": vurl, "error": "download failed"}); continue
        try:
            grades = grade_chunk(local, q_python, q_repo, config, frames_dir, td, log)
        except Exception as e:                                  # noqa: BLE001
            rec["videos"].append({"video": vurl, "error": str(e)[:300]})
            local.unlink(missing_ok=True)                       # never leave the chunk behind
            continue
        if overlap and overlap.get("enable", True):
            subs = split_by_quality_and_overlap(
                local, grades, log=log,
                orb_features=overlap.get("orb_features", 1200),
                downscale=overlap.get("downscale", 2),
                ransac_thr=overlap.get("ransac_thr", 4.0),
                min_inlier_ratio=overlap.get("min_inlier_ratio", 0.15),
                bridge_max_gap=overlap.get("bridge_max_gap", 8),
                min_scene_len=overlap.get("min_scene_len", 6))
        else:
            subs = split_subvideos(grades)
        # match each GT still -> abs frame index, then locate its sub-video
        matched = {}
        low_ncc = []
        fps = 30.0
        for pm in pmlist:
            m = match_still(gt[pm.n], local)
            if m is None:
                continue
            idx, fps, t_s, ncc, _frame, _nfr = m
            if ncc < min_ncc:                      # wrong-frame match -> do not place it
                low_ncc.append({"n": pm.n, "ncc": round(ncc, 4), "frame_index_in_chunk": idx})
                continue
            matched[pm.n] = (idx, t_s, ncc, pm)
        vrec = {"video": vurl, "fps": round(fps, 3), "n_frames": len(grades),
                "bad_frames": bad_frame_ranges(grades),   # index ranges, video untouched
                "grades_rle": rle_grades(grades),         # full quality structure
                "sub_videos": []}
        for si, (a, b) in enumerate(subs):
            sv = {"sub_index": si, "start_frame": a, "end_frame": b,
                  "start_s": round(a / fps, 3) if fps else None,
                  "end_s": round(b / fps, 3) if fps else None, "gt_frames": []}
            for n, (idx, t_s, ncc, pm) in matched.items():
                if a <= idx <= b:
                    sv["gt_frames"].append({
                        "n": n, "file": gt[n].name, "lat": pm.lat, "lon": pm.lon,
                        "frame_index_in_chunk": idx, "frame_index_in_subvideo": idx - a,
                        "time_s": round(t_s, 3), "ncc": round(ncc, 4)})
            vrec["sub_videos"].append(sv)
        # GT frames that matched to a bad frame (in no sub-video)
        in_sub = {n for sv in vrec["sub_videos"] for f in sv["gt_frames"] for n in [f["n"]]}
        drop = [n for n in matched if n not in in_sub]
        if drop:
            vrec["gt_on_bad_frames"] = sorted(drop)
        if low_ncc:
            vrec["gt_low_ncc"] = sorted(low_ncc, key=lambda d: d["n"])
        rec["videos"].append(vrec)
        local.unlink(missing_ok=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kmz", required=True, help="s3:// KMZ prefix (city) or a single .kmz URL")
    ap.add_argument("--gt-flat", required=True)
    ap.add_argument("--out-dir", required=True, help="write one JSON per KMZ here (e.g. gt_cramatorsc)")
    ap.add_argument("--quality-python", required=True)
    ap.add_argument("--quality-repo", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap KMZ processed (smoke)")
    # continuity-aware split (ORB+RANSAC overlap). On by default; --no-overlap-split
    # falls back to the legacy quality-only split.
    ap.add_argument("--overlap-split", dest="overlap_split", action="store_true", default=True)
    ap.add_argument("--no-overlap-split", dest="overlap_split", action="store_false")
    ap.add_argument("--min-inlier-ratio", type=float, default=0.15,
                    help="below this RANSAC inlier ratio between adjacent KEEP frames = a cut")
    ap.add_argument("--bridge-max-gap", type=int, default=8,
                    help="max bad-frame dropout (frames) to bridge inside one sub-video")
    ap.add_argument("--min-scene-len", type=int, default=6, help="drop sub-videos shorter than this")
    ap.add_argument("--orb-features", type=int, default=1200)
    ap.add_argument("--overlap-downscale", type=int, default=2, help="downscale frames for ORB speed")
    ap.add_argument("--ransac-thr", type=float, default=4.0)
    ap.add_argument("--min-ncc", type=float, default=0.9,
                    help="drop GT-still->frame matches below this NCC (wrong-frame matches "
                         "that produce impossible interpolation speeds)")
    args = ap.parse_args(argv)

    overlap = {
        "enable": args.overlap_split,
        "min_inlier_ratio": args.min_inlier_ratio,
        "bridge_max_gap": args.bridge_max_gap,
        "min_scene_len": args.min_scene_len,
        "orb_features": args.orb_features,
        "downscale": args.overlap_downscale,
        "ransac_thr": args.ransac_thr,
    }

    gt = gt_index(Path(args.gt_flat).expanduser())
    out_dir = Path(args.out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    q_repo = Path(args.quality_repo).expanduser()
    kmz_keys = list_kmz(args.kmz)
    print(f"[gt] {len(gt)} frames | [kmz] {len(kmz_keys)} KMZ under {args.kmz}", flush=True)

    def log(m):
        print(m, flush=True)

    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        for i, kurl in enumerate(kmz_keys):
            if args.limit and i >= args.limit:
                break
            rec = process_kmz(kurl, gt, args.quality_python, q_repo, td, log, overlap, args.min_ncc)
            nsv = sum(len(v.get("sub_videos", [])) for v in rec["videos"])
            ngf = sum(len(s["gt_frames"]) for v in rec["videos"] for s in v.get("sub_videos", []))
            name = f"{rec['city']}_{rec['version']}.json"
            (out_dir / name).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] {kurl}: {len(rec['videos'])} videos, {nsv} sub-videos, {ngf} GT frames -> {name}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
