"""CSV-driven MOTION-BUDGET frame selection for whole-video geo-localization.

Unlike :mod:`clip_search.sample` (GT-centred ±window frames), this picks frames with the
**production keyframe selector** — the quality-gated motion budget from
``pipeline1_interframe.build_label_set`` (``place_in_motion``): DINOv2 quality-gate → good→good
Farneback motion → split on quality gaps (``max_gap``) → keyframes spaced evenly in CUMULATIVE
motion. Every selected frame is tagged with its VIDEO's known GT lat/lon, so per-frame and
per-video ``distR`` are both measured against that point.

Input = the labelled-detections CSV (``mapforml_public_Detection.csv``; NO header, 31 columns). We
keep one station zone (default the Kupyansk node ``0-17-34-138``), download each chunk once, select
frames, and write a ``clips_manifest.json``-COMPATIBLE manifest (one "clip" == one video) so the
existing downstream runs unchanged: ``map_extract.extract --images`` → ``coarse_rank_gallery`` →
``pyramid_newformat.locate`` → ``clip_search.aggregate``.

    python -m clip_search.sample_motion --csv mapforml_public_Detection.csv \
        --zone 0-17-34-138 --out-dir ~/work/kup_vidloc --quality-gate good --target-k 8 \
        --checkpoint ~/work/RevisitAnything/quality-cls/checkpoints/head.pt
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- repo root on sys.path (nested geoLocalisation repo) so sibling packages import ----
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# CSV column indices (0-based; the file has NO header and 31 columns)
KUP_ZONE = "0-17-34-138"
C_ID, C_ZONE, C_VIDEO = 0, 4, 5
C_PRED_LAT, C_PRED_LON = 8, 9
C_GT_LAT, C_GT_LON = 27, 28          # verified / corrected pair = the labelled GT
C_MATCHED = 29                       # "true" = predicted == verified


# ------------------------------- pure helpers (testable) -------------------------------

def _fmt_ll(x: float) -> str:
    """lat/lon → filename-safe token that ``map_extract._parse_image_stem`` round-trips via float()."""
    return f"{x:.7f}"


def frame_stem(gidx: int, lat: float, lon: float) -> str:
    """``{gidx:08d}_{lat}_{lon}`` — parsed by ``map_extract.helpers._parse_image_stem``."""
    return f"{gidx:08d}_{_fmt_ll(lat)}_{_fmt_ll(lon)}"


def make_qid(city: str, stem: str) -> str:
    """Query id as ``QueryGridStore`` builds it: ``<city>:<stem>``."""
    return f"{city}:{stem}"


def parse_csv(path, zone: str | None = KUP_ZONE) -> list[dict]:
    """Parse the labelled-detections CSV → one record per (unique) video in ``zone``.

    Each record: ``{video_id, video_url, zone, gt_lat, gt_lon, pred_lat, pred_lon, matched}``.
    GT = verified pair (cols 27/28); predicted = cols 8/9; ``matched`` = col 29 == "true".
    Rows with a non-s3 video, un-parseable GT, or a different zone are skipped. Duplicate video
    URLs keep the first row.
    """
    out: list[dict] = []
    seen: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        for row in _csv.reader(f):
            if len(row) <= C_GT_LON:
                continue
            if zone and row[C_ZONE].strip() != zone:
                continue
            vurl = row[C_VIDEO].strip()
            if not vurl.startswith("s3://") or vurl in seen:
                continue
            try:
                gt_lat, gt_lon = float(row[C_GT_LAT]), float(row[C_GT_LON])
            except ValueError:
                continue

            def _f(i):
                try:
                    return float(row[i])
                except (ValueError, IndexError):
                    return None

            seen.add(vurl)
            out.append({
                "video_id": row[C_ID].strip(),
                "video_url": vurl,
                "zone": row[C_ZONE].strip(),
                "gt_lat": gt_lat, "gt_lon": gt_lon,
                "pred_lat": _f(C_PRED_LAT), "pred_lon": _f(C_PRED_LON),
                "matched": (row[C_MATCHED].strip().lower() == "true") if len(row) > C_MATCHED else None,
            })
    return out


def build_segments(grays, labels, *, quality_gate: set | None, max_gap: int) -> list:
    """Production segmenting (reuses ``build_label_set``): if ``quality_gate`` given, keep gated
    frames and split good→good runs on gaps > ``max_gap``; else one segment with adjacent motion.
    Returns ``[(positions:list[int], cum_motion:list[float]), ...]`` (positions index ``grays``)."""
    from pipeline1_interframe.build_label_set import build_gated_segments, fb
    n = len(grays)
    if quality_gate:
        gated = [p for p in range(n) if labels[p] in quality_gate]
        return build_gated_segments(grays, gated, max_gap)
    cum, acc = [0.0], 0.0
    for i in range(1, n):
        acc += fb(grays[i - 1], grays[i])
        cum.append(acc)
    return [(list(range(n)), cum)]


def select_positions(segments, *, target_k: int | None = None, frames_per_key: float = 50.0) -> list[int]:
    """Keyframe positions (into the original frame list) via the production motion budget
    (``place_in_motion``). ``target_k`` (if set) binary-searches ``frames_per_key`` to hit ~k total.
    Returns sorted unique positions. Pure except for the reused ``build_label_set`` helpers."""
    import numpy as np
    from pipeline1_interframe.build_label_set import fpk_for_k, place_in_motion
    steps = [cum[i] - cum[i - 1] for _, cum in segments for i in range(1, len(cum))]
    nz = [s for s in steps if s > 0]
    mu = float(np.median(nz)) if nz else 0.0
    fpk = frames_per_key
    if target_k is not None and mu > 0:
        fpk = fpk_for_k(segments, mu, target_k)
    budget = fpk * mu
    picked: list[int] = []
    for pos, cum in segments:
        for lp in place_in_motion(cum, budget):
            picked.append(pos[lp])
    return sorted(set(picked))


def clip_entry(rec: dict, city: str, frames_meta: list[dict]) -> dict:
    """Assemble one ``clips_manifest`` clip (== one video) from a CSV record + its frame metas."""
    return {
        "clip_id": f"{city}_{rec['video_id']}", "city": city, "video_id": rec["video_id"],
        "video": rec["video_url"], "gt_lat": rec["gt_lat"], "gt_lon": rec["gt_lon"],
        "pred_lat": rec.get("pred_lat"), "pred_lon": rec.get("pred_lon"), "matched": rec.get("matched"),
        "n_selected": len(frames_meta), "frames": frames_meta,
    }


# ------------------------------- I/O runner (needs cv2 / quality GPU) -------------------------------

def _download(vurl: str, dest: Path) -> bool:
    if dest.exists():
        return True
    subprocess.run(["aws", "s3", "cp", vurl, str(dest)], capture_output=True)
    return dest.exists()


def process_video(rec: dict, local_video: Path, out_frames_dir: Path, gidx: int, *, city: str,
                  quality_gate: set | None, target_k: int | None, frames_per_key: float,
                  max_gap: int, max_side: int, decode_every: int, jpeg_quality: int,
                  backend=None, log=print) -> tuple[dict, int]:
    """Select + write frames for ONE already-downloaded video. Returns (clip_entry, next_gidx)."""
    import cv2
    from pipeline1_interframe.build_label_set import to_grays
    from video_label_pipeline.video_io import read_all_frames

    fps, _, kept = read_all_frames(local_video, every=decode_every)
    if not kept:
        log(f"  !! no frames decoded: {local_video.name}")
        return clip_entry(rec, city, []), gidx
    frames = [k[2] for k in kept]
    orig_idx = [k[0] for k in kept]

    labels = None
    if quality_gate:
        labels = [lab for lab, _ in backend.score(frames)]

    grays = to_grays(frames, max_side)
    segments = build_segments(grays, labels, quality_gate=quality_gate, max_gap=max_gap)
    positions = select_positions(segments, target_k=target_k, frames_per_key=frames_per_key)

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    frames_meta: list[dict] = []
    for p in positions:
        stem = frame_stem(gidx, rec["gt_lat"], rec["gt_lon"])
        fpath = out_frames_dir / f"{stem}.jpg"
        cv2.imwrite(str(fpath), frames[p], [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        frames_meta.append({"global_idx": gidx, "frame_index": int(orig_idx[p]), "stem": stem,
                            "qid": make_qid(city, stem), "file": fpath.name,
                            "quality": (labels[p] if labels is not None else None)})
        gidx += 1
    log(f"  {rec['video_id']}: {len(frames)} frames -> {len(frames_meta)} keyframes")
    return clip_entry(rec, city, frames_meta), gidx


def run(csv_path, out_dir, *, zone: str | None = KUP_ZONE, city: str = "kup",
        quality_gate: set | None = None, target_k: int | None = None, frames_per_key: float = 50.0,
        max_gap: int = 20, max_side: int = 384, decode_every: int = 1, jpeg_quality: int = 95,
        checkpoint: str | None = None, local_video_dir: str | None = None, limit: int = 0,
        log=print) -> dict:
    """Full run: parse CSV → per video download/select/write frames → one ``clips_manifest.json``."""
    recs = parse_csv(csv_path, zone=zone)
    if limit:
        recs = recs[:limit]
    log(f"[csv] {len(recs)} videos in zone {zone!r}")
    out_dir = Path(out_dir).expanduser()
    frames_dir = out_dir / "frames"

    backend = None
    if quality_gate:                                    # DINOv2 quality model (GPU) — construct once
        from video_label_pipeline.quality import QualityClsBackend
        backend = QualityClsBackend(checkpoint=checkpoint)

    clips: list[dict] = []
    gidx = 0
    lvdir = Path(local_video_dir).expanduser() if local_video_dir else None
    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        for rec in recs:
            name = Path(rec["video_url"]).name
            if lvdir is not None and (lvdir / name).exists():
                local, downloaded = lvdir / name, False
            else:
                local, downloaded = td / name, True
                if not _download(rec["video_url"], local):
                    log(f"  !! download failed: {rec['video_url']}")
                    clips.append(clip_entry(rec, city, []))
                    continue
            entry, gidx = process_video(
                rec, local, frames_dir, gidx, city=city, quality_gate=quality_gate,
                target_k=target_k, frames_per_key=frames_per_key, max_gap=max_gap, max_side=max_side,
                decode_every=decode_every, jpeg_quality=jpeg_quality, backend=backend, log=log)
            clips.append(entry)
            if downloaded:
                local.unlink(missing_ok=True)

    n_frames = sum(len(c["frames"]) for c in clips)
    manifest = {
        "meta": {"source": "csv", "csv": str(csv_path), "zone": zone, "city": city,
                 "selector": {"quality_gate": sorted(quality_gate) if quality_gate else None,
                              "target_k": target_k, "frames_per_key": frames_per_key,
                              "max_gap": max_gap, "max_side": max_side, "decode_every": decode_every},
                 "n_clips": len(clips), "n_frames": n_frames, "frames_dir": str(frames_dir)},
        "clips": clips}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    log(f"[ok] {len(clips)} videos, {n_frames} keyframes -> {frames_dir} + clips_manifest.json")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=r"C:\Users\CR-L-0040\Desktop\teory\mapforml_public_Detection.csv",
                    help="labelled-detections CSV (no header, 31 cols)")
    ap.add_argument("--out-dir", default=None, help="writes frames/ + clips_manifest.json here")
    ap.add_argument("--zone", default=KUP_ZONE, help="keep only this station-node zone (col 4)")
    ap.add_argument("--city", default="kup", help="city prefix for qids / gallery")
    ap.add_argument("--quality-gate", default=None,
                    help="restrict keyframes to these labels, e.g. 'good' or 'good,usable' (needs --checkpoint)")
    ap.add_argument("--checkpoint", default=None, help="quality-cls checkpoint (head.pt) for --quality-gate")
    ap.add_argument("--target-k", type=int, default=None, help="pick budget for ~this many keyframes/video")
    ap.add_argument("--frames-per-key", type=float, default=50.0, help="motion budget = this × median step")
    ap.add_argument("--max-gap", type=int, default=20, help="gated: gap>this frames splits a propagation run")
    ap.add_argument("--max-side", type=int, default=384, help="downscale for optical flow")
    ap.add_argument("--decode-every", type=int, default=1, help="decode every Nth frame (1 = faithful)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--local-video-dir", default=None, help="reuse pre-downloaded chunks by basename")
    ap.add_argument("--limit", type=int, default=0, help="cap videos (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="only parse + count, no download/decode")
    args = ap.parse_args(argv)

    gate = set(s.strip() for s in args.quality_gate.split(",")) if args.quality_gate else None
    if gate and not args.checkpoint:
        print("[warn] --quality-gate set without --checkpoint; quality scoring will fail", flush=True)

    if args.dry_run:
        recs = parse_csv(args.csv, zone=args.zone)
        print(f"[dry-run] {len(recs)} videos in zone {args.zone!r} "
              f"({sum(1 for r in recs if r['matched'] is False)} with corrected GT)")
        return 0

    if not args.out_dir:
        ap.error("--out-dir is required (unless --dry-run)")

    run(args.csv, args.out_dir, zone=args.zone, city=args.city, quality_gate=gate,
        target_k=args.target_k, frames_per_key=args.frames_per_key, max_gap=args.max_gap,
        max_side=args.max_side, decode_every=args.decode_every, jpeg_quality=args.jpeg_quality,
        checkpoint=args.checkpoint, local_video_dir=args.local_video_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
