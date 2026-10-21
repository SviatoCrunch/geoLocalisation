"""Clip windows around each GT + frame extraction (no video kept, no DINO here).

A **clip** = a window of up to ``window`` (default 32) CONSECUTIVE frames centred on a GT frame,
clipped to the GT's sub-video (so it may be shorter near a sub-video edge — "може бути менше").
Input is one or more ``s3_gt_sync.subvideo_index`` JSONs (GT already matched to its exact
``frame_index_in_chunk`` + lat/lon inside a sub-video). We:

  1. build the window for every GT (pure, testable: :func:`build_clips`);
  2. download each referenced video ONCE and grab only the needed frames (one sequential pass);
  3. write ``frames/{global_idx}_{gt_lat}_{gt_lon}.jpg`` (so ``map_extract --images`` parses lat/lon =
     the clip's GT) + ``clips_manifest.json`` mapping ``qid = "<city>:<stem>"`` → clip.

Every frame of a clip is tagged with the CLIP's GT lat/lon, so per-frame and clip-aggregate error are
both measured against that GT. The video and extracted temp download are removed after each video.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def _city_ver(rec: dict) -> tuple[str, str]:
    return str(rec.get("city", "unknown")), str(rec.get("version", "0"))


def _fmt_ll(x: float) -> str:
    """lat/lon → filename-safe token that ``map_extract._parse_image_stem`` round-trips via float()."""
    return f"{x:.7f}"


def window_bounds(gt_idx: int, sub_start: int, sub_end: int, window: int = 32) -> tuple[int, int]:
    """[a, b] inclusive frame range: ``window`` frames with GT centred, clipped to the sub-video.

    Even ``window`` puts the GT at position ``window//2`` (so 16 before + GT + 15 after for 32). Clipping
    to the sub-video can shorten it; we do NOT re-expand the other side (the result may be < ``window``).
    """
    half = window // 2
    a = max(sub_start, gt_idx - half)
    b = min(sub_end, gt_idx + (window - half) - 1)
    return a, b


def build_clips(index_json_paths, window: int = 32) -> list[dict]:
    """Parse ``subvideo_index`` JSON(s) → one clip dict per GT frame (pure; no video I/O).

    clip = {clip_id, city, version, video, gt_n, gt_lat, gt_lon, sub_index, gt_frame_index,
            start_frame, end_frame, frame_indices:[...]}. GT frames on bad frames / low-ncc (not inside
    any sub-video) are skipped (they have no clip window). ``clip_id`` is unique per (city, version, n).
    """
    clips: list[dict] = []
    for jp in index_json_paths:
        rec = json.loads(Path(jp).expanduser().read_text(encoding="utf-8"))
        if not isinstance(rec, dict) or "videos" not in rec:
            continue
        city, ver = _city_ver(rec)
        for v in rec["videos"]:
            vurl = v.get("video")
            if not vurl or v.get("error"):
                continue
            for sv in v.get("sub_videos", []):
                a0, b0 = int(sv["start_frame"]), int(sv["end_frame"])
                for gf in sv.get("gt_frames", []):
                    gi = int(gf["frame_index_in_chunk"])
                    a, b = window_bounds(gi, a0, b0, window)
                    clips.append({
                        "clip_id": f"{city}_{ver}_n{gf['n']}",
                        "city": city, "version": ver, "video": vurl,
                        "gt_n": gf["n"], "gt_lat": float(gf["lat"]), "gt_lon": float(gf["lon"]),
                        "sub_index": int(sv["sub_index"]), "gt_frame_index": gi,
                        "start_frame": a, "end_frame": b,
                        "frame_indices": list(range(a, b + 1)),
                    })
    return clips


def _download(vurl: str, dest: Path) -> bool:
    if dest.exists():
        return True
    subprocess.run(["aws", "s3", "cp", vurl, str(dest)], capture_output=True)
    return dest.exists()


def extract_frames(clips: list[dict], out_dir: Path, *, jpeg_quality: int = 95, log=print) -> dict:
    """Download each video once, grab the union of needed frames, write query jpgs + manifest.

    Returns the manifest dict (also written to ``out_dir/clips_manifest.json``). Each clip gains a
    ``frames`` list of {global_idx, frame_index, stem, qid, file}. Frames that could not be read
    (seek past EOF / decode error) are recorded in the clip's ``missing_frames``.
    """
    import cv2

    out_dir = Path(out_dir).expanduser()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    by_video: dict[str, list[dict]] = {}
    for c in clips:
        by_video.setdefault(c["video"], []).append(c)

    gidx = 0
    manifest_clips: list[dict] = []
    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        for vurl, vclips in by_video.items():
            local = td / Path(vurl).name
            if not _download(vurl, local):
                log(f"[skip] download failed: {vurl}")
                for c in vclips:
                    manifest_clips.append({**c, "frames": [], "missing_frames": c["frame_indices"],
                                           "error": "download failed"})
                continue
            need = sorted({i for c in vclips for i in c["frame_indices"]})
            log(f"[vid] {Path(vurl).name}: {len(vclips)} clips, {len(need)} unique frames")
            want = set(need)
            last = need[-1] if need else -1
            cache: dict[int, "any"] = {}
            cap = cv2.VideoCapture(str(local))
            i = -1
            while i < last:
                ok, fr = cap.read()
                if not ok:
                    break
                i += 1
                if i in want:
                    cache[i] = fr
            cap.release()

            for c in vclips:
                frames_meta, missing = [], []
                for fi in c["frame_indices"]:
                    fr = cache.get(fi)
                    if fr is None:
                        missing.append(fi)
                        continue
                    stem = f"{gidx:08d}_{_fmt_ll(c['gt_lat'])}_{_fmt_ll(c['gt_lon'])}"
                    fpath = frames_dir / f"{stem}.jpg"
                    cv2.imwrite(str(fpath), fr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
                    frames_meta.append({"global_idx": gidx, "frame_index": fi, "stem": stem,
                                        "qid": f"{c['city']}:{stem}", "file": fpath.name})
                    gidx += 1
                entry = {**c, "frames": frames_meta}
                if missing:
                    entry["missing_frames"] = missing
                manifest_clips.append(entry)
            local.unlink(missing_ok=True)

    n_frames = sum(len(c["frames"]) for c in manifest_clips)
    manifest = {"meta": {"window": None, "n_clips": len(manifest_clips), "n_frames": n_frames,
                         "frames_dir": str(frames_dir)},
                "clips": manifest_clips}
    (out_dir / "clips_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    log(f"[ok] {len(manifest_clips)} clips, {n_frames} frames -> {frames_dir} + clips_manifest.json")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", nargs="+", required=True, help="subvideo_index JSON(s) (e.g. kram_2.*.json)")
    ap.add_argument("--out-dir", required=True, help="writes frames/ + clips_manifest.json here")
    ap.add_argument("--window", type=int, default=32, help="frames per clip (GT centred, clipped to sub-video)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--dry-run", action="store_true", help="only build + count clips, no video download")
    args = ap.parse_args(argv)

    clips = build_clips(args.index, window=args.window)
    print(f"[clips] {len(clips)} clips from {len(args.index)} index json(s), window={args.window}", flush=True)
    if not clips:
        print("[clip_search] no clips (no sub-videos with GT frames)"); return 2
    if args.dry_run:
        lens = [len(c["frame_indices"]) for c in clips]
        print(f"[dry-run] frames/clip: min={min(lens)} max={max(lens)} total={sum(lens)}")
        return 0
    out = Path(args.out_dir).expanduser()
    manifest = extract_frames(clips, out, jpeg_quality=args.jpeg_quality)
    manifest["meta"]["window"] = args.window
    (out / "clips_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
