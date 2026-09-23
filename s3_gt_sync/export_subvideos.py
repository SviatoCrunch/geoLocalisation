"""Verify the sub-video split by EYE: dump every full video + each of its sub-videos as clips.

Reads a ``subvideo_index`` JSON (``<city>_<ver>.json``) and, per video, downloads the
full chunk and cuts one clip per sub-video (``start_frame..end_frame`` inclusive, the
SAME absolute cv2 decode indices the split was computed on). Names are built so a clip
always matches its parent chunk::

    <out-dir>/<chunk_stem>/
        <chunk_stem>.mp4                              # full video (as downloaded)
        <chunk_stem>__sub00_f0-118_first.mp4
        <chunk_stem>__sub01_f130-260_overlap_cut.mp4
        ...

Each sub-clip's name carries its ``sub_index``, absolute frame range and boundary reason,
so you can line the clip up against the JSON row without opening it. The frame range in
the name is exactly ``start_frame``/``end_frame`` from the JSON.

Run (from ~/work/geoLocalisation; cv2/numpy here)::

    uv run --with opencv-python --with numpy python -m s3_gt_sync.export_subvideos \
      --json /home/ubuntu/work/gt_cramatorsc/kram_2.7.json \
      --out-dir /home/ubuntu/work/subvideo_check/kram_2.7
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2

_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


# ------------------------- pure core (testable) -------------------------

def _safe(s: str) -> str:
    return _SAFE.sub("_", str(s)).strip("_") or "x"


def clip_name(stem: str, sv: dict) -> str:
    """Sub-video clip filename that MATCHES its parent chunk stem.

    ``<stem>__sub{NN}_f{start}-{end}_{boundary}.mp4`` — shares the chunk stem prefix so
    clip and full video sort together; frame range/boundary come straight from the JSON."""
    return (f"{stem}__sub{int(sv['sub_index']):02d}"
            f"_f{sv['start_frame']}-{sv['end_frame']}"
            f"_{_safe(sv.get('boundary', 'na'))}.mp4")


def build_plan(video_rec: dict, stem: str) -> list[dict]:
    """[{sub_index, start, end, name}] for every sub-video, sorted by start frame."""
    plan = []
    for sv in video_rec.get("sub_videos", []):
        plan.append({"sub_index": int(sv["sub_index"]),
                     "start": int(sv["start_frame"]), "end": int(sv["end_frame"]),
                     "name": clip_name(stem, sv)})
    return sorted(plan, key=lambda d: d["start"])


def owner(idx: int, plan: list[dict], j: int) -> int:
    """Advance pointer j so plan[j] is the (disjoint, sorted) clip owning frame idx, or
    the first clip not yet ended. Returns the pointer; caller checks start<=idx<=end."""
    while j < len(plan) and plan[j]["end"] < idx:
        j += 1
    return j


# ------------------------------- I/O runner -------------------------------

def _download(url: str, dst: Path, log) -> bool:
    if dst.exists():
        return True
    log(f"  downloading {Path(url).name} ...")
    subprocess.run(["aws", "s3", "cp", url, str(dst)], capture_output=True)
    return dst.exists()


def export_video(url: str, out_dir: Path, video_rec: dict, fourcc: str, keep_full: bool,
                 log=print) -> dict:
    """Download one chunk and write a clip per sub-video. Returns a small report row."""
    stem = Path(url).stem
    vdir = out_dir / _safe(stem)
    vdir.mkdir(parents=True, exist_ok=True)
    full = vdir / f"{_safe(stem)}.mp4"
    if not _download(url, full, log):
        return {"video": stem, "error": "download failed"}

    plan = build_plan(video_rec, _safe(stem))
    fps = float(video_rec.get("fps") or 30.0)
    cap = cv2.VideoCapture(str(full))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    cc = cv2.VideoWriter_fourcc(*fourcc)

    writers: dict[int, cv2.VideoWriter] = {}
    counts = {p["sub_index"]: 0 for p in plan}
    idx, j = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if not plan or idx > plan[-1]["end"]:
            break                                    # past the last sub-video -> done
        j = owner(idx, plan, j)
        if j >= len(plan):
            break
        p = plan[j]
        if p["start"] <= idx <= p["end"]:
            wr = writers.get(p["sub_index"])
            if wr is None:
                fh, fw = frame.shape[0], frame.shape[1]
                wr = cv2.VideoWriter(str(vdir / p["name"]), cc, fps,
                                     (w or fw, h or fh))
                writers[p["sub_index"]] = wr
            wr.write(frame)
            counts[p["sub_index"]] += 1
    cap.release()
    for wr in writers.values():
        wr.release()
    if not keep_full:
        full.unlink(missing_ok=True)

    log(f"  {stem}: {len(plan)} sub-videos -> {sum(counts.values())} frames written")
    return {"video": stem, "full_kept": keep_full, "sub_videos": len(plan),
            "clips": [{"name": p["name"], "sub_index": p["sub_index"],
                       "start_frame": p["start"], "end_frame": p["end"],
                       "frames_written": counts[p["sub_index"]]} for p in plan]}


def run(json_path, out_dir, fourcc="mp4v", keep_full=True, log=print) -> dict:
    rec = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"json": str(json_path), "out_dir": str(out_dir), "videos": []}
    for v in rec.get("videos", []):
        url = v.get("video")
        if not url or "error" in v:
            continue
        report["videos"].append(export_video(url, out_dir, v, fourcc, keep_full, log))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="subvideo_index JSON (<city>_<ver>.json)")
    ap.add_argument("--out-dir", required=True, help="one folder per chunk is created here")
    ap.add_argument("--fourcc", default="mp4v", help="VideoWriter fourcc (default mp4v)")
    ap.add_argument("--no-full", dest="keep_full", action="store_false", default=True,
                    help="do NOT keep the full downloaded chunk (clips only)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    rep = run(args.json, args.out_dir, args.fourcc, args.keep_full)
    nclip = sum(len(v.get("clips", [])) for v in rep["videos"])
    print(f"\n== {Path(args.json).name}: {len(rep['videos'])} videos, {nclip} sub-video clips "
          f"-> {rep['out_dir']} ==")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
