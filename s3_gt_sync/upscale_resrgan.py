"""Batch Real-ESRGAN x4 over a tree of mp4s (e.g. the export_subvideos output).

Self-contained, ffmpeg-free: per video we cv2-decode -> numbered PNG frames -> the
verified Real-ESRGAN frame-folder mode (``inference_realesrgan.py``, run in its OWN
env via subprocess) -> re-encode the ``*_out.png`` frames back to mp4 at the ORIGINAL
fps. The output tree mirrors the input (``<out-root>/<same rel path>``), so an x4 clip
keeps the same name as its source and still matches its parent chunk.

Speed is reported per video (total + Real-ESRGAN-only frames/sec) and as a batch total,
so you can see throughput on the box.

Real-ESRGAN lives in its own venv (see the FPV-restoration teardown: only the x4
upscaler was kept). Point --resrgan-python / --resrgan-repo at it.

Run (on the server)::

    uv run --with opencv-python --with numpy python -m s3_gt_sync.upscale_resrgan \
      --root /home/ubuntu/work/subvideo_check \
      --out-root /home/ubuntu/work/subvideo_check_x4 \
      --resrgan-python /home/ubuntu/work/resrgan/.venv/bin/python \
      --resrgan-repo   /home/ubuntu/work/resrgan/Real-ESRGAN \
      --which full            # full|sub|all  (default full = the big videos)
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import cv2


# ------------------------- pure core (testable) -------------------------

def is_sub_clip(path) -> bool:
    """A sub-video clip is named ``<stem>__subNN_...mp4`` by export_subvideos."""
    return "__sub" in Path(path).name


def want(path, which: str) -> bool:
    """which = full|sub|all. 'full' = the big per-chunk videos (no __sub marker)."""
    if which == "all":
        return True
    if which == "sub":
        return is_sub_clip(path)
    return not is_sub_clip(path)                          # 'full'


def list_videos(root, which: str) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.mp4") if want(p, which))


def out_frame_order(names: list[str]) -> list[str]:
    """Sort Real-ESRGAN outputs (``000001_out.png``) by their numeric stem so the
    re-encoded video keeps temporal order regardless of filesystem listing."""
    def key(n):
        stem = Path(n).stem
        digits = stem.split("_")[0]
        return int(digits) if digits.isdigit() else stem
    return sorted(names, key=key)


def fps_of(n_frames: int, seconds: float) -> float:
    """frames / second, guarded against div-by-zero."""
    return round(n_frames / seconds, 2) if seconds > 0 else 0.0


# ------------------------------- I/O runner -------------------------------

def _decode_to_frames(video, frames_dir: Path) -> tuple[int, float, int, int]:
    """Write every frame as zero-padded PNG. Returns (n, fps, w, h)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        cv2.imwrite(str(frames_dir / f"{n:06d}.png"), frame)
        n += 1
    cap.release()
    return n, fps, w, h


def _run_esrgan(py, repo, in_dir: Path, out_dir: Path, model: str, outscale: int,
                tile: int, fp32: bool) -> None:
    cmd = [py, "-u", "inference_realesrgan.py", "-n", model, "-i", str(in_dir),
           "-o", str(out_dir), "--outscale", str(outscale)]
    if tile:
        cmd += ["--tile", str(tile)]
    if fp32:
        cmd += ["--fp32"]
    # stream Real-ESRGAN's own per-frame progress live (do NOT capture) so a long
    # video isn't a silent black box; -u keeps its stdout unbuffered.
    p = subprocess.run(cmd, cwd=str(repo))
    if p.returncode != 0:
        raise RuntimeError(f"Real-ESRGAN exited {p.returncode} (see output above)")


def _encode(frames_dir: Path, dst: Path, fps: float, fourcc: str) -> int:
    names = out_frame_order([p.name for p in frames_dir.glob("*.png")])
    if not names:
        return 0
    first = cv2.imread(str(frames_dir / names[0]))
    h, w = first.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    for nm in names:
        wr.write(cv2.imread(str(frames_dir / nm)))
    wr.release()
    return len(names)


def upscale_one(video: Path, root: Path, out_root: Path, py, repo, model, outscale,
                tile, fp32, fourcc, log=print) -> dict:
    rel = video.relative_to(root)
    dst = out_root / rel
    if dst.exists():
        log(f"  skip (exists) {rel}")
        return {"video": str(rel), "skipped": True}
    log(f"  >> {rel}: decoding ...")
    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        fin, fout = td / "in", td / "out"
        fin.mkdir(); fout.mkdir()
        t0 = time.perf_counter()
        n, fps, w, h = _decode_to_frames(video, fin)
        if n == 0:
            log(f"  empty {rel}")
            return {"video": str(rel), "error": "no frames"}
        t1 = time.perf_counter()
        log(f"     decoded {n} frames {w}x{h} in {round(t1 - t0, 1)}s "
            f"({fps_of(n, t1 - t0)} fps); Real-ESRGAN x{outscale} ...")
        _run_esrgan(py, repo, fin, fout, model, outscale, tile, fp32)
        t2 = time.perf_counter()
        log(f"     esrgan done in {round(t2 - t1, 1)}s ({fps_of(n, t2 - t1)} fps); encoding ...")
        nw = _encode(fout, dst, fps, fourcc)
        t3 = time.perf_counter()
    total_s = round(t3 - t0, 2)
    esr_s = round(t2 - t1, 2)
    esr_fps = fps_of(n, t2 - t1)                          # the GAN throughput
    total_fps = fps_of(n, t3 - t0)
    log(f"  {rel}: {n} frames {w}x{h} -> x{outscale} | "
        f"esrgan {esr_s}s ({esr_fps} fps) | total {total_s}s ({total_fps} fps)")
    return {"video": str(rel), "frames": n, "written": nw,
            "src_size": [w, h], "out_size": [w * outscale, h * outscale],
            "decode_s": round(t1 - t0, 2), "esrgan_s": esr_s, "encode_s": round(t3 - t2, 2),
            "total_s": total_s, "esrgan_fps": esr_fps, "total_fps": total_fps,
            "out": str(dst)}


def run(root, out_root, py, repo, which="full", model="RealESRGAN_x4plus", outscale=4,
        tile=0, fp32=False, fourcc="mp4v", limit=0, log=print) -> list[dict]:
    root, out_root = Path(root), Path(out_root)
    repo = Path(repo)
    vids = list_videos(root, which)
    if limit:
        vids = vids[:limit]                              # measure on a few clips first
    log(f"[esrgan] {len(vids)} videos ({which}) under {root} -> {out_root}")
    rep = []
    for v in vids:
        try:
            rep.append(upscale_one(v, root, out_root, py, repo, model, outscale,
                                   tile, fp32, fourcc, log))
        except Exception as e:                            # noqa: BLE001 - keep batch going
            log(f"  ERROR {v.name}: {str(e)[:200]}")
            rep.append({"video": str(v.relative_to(root)), "error": str(e)[:300]})
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="tree of mp4s (e.g. subvideo_check)")
    ap.add_argument("--out-root", required=True, help="mirrored output tree")
    ap.add_argument("--resrgan-python", required=True, help="python of the resrgan venv")
    ap.add_argument("--resrgan-repo", required=True, help="cloned Real-ESRGAN repo dir")
    ap.add_argument("--which", choices=["full", "sub", "all"], default="full")
    ap.add_argument("--model", default="RealESRGAN_x4plus")
    ap.add_argument("--outscale", type=int, default=4)
    ap.add_argument("--tile", type=int, default=0, help="tile size to bound VRAM (0=off)")
    ap.add_argument("--fp32", action="store_true", help="full precision (default fp16)")
    ap.add_argument("--fourcc", default="mp4v")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N videos (0=all) - for speed measurement")
    args = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)

    rep = run(args.root, args.out_root, args.resrgan_python, args.resrgan_repo,
              args.which, args.model, args.outscale, args.tile, args.fp32, args.fourcc,
              limit=args.limit, log=log)
    done = [r for r in rep if r.get("written")]
    tot_frames = sum(r["frames"] for r in done)
    tot_esr = sum(r["esrgan_s"] for r in done)
    tot_all = sum(r["total_s"] for r in done)
    err = sum(1 for r in rep if r.get("error"))
    print(f"\n== upscale x{args.outscale}: {len(done)} videos, {tot_frames} frames | "
          f"esrgan {round(tot_esr, 1)}s ({fps_of(tot_frames, tot_esr)} fps) | "
          f"total {round(tot_all, 1)}s ({fps_of(tot_frames, tot_all)} fps) | "
          f"{sum(1 for r in rep if r.get('skipped'))} skipped, {err} errors ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
