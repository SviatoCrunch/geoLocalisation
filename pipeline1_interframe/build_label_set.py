"""Minimum-keyframe label set for ONE video (SegProp-style motion budget).

Goal: label a whole clip through the FEWEST frames. What breaks mask propagation
is how far the scene SHIFTS between labelled frames, not how many frames pass, so
keyframes are placed uniformly in CUMULATIVE OPTICAL-FLOW motion (denser where the
clip moves fast).

IMPORTANT for analog FPV: RF breakup frames read as huge "motion" (noise), which
fools a naive motion budget into clustering keyframes on CORRUPTED frames. So the
selection is QUALITY-GATED: candidates are restricted to good (optionally usable)
frames, motion is measured good->good (skipping noise), and a long run of bad
frames splits the sequence into separate propagation segments (``--max-gap``).

Pipeline (reuses the tested ``video_label_pipeline`` modules already on the box):
    decode -> DINOv2 quality (good/usable/bad) -> keep gated frames
    -> motion good->good (Farneback) -> motion-budget keyframes per run
    -> manifest.json + label_plan.json + keyframes/*.png (+ optional contact_sheet).

    cd /home/ubuntu/work/geoLocalisation
    PYTHONPATH=/home/ubuntu/work/RevisitAnything/quality-cls/src uv run --python 3.12 \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --with torch==2.5.1+cu121 --with torchvision==0.20.1 \
        --with opencv-python-headless --with numpy --with pillow --with pyyaml --with pandas \
        python build_label_set.py --video <mp4> --out <dir> \
        --quality-gate good --target-k 20 --montage \
        --checkpoint /home/ubuntu/work/RevisitAnything/quality-cls/checkpoints/head.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # geoLocalisation/
from video_label_pipeline.segment import segment_frames, to_feature
from video_label_pipeline.video_io import read_all_frames

_FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def to_grays(frames_bgr, max_side):
    """Downscaled grayscale frames for optical flow (compute once, reuse)."""
    grays, scale = [], None
    for bgr in frames_bgr:
        h, w = bgr.shape[:2]
        if scale is None:
            scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        grays.append(g)
    return grays


def fb(ga, gb):
    """Median optical-flow magnitude between two gray frames."""
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, **_FB)
    return float(np.median(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))


def place_in_motion(cum, budget):
    """Shot-local positions spaced ~evenly in cumulative motion ``cum`` (non-
    decreasing, starts at 0) so each gap holds <= ``budget`` of shift. Keeps both
    endpoints. Returns sorted 0-based positions into the segment."""
    n = len(cum)
    if n <= 2 or cum[-1] <= 1e-9 or budget <= 1e-9:
        return list(range(n)) if n <= 2 else [0, n - 1]
    total = cum[-1]
    n_int = max(1, math.ceil(total / budget))
    targets = [j * total / n_int for j in range(n_int + 1)]
    sel, j = [], 0
    for t in targets:
        while j + 1 < n and cum[j + 1] < t:
            j += 1
        k = j + 1 if (j + 1 < n and abs(cum[j + 1] - t) < abs(cum[j] - t)) else j
        sel.append(k)
    sel = sorted(set(sel))
    if sel[0] != 0:
        sel.insert(0, 0)
    if sel[-1] != n - 1:
        sel.append(n - 1)
    return sel


def seg_count(segments, budget):
    return sum(len(place_in_motion(cum, budget)) for _, cum in segments)


def fpk_for_k(segments, mu, target_k):
    """Binary-search frames/key G whose total keyframe count is <= target_k and
    closest to it (count is monotone non-increasing in G)."""
    if mu <= 0:
        return 1.0
    lo, hi = 1.0, 8000.0
    for _ in range(46):
        mid = (lo + hi) / 2.0
        if seg_count(segments, mid * mu) > target_k:
            lo = mid
        else:
            hi = mid
    return hi


def build_gated_segments(grays, gated_pos, max_gap):
    """Split gated frame positions into runs (break when the temporal gap to the
    next gated frame exceeds ``max_gap`` — a long corrupted stretch can't be
    propagated across). Each run carries cumulative good->good Farneback motion."""
    if not gated_pos:
        return []
    runs, cur = [], [gated_pos[0]]
    for prev, p in zip(gated_pos, gated_pos[1:]):
        if (p - prev) <= max_gap:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    segs = []
    for run in runs:
        cum, acc = [0.0], 0.0
        for a, b in zip(run, run[1:]):
            acc += fb(grays[a], grays[b])
            cum.append(acc)
        segs.append((run, cum))
    return segs


def make_montage(frames, selected, cols=5, cell_w=260):
    cells = []
    for s in selected:
        im = frames[s["_pos"]]
        h, w = im.shape[:2]
        th = cv2.resize(im, (cell_w, max(1, int(h * cell_w / w))))
        tag = f"#{s['frame_idx']}" + (f" {s['quality']}" if "quality" in s else "")
        cv2.rectangle(th, (0, 0), (cell_w, 18), (0, 0, 0), -1)
        cv2.putText(th, tag, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(th)
    if not cells:
        return None
    ch, cw = cells[0].shape[:2]
    rows = math.ceil(len(cells) / cols)
    grid = np.zeros((rows * ch, cols * cw, 3), np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        grid[r * ch:(r + 1) * ch, cc * cw:(cc + 1) * cw] = c
    return grid


def rle(labels):
    """Run-length encode a label list -> [[start, end, label], ...]."""
    runs = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][2] == lab:
            runs[-1][1] = i
        else:
            runs.append([i, i, lab])
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-key", type=float, default=50.0)
    ap.add_argument("--target-k", type=int, default=None, help="pick budget for ~this many keyframes")
    ap.add_argument("--quality-gate", default=None,
                    help="restrict keyframes to these labels, e.g. 'good' or 'good,usable' (forces --quality)")
    ap.add_argument("--max-gap", type=int, default=20,
                    help="gated: >this many frames to the next kept frame splits a propagation run")
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--adaptive-thr", type=float, default=3.0)
    ap.add_argument("--min-content", type=float, default=15.0)
    ap.add_argument("--content-window", type=int, default=2)
    ap.add_argument("--content-reduce", type=int, default=4)
    ap.add_argument("--min-seg", type=int, default=2)
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args(argv)

    gate = set(s.strip() for s in args.quality_gate.split(",")) if args.quality_gate else None
    need_quality = args.quality or gate is not None

    video = Path(args.video)
    outdir = Path(args.out) / video.stem
    kdir = outdir / "keyframes"

    print(f"decoding {video.name} ...", flush=True)
    fps, n_total, kept = read_all_frames(video, every=1)
    if not kept:
        print("!! no frames decoded"); return 1
    frames = [f[2] for f in kept]
    idxs = [f[0] for f in kept]
    n = len(frames)
    dur = n / fps if fps > 0 else 0.0
    print(f"  {n} frames @ {fps:.2f} fps  ({dur:.1f}s)", flush=True)

    # content shots (reported for reference)
    feats = [to_feature(f, args.width, args.content_reduce) for f in frames]
    shots, _, cuts = segment_frames(feats, args.adaptive_thr, args.min_content,
                                    args.content_window, args.min_seg)

    grays = to_grays(frames, args.max_side)

    # quality
    labels = None
    if need_quality:
        print("  scoring quality (DINOv2) ...", flush=True)
        from video_label_pipeline.quality import QualityClsBackend
        backend = QualityClsBackend(checkpoint=args.checkpoint)
        labels = [lab for lab, _ in backend.score(frames)]
        qc = {k: labels.count(k) for k in ("good", "usable", "bad")}
        print(f"  quality: {qc}", flush=True)

    # build segments (positions into kept frames) + cumulative motion
    if gate is not None:
        gated_pos = [p for p in range(n) if labels[p] in gate]
        print(f"  gate={sorted(gate)}: {len(gated_pos)}/{n} candidate frames; "
              f"max-gap {args.max_gap} frames", flush=True)
        print("  measuring good->good optical-flow motion ...", flush=True)
        segments = build_gated_segments(grays, gated_pos, args.max_gap)
    else:
        print("  measuring adjacent optical-flow motion ...", flush=True)
        deltas = [0.0] + [fb(grays[i - 1], grays[i]) for i in range(1, n)]
        segments = []
        for pos in shots:
            cum, acc = [0.0], 0.0
            for p in pos[1:]:
                acc += deltas[p]
                cum.append(acc)
            segments.append((pos, cum))

    steps = [cum[i] - cum[i - 1] for _, cum in segments for i in range(1, len(cum))]
    nz = [s for s in steps if s > 0]
    mu = float(np.median(nz)) if nz else 0.0
    C_total = float(sum(steps))
    print(f"  segments: {len(segments)}  (sizes {[len(p) for p, _ in segments][:12]}"
          f"{'...' if len(segments) > 12 else ''})", flush=True)
    print(f"  motion px@{args.max_side}: step-median {mu:.2f}  cumulative {C_total:.0f}", flush=True)

    # budget -> keyframe-count table
    print("\n  frames/key  budget_px   keyframes", flush=True)
    table = []
    for G in [25, 50, 75, 100, 150, 200]:
        k = seg_count(segments, G * mu)
        table.append({"frames_per_key": G, "budget_px": round(G * mu, 2), "keyframes": k})
        print(f"    {G:>6}    {G * mu:>8.1f}    {k:>6}", flush=True)

    # selection
    fpk = args.frames_per_key
    if args.target_k is not None:
        fpk = fpk_for_k(segments, mu, args.target_k)
    budget = fpk * mu

    selected = []
    per_seg = []
    for si, (pos, cum) in enumerate(segments):
        local = place_in_motion(cum, budget)
        for lp in local:
            p = pos[lp]
            rec = {"frame_idx": int(idxs[p]), "timestamp_s": round(idxs[p] / fps, 3) if fps > 0 else None,
                   "segment": si + 1, "cum_motion_px": round(cum[lp], 2), "_pos": p}
            if labels is not None:
                rec["quality"] = labels[p]
            selected.append(rec)
        per_seg.append({"segment": si + 1, "start_frame": int(idxs[pos[0]]), "end_frame": int(idxs[pos[-1]]),
                        "n_candidate_frames": len(pos), "cum_motion_px": round(cum[-1], 2),
                        "keyframes": len(local)})
    print(f"\n  CHOSEN frames/key={fpk:g} -> budget {budget:.1f}px -> {len(selected)} keyframes "
          f"over {len(segments)} segment(s)"
          f"{' (target_k=' + str(args.target_k) + ')' if args.target_k else ''}", flush=True)
    print("  keyframe indices: " + ", ".join(str(s["frame_idx"]) for s in selected), flush=True)

    # write
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "video": str(video), "stem": video.stem, "fps": round(fps, 3),
        "n_frames": n, "duration_s": round(dur, 2),
        "content_shots": len(shots), "content_cuts": sorted(int(c) for c in cuts),
        "quality_gate": sorted(gate) if gate else None, "max_gap": args.max_gap,
        "motion_px_at": args.max_side, "motion_step_median": round(mu, 3), "motion_cumulative": round(C_total, 2),
        "quality": None if labels is None else {k: labels.count(k) for k in ("good", "usable", "bad")},
        "quality_runs": None if labels is None else [[a, b, l] for a, b, l in rle(labels)],
        "n_segments": len(segments), "segments": per_seg, "budget_table": table,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    plan = {"video": str(video), "stem": video.stem, "quality_gate": sorted(gate) if gate else None,
            "frames_per_key": round(fpk, 3), "target_k": args.target_k, "budget_px": round(budget, 2),
            "n_keyframes": len(selected),
            "keyframes": [{k: v for k, v in s.items() if k != "_pos"} for s in selected]}
    (outdir / "label_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.montage:
        grid = make_montage(frames, selected)
        if grid is not None:
            cv2.imwrite(str(outdir / "contact_sheet.png"), grid)
            print(f"  wrote contact_sheet.png ({grid.shape[1]}x{grid.shape[0]})", flush=True)

    if not args.no_extract:
        kdir.mkdir(parents=True, exist_ok=True)
        for old in kdir.glob("*.png"):
            old.unlink()
        for s in selected:
            cv2.imwrite(str(kdir / f"{s['frame_idx']:06d}.png"), frames[s["_pos"]])
        print(f"  wrote {len(selected)} PNG -> {kdir}", flush=True)

    print(f"\nwrote manifest.json + label_plan.json -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
