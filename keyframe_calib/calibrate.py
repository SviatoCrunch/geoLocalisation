"""Runner + CLI: measure the SegProp F-measure(delta) degradation curve on a
densely-labelled calibration segment and report the MINIMUM number of manual masks.

You provide ONE short, continuous, densely-labelled segment (every frame, or a fixed
stride) of your own footage. For each candidate keyframe spacing ``delta`` we keep
only frames on the uniform delta-grid as "manual" keyframes and propagate their masks
to the in-between frames with the FAITHFUL SegProp vote (keyframe_calib.segprop:
forward/backward key + cur projections, exp(-beta*delta) temporal weighting, optional
per-CC homography votes), then score the filled frames against the dense GT. The
largest delta that still clears ``--threshold`` is ``delta*``; ``masks_for_span``
turns it into the mask budget for a full video (each shot resets propagation).

Frame source is EITHER a folder of pre-extracted frames (``--frames-dir``) OR a video
file (``--video``) — with a video the frames are pulled directly by the mask indices,
so no manual extraction is needed. To create a segment to label from a video first:
``python -m keyframe_calib.extract --video clip.mp4 --start 1000 --count 200 --out calib/frames``

    # from a video
    python -m keyframe_calib.calibrate \
        --video calib/clip.mp4 --masks-dir calib/masks --num-classes 3 \
        --deltas 10,25,50,75,100,150 --beta 1.0 --flow-backend raft \
        --key-homography-weight 0 --metric mean_f1 --threshold 0.90 --out calib.json

    # or from a frames folder
    python -m keyframe_calib.calibrate --frames-dir calib/frames --masks-dir calib/masks ...

Metric ``mean_f1`` is the per-class F-measure averaged over classes present in GT —
the same quantity SegProp/Ruralscapes report. GT masks are single-channel PNGs of
class ids named with the frame index (e.g. ``000123.png`` / ``frame_123.png``); the
video/frames share that index. Flow for every adjacent pair is computed ONCE and reused.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import flow as flow_mod
from . import segprop
from .metrics import evaluate
from .propagate import bracket, masks_for_span, pick_delta_star, uniform_keyframes

_IDX = re.compile(r"(\d+)")


def _index_of(path: Path) -> int | None:
    m = _IDX.findall(path.stem)
    return int(m[-1]) if m else None


def _load_gray(path: Path, downscale: int) -> np.ndarray:
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    if downscale > 1:
        g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale),
                       interpolation=cv2.INTER_AREA)
    return g


def _load_mask(path: Path, downscale: int) -> np.ndarray:
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    if downscale > 1:
        m = cv2.resize(m, (m.shape[1] // downscale, m.shape[0] // downscale),
                       interpolation=cv2.INTER_NEAREST)
    return m.astype(np.int64)


def _collect(frames_dir: Path, masks_dir: Path):
    """Frame indices that have BOTH a frame and a GT mask, sorted; + path maps."""
    fmap, mmap = {}, {}
    for p in sorted(frames_dir.iterdir()):
        i = _index_of(p)
        if i is not None:
            fmap.setdefault(i, p)
    for p in sorted(masks_dir.iterdir()):
        i = _index_of(p)
        if i is not None:
            mmap.setdefault(i, p)
    idx = sorted(set(fmap) & set(mmap))
    return idx, fmap, mmap


def _mask_indices(masks_dir: Path):
    """Sorted frame indices present in the masks folder (the calibration segment)."""
    mmap = {}
    for p in sorted(masks_dir.iterdir()):
        i = _index_of(p)
        if i is not None:
            mmap.setdefault(i, p)
    return sorted(mmap), mmap


def _to_segprop_field(flow_xy: np.ndarray, device):
    """Our (H,W,2)=(dx,dy) optical-flow field -> SegProp (H,W,2)=(dy,dx) torch tensor."""
    import torch
    f = np.stack([flow_xy[..., 1], flow_xy[..., 0]], axis=-1)  # (dy, dx)
    return torch.from_numpy(np.ascontiguousarray(f)).float().to(device)


def _chain(fields, device):
    """Stack a list of per-step SegProp fields into (nsteps,H,W,2). Empty -> (0,H,W,2)."""
    import torch
    if not fields:
        return None
    return torch.stack(fields, 0).to(device)


def propagate_frame(t, keys, gt_oh, flows, grays, *, beta, num_classes, key_weight, cur_weight,
                    key_homography_weight, hom_estimator, device):
    """SegProp vote for interior frame ``t`` from its bracketing keyframes.

    ``flows`` maps base index j -> (fab, fba) fields for pair (j, j+1), already in
    SegProp (dy,dx) torch form: fab = j->j+1, fba = j+1->j. Builds the four traces
    (start->cur, end->cur, cur->end, cur->start) and calls segprop.vote_frame."""
    import torch

    left, right = bracket(keys, t)
    fwd = _chain([flows[j][0] for j in range(left, t)], device)            # start->cur (fab)
    bkw = _chain([flows[j][1] for j in range(right - 1, t - 1, -1)], device)  # end->cur (fba)
    curf = _chain([flows[j][0] for j in range(t, right)], device)          # cur->end (fab)
    curb = _chain([flows[j][1] for j in range(t - 1, left - 1, -1)], device)  # cur->start (fba)
    delta = (t - left) / (right - left)
    vote = segprop.vote_frame(gt_oh[left], gt_oh[right], fwd, bkw, curf, curb,
                              delta=delta, beta=beta, key_weight=key_weight,
                              cur_weight=cur_weight, key_homography_weight=key_homography_weight,
                              hom_estimator=hom_estimator)
    return vote.argmax(2).cpu().numpy().astype(np.int64)


def calibrate(masks_dir, num_classes, deltas, *, frames_dir=None, video=None, beta=1.0,
              flow_backend="farneback", device=None, downscale=1, key_weight=1.0, cur_weight=1.0,
              key_homography_weight=0.0, hom_estimator="lmeds", metric="mean_f1",
              threshold=0.90, log=print):
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (frames_dir is None) == (video is None):
        raise ValueError("provide exactly ONE frame source: frames_dir OR video")
    masks_dir = Path(masks_dir)

    # frame indices come from the masks (the densely-labelled calibration segment)
    if frames_dir is not None:
        frames_dir = Path(frames_dir)
        idx, fmap, mmap = _collect(frames_dir, masks_dir)
    else:
        idx, mmap = _mask_indices(masks_dir)
    if len(idx) < 3:
        raise ValueError("need >=3 densely-labelled frames in the calibration segment")
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError("calibration frames must be a CONTIGUOUS index range "
                         "(one shot, dense GT); found gaps")
    n = len(idx)
    src = f"frames_dir={frames_dir}" if frames_dir is not None else f"video={video}"
    log(f"calibration segment: {n} contiguous frames [{idx[0]}..{idx[-1]}], {src}, device={device}")

    if frames_dir is not None:
        grays = [_load_gray(fmap[i], downscale) for i in idx]
    else:
        from . import video as video_mod
        gmap = video_mod.read_frames_gray(video, idx, downscale)
        grays = [gmap[i] for i in idx]
    gt_idx = [_load_mask(mmap[i], downscale) for i in idx]
    gt_oh = [segprop.classmap_logical(torch.from_numpy(g).to(device), num_classes).float()
             for g in gt_idx]

    log(f"computing {n - 1} adjacent-pair flows ({flow_backend})...")
    flows = {}
    for j in range(n - 1):
        fab, fba = flow_mod.pair_flows(grays[j], grays[j + 1], backend=flow_backend, device=device)
        flows[j] = (_to_segprop_field(fab, device), _to_segprop_field(fba, device))

    curve = []
    for delta in deltas:
        keys = uniform_keyframes(n, delta)
        keyset = set(keys)
        preds, gts = [], []
        for t in range(n):
            if t in keyset:
                continue
            preds.append(propagate_frame(t, keys, gt_oh, flows, grays, beta=beta,
                                         num_classes=num_classes, key_weight=key_weight,
                                         cur_weight=cur_weight,
                                         key_homography_weight=key_homography_weight,
                                         hom_estimator=hom_estimator, device=device))
            gts.append(gt_idx[t])
        miou, mf1 = evaluate(preds, gts, num_classes, ignore=255)
        row = {"delta": int(delta), "keyframes": len(keys), "scored_frames": len(preds),
               "mean_f1": round(mf1, 4), "mean_iou": round(miou, 4)}
        curve.append(row)
        log(f"  delta={delta:4d}  keys={len(keys):3d}  mF1={mf1:.4f}  mIoU={miou:.4f}")

    d_star = pick_delta_star(curve, threshold, metric)
    report = {
        "source": str(frames_dir) if frames_dir is not None else str(video),
        "masks_dir": str(masks_dir), "num_classes": num_classes,
        "beta": beta, "flow_backend": flow_backend, "downscale": downscale,
        "key_weight": key_weight, "cur_weight": cur_weight,
        "key_homography_weight": key_homography_weight, "metric": metric, "threshold": threshold,
        "segment_frames": n, "index_range": [idx[0], idx[-1]], "curve": curve, "delta_star": d_star,
    }
    if d_star is not None:
        report["masks_per_1000_frames"] = masks_for_span(1000, d_star)
        report["note"] = (f"1 mask / {d_star} frames keeps {metric} >= {threshold}; "
                          f"for a shot of N frames budget masks_for_span(N, {d_star}).")
    else:
        report["note"] = (f"NO tested delta reaches {metric} >= {threshold}; label denser, "
                          f"raise flow quality (raft) or add homography votes, or lower threshold.")
    return report


def _parse_deltas(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--frames-dir", help="folder of pre-extracted frames (index in filename)")
    grp.add_argument("--video", help="video file; frames pulled by the mask indices")
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--deltas", default="10,25,50,75,100,150", type=_parse_deltas,
                    help="candidate keyframe spacings (frames) to sweep")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="SegProp dist_weighting_beta (0 = equal weighting)")
    ap.add_argument("--flow-backend", choices=["farneback", "raft"], default="farneback")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--key-weight", type=float, default=1.0)
    ap.add_argument("--cur-weight", type=float, default=1.0)
    ap.add_argument("--key-homography-weight", type=float, default=0.0,
                    help="per-CC homography key votes (SLOW; better road/river edges)")
    ap.add_argument("--hom-estimator", choices=["lmeds", "ransac", "magsac"], default="lmeds",
                    help="original SegProp uses lmeds")
    ap.add_argument("--metric", choices=["mean_f1", "mean_iou"], default="mean_f1")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rep = calibrate(args.masks_dir, args.num_classes, args.deltas, frames_dir=args.frames_dir,
                    video=args.video, beta=args.beta, flow_backend=args.flow_backend,
                    device=args.device, downscale=args.downscale, key_weight=args.key_weight,
                    cur_weight=args.cur_weight, key_homography_weight=args.key_homography_weight,
                    hom_estimator=args.hom_estimator, metric=args.metric, threshold=args.threshold)
    print("\n== keyframe calibration (SegProp) ==")
    print(f"delta* = {rep['delta_star']} frames  ({rep['note']})")
    if rep.get("delta_star"):
        print(f"~{rep['masks_per_1000_frames']} manual masks per 1000-frame shot")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
