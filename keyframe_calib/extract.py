"""CLI: dump a contiguous segment of frames from a video, ready for dense labelling.

    python -m keyframe_calib.extract --video clip.mp4 --start 1000 --count 200 --out calib/frames

Label the resulting PNGs (one class-id mask per frame, same ``{index:06d}.png`` name)
into a masks folder, then run keyframe_calib.calibrate on the video + that masks folder.
"""
from __future__ import annotations

import argparse

from .video import extract_segment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--count", type=int, required=True, help="number of frames in the segment")
    ap.add_argument("--stride", type=int, default=1, help="keep every k-th frame")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--out", required=True, help="output frames folder")
    args = ap.parse_args(argv)
    written = extract_segment(args.video, args.start, args.count, args.out,
                              downscale=args.downscale, stride=args.stride)
    print(f"done: {len(written)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
