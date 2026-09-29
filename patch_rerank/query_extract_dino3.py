"""Extract DINOv3-sat token grids for drone query frames → QueryGridStore-format H5.

The fine reranker matches map cells (DINOv3-sat) against the query by mutual-NN, so the query MUST be
embedded with the SAME model. This reads drone frames from a folder, embeds each with DINOv3-sat
(``map_dino3``), and writes one H5 group per frame in the layout ``query_io.QueryGridStore`` expects:
``<stem>/ift_dino`` (D, N) + attrs ``patch_grid_h/w``, ``lat``, ``lon``, ``filename``. No sky filter
(all tokens kept). lat/lon are parsed from the filename stem (``<idx>_<lat>_<lon>``), so the query id
``<city>:<stem>`` matches the coarse shortlist.

Run::

    HF_TOKEN=… uv run --python 3.11 --with "torch==2.5.1" --with transformers --with opencv-python-headless \
      --with h5py --with numpy --with tqdm python -m patch_rerank.query_extract_dino3 \
      --frames /home/ubuntu/work/gt_kup/GT_flat --city kup --output-px 512 --device cuda --amp \
      --out /home/ubuntu/work/out/gallery_h5/kup/query_kup_dinov3sat.h5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .map_dino3 import DEFAULT_MODEL, build_dinov3_extractor, extract_grids_v3


def _latlon_from_stem(stem: str):
    """``<idx>_<lat>_<lon>`` → (lat, lon); NaN if the last two underscore tokens aren't floats."""
    parts = stem.split("_")
    try:
        return float(parts[-2]), float(parts[-1])
    except (ValueError, IndexError):
        return float("nan"), float("nan")


def _read_rgb(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)          # BGR (H,W,3) uint8
    if im is None:
        return None
    return np.ascontiguousarray(im[:, :, ::-1])           # → RGB


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of drone frames (<idx>_<lat>_<lon>.jpg)")
    ap.add_argument("--city", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--output-px", type=int, default=512, help="square resize (multiple of patch 16)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import h5py
    from tqdm import tqdm

    frames = sorted(Path(args.frames).expanduser().glob(args.glob))
    if not frames:
        raise SystemExit(f"no frames matching {args.glob} in {args.frames}")
    ext = build_dinov3_extractor(args.model, args.device)
    print(f"[dino3] {ext['model_id']} patch={ext['patch']} D={ext['D']} n_prefix={ext['n_prefix']} "
          f"| {len(frames)} frames output_px={args.output_px}", flush=True)

    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    h = w = args.output_px // ext["patch"]
    n_ok = n_nogeo = 0
    with h5py.File(out, "w") as fout:
        fout.attrs.update({"backbone": ext["model_id"], "patch": ext["patch"], "D": ext["D"],
                           "output_px": args.output_px, "city": args.city})
        buf_imgs, buf_meta = [], []                        # (stem, filename, lat, lon)

        def _flush():
            if not buf_imgs:
                return
            grids = extract_grids_v3(buf_imgs, ext, args.output_px, args.device, amp=args.amp)
            for (stem, fn, la, lo), g in zip(buf_meta, grids):
                hh, ww, D = g.shape
                feat = g.reshape(hh * ww, D).numpy().astype(np.float32)   # (N, D)
                grp = fout.create_group(stem)
                grp.create_dataset("ift_dino", data=feat.T)               # (D, N) — QueryGridStore reads .T
                grp.attrs["patch_grid_h"] = hh; grp.attrs["patch_grid_w"] = ww
                grp.attrs["lat"] = la; grp.attrs["lon"] = lo; grp.attrs["filename"] = fn
            buf_imgs.clear(); buf_meta.clear()

        for fp in tqdm(frames, desc=f"query:{args.city}", unit="frame"):
            rgb = _read_rgb(fp)
            if rgb is None:
                continue
            la, lo = _latlon_from_stem(fp.stem)
            if not (math.isfinite(la) and math.isfinite(lo)):
                n_nogeo += 1
            buf_imgs.append(rgb); buf_meta.append((fp.stem, fp.name, la, lo)); n_ok += 1
            if len(buf_imgs) >= args.batch:
                _flush()
        _flush()
    print(f"[ok] {n_ok} query frames ({n_nogeo} without parseable lat/lon) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
