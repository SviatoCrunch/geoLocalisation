"""Evaluate a trained checkpoint on a split: per-class IoU + mIoU (+ optional colorized preds).

    python -m segformer_seg.evaluate \
      --ckpt      /home/ubuntu/work/geoLocalisation/segformer_seg/runs/b3_v1/best \
      --manifests /home/ubuntu/work/geoLocalisation/segformer_seg/manifests \
      --split test [--save-vis /path/to/vis]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CLASSES
from .dataset import SegDataset
from .label_maps import colorize
from .metrics import ConfusionMatrix


@torch.no_grad()
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="dir saved by train.py (best/)")
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--background-mode", default="class", choices=["class", "ignore"])
    ap.add_argument("--save-vis", default=None, help="dir to dump colorized pred|gt panels")
    args = ap.parse_args(argv)

    from transformers import SegformerForSemanticSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(args.ckpt).to(device).eval()
    ds = SegDataset(Path(args.manifests) / f"manifest_{args.split}.jsonl", train=False,
                    crop=args.crop, background_mode=args.background_mode)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    cm = ConfusionMatrix()
    vis = Path(args.save_vis).expanduser() if args.save_vis else None
    if vis:
        vis.mkdir(parents=True, exist_ok=True)
    k = 0
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        labels = batch["labels"].numpy()
        logits = torch.nn.functional.interpolate(
            model(pixel_values=pv).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy()
        cm.update(pred, labels)
        if vis:
            from PIL import Image
            for b in range(pred.shape[0]):
                panel = np.concatenate([colorize(pred[b]), colorize(labels[b].astype(np.uint8))], axis=1)
                Image.fromarray(panel).save(vis / f"{args.split}_{k:04d}.png"); k += 1

    summ = cm.summary()
    print(f"[{args.split}] mIoU {summ['mIoU']:.4f}")
    for c, v in summ["per_class_iou"].items():
        print(f"    {c:11} IoU {'—' if v is None else f'{v:.4f}'}")
    out = Path(args.manifests) / f"eval_{args.split}.json"
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
