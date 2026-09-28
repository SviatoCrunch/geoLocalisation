"""Train SegFormer (MiT-B3) on the mixed manifest. HuggingFace + custom balanced loss.

Recipe (SegFormer paper + mmseg defaults, adapted to this small, heavily imbalanced set):
  * ``nvidia/mit-b3`` ImageNet-pretrained encoder + a fresh 7-class decode head;
  * AdamW, lr 6e-5, weight-decay 1e-2, **poly** LR decay (power 1.0), warmup;
  * 512x512 multi-scale crops (see dataset), AMP, grad-clip;
  * **weighted-CE + Dice** loss (median-freq weights from build_report.json) and a
    **rare-class WeightedRandomSampler** — both target the tower/bridge/water tail;
  * per-epoch val mIoU, best checkpoint saved.

Needs: torch, transformers, albumentations, opencv-python. Run in a torch env::

    python -m segformer_seg.train \
      --manifests /home/ubuntu/work/geoLocalisation/segformer_seg/manifests \
      --out       /home/ubuntu/work/geoLocalisation/segformer_seg/runs/b3_v1 \
      --epochs 120 --batch 8 --lr 6e-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .class_weights import compute_weights, load_counts
from .config import CLASSES, IGNORE_INDEX, NUM_CLASSES
from .dataset import SegDataset, read_manifest, sampler_weights
from .losses import CombinedLoss
from .metrics import ConfusionMatrix


def build_model(model_name: str):
    from transformers import SegformerForSemanticSegmentation

    id2label = {i: c for i, c in enumerate(CLASSES)}
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=NUM_CLASSES, id2label=id2label,
        label2id={c: i for i, c in enumerate(CLASSES)}, ignore_mismatched_sizes=True)


def _logits_to_label(logits: torch.Tensor, size) -> torch.Tensor:
    up = torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return up


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = ConfusionMatrix()
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        labels = batch["labels"]
        logits = _logits_to_label(model(pixel_values=pv).logits, labels.shape[-2:])
        pred = logits.argmax(1).cpu().numpy()
        cm.update(pred, labels.numpy())
    return cm.summary()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True, help="dir with manifest_{train,val,test}.jsonl + build_report.json")
    ap.add_argument("--out", required=True, help="run dir (checkpoints + metrics)")
    ap.add_argument("--model", default="nvidia/mit-b3")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    # balance knobs
    ap.add_argument("--weight-scheme", default="median", choices=["median", "inverse", "none"])
    ap.add_argument("--weight-clip", type=float, default=15.0)
    ap.add_argument("--dice-w", type=float, default=1.0, help="Dice weight (0 = CE only)")
    ap.add_argument("--rare-boost", type=float, default=5.0, help="oversampling factor for rare-class frames (1 = off)")
    ap.add_argument("--background-mode", default="class", choices=["class", "ignore"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = Path(args.manifests)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr_ds = SegDataset(man / "manifest_train.jsonl", train=True, crop=args.crop, background_mode=args.background_mode)
    va_ds = SegDataset(man / "manifest_val.jsonl", train=False, crop=args.crop, background_mode=args.background_mode)

    # rare-class oversampling
    if args.rare_boost > 1.0:
        w = sampler_weights(read_manifest(man / "manifest_train.jsonl"), args.rare_boost)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # loss weights (background weighted down; keep as class unless --background-mode ignore)
    counts = load_counts(man / "build_report.json")
    weights = compute_weights(counts, scheme=args.weight_scheme, clip=args.weight_clip,
                              ignore_background=(args.background_mode == "ignore"))
    print("[weights]", {c: round(w, 3) for c, w in zip(CLASSES, weights)}, flush=True)
    criterion = CombinedLoss(class_weights=weights, dice_w=args.dice_w).to(device)

    model = build_model(args.model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_iters = args.epochs * max(1, len(tr_loader))
    warmup = int(args.warmup_frac * total_iters)

    def lr_at(it):  # linear warmup -> poly(power=1) decay
        if it < warmup:
            return it / max(1, warmup)
        return max(0.0, (1 - (it - warmup) / max(1, total_iters - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    amp_on = (not args.no_amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    best_miou, hist = -1.0, []
    it = 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for batch in tr_loader:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = _logits_to_label(model(pixel_values=pv).logits, labels.shape[-2:])
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); sched.step()
            run += loss.item(); it += 1
        m = evaluate(model, va_loader, device)
        row = {"epoch": ep, "train_loss": run / max(1, len(tr_loader)),
               "lr": sched.get_last_lr()[0], **m}
        hist.append(row)
        print(f"[ep {ep:3d}] loss {row['train_loss']:.4f} | val mIoU {m['mIoU']:.4f} "
              f"| {', '.join(f'{k}:{v:.2f}' for k, v in m['per_class_iou'].items() if v is not None)}",
              flush=True)
        if m["mIoU"] > best_miou:
            best_miou = m["mIoU"]
            model.save_pretrained(out / "best")
            (out / "best_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"[done] best val mIoU {best_miou:.4f} -> {out/'best'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
