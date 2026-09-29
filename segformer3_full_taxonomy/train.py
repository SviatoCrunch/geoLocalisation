"""Train SegFormer-B3 on SkyScenes full taxonomy (22 classes). HF + weighted-CE+Dice.

Leakage-free split by TOWN (``--val-towns`` held out). Resumable checkpoints (model, optim,
scheduler, scaler, epoch, best mIoU). Modes: ``--max-train-samples N`` for smoke/tiny-overfit
before any long run; ``--overfit`` validates on the (tiny) train set to prove the loss can
memorise a few masks.

Verified server recipe (Tesla T4, CUDA 12.4)::

    uv run --extra-index-url https://download.pytorch.org/whl/cu121 \
      --with "torch==2.5.1+cu121" --with transformers --with albumentations \
      python -m segformer3_full_taxonomy.train \
      --prepared-root /home/ubuntu/work/datasets/SkyScenes/prepared/H_35_P_0_ClearNoon \
      --out /home/ubuntu/work/out/segformer3_full_taxonomy/runs/e1_smoke \
      --val-towns Town07 Town10HD --epochs 1 --batch 4 --max-train-samples 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SkyScenesDataset, town_split
from .external import evaluate_external, external_pairs
from .losses import CombinedLoss
from .metrics import ConfusionMatrix
from .taxonomy import CLASSES, NUM_CLASSES


def build_model(model_name: str):
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=NUM_CLASSES, id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}, ignore_mismatched_sizes=True,
        use_safetensors=True)


def _up(logits, size):
    return torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = ConfusionMatrix()
    for b in loader:
        logits = _up(model(pixel_values=b["pixel_values"].to(device)).logits, b["labels"].shape[-2:])
        cm.update(logits.argmax(1).cpu().numpy(), b["labels"].numpy())
    return cm.summary()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-towns", nargs="+", default=["Town07", "Town10HD"])
    ap.add_argument("--model", default="nvidia/mit-b3")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", default="standard", choices=["standard", "analog"],
                    help="train augmentation: 'analog' = heavy analog-FPV domain randomization")
    ap.add_argument("--save-best-ext", action="store_true",
                    help="also save best_ext/ by external macro (exploratory; contaminates the "
                         "test if used to pick the final model — keep best/ as the official one)")
    ap.add_argument("--dice-w", type=float, default=1.0)
    ap.add_argument("--external-root", default=None,
                    help="gt_cramatorsc root -> monitor road/railtrack/water on GT_flat_mask "
                         "(NOT used for checkpoint selection)")
    ap.add_argument("--external-every", type=int, default=1,
                    help="run the external monitor every N epochs (and on the last) to cut overhead")
    ap.add_argument("--max-train-samples", type=int, default=0, help="cap train set (smoke/tiny)")
    ap.add_argument("--overfit", action="store_true", help="validate on the (tiny) train set")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr_pairs, va_pairs = town_split(args.prepared_root, args.val_towns)
    if args.max_train_samples:
        tr_pairs = tr_pairs[:args.max_train_samples]
    if args.overfit:
        va_pairs = tr_pairs
    ext_pairs = external_pairs(args.external_root) if args.external_root else []
    print(f"[data] train {len(tr_pairs)} | val {len(va_pairs)} | val_towns {args.val_towns}"
          f" | external {len(ext_pairs)}", flush=True)

    tr = DataLoader(SkyScenesDataset(tr_pairs, True, args.crop, aug=args.aug), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=len(tr_pairs) > args.batch)
    va = DataLoader(SkyScenesDataset(va_pairs, False, args.crop), batch_size=args.batch,
                    shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model).to(device)
    criterion = CombinedLoss(dice_w=args.dice_w).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(tr) // args.grad_accum)
    total_iters = args.epochs * steps_per_epoch
    warmup = int(args.warmup_frac * total_iters)

    def lr_at(it):
        return it / max(1, warmup) if it < warmup else max(0.0, (1 - (it - warmup) / max(1, total_iters - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    amp_on = (not args.no_amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    start_ep, best_miou, best_ext, hist = 0, -1.0, -1.0, []
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best_miou = ck["epoch"] + 1, ck["best_miou"]
        print(f"[resume] from epoch {start_ep} (best mIoU {best_miou:.4f})", flush=True)

    for ep in range(start_ep, args.epochs):
        model.train(); run = 0.0
        optim.zero_grad(set_to_none=True)
        for i, b in enumerate(tr):
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = _up(model(pixel_values=b["pixel_values"].to(device)).logits, b["labels"].shape[-2:])
                loss = criterion(logits, b["labels"].to(device)) / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update(); sched.step(); optim.zero_grad(set_to_none=True)
            run += loss.item() * args.grad_accum
        m = evaluate(model, va, device)
        row = {"epoch": ep, "train_loss": run / max(1, len(tr)), "lr": sched.get_last_lr()[0], **m}
        ext_str = ""
        if ext_pairs and (ep % args.external_every == 0 or ep == args.epochs - 1):
            ext = evaluate_external(model, ext_pairs, device, args.crop)
            row["external"] = ext
            ext_str = (" | EXT[road:{road} rail:{railtrack} water:{water} macro:{macro_target}]"
                       .format(**{k: ("—" if v is None else f"{v:.3f}") for k, v in ext.items()}))
            if args.save_best_ext and (ext["macro_target"] or -1) > best_ext:
                best_ext = ext["macro_target"]; model.save_pretrained(out / "best_ext")
                (out / "best_ext_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        hist.append(row)
        top = ", ".join(f"{k}:{v:.2f}" for k, v in m["per_class_iou"].items() if v is not None and v > 0.05)
        print(f"[ep {ep:3d}] loss {row['train_loss']:.4f} | val mIoU {m['mIoU']:.4f} | {top}{ext_str}", flush=True)
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best_miou": best_miou}, out / "last.pt")
        if m["mIoU"] > best_miou:
            best_miou = m["mIoU"]; model.save_pretrained(out / "best")
            (out / "best_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"[done] best val mIoU {best_miou:.4f} -> {out/'best'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
