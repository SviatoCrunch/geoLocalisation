# segformer_seg — SegFormer (MiT-B3) drone-frame scene parser

Isolated module (git + server). Semantic segmentation over drone stills with a **6-class
union** taxonomy chosen for this project:

| id | class | source |
|----|-------|--------|
| 0 | background | implicit ("none of the above") |
| 1 | bridge | new COCO only |
| 2 | railway | server GT (`Railways`) + COCO |
| 3 | road | server GT + COCO |
| 4 | sky | server GT + COCO |
| 5 | tower | new COCO only |
| 6 | water | server GT + COCO |

Server Label-Studio classes `Building`/`Target` are **dropped** (not in this taxonomy and
never rasterised to PNG). `bridge`/`tower` are supervised only by the new COCO frames; in
server stills those pixels are background — which is correct.

## Data mixing (path-level, no folder merge)

`build_manifest.py` records paths (+ inline COCO polygons) into JSONL. It combines:
- **new COCO** (`segment.v1i.coco`) — hand-labelled frames, polygons;
- **server GT** (`gt_cramatorsc`) — per-class binary PNGs, via the clean `gt_mask_audit.csv`
  view, **excluding** the N listed in `covered_gt.csv` (already labelled in COCO, so the
  same frame is never supervised twice).

Everything dropped (covered-excluded, missing PNGs, stills-without-mask, unresolved uuid-N
orphans, dropped categories) is logged to `build_report.json`.

```bash
# ON THE SERVER (files live there). Put the COCO export somewhere first, e.g.
#   gt_cramatorsc/new_coco/train/{_annotations.coco.json, *.jpg}
python3 -m segformer_seg.build_manifest \
  --gt-root  /home/ubuntu/work/gt_cramatorsc \
  --coco-dir /home/ubuntu/work/gt_cramatorsc/new_coco/train \
  --covered  /home/ubuntu/work/gt_cramatorsc/covered_gt.csv \
  --out-dir  /home/ubuntu/work/geoLocalisation/segformer_seg/manifests
```

Outputs: `manifest_{train,val,test}.jsonl`, `build_report.json` (incl. train pixel
histogram + median-frequency class weights for balanced loss).

## Training (Stage 2)

Runs via `uv` overlays (this repo's uv project is CPU-only — torch is added per-run). The
server driver is CUDA 12.4, so pin a **cu121** wheel; transformers>=5 needs **torch>=2.5**
and loads via **safetensors** (avoids the torch.load CVE block). Verified working recipe:

```bash
# from ~/work/geoLocalisation — TRAIN (GPU)
uv run --extra-index-url https://download.pytorch.org/whl/cu121 \
  --with "torch==2.5.1+cu121" --with transformers --with albumentations \
  python -m segformer_seg.train \
  --manifests /home/ubuntu/work/geoLocalisation/segformer_seg/manifests \
  --out       /home/ubuntu/work/geoLocalisation/segformer_seg/runs/b3_v1 \
  --epochs 120 --batch 8 --lr 6e-5

# EVAL best checkpoint on test (+ optional colorized pred|gt panels)
uv run --extra-index-url https://download.pytorch.org/whl/cu121 \
  --with "torch==2.5.1+cu121" --with transformers --with albumentations \
  python -m segformer_seg.evaluate \
  --ckpt      /home/ubuntu/work/geoLocalisation/segformer_seg/runs/b3_v1/best \
  --manifests /home/ubuntu/work/geoLocalisation/segformer_seg/manifests \
  --split test --save-vis /home/ubuntu/work/geoLocalisation/segformer_seg/runs/b3_v1/vis_test
```

Balance handling (all knobs on `train.py`): median-freq **weighted-CE + Dice** loss
(`--weight-scheme`, `--weight-clip`, `--dice-w`), **rare-class oversampling**
(`--rare-boost`, boosts frames with bridge/tower/water/railway), 512² multi-scale crops,
AMP, poly LR. `--background-mode ignore` drops the background class from supervision.

## Status
- **Stage 1:** taxonomy/config, label-map composition, manifest builder — DONE, verified on
  server (367 imgs: 301 server + 66 COCO; train/val/test = 293/37/37; pixel hist in
  `build_report.json`). Severe imbalance: bg 76% … tower 0.04%.
- **Stage 2:** `dataset.py` (compose + augment), `class_weights.py`, `losses.py` (CE+Dice),
  `metrics.py` (mIoU), `train.py` (`nvidia/mit-b3`), `evaluate.py` — code + unit/smoke tests
  GREEN locally. Model forward not smoke-tested on this dev box (transformers import blocked
  by a local DLL policy); runs on the server torch env.

## Tests
```bash
cd geoLocalisation && python -m pytest segformer_seg/tests -q
```
