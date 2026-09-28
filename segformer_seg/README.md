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

## Status
- **Stage 1 (this):** taxonomy/config, label-map composition (`label_maps.py`), manifest
  builder, unit tests. Run `build_manifest` on the server and check `build_report.json`.
- **Stage 2 (next):** `dataset.py` (on-the-fly label compose + augmentation), `class_weights.py`,
  `train.py` (HF `SegformerForSemanticSegmentation`, `nvidia/mit-b3`), `evaluate.py` (mIoU).

## Tests
```bash
cd geoLocalisation && python -m pytest segformer_seg/tests -q
```
