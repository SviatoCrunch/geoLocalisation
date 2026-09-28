# segformer3_full_taxonomy

Full-taxonomy SegFormer-B3 semantic segmentation, isolated from `segformer_seg` (which is not
touched). Trains on **all** classes of a verified master taxonomy from source datasets and
keeps Kramatorsk `GT_flat_mask` as an **external validation set only**.

## Decisions locked so far
- **Master taxonomy = SkyScenes 28-class** (CARLA lineage) — to be *verified against the real
  downloaded metadata* before it is written to a versioned config. Not invented from web text.
- **Sources (open access, license set aside for now):**
  - `SkyScenes` (MIT, HF `hoffman-lab/SkyScenes`) — E1, has railtrack/bridge/water/road/building.
  - `FlyAwareV2` (open, SynDrone/CARLA lineage) — E2. Synthetic part is ~290 GB → subset.
  - `Mid-Air` (open, 14-class rural forward-view, has "Train Track") — E4 partial-label only.
- **External validation:** `gt_cramatorsc/GT_flat_mask` (per-class binary PNGs, keyed by
  `<lat>_<lon>`). NEVER used for train / aug / split / class-weights / pseudo-labels. Scored
  only on **railway, road, water**; a target class absent from all of GT → N/A, not IoU 0.
- **Hardware:** Tesla T4 15 GB, CUDA 12.4 driver. SegFormer-B3 @512² batch 8 AMP fits.

## Blocking now
- Source datasets are NOT on the server yet; disk being expanded (SkyScenes subset + FlyAwareV2
  synth do not fit the current 51 GB free).

## Stage 1 (this): data audit
`tools/audit_segmentation_data.py` — config-driven, no torch. Reports counts, resolutions,
unique mask IDs/colours + pixel histograms, image↔mask pairing (unmatched/corrupt/empty), and
SHA-256 of file lists. Handles `single` (index|rgb|auto) and `per_class_binary` (external val).

```bash
# external val can be audited NOW; SkyScenes block runs after download
python3 -m segformer3_full_taxonomy.tools.audit_segmentation_data \
    --config segformer3_full_taxonomy/configs/audit_example.yaml
```
Outputs `audit.json` + `audit.md` under `out/segformer3_full_taxonomy/audit/`.

## Next stages (per the §11 order, after audit)
`configs/full_taxonomy/` (versioned taxonomy + per-source mapping YAML, loader errors on
unknown IDs) → `dataset_adapters/` (+ partial-label) → leakage-free town/scene split →
`tools/train_full_taxonomy.py` (reuses segformer_seg losses/metrics; resumable ckpts,
grad-accum, source- & class-balanced samplers) → `tools/eval_target_classes.py` (external
val: railway/road/water IoU/Dice/PR + confusion + qualitative overlays). Smoke (8–16) and
tiny-overfit (2–4) gate any long run.

## Tests
```bash
cd geoLocalisation && python -m pytest segformer3_full_taxonomy/tests -q
```
