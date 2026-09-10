# sky_filter

Precompute **sky keep-masks** for UAV frames — the **first gate** of the drone pipeline.
Before ANY operation on a drone frame (footprint estimation, DINOv2 tokens, patch overlap,
global VLAD, and later the main-training query encoder) the sky is removed. Masks are
computed **once**, cached per `frame_id`, and reused everywhere → one consistent source.

Sky is **DROPPED** (keep-mask; sky tokens removed before global/patch, as in Stage-2 /
`build_query_tokens`). Applies to **UAV frames only** — satellite crops have no sky.

## Backends (precedence: precomputed → neural)
- **`precomputed`** — read a ready-made mask from the store (data-ready, no model).
- **`neural`** — SegFormer/UAVid segmentation, sky class → not-kept (heavy, **injected**,
  server-only, lazy torch/transformers import).
- (extendable via `register_masker`, e.g. a heuristic fallback).

The `CascadingMasker` tries backends in `cfg.backend_order`; the first non-`None` mask
wins and records its backend. A frame with a ready mask **never** re-runs the model.

## Primary source: GT masks (`ingest_sky_masks`)
The main path is **precomputed where a GT sky mask exists, neural where it doesn't**.
`ingest_sky_masks(mask_dir, city, store)` reads `GT_flat_mask/*__Sky.png` (from
`s3_gt_sync.make_masks`), **inverts** sky→keep (`keep = ~(sky>0)`), and stores one
keep-mask per `frame_id = "<city>:<stem>"`. Frames without a `__Sky.png` are simply not
stored → the resolver falls through to the neural backend. Corrupt `lat_lon` stems (e.g.
a `8.59…` latitude missing the leading `4`) are skipped + reported.

```python
from sky_filter import MaskStore, ingest_sky_masks
store = MaskStore("~/work/sky_masks")
ingest_sky_masks("~/work/gt_cramatorsc/GT_flat_mask", "cramatorsc", store)   # GT → precomputed
```

## Flow
```python
from sky_filter import SkyFilterConfig, MaskStore, precompute
from sky_filter.maskers import build_resolver
from sky_filter.maskers.neural import NeuralSkyMasker     # server-only (torch/transformers)

cfg   = SkyFilterConfig(mask_store="~/work/sky_masks")
store = MaskStore(cfg.mask_store)
res   = build_resolver(cfg, store=store, neural_impl=NeuralSkyMasker(cfg))
precompute(frame_ids, image_loader, res, store)           # writes masks + manifest (fingerprint)
```

Downstream (drone side) reads the cached mask and drops sky tokens:
```python
from sky_filter import reduce_to_grid, apply_drop
keep_px    = store.load(frame_id)                    # (H,W) bool, True=ground
token_keep = reduce_to_grid(keep_px, gh, gw, cfg.cell_sky_max)   # → token grid
ground_tokens = apply_drop(dino_token_grid, token_keep)         # (Nkept, D)
```

## Artifacts
- `masks/<frame_id>.npz` — pixel keep-mask per frame.
- `manifest.json` — per-frame `{backend, version, shape, sky_fraction, sha}`; `fingerprint()`
  is the provenance hash downstream asserts against.

## Reduction rule (DROP)
`reduce_to_grid` drops a token cell if its sky fraction exceeds `cell_sky_max`
(`0.0` = drop on ANY sky; default `0.5` = drop majority-sky cells).

## Integration
`sky_filter` sits **before** `footprint_assoc`: a sky-aware extractor applies the keep-mask
to the UAV frame's tokens (drop sky) before building `QueryFeatures`; satellite crops are
untouched. The same cached mask feeds the main-training query encoder — identical to how
`query_tokens.h5` is already sky-filtered. (That wiring is the integration bridge, added
next.)

## Isolation
numpy-only core (reduce/drop, store, resolver, precompute); the neural backend is behind a
protocol and imported lazily. No dependency on RevisitAnything or sibling packages.

## Tests
```bash
python -m pytest sky_filter -q
```
Reduce/drop semantics, store roundtrip + manifest fingerprint, resolver precedence
(precomputed hit / neural fallback / all-miss raises), and the precompute pipeline
(reproducible, skips existing, prefers precomputed).
