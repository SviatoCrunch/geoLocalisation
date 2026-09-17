# native_map_pyramid — native hierarchical map-cell features (A/B experiment)

Adds ONE experimental map-cell **feature source** to the e2c/SuperVLAD cell model. It changes
*only* where the n=[8,4,2,1] map-cell DINOv2 tokens come from. The model tail (frozen VLAD
assignment + trainable `PerGroupProjection` / `map_head` / `ScaleGate` / τ(n)) and the whole
UAV/query branch, loss, mining, split, optimizer and metrics are **unchanged**.

## The two modes (`--map-pyramid-source`)

| | `legacy_token_partition` (default) | `native_hierarchical` |
|---|---|---|
| n=1 (1000 m) | 1000 m DINO grid | native 1000 m COG crop → DINO |
| n=2 (500 m) | 2×2 token split of the 1000 m grid | **4 independent 500 m COG crops** → DINO |
| n=4 (250 m) | 4×4 token split of the 1000 m grid | **16 independent 250 m COG crops** → DINO |
| n=8 (125 m) | 8×8 token split of the 1000 m grid | **2×2 quadrants of each 250 m DINO grid** |
| DINO views / tile | 1 | **21** = 1×1000 + 4×500 + 16×250 |

n=8 is **never** a separate 125 m DINO forward — it reuses the 250 m grids' quadrants, so a
125 m cell carries its 250 m parent's context. Child bounds are read at exact EPSG:3857 map
bounds from the COG (never by subdividing a resized RGB tile).

### Cell order (pinned, tested)
Cells are row-major with `i`=row-from-north, `j`=col-from-west. For a 250 m parent `(i,j)` and
quadrant `(dy,dx)` (dy=0 north, dx=0 west): `global_row=2i+dy`, `global_col=2j+dx`,
`flat = global_row*8 + global_col`. This matches the legacy `region_ids` layout. See
`geometry.py` + `tests/test_geometry_order.py`.

## Caching point (and why)

The frozen part of the map path is: L2-normalize tokens → `alpha = agg.alpha` (frozen VLAD
assignment) → `value = agg.value` (identity ψ). The **trainable** `PerGroupProjection` (and
`map_head`) run *after*. Because the per-group projection `A_k` is bias-free linear,
`A_k(Σ α_k·value) = Σ α_k·A_k(value)` — so we cache the **per-cell per-group residual sum**
`S[cell,k] = Σ_{tok∈cell} α_k·value` (the last frozen quantity) and replay the trainable tail
every step. Map projections are trained every step and are **never** cached.

`build_V_from_cell_sums(S)` reproduces `build_V(grid)` bit-for-bit on legacy `S`
(`tests/test_tail_parity.py`).

## Cache schema & fingerprint (`cache.py`)

HDF5: `S` `(M, 85, K, d_value)` **fp16** (85 = 64+16+4+1 cells, scales concatenated in order
(8,4,2,1), row-major within each), `tile_id` `(M,)` strings, attrs `fingerprint`,
`identity_json`, `meta_json`. Strict identity fields: schema version, mode, backbone,
dino layer/facet, `output_px`, tile size, patch size, projection kind/seed/in/out dims,
`n_groups`, `d_value`, scale layout, cell counts, CRS, resampling, token order, and the
**VLAD dictionary id** (sha256 of `assign_weight`). `NativeCellCache.assert_compatible` raises
`CacheIncompatibleError` naming the first mismatching field — no silent reads. The chosen mode
+ fingerprint are recorded in the cache, the run's `run_config.json`, the checkpoint's
`resolved_config`, and each metrics row.

## Trainable / frozen parameters (runtime-audited; identical in both modes)

| module | params | grad |
|---|---|---|
| `agg.assign` (VLAD) | 32,768 | **frozen** |
| `group_proj` (32×32×1024) | 1,048,576 | trainable |
| `drone_head` | 657,152 | trainable |
| `map_head` | 657,152 | trainable |
| `scale_gate` | 33,412 | trainable |
| `rho` (τ for n=8,4,2) | 3 | trainable |
| **total trainable** | **2,396,295 (≈2.40M)** | |

The native source adds **zero** trainable params; `state_dict` keys/shapes are identical
(`tests/test_tail_parity.py::test_native_source_adds_no_parameters_and_same_state_dict`).

## Storage / compute

- Cache: `85·32·1024·2 B ≈ 5.31 MiB/tile` fp16 → **~13.6 GiB for 2615 tiles**.
- Prep compute: 21 DINO views/tile (batched, shared backbone), one-off.
- Token/value dim after `RandomProjector` is **1024** (not the native 1536).

## Commands (actual repo modules/paths)

Cache build (server; GPU + COG). Reuses the SAME `RandomProjector(1024, seed)` as the galleries:
```bash
cd ~/work/geoLocalisation
uv run --python 3.11 --with "torch==2.5.1" --with torchvision --with h5py --with numpy \
       --with rasterio --with pyproj --with pillow --with tqdm \
  python -m native_map_pyramid.build_cache \
    --index  ~/work/out/gallery_h5/geo_iso_noverlap/tiles/tiles_index_dense.h5 \
    --cog    kramatorsc=~/work/tif/Krum_esri_48.5683N_37.6272E_z18_cog.tif \
             kup=~/work/tif/Kup_esri_z18.tif \
             liman_day=~/work/tif/Liman_day_esri_z18.tif \
    --assign ~/work/out/gallery_h5/geo_iso_noverlap/dict/k32.pt \
    --backbone dinov2_vitg14 --dino-facet value --output-px 840 \
    --desc-dim 1024 --proj-seed 0 --tile-size-m 1000 \
    --out ~/work/out/gallery_h5/geo_iso_noverlap/native_cache/S_native.h5 --device cuda --amp
```

Smoke tests (CPU, no GPU/COG):
```bash
cd ~/work/geoLocalisation && python -m pytest native_map_pyramid/tests -q
```

Baseline **A** (legacy; default flag can be omitted):
```bash
python -m geo_e2c_train.train --map-pyramid-source legacy_token_partition \
  --split-config $OUT/geo_iso_noverlap/config.yaml --split-json $OUT/geo_iso_noverlap/split.json \
  --galleries kramatorsc=$OUT/geo_iso_noverlap/maps/map_dinov2_kramatorsc_s250m_d1024.h5 \
              kup=$OUT/geo_iso_noverlap/maps/map_dinov2_kup_s250m_d1024.h5 \
              liman_day=$OUT/geo_iso_noverlap/maps/map_dinov2_liman_day_s250m_d1024.h5 \
  --queries   kramatorsc=$OUT/geo_iso_noverlap/queries/query_kramatorsc_d1024.h5 \
              kup=$OUT/geo_iso_noverlap/queries/query_kup_d1024.h5 \
              liman_day=$OUT/geo_iso_noverlap/queries/query_liman_day_d1024.h5 \
  --assign $OUT/geo_iso_noverlap/dict/k32.pt --agg supervlad --pyramid-mode cell \
  --k 32 --d-token 1024 --scales 8 4 2 1 --seed 0 \
  --out ~/work/out/runs_native_ab/A_legacy
```

Experiment **B** (native; ONLY `--map-pyramid-source` + `--native-cache` differ — same split, dict,
seed, loss, mining, optimizer, epochs):
```bash
python -m geo_e2c_train.train --map-pyramid-source native_hierarchical \
  --native-cache ~/work/out/gallery_h5/geo_iso_noverlap/native_cache/S_native.h5 \
  --native-backbone dinov2_vitg14 --native-dino-facet value --native-output-px 840 --native-proj-seed 0 \
  --split-config $OUT/geo_iso_noverlap/config.yaml --split-json $OUT/geo_iso_noverlap/split.json \
  --galleries ... (same as A) --queries ... (same as A) \
  --assign $OUT/geo_iso_noverlap/dict/k32.pt --agg supervlad --pyramid-mode cell \
  --k 32 --d-token 1024 --scales 8 4 2 1 --seed 0 \
  --out ~/work/out/runs_native_ab/B_native
```
(`--galleries`/`--queries` stay required — queries are always token grids; galleries feed the
val/test query encoder path and the loader identity.)

Evaluation of a saved checkpoint (Recall@K, median rank, geographic recall):
```bash
python -m geo_e2c_train.evaluate_ckpt --ckpt ~/work/out/runs_native_ab/B_native/best.pt \
  --split-config $OUT/geo_iso_noverlap/config.yaml --split-json $OUT/geo_iso_noverlap/split.json \
  --galleries ... --queries ... --assign $OUT/geo_iso_noverlap/dict/k32.pt --which val test
```

## Expected shapes
Per tile cache `S`: `(85, 32, 1024)`. After the tail: V `n=8:(M,64,256)`, `n=4:(M,16,256)`,
`n=2:(M,4,256)`, `n=1:(M,1,256)`, each L2-normalized.

## Known limits (need the GPU/COG server)
- Real `build_cache` run + numeric spot-check vs a legacy tile (cell-context differs by design).
- The full A/B training run and its logged β / τ / per-scale stats.
- Standalone `evaluate_ckpt.py` / `export_shortlist.py` currently use the **legacy** source
  (dispatch was added only at `train.py` + `eval.py:build_gallery_V`); the in-training val/test
  eval already honors `native_hierarchical`. Wiring the standalone CLIs is a small follow-up.
