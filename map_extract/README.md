# map_extract — GeoTIFF → DINO features → HDF5 (isolated)

Self-contained map/gallery builder. Tiles an EPSG:3857 GeoTIFF into a concentric
context pyramid, runs a DINOv2/v3 backbone per level, and writes one HDF5 group per
`(tile, level)` — the same schema the rest of the pipeline (VLAD build, split) reads.

Vendored from RevisitAnything so this folder has **no** cross-repo imports:

| module | what | heavy deps |
|---|---|---|
| `geometry.py` | true-metre ↔ EPSG:3857 scaling (pure) | — (math) |
| `fishnet.py` | strict metric grid + local `BoundingBox` (torchgeo dropped) | pyproj/shapely (lazy, AOI only) |
| `pyramid.py` | `PyramidConfig` + `PyramidContextSampler` | rasterio/pyproj (lazy) |
| `dino.py` | `build_dino_extractor`, `RandomProjector`, DINOv2/v3 | torch (+torchvision) |
| `helpers.py` | preprocess, COG check, KMZ AOI, preview | torchvision; rasterio/cv2/shapely (lazy) |
| `sky.py` | ONNX sky segmenter (`--images --segment_sky`) | onnxruntime/cv2 (lazy) |
| `extract.py` | GPU extractor + CLI (`extract`, `extract_images`) | torch, h5py, rasterio |

## The `--true_meters` flag (why it exists)

The sampler steps `--tile_size_m` / `--stride_m` in the TIF CRS (EPSG:3857), whose unit
is a metre inflated by `1/cos(lat)`. So **without** the flag, `--stride_m 250` yields a
grid only `250·cos(lat) ≈ 162 m` on the ground at lat 49 (and omitting `--stride_m`
defaults to a `tile_size_m`-sized grid — e.g. a 1000-unit = 662 m grid).

With `--true_meters`, `--tile_size_m`/`--stride_m` are TRUE ground metres: both are scaled
by `1/cos(centre_lat)` into EPSG:3857 units before the grid is built, and the stored
`window_size_m` is divided back to the real footprint. So the ground stride is exactly
250 m. Off by default → legacy builds are byte-identical (`true_scale = 1.0`).

## Build a gallery (true 250 m stride, 1000 m tiles)

```bash
python -m map_extract.extract \
    --tif /path/to/city_z18_cog.tif --out out/map_city.h5 \
    --true_meters --tile_size_m 1000 --stride_m 250 \
    --levels 1 --output_px 224 --backbone dinov2_vitg14 \
    --desc_dim <same-as-query> --proj_seed <same-as-query> --device cuda
```

`--desc_dim` / `--proj_seed` **must match** the drone-frame (query) extraction, or the
projected descriptors are not comparable.

## Tests

`python -m pytest map_extract/tests` — geometry/grid/config only (no GPU or geo stack
needed; shapely-gated AOI test skips if shapely is absent).
