# single_frame_locate

Faithful **standalone** reproduction of the production vision_project service GPU localization path,
for a **single drone frame** → top-K map cells (JSON).

It reproduces exactly what the live service (`Crunch-is/vision_project`, branch
`server/e2c-live-snapshot`, `service/siam_tools/siam.py` → `SiamEngine.localize`) runs on the GPU —
only sliced to one frame and stripped of the FastAPI / SQS / pair-render / GPU-lock glue. The GPU
compute is identical.

## Pipeline

**Stage A — coarse (e2c SuperVLAD), GPU.** One DINOv2-vitg14 forward on the frame →
`RandomProjector(d=1024, seed=0)` → sky filter → token grid `(N,D)` + patch coords `xy`, plus the
query embedding `q`. SuperVLAD cell-pyramid `score(q, V)` against the gallery `V` (built once from the
map token H5) → `argsort(-sims)[:K]` → **coarse top-K candidate cells** (`--k`, default 100).

**Stage B — fine rerank (GPU MAGSAC++), GPU.** `rerank_multiframe([tokens], [xy], cell_ids, store)`:
each candidate cell's packed `(P, L, H, W, D)` fp16 pyramid block → fp16 batched match `Q @ blockᵀ`
over all (position × level) grids → mutual-NN → **one batched `estimate_homography_magsacpp_batch`**
over all crops → inlier counts. Per position: `level_agg='sum'` over levels; per cell:
`cell_agg='mean'` over positions. Best position → point (lat/lon); best level → `level_m`. →
**top-K distinct cells** (`--topk`, default 10).

GPU MAGSAC++ config (from env, service defaults): `hyps=256, irls=1, closed_form, fp32,
sigma_max=2.0, inlier_threshold=--reproj (2.0)`. Matching is fp16 on cuda, fp32 on cpu.

## What is vendored vs new

- **Byte-for-byte from the server** (`git show vision/server/e2c-live-snapshot:service/...`):
  - `rerank/` — `matcher.py`, `reranker.py` (incl. `rerank_multiframe`), `cell_store.py`,
    `magsacpp_torch/` (GPU MAGSAC++), `newformat/` (packed `/features` store reader + adapter).
  - `e2c_backend.py` — the service `stage_e2c_adapter.py`; the **only** change is the sys.path
    bootstrap (`_GEO_DIR = <geoLocalisation>` so `siam_e2c_model` + `map_extract` import as a
    sibling package instead of via the service `engine`).
- **New** — `locate.py` (`SingleFrameLocator`, the single-frame slice of `SiamEngine.localize`) + CLI.

## Run (kram)

```bash
cd geoLocalisation
python -m single_frame_locate.locate \
  --image  frame.jpg \
  --ckpt   /path/run_supervlad_cell_w1_best.pt \
  --map-h5 /path/kram20_prodaction.h5 \
  --assign /path/k32.pt \
  --store  /path/pyramid_newformat/kram \
  --city kram --k 100 --topk 10 --device cuda \
  --index-cache /tmp/kram_V.pt \
  --out result.json
```

- `--store` is the **packed `/features`** pyramid store (local dir or `s3://…/`). Path TBD per run.
- `--index-cache` caches the gallery `V` (first run builds it from `--map-h5`; reuse on later runs).
- Artifacts/galleries for kram live on S3 (`s3://geo-reference/embeddings/kram/`,
  `.../tif/`), see the deploy notes.

## Parity

The compute matches production because the compute files are the production files. Cross-check vs a
service response on the same frame: same coarse top-K cell ids, same reranked `cell_score` /
`level_m` / lat-lon. (cv2 vs torch MAGSAC are close-but-not-bitwise; here BOTH service and this tool
use the torch GPU backend via `rerank_multiframe`, so they agree up to fp16-match/RNG noise.)
