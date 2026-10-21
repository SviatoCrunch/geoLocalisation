# clip_search — clip-level geo-localization (GPU scheme)

Localize a **clip** (≤32 consecutive frames, GT centred, clipped to its sub-video) instead of a single
still. Each clip's frames are searched with the **existing GPU pipeline** (coarse SuperVLAD →
`pyramid_newformat.locate` batched Torch MAGSAC++ over the packed `*_newformat` store); this package only
adds the clip windowing/extraction (step 1) and the clip aggregation + KMZ (step 5). Nothing in the
search path is re-implemented.

```
subvideo_index JSON ─1─▶ frames/ + clips_manifest.json ─2─▶ query H5 ─3─▶ coarse shortlist
                                                                               │
          KMZ + distR ◀─5─ clip_search_results.json ◀─────── locate (GPU fine) ◀─4─┘
```

Steps **1** and **5** are torch-free (`cv2`+`numpy`). Steps **2–4** are the existing tools, run in their
own GPU env exactly as yesterday.

---

## 1. Clip windows + frame extraction (no GPU)

```bash
cd ~/work/geoLocalisation
uv run --with opencv-python --with numpy python -m clip_search.sample \
  --index /home/ubuntu/work/gt_cramatorsc/kram_2.*.json \
  --out-dir /home/ubuntu/work/gt_cramatorsc/clips \
  --window 32
```
Writes `clips/frames/{idx}_{lat}_{lon}.jpg` (lat/lon = the clip's GT) + `clips/clips_manifest.json`.
`--dry-run` just counts clips/frames. Downloads each video once, grabs only the needed frames, deletes
the video after.

## 2. Embed the frames → one query H5 (GPU)  — existing `map_extract`

Use the **same** backbone/proj as the kram gallery (DINOv2 ViT-g/14, p448 → 32×32 grid, d1024,
`proj_seed 0`, sky-filtered), so `QueryGridStore` grids align with the `*_newformat` cells.

```bash
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with pillow --with tqdm \
  python -m map_extract.extract --images /home/ubuntu/work/gt_cramatorsc/clips/frames \
  --out /home/ubuntu/work/gt_cramatorsc/clips/query_clips_d1024.h5 \
  --backbone dinov2_vitg14 --desc_dim 1024 --proj_seed 0 --segment_sky \
  --device cuda --amp
```

## 3. Coarse shortlist — top-100 cells per frame (GPU)  — existing `coarse_rank_gallery`

```bash
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
  python -m geo_e2c_train.coarse_rank_gallery \
  --ckpt   <KRAM_E2C_CKPT>_best.pt \
  --assign <DICT>/k32.pt \
  --gallery kram=<KRAM_COARSE_GALLERY>.h5 \
  --queries kram=/home/ubuntu/work/gt_cramatorsc/clips/query_clips_d1024.h5 \
  --city kram --k 100 --device cuda \
  --out /home/ubuntu/work/gt_cramatorsc/clips/shortlist_clips_k100.json
```

## 4. GPU fine rerank — the scheme from yesterday (`pyramid_newformat.locate`)

`--topk 100` so each frame keeps all 100 fine-scored cells for the aggregation sweep. `--dataset-dir`
is a local `aws s3 sync` of the packed store (read-bound EBS → local is faster) or the S3 URL directly.

```bash
# packed kram store: s3://geo-reference/embeddings/kram_newformat/pyramid_dinov2vitg14_p448/kram
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with boto3 --with tqdm \
  python -m pyramid_newformat.locate \
  --queries kram=/home/ubuntu/work/gt_cramatorsc/clips/query_clips_d1024.h5 \
  --indices /home/ubuntu/work/gt_cramatorsc/clips/shortlist_clips_k100.json \
  --dataset-dir /home/ubuntu/work/mpp_local/kram \
  --topk 100 --device cuda \
  --out /home/ubuntu/work/gt_cramatorsc/clips/locate_clips.json
```

## 5. Clip aggregation → top-10 + KMZ + distR sweep (no GPU)

```bash
uv run --with numpy python -m clip_search.aggregate \
  --locate   /home/ubuntu/work/gt_cramatorsc/clips/locate_clips.json \
  --manifest /home/ubuntu/work/gt_cramatorsc/clips/clips_manifest.json \
  --out-dir  /home/ubuntu/work/gt_cramatorsc/clips/out \
  --topn 10 20 30 40 50 --topk 10
```
For each clip and each per-frame cut `N ∈ {10,20,30,40,50}`, aggregates cells two ways across the clip:
**sum** (Σ fine `cell_score`) and **votes** (#frames with the cell in top-N), takes the top-10, and writes
`out/kmz/{sum,votes}_top{N}.kmz` + `out/clip_search_results.json` (top-1 / top-10 `distR@{250,500,1000}m`
per agg×N). A compact table is printed to stdout.

---

### Artifacts to fill in (confirm the server paths)
- `<KRAM_E2C_CKPT>_best.pt` — trained e2c SuperVLAD checkpoint (d1024).
- `<DICT>/k32.pt` — VLAD assign dict.
- `<KRAM_COARSE_GALLERY>.h5` — kram coarse cell gallery (group-per-tile `ift_dino`, `lat`/`lon` attrs).
- `--dataset-dir` — local sync of `kram_newformat/pyramid_dinov2vitg14_p448/kram`.
