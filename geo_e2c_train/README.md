# geo_e2c_train — DSS training loop (glue)

Wires the three isolated packages into one training run (no new math):

- **`siam_e2c_model`** — `build_e2c_model` (agg ∈ supervlad/vlad/residual × pyramid_mode ∈ cell/concentric);
- **`geo_train_batching`** — `build_split_relevance` (positives = the split's) → canonical pairs →
  neighbour cache → logical-batch planner → symmetric multi-positive CE (the proven e2c/DSS loss);
- **`geo_split_no_overlap`** — the leakage-free split (`split.json` + its config).

Data comes from `map_extract` H5s: **tile grids** (raw d1024 galleries, group-per-tile) and **drone
query tokens** (`map_extract --images` output). Tile grids are cached in RAM (`--grid-size` optional
resample); a tqdm bar tracks epochs; eval = Recall@1/5/10/20 + median rank on val/test.

## Prerequisites

1. Split built (`geo_split_no_overlap` → `split_pyr250/split.json`) + its config yaml.
2. Dictionary `dict/k32.pt` (`vlad_vocab.fit`; assign_weight + centroids, D=1024).
3. **Drone query tokens at d1024** — extract once per city (same backbone/desc_dim/proj_seed as the
   galleries; sky handled by `--segment_sky` if wanted):
   ```bash
   uv run --python 3.11 --with "torch==2.5.1" --with "torchvision==0.20.1" --with h5py \
          --with numpy --with tqdm --with pillow \
     python -m map_extract.extract --images /home/ubuntu/work/gt_cramatorsc/GT_flat \
       --out /home/ubuntu/work/out/gallery_h5/query_kramatorsc_d1024.h5 \
       --backbone dinov2_vitg14 --desc_dim 1024 --proj_seed 0 --amp --device cuda
   # repeat for kup, liman_day
   ```

## Run (residual + classic cell apex — the A/B baseline)

```bash
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with shapely \
       --with pyyaml --with tqdm python -m geo_e2c_train.train \
  --split-config /home/ubuntu/work/split_pyr250.yaml \
  --split-json   /home/ubuntu/work/out/gallery_h5/split_pyr250/split.json \
  --galleries kramatorsc=/home/ubuntu/work/out/gallery_h5/map_dinov2_kramatorsc_s250m_d1024.h5 \
              kup=/home/ubuntu/work/out/gallery_h5/map_dinov2_kup_s250m_d1024.h5 \
              liman_day=/home/ubuntu/work/out/gallery_h5/map_dinov2_liman_day_s250m_d1024.h5 \
  --queries kramatorsc=/home/ubuntu/work/out/gallery_h5/query_kramatorsc_d1024.h5 \
            kup=/home/ubuntu/work/out/gallery_h5/query_kup_d1024.h5 \
            liman_day=/home/ubuntu/work/out/gallery_h5/query_liman_day_d1024.h5 \
  --assign /home/ubuntu/work/out/gallery_h5/dict/k32.pt \
  --agg residual --pyramid-mode cell --k 32 --d-token 1024 --scales 8 4 2 1 \
  --epochs 40 --b-log 32 --out /home/ubuntu/work/out/gallery_h5/split_pyr250/run_residual_cell
```

The **A/B pair** (isolate pyramid shape) is `--agg supervlad --pyramid-mode cell` vs
`--agg supervlad --pyramid-mode concentric --concentric-sizes 1000 840 710 600 500 420 350 300 250`,
everything else identical (same split, dict, seed, optimizer, epochs). Outputs: `metrics.jsonl`,
`best.pt` (+ `resolved_config`), `best.json`.

## Tests

`python -m pytest geo_e2c_train/tests` — end-to-end smoke on synthetic galleries/queries/split for
residual/cell and supervlad/concentric (2 epochs + eval), CPU. Skips if h5py/shapely absent.
