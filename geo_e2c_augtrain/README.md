# geo_e2c_augtrain

Retrain the e2c retriever (the `supervlad_cell_w1` recipe) **with full drone-image augmentation**,
eval every 3 epochs, and a **limited-map evaluation** that grows the search area (50, 60, 70, 80, 90,
100 km² around each query) to see how coarse retrieval behaves **without reranking**.

Isolated package: it **reuses and never edits** `geo_e2c_train` (batching / loss / model / eval),
`siam_e2c_model`, `geo_train_batching`, `map_extract` (DINO extractor + sky), `sky_filter`, and the
`siam_model_stage3` UAV-augment recipe.

## The key architectural fact (why this package exists)

`geo_e2c_train.train` trains on **FROZEN pre-extracted DINO tokens** read from the query H5
(`ift_dino`). The model never sees a raw image and never runs DINO during training. So **image-level
augmentation is impossible there** — you cannot rotate/jitter/mask a token grid that was baked once.

Real drone augmentation therefore requires a **per-epoch re-encode**: each epoch, take the raw drone
frame → augment (geometric + photometric + affine-fill) → **sky mask + (optional) road filter** →
run DINO → tokens → `model.encode_query`. This is the `siam_model_stage3` recipe, ported to feed the
e2c trainer instead of the stage3 model. It is slower per epoch (a DINO forward per query per epoch),
which is the cost of genuine augmentation.

## Modules (planned)

- `augmented_query_store.py` — drop-in replacement for `geo_e2c_train.QueryTokenStore` whose
  `tokens(qid)` returns FRESH per-epoch tokens: raw frame → `uav_augment` (stage3 recipe) → sky mask
  (`map_extract.sky` / `sky_filter`) → DINO (`map_extract.dino`) → same projection as the frozen
  query H5 (seed read from the H5 attrs so the space matches the gallery). Deterministic per
  `(epoch, qid)`. Batched re-encode to amortise the DINO forward.
- `train_aug.py` — thin fork of `geo_e2c_train.train` that swaps `QueryTokenStore` for the augmented
  store and re-encodes at each epoch boundary. Everything else (split relevance, neighbour cache,
  logical-batch planner, symmetric multi-positive CE, checkpointing) is imported unchanged. Default
  `--eval-every 3`.
- `eval_limited_map.py` — evaluation only: for each query restrict the gallery to the tiles inside a
  square area A (km²) centred on the query, sweep `--areas-km2 50 60 70 80 90 100 …`, and report
  Recall@K + median rank + distR@250/500/1000 m as the area grows — **coarse retrieval, no rerank**.

## Constraints / open decisions (confirm before big build)

1. Per-epoch DINO re-encode is the only way to augment images — confirm the slower epochs are OK.
2. Only the QUERY (drone) side is augmented + re-encoded; the map gallery stays frozen tokens.
3. Augmentation recipe = stage3 `stage3_strong` (stronger + affine_fill) + sky mask; road filter
   optional (SegFormer/UAVid, `project_road_filter_drone`).
4. Raw drone frames: `~/work/gt_cramatorsc|gt_kup|gt_liman` (paths in `multicity_split.json`).
5. Projection must match the frozen query H5 (read `projection_seed` / backbone / desc_dim from the
   H5 attrs) so query and gallery live in the same space.
