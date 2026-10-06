# cv2.MAGSAC++ rerank — full diagnostics archive

`search_pyramid_s3.py --diag-dir <DIR>` records **every intermediate result** of the existing
`cpu_magsac` (= `cv2.USAC_MAGSAC`) pyramid rerank, for the whole Kramatorsk test set, **without
changing a single score, rank, best position/level or predicted lat/lon**. It is the same search —
one DINO forward, coarse e2c-SuperVLAD top-100, every candidate's `position × level` grid matched by
mutual-NN and verified by one `cv2.findHomography(rm, qm, USAC_MAGSAC, ransacReprojThreshold=2.0)`
call, `level_agg=sum`, `cell_agg=mean`, point = best position reported at its best level — but now the
inputs/outputs of every cv2 call are persisted.

The archive, not the top-5 table, is the deliverable. The top-5 metrics are still printed/written for a
coverage sanity check.

## Run (full Kram, on the EC2 geo box where cv2/GPU/S3/data live)

```bash
cd /home/ubuntu/work/geoLocalisation
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with boto3 \
       --with tqdm --with opencv-python-headless \
  python -m patch_rerank.search_pyramid_s3 \
    --queries kram=/home/ubuntu/work/out/gallery_h5/geo_iso_noverlap/queries/query_kramatorsc_d1024.h5 \
    --shortlist /home/ubuntu/work/out/shortlist_kram_k100.json \
    --index-uri s3://geo-reference/embeddings/kram/pyramid_dinov2vitg14_p448/kram/_index.json \
    --k-coarse 100 --topk 5 --level-agg sum --cell-agg mean --device cuda \
    --cache-cap 500 \
    --diag-dir /home/ubuntu/work/diag_kram_cv2_magsac \
    --out /home/ubuntu/work/diag_kram_cv2_magsac/search.json
```

Notes:
- The `--queries` city prefix (`kram=`) must match the keys in `shortlist_kram_k100.json`
  (e.g. `kram:100_48.59…`). Use whatever prefix the shortlist uses.
- `--diag-dir` **forces** `--verify-backend cpu_magsac` and `--execution-order query-major` (so each
  query's arrays live in one file and resume is per-query). It is NOT the experimental `torch_magsacpp`
  path — this archives the production cv2 path only.
- It **asserts** `cv2.USAC_MAGSAC` exists *and* runs a probe fit before starting — if this OpenCV build
  lacks MAGSAC the run aborts (no silent RANSAC fallback; a MAGSAC run never gets mislabelled).
- Set `--cache-cap` ≥ the number of unique Kram cells (~440) so query-major still downloads each cell
  only once; otherwise cells re-download across queries.
- The full run processes all 100 coarse candidates per query (no `--k-coarse 30`, no `--max-queries`).
- **Resume**: re-run the same command; finished queries (in `completed.json`) are skipped. It refuses
  to append to a `--diag-dir` built with a different config (fingerprint guard). `--no-resume` reprocesses.

### Small verification run first

```bash
#   --max-queries 3  on the same command, into a throwaway --diag-dir
```
Then check (see the reader below): (1) the diag `search.json` top-k equals a plain run without
`--diag-dir`; (2) every `candidate × position × level` has a `records.jsonl` row incl. `cv2_called=false`
skips; (3) each row's `level_score == n_inliers / n_query_patches`; (4) `aggregation.jsonl`
`position_score == Σ level_scores`, `cell_score_mean == mean(position_scores)`; (5) any row's
`arrays` group reloads `H/mask/rm/qm` with no re-matching / DINO / S3.

## Files under `--diag-dir`

Everything is written **incrementally** (append per record / per query), so a killed run keeps its
finished part.

| file | one row/group per | holds |
|---|---|---|
| `manifest.json` | the run | schema_version, run_id, git commit, config **fingerprint**, full config, cv2 version + `USAC_MAGSAC` int, store index_uri + store config, query paths, shortlist path, seed |
| `records.jsonl` | query × cell × position × level | the scalars + status of every geometric check (incl. skips) + pointer to its arrays |
| `aggregation.jsonl` | query × cell | all level scores per position → position_score (Σ levels) → cell_score (mean positions) + best pos/level |
| `summary.jsonl` | query | full coarse top-100, full reranked list (all candidates), final top-K, GT + distance metrics |
| `arrays/<query>.h5` | query × cell × position × level | `qm, rm, q_idx, r_idx, H, mask, residuals` (the per-check geometry) |
| `completed.json` | the run | fingerprint + finished query ids (resume marker) |

### `records.jsonl` row

Key = `(query_id, cell_id, position_id, level_id)`.

```
query_id, cell_id, coarse_rank            # coarse_rank = rank of this cell in the coarse top-100
position_id, level_id, level_m            # which pyramid position + concentric level
crop_key, cell_lat, cell_lon, pos_lat, pos_lon, tile_m   # map-crop geo metadata
map_grid_h, map_grid_w, query_grid_h, query_grid_w, query_n_keep, patch_px  # grid/patch metadata
n_query_patches                           # the score DENOMINATOR
n_map_patches, n_mutual                   # #map tokens, #mutual-NN correspondences
cv2_called (bool), status, fail_reason    # ok | no_consensus | skipped_insufficient_matches
method_int, reproj_thresh                 # cv2 method (USAC_MAGSAC int) + threshold
n_inliers, level_score                    # level_score = n_inliers / n_query_patches (verbatim)
verify_s                                  # wall time of this one cv2 call
h5, h5_group                              # pointer to the arrays (null if cv2 not called): h5 is the
                                          #   FINAL location — s3://… when --diag-arrays-s3 is set,
                                          #   else local arrays/<query>.h5 ; h5_group = <cell>/p{i}/l{L}
```

Three states are distinguished: **skipped** (`cv2_called=false`, `< 4` mutual matches),
**no_consensus** (cv2 ran, returned no mask), **ok** (may still be a low score).

### `arrays/<query>.h5` group `"<cell>/p{i}/l{L}"`

Datasets (all in the exact order passed to cv2):
- `qm (M,2) f8`, `rm (M,2) f8` — matched query / map-crop patch coordinates;
- `q_idx (M,) i8`, `r_idx (M,) i8` — which query / map grid token each match came from;
- `H (3,3) f8` — the homography, **unrounded**, direction **rm (map crop) → qm (query frame)**
  (group attr `H_direction`); absent if no H;
- `mask (M,) u8` — cv2 inlier mask, same order (`mask.sum() == n_inliers`);
- `residuals (M,) f8` — per-correspondence reprojection residual ‖π(H·rm) − qm‖ (diagnostic only;
  does not affect the mask).
Group attrs: `n_inliers, level_score, status, method_int, reproj_thresh, H_direction`.

### `aggregation.jsonl` row

Key = `(query_id, cell_id)`. `positions[i]` = `{position_id, pos_lat, pos_lon, level_scores{L→score},
position_score, best_level_m}`; plus `position_scores[]`, `n_positions`, `cell_score_mean/min/max`,
`best_position_id`, `best_level_m`, `best_h5_group`. `position_score` is Σ over levels (because
`--level-agg sum`); the service cell score is `cell_score_mean`.

### `summary.jsonl` row

Key = `query_id`. `gt [lat,lon]|null`, `coarse_top[{cell_id,coarse_rank}]` (all 100),
`reranked_top[{cell_id,cell_score,rank,lat,lon,level_m,dist_m}]` (ALL candidates, not just top-k),
`final_topk[...]`, `metrics{fine_dist_m, fine_dist_topk_m}`.

> Full-set top-5 metrics: on a resumed run, `search.json` only covers the queries processed *this*
> invocation. Aggregate `summary.jsonl` across the whole archive for the complete-set numbers.

## Read one query / candidate / position / level

```python
import json, os, tempfile, h5py, numpy as np
DIR = "/home/ubuntu/work/diag_kram_cv2_magsac"

# read a FINALIZED query only (an in-progress query's arrays H5 is mid-write and unreadable)
done = set(json.load(open(f"{DIR}/completed.json"))["queries"])
row = next(json.loads(l) for l in open(f"{DIR}/records.jsonl")
           if (r := json.loads(l))["cv2_called"] and r["query_id"] in done)
print(row["query_id"], row["cell_id"], row["h5_group"],
      "score", row["level_score"], "=", row["n_inliers"], "/", row["n_query_patches"])

# row["h5"] is the final location: s3://… (when offloaded) or local arrays/<q>.h5
uri = row["h5"]
if uri.startswith("s3://"):
    import boto3
    b, key = uri[5:].split("/", 1); local = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    boto3.client("s3").download_file(b, key, local)
else:
    local = os.path.join(DIR, uri)
with h5py.File(local, "r") as f:                               # no re-matching / DINO / fetch needed
    g = f[row["h5_group"]]
    qm, rm, mask, H = g["qm"][:], g["rm"][:], g["mask"][:], g["H"][:]
    assert int(mask.sum()) == row["n_inliers"]                 # reproduce the inlier count
    proj = (np.c_[rm, np.ones(len(rm))] @ H.T);  proj = proj[:, :2] / proj[:, 2:3]
    print("reproj residual (inliers):", np.linalg.norm((proj - qm)[mask.astype(bool)], axis=1))
```

## What "all cv2.MAGSAC++ intermediate results" means here

Everything the standard OpenCV Python API exposes for **each** call: correspondences → H + inlier mask
→ inliers → level score → position score → cell score. It does **not** include MAGSAC's internal random
hypotheses, every USAC iteration, or the internal σ-consensus losses — the Python API does not return
those; extracting them needs an instrumented C++ OpenCV build (a separate task). This archive is the
complete set of *available* results of every existing cv2 call, not an internal USAC trace.
