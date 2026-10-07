# patch_rerank_local — isolated local fine-reranker (no S3)

Reranks a coarse shortlist against **local** cell pyramids (the production case: the map is downloaded
once, not fetched per query), on **GPU or CPU**. Same score (`inliers / n_query_patches`) and same
level→position→cell aggregation as the S3 reranker, so numbers match bit-for-bit.

## Layout
| file | role |
|---|---|
| `local_cell_store.py` | `LocalCellStore` / `CellData` — read local cell H5s (same schema as the S3 store), bounded open-handle LRU, no boto3 |
| `rerank_local.py` | driver: cell-major / query-major, `--verify-backend cpu_magsac` (cv2 only; GPU MAGSAC++ = `magsacpp_torch.search`), `--device {cuda,cpu}` |
| `curve.py` | top-K curve (distR@250/500/1000 + gtcell_recall) from a run's `--out` + `--dump-scores` |
| `tests/` | synthetic store + curve tests (no cv2 / no cuda) |

Scoring (`_load_cell`, `_score_loaded`, `_aggregate`) is copied verbatim from
`patch_rerank.search_pyramid_s3`; geometry is reused from the pure `patch_rerank.matcher` /
`patch_rerank.gpu_verify`. No network, no diagnostics archive, no `--diag-dir` forcing.

## Cell H5 schema (per cell, same as `build_cell_store_s3`)
Datasets `lat` / `lon` (per sub-position), attrs `levels_m` (list) + `n_positions`, pyramid grids
`p{i}/l{L}` shaped `(h, w, D)`. Address by `cell_id` (the shortlist ids, e.g. `kup:73_lvl0`).
A `cells_dir` either has an `_index.json` manifest (resolve by key basename) or is a flat/`cells/` dir
of `*.h5` (cell_id = filename stem).

## Get the cells local once (example: kup p448)
```bash
mkdir -p ~/work/local_pyr/kup
aws s3 sync s3://geo-reference/embeddings/kup/pyramid_dinov2vitg14_p448/kup/ ~/work/local_pyr/kup/
# → ~/work/local_pyr/kup/_index.json + ~/work/local_pyr/kup/cells/*.h5
```

## Run (matches the canonical kup run, but local)
```bash
cd /home/ubuntu/work/geoLocalisation
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
       --with opencv-python-headless python -m patch_rerank_local.rerank_local \
  --queries kup=/home/ubuntu/work/out/gallery_h5/geo_iso_noverlap/queries/query_kup_d1024.h5 \
  --shortlist /home/ubuntu/work/out/gallery_h5/kup/shortlist_kup_prod_k100.json \
  --cells-dir ~/work/local_pyr/kup \
  --k-coarse 100 --topk 5 --level-agg sum --cell-agg mean \
  --execution-order cell-major --verify-backend cpu_magsac --device cuda \
  --dump-scores ~/work/kup_scores_local.json --out ~/work/kup_search_local.json

python -m patch_rerank_local.curve --search ~/work/kup_search_local.json \
                                   --dump ~/work/kup_scores_local.json
```
CPU-only box: `--device cpu`. For the **GPU MAGSAC++** reranker use `magsacpp_torch.search --store-dir`
(reads the same local store); the gpu_batch / gpu_kornia verifiers were removed from this package.

## Tests
```bash
cd /home/ubuntu/work/geoLocalisation
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with pytest \
  pytest patch_rerank_local/tests -q
```
