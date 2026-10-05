# magsacpp_torch — MAGSAC++ homography on GPU (PyTorch)

A self-contained, isolated implementation of the MAGSAC++ marginalized robust homography estimator
as PyTorch tensor ops, for the UAV→satellite geo-localisation reranker. It lives under
`geoLocalisation/experiments/` and does **not** touch any production backend. See `REPORT.md` for the
source-pinned math, verification status and divergences.

> Status: the math core (loss / σ-consensus++ weights / IRLS) follows the paper + `danini/magsac`
> and is verified locally against the analytic incomplete-gamma functions and on synthetic data. It is
> **not yet** validated bit-for-bit against the C++ oracle, so it is not called "MAGSAC++-equivalent"
> until the oracle parity in `REPORT.md` §6 passes. The production reranker default is unchanged.

## Install / environment

Runs on the repo's `uv` env. Core needs only `torch` (+ `numpy`); `scipy` is used by tests. The
applied-baseline compare needs `opencv-python`; the data commands need `h5py`/`rasterio`/`boto3`
(already in the repo `pyproject.toml`).

```bash
# from geoLocalisation/
uv run python -c "import torch, scipy; print(torch.__version__)"
```

All commands below are run from **`geoLocalisation/experiments/`** so that `magsacpp_torch` is the
top-level package.

## Python API

```python
import torch
from magsacpp_torch import estimate_homography_magsacpp, MagsacppConfig

# points1 = source (map-crop) coords, points2 = dest (query) coords  -> H maps map-crop -> query,
# residual in query patch-grid units (same convention as patch_rerank/matcher.py).
cfg = MagsacppConfig(sigma_max=1.5, inlier_threshold=2.0, max_hypotheses=2000, irls_iters=3)
res = estimate_homography_magsacpp(
    points1, points2,              # (N,2) or (B,N,2) torch tensors / array-likes
    valid_mask=None,               # (N,) / (B,N) bool for padding ragged batches
    config=cfg,
    generator=torch.Generator().manual_seed(0),
    hypothesis_indices=None,       # (S,4)/(B,S,4) recorded schedule for parity/replay
    return_trace=False,
)
res.H            # (3,3) map->query, NaN on failure
res.success, res.status      # status in {ok, n_lt_4, all_invalid, no_valid_hypothesis, degenerate}
res.score, res.total_loss    # quality = 1/total_loss (higher better); total_loss (lower better)
res.inlier_mask, res.inlier_count    # explicit threshold, NOT used by the score
res.n_hypotheses, res.n_irls_steps
```

`sigma_max` must match the problem's noise scale — see `REPORT.md` §4 (oversizing it drifts the IRLS
fit). Everything in the fit/score/refine path is pure Torch under `torch.inference_mode()`; it runs on
CPU or CUDA by moving the input tensors to the device.

## CLI

```bash
cd geoLocalisation/experiments

# 1. audit: config, pinned refs, environment, expected data paths (runs anywhere)
python -m magsacpp_torch.cli audit

# 2. parity:
python -m magsacpp_torch.cli parity gamma                 # analytic vs LUT + danini constant
python -m magsacpp_torch.cli parity synthetic --pairs 50  # torch vs cv2 USAC_MAGSAC (needs opencv)
python -m magsacpp_torch.cli parity oracle --records traces.json   # vs C++/cv2 oracle (server)

# 3. benchmark (estimator-only; CUDA uses events+warmup):
python -m magsacpp_torch.cli benchmark --device cuda --dtype float64 --pairs 64
python -m magsacpp_torch.cli benchmark --device cuda --dtype float32 --pairs 64

# 4. tests
python -m pytest magsacpp_torch/tests -q
```

`cache` and `evaluate` require the real reranker data (query H5 + token store + shortlist) and run on
the GPU server — see below.

## Data (server) — paths & contract

From the audit of `patch_rerank` (see `REPORT.md` §2 for exact files):
- Query grids: `queries/query_<city>_*.h5` (per-frame `ift_dino` (D,N), `patch_grid_h/w`, `lat/lon`,
  optional `keep_indices` sky mask). Cities: `kup`, `kramatorsc` (+ `liman_day/night`).
- Token stores: `<city>/rerank_store_<city>.h5` (local) or S3 per-cell
  `s3://geo-reference/embeddings/<city>/rerank_cells_*/…` with an `_index.json` manifest.
- Dense / checker indices: `dict/tiles_index_dense.h5`, `dict/tiles_index_checker1000.h5`.
- Shortlist: `shortlist_k70.json` (`{"shortlist": {"<city>:<stem>": {"cells":[…]}}}`).
- CRS EPSG:3857; pyramid levels in metres (e.g. 1000…300); residuals in query patch-grid units;
  geodesic error via haversine (R=6371000); metric distR@250m top1/top5.

Correspondences come from `patch_rerank.matcher.matched_coords` (mutual-NN on L2-normalised DINO
tokens). GT is used only for evaluation — never fed to fitting/scoring.

### `cache` (server): immutable correspondence cache
Build one record per (city, query_id, candidate_id, level) with `xy_uav`/`xy_map`, confidence, valid
mask, both image sizes, coord transforms, GT+metadata kept separate, and a fingerprint of the input
arrays + matcher config. This cache is what both the Torch core and the cv2 baseline consume, so the
comparison is apples-to-apples. (CLI stub prints the schema; wire it to `matched_coords` on the box.)

### `evaluate` (server): quality vs cv2 baseline
Run the Torch estimator over the cache, reproduce the rerank score `inliers / n_query_tokens`, the
sub-tile lat/lon (`matcher.grid_to_latlon`) and distR@250m top1/top5 + median/p90/p95 geodesic error,
and compare against `cv2.USAC_MAGSAC` on the **same** cache and ranker. Report per city / level / N /
overlap, with paired outcomes and the Stage-A ceiling (if GT cell isn't shortlisted, Stage-B can't fix
it).

## Building the C++ oracle (server, for math parity)

The independent oracle is the instrumented `danini/magsac` C++ (a Python recomputation is **not**
independent — see `oracle.py::PythonReference`). Minimal recipe:

1. Clone at the pinned commit:
   `git clone https://github.com/danini/magsac && cd magsac && git checkout d259f8b && git submodule update --init` (graph-cut-ransac @ 9fa075d).
2. Add a **test-only** wrapper that, for a fixed correspondence array + a recorded minimal-sample
   schedule, exports: normalized coords & T, each minimal H, residuals, per-hypothesis loss, the
   σ-consensus++ weights, one weighted refit, the IRLS sequence, the chosen best hypothesis and the
   final H + inlier mask — as JSON matching the `oracle.py` record schema. Patch points:
   `MAGSAC::getModelQualityPlusPlus`, `MAGSAC::sigmaConsensusPlusPlus`,
   `solver_homography_four_point.h`. Do not change the oracle math; document the patch.
3. Emit records with the same `points1/points2/schedule/config` you pass to the Torch replay, fill the
   `oracle` block, then: `python -m magsacpp_torch.cli parity oracle --records traces.json`.

`oracle.py::dump_schedule` builds the pre-oracle record; `run_torch_replay` / `compare_trace` do the
Torch side and the comparison (projection-space geometry + sign/scale-normalised matrix diff + loss /
best-hypothesis / mask agreement).

## Order of work (brief §13)

audit → fix oracle/inputs → correctness float64 → controlled parity → GPU parity → real-data quality →
optimization → benchmark → opt-in integration → report. We are through **correctness float64** locally;
the remaining steps are the `REPORT.md` §6 Pending list on the server.
