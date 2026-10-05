# REPORT — Torch MAGSAC++ homography core

Status date: 2026-10-05. This module implements the MAGSAC++ marginalized robust homography
estimator as PyTorch tensor ops, with the math core taken from the paper + `danini/magsac`. It is
**not yet certified MAGSAC++-equivalent**: the scoring / σ-consensus++ / IRLS have been verified
locally against the analytic incomplete-gamma functions and on synthetic data, but **not** bit-for-bit
against the C++ oracle. Do not flip any production backend until §"Pending" clears on the server.

---

## 1. Pinned references & licenses

| Role | Source | Ref / commit | License |
|---|---|---|---|
| **Math core** (designated) | `danini/magsac` | `d259f8b3a8925025e45667241fb68629b07603bb` (branch `master`) | BSD-3-Clause, © CTU |
| ↳ estimators / solvers / samplers / gamma LUT | submodule `danini/graph-cut-ransac` | `9fa075dc76d0e632ab3d73297c76ac3d4a29decd` | BSD-3-Clause |
| Paper | Barath et al., *MAGSAC++…*, CVPR 2020 | arXiv:1912.05909 | — |
| **Applied baseline** (NOT a math ref) | OpenCV USAC `cv2.USAC_MAGSAC` | opencv `4.x` `62587ae9976b28cfa61ad940d0e7f610b8742ee4` | Apache-2.0 |

We designate **danini/magsac as the math-core reference** (it matches the paper's n=4 convention and
has the full-resolution gamma LUT) and keep **OpenCV USAC as a separate applied baseline**. OpenCV
uses different constants (DoF=2, k=3.04, C=0.5) and must not have its constants mixed into the core.

---

## 2. Formula → source → code mapping

All quantities are in terms of the **squared forward reprojection residual** `sq` (px², dest/query
units). With `n=4`, `k=3.64`, `C=0.25`, `σ=σ_max`: `a1=(n+1)/2=2.5`, `a2=(n-1)/2=1.5`,
`t=sq/(2σ²)`, `tk=k²/2`.

### 2.1 σ-consensus++ weight (IRLS)  — `gamma.py::GammaMath.weight`
```
w(sq) = C·2^a2/σ · ( Γ_up(a2, t) − Γ_up(a2, tk) )     for sq ≤ (kσ)²,   else 0
```
- Paper Eq. (2) (marginal inlier density). Source: danini `MAGSAC::sigmaConsensusPlusPlus`
  (`magsac.h` ~L835–865), `one_over_sigma = C·2^((DoF−1)/2)/σ`, `stored_gamma_values` = **upper**
  incomplete Γ(a2,·), `gamma_k = getUpperIncompleteGammaOfK()`.
- **Verified**: our `Γ_up(1.5, k²/2) = 0.00365726083409…` reproduces danini's hard-coded
  `getUpperIncompleteGammaOfK() = 0.0036572608340910764` to ~1e-15 (`test_gamma.py`).

### 2.2 Marginalized loss / quality  — `gamma.py::GammaMath.loss`
The loss is the integral of the weight, `ρ(r)=∫₀ʳ x·w(x)dx`; carried out analytically:
```
ρ(sq) = C·2^a2·σ · [ γ_low(a1, t) + t·( Γ_up(a2, t) − Γ_up(a2, tk) ) ]   for sq ≤ (kσ)²
ρ(sq) = C·2^a2·σ ·   γ_low(a1, tk)   (= ρ_sat, the outlier loss)         for sq > (kσ)²
```
`ρ(0)=0`, monotone non-decreasing, continuous at the cutoff (2nd term →0). Total loss
`L=Σρ`; quality `Q=1/L` is **maximized** ⇔ `L` is **minimized** (we rank by `L`).
- Paper Eq. (3); danini `MAGSAC::getModelQualityPlusPlus` (`magsac.h` ~L947–1045), which accumulates
  `σ²/2·stored_lower_incomplete[x] + sq/4·(stored_complete[x] − gamma_value_of_k)` scaled by
  `2^((DoF+1)/2)/σ`. Algebraically this equals our form **iff** danini's `stored_complete_gamma_values`
  is the **upper**-incomplete Γ(a2,·) (its name is misleading). **This equivalence is asserted here
  from the derivation + the weight match; bit-exact confirmation is a Pending oracle item.**
- ⚠️ A naïve transcription of the printed paper 2nd term as a *lower*-incomplete γ gives a loss that
  **dips negative** near r=0 (non-monotone) — that was caught by `test_loss_monotone_increasing_inside`
  and is why the derived (upper-Γ) form is used.

### 2.3 Minimal solver  — `solver.py::solve_minimal_h`
4-point (or N≥4) Hartley-normalized DLT. The homography is the **true 9-vector right nullspace** of
the 8×9 design matrix, taken as `Vh[-1]` from `torch.linalg.svd(A, full_matrices=True)` (Vh is 9×9).
- danini uses Gaussian elimination on 8×9 (`solver_homography_four_point.h`); OpenCV uses SVD
  (`homography_solver.cpp` `HomographyMinimalSolver4ptsImpl`). We use full-matrix SVD.
- ⚠️ `full_matrices=False` returns a 9×8 Vh and **omits the 9th right singular vector** → wrong null
  space. Guarded by `test_minimal_solver_needs_full_matrices_nullspace`.

### 2.4 Weighted non-minimal fit / IRLS  — `solver.py::solve_weighted_h`, `estimator.py`
Weighted **algebraic** (DLT) least squares: each correspondence's two design rows scaled by its
σ-consensus++ weight; homogeneous SVD nullspace. IRLS = residual→weight→refit, `irls_iters` steps
(danini default **1**), accept a refit only if total loss did not increase (`irls_require_improvement`).
- danini does weighted algebraic DLT too (paper Eq. 1) but with an **inhomogeneous h33≡1** system via
  column-pivoted QR; we use the homogeneous SVD form (no h33≡1 limitation when true h33≈0). **This is a
  refinement-path divergence to be measured against the oracle**, documented, not a core-math change.

### 2.5 Residual  — `solver.py::forward_sq_residual`
Forward reprojection error src→dst only, **squared**, dest units — matching danini
`homography_estimator.h::squaredResidual` and OpenCV `ReprojectionErrorForwardImpl` (both forward, not
symmetric/Sampson; danini's "Sampson" comment is wrong boilerplate). Negative homogeneous denominators
are kept (valid); `|w|<min_abs_denominator` ⇒ residual `+inf` (near-infinity, outlier); non-finite ⇒
`+inf` (never NaN, never silently clamped). Guarded by `test_residual_*`.

### 2.6 Sampling  — `estimator.py::_sample_indices`
Uniform sampling **without replacement** inside the minimal sample (`torch.multinomial(..., replacement=False)`).
- **Divergence (flagged):** the paper/danini default is **P-NAPSAC** (localised PROSAC). Our uniform
  sampler changes which hypotheses are drawn, so end-to-end runs are not bit-comparable to danini; math
  parity therefore uses a **recorded schedule** (`hypothesis_indices`), not a shared seed (brief §8).

### 2.7 Inlier mask (reporting only)  — `estimator.py`
`sq ≤ inlier_threshold²` (default 2.0 px, = cv2 `ransacReprojThreshold` in `patch_rerank/matcher.py`).
Kept **separate** from the score (brief §5.9); it never feeds hypothesis selection.

---

## 3. danini ⟷ OpenCV divergences (do not mix)

| Quantity | danini (core) | OpenCV (baseline) |
|---|---|---|
| DoF `n` | **4** | 2 |
| quantile `k` | **3.64** | 3.04 |
| `C(n)` | **0.25** | 0.5 |
| `Γ_up(a2,k²/2)` | 0.0036572608340910764 | 0.00419 |
| σ_max default | 10.0 (`maximum_threshold`) | 7.5 (clamped) |
| score sign | maximize `1/L` | minimize negated loss |
| LUT | 36k/100k entries, step 1e-4, round-indexed | 500 entries, interpolated, truncation-indexed |
| minimal nullspace | Gaussian elimination | SVD |
| sampler | selectable (P-NAPSAC/PROSAC/…) | uniform |

**Internal danini inconsistency (confirmed in source):** the scorer uses `σ_max = maximum_threshold/k`
while the refit uses `σ_max = maximum_threshold` directly. We therefore expose `sigma_max` explicitly
rather than deriving it, so the chosen value is used consistently in both paths. See
`config.MagsacppConfig`.

---

## 4. Choosing σ_max (important)

`σ_max` is the noise-scale upper bound of the marginalization; the hard cutoff is `τ = k·σ_max`. It
must be set to the problem's noise scale. Oversizing it (e.g. the raw danini default 10 px on tight
patch-grid coordinates where the inlier threshold is ~2 px) lets σ-consensus++ assign weight to far
points and **drifts the IRLS fit** — observed directly: a 50 %-outlier synthetic case recovered the
wrong model at `σ_max=10` and the correct model at `σ_max=1`. For the reranker (residuals in query
patch-grid units, threshold ≈ 2), start `σ_max` near the threshold (≈1–2) and sweep.

---

## 5. Verified locally (this machine, CPU float64, torch 2.12.1)

- Incomplete-gamma analytic path == scipy (`test_analytic_matches_scipy_lower_upper`).
- danini `gamma_value_of_k` reproduced to 1e-15 (`test_reproduces_danini_gamma_value_of_k`).
- Loss: zero at origin, monotone, saturates & continuous at cutoff; invalid residual → finite outlier
  loss (not NaN). Weight: ≥0, decreasing, zero beyond cutoff.
- LUT vs analytic: loss & weight max|Δ| ≈ 1.7e-10 / 3.8e-10 (`parity gamma`).
- Minimal solver recovers known H exactly (reprojection < 1e-6); full-matrices nullspace required.
- Weighted solver recovers H with outliers down-weighted.
- Estimator: clean/20 %/50 % outliers recovered; 80 % recovered with a larger budget; N<4 / all-invalid
  / collinear return correct failure status (no bogus success); **chunk-size and padding invariance**;
  replay schedule deterministic; batched API.
- 26/26 tests pass (`pytest magsacpp_torch/tests`). CLI `audit`, `parity gamma`, `parity synthetic`,
  `benchmark --device cpu` run (≈55 ms/pair float64 CPU estimator-only at n=200, 1000 hyps).

## 6. Pending (requires the GPU server / real data / C++ toolchain)

1. **Math-core oracle parity** — build the instrumented `danini/magsac` C++ (or cv2) oracle, dump
   traces on a recorded schedule, run `cli.py parity oracle --records …`. Targets (well-conditioned):
   float64 scalar loss/weights `atol=1e-8 rtol=1e-6`, projections ≤1e-5 px (tighten/justify per §8 of
   the brief before the final test). **Until this passes, "MAGSAC++-equivalent" is unproven.**
2. **GPU float64 then float32 parity** — rerun the synthetic + oracle checks on CUDA; float32 fast mode
   targets `atol=1e-5 rtol=1e-4`, projections ≤1e-2 px.
3. **cv2 applied-baseline compare** — `parity synthetic` with opencv installed (absent on this box).
4. **Real-data quality** — both cities (kup / kramatorsc), all active pyramid levels, true & false
   shortlisted candidates; distR@250m top1/top5, median/p90/p95 geodesic error vs the cv2 baseline on
   the SAME cached correspondences and ranker (`cache` → `evaluate`). Report per city/level/N/overlap.
5. **Benchmark on the actual GPU** (confirm it's a T4 first): float64 vs float32 cost, batched SVD cost,
   estimator-only / +transfer / full Stage-B; median/p90/p95 ms/pair, pairs/s, peak VRAM.
6. **Opt-in integration** — add a `torch_magsacpp` choice next to `cpu_magsac`/`gpu_kornia`/`gpu_batch`
   in `patch_rerank` only after 1–4; keep the production default unchanged; make any fallback explicit.

## 7. Known limitations / not verified

- Loss-form equivalence to danini's stored table is derivation-based (§2.2); bit-exact match is a
  Pending oracle item.
- Weighted refit uses homogeneous SVD, not danini's h33≡1 QR (§2.4) — a measured divergence.
- Sampler is uniform, not P-NAPSAC (§2.6) — end-to-end runs differ by design; use recorded schedules
  for parity.
- Adaptive early-stopping is intentionally not used (fixed hypothesis budget) so replay/parity is
  deterministic; an adaptive production mode is future work (brief §8).
- No cross-pair batching in the core loop yet (per-pair, hypotheses chunked). Cross-pair batching is
  the throughput optimization (cf. `patch_rerank/gpu_verify.ransac_homography_batch`).
