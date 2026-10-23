"""Configuration + pinned reference constants for the Torch MAGSAC++ homography core.

STATUS: the scoring / sigma-consensus++ / IRLS math implemented here reproduces the
*paper + danini/magsac* closed form (see constants below and ``REPORT.md``), but it has
NOT yet been checked bit-for-bit against the C++ oracle. Per the task brief, do not call a
run "MAGSAC++-equivalent" until :mod:`oracle` parity passes on the server. Until then this is
"a Torch marginalized robust homography estimator following the MAGSAC++ formulas".

Math-core reference (designated): ``danini/magsac`` @ ``d259f8b3a8925025e45667241fb68629b07603bb``
(BSD-3-Clause, (c) Czech Technical University), whose estimators/solvers/gamma LUT live in the
submodule ``danini/graph-cut-ransac`` @ ``9fa075dc76d0e632ab3d73297c76ac3d4a29decd``. Paper:
Barath et al., "MAGSAC++, a fast, reliable and accurate robust estimator", CVPR 2020,
arXiv:1912.05909. These constants are for the HOMOGRAPHY estimator (n=4).

Separate APPLIED baseline (NOT a math reference, different constants): OpenCV USAC
``cv2.USAC_MAGSAC`` @ opencv ``4.x`` ``62587ae9976b28cfa61ad940d0e7f610b8742ee4`` (Apache-2.0).
OpenCV uses DoF=2, k=3.04, C=0.5 for homography and a 500-entry interpolated LUT -- do NOT mix
its constants with the danini ones (see ``REPORT.md`` divergence table).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

# --- danini/magsac homography constants (math-core reference) ---------------------------------
# DoF of the residual distribution used in the marginalization. danini sets n=4 for homography
# ("(x1,y1,x2,y2) -> 4", graph-cut-ransac estimators.h:43). OpenCV uses 2; we follow danini.
DANINI_DOF = 4
# chi quantile at confidence 0.99 for the cutoff tau = k * sigma_max (getSigmaQuantile(),
# estimators.h:38). danini=3.64, OpenCV=3.04.
DANINI_QUANTILE_K = 3.64
# normalizer C(n) = 1 / (2^(n/2) * Gamma(n/2)); for n=4 this is 1/(4*Gamma(2)) = 0.25
# (getC(), estimators.h:48). OpenCV hardcodes 0.5.
DANINI_C = 0.25
# danini maximum_threshold default (magsac.h:34). This is sigma_max in the refit path
# (sigmaConsensusPlusPlus uses sigma_max = maximum_threshold directly); the SCORER path uses
# sigma_max = maximum_threshold / k. That internal inconsistency is real in danini -- we expose
# ``sigma_max`` explicitly instead of deriving it, so there is no hidden inconsistency here.
DANINI_MAXIMUM_THRESHOLD = 10.0


@dataclass(frozen=True)
class MagsacppConfig:
    """Immutable configuration for :func:`magsacpp_torch.estimator.estimate_homography_magsacpp`.

    The gamma-math constants default to the danini/magsac homography values. ``sigma_max`` and
    ``inlier_threshold`` are kept SEPARATE on purpose (task brief sec.5.9): ``sigma_max`` drives the
    marginalized score/weights, while ``inlier_threshold`` is the explicit (hard) reprojection
    threshold used ONLY to report an inlier mask/count -- it never enters the score.
    """

    # --- robust-score / weight math (danini homography defaults) ---
    dof: int = DANINI_DOF
    quantile_k: float = DANINI_QUANTILE_K
    normalizer_c: float = DANINI_C
    sigma_max: float = DANINI_MAXIMUM_THRESHOLD  # noise-scale upper bound for marginalization (px)

    # --- explicit inlier mask threshold (reporting only, NOT used by the score) ---
    # Matches the cv2 ``ransacReprojThreshold`` used by the production reranker (matcher.py = 2.0 px,
    # error in query patch-grid units). Forward reprojection error <= this => inlier.
    inlier_threshold: float = 2.0

    # --- RANSAC / sampling loop ---
    min_sample: int = 4           # homography minimal sample
    max_hypotheses: int = 1000    # hypothesis budget per pair
    confidence: float = 0.99      # adaptive-stop confidence (relaxed termination)
    # sampler: uniform sampling WITHOUT replacement inside the minimal sample. This DIFFERS from the
    # paper's P-NAPSAC sampler -- flagged explicitly (task brief sec.5). Parity runs must use a
    # recorded ``hypothesis_indices`` schedule, not a shared seed (sec.8).

    # --- local optimization / IRLS (sigma-consensus++) ---
    irls_iters: int = 1           # danini number_of_irwls_iters default = 1 (magsac.h:39)
    irls_require_improvement: bool = True   # accept a refit only if total loss did not increase

    # --- degeneracy / numerical guards ---
    # points with |homogeneous denominator| below this are treated as projecting near infinity:
    # their residual is +inf (outlier), weight 0, never an inlier. We do NOT clamp negative
    # denominators to +eps (sec.5): a negative w is a valid projection.
    min_abs_denominator: float = 1e-12
    min_singular_ratio: float = 1e-7   # near-singular minimal/refit model rejection (cond proxy)
    # minimal 4-pt solver: "svd" (full-matrices nullspace) or "closed_form" (adjugate, no SVD -- the
    # GPU-fast path; avoids thousands of tiny per-hypothesis SVDs). Math-equivalent up to scale/sign.
    minimal_solver: str = "svd"
    collinearity_eps: float = 1e-8     # minimal-sample triplet-area degeneracy guard

    # --- precision / device / optimization ---
    dtype: torch.dtype = torch.float64   # float64 correctness mode; float32 is the fast mode
    use_lut: bool = False                # analytic torch.special path by default; LUT = optimization
    # memory budget for batched work tensors (elements of the (pairs*hyps, points) grids). Results
    # must not change with chunk size beyond the declared tolerance.
    points_budget: int = 24_000_000

    # provenance string carried into result traces / reports
    reference: str = field(default=(
        "danini/magsac@d259f8b (graph-cut-ransac@9fa075d), paper arXiv:1912.05909; "
        "homography n=4, k=3.64, C=0.25"
    ))

    def sigma_max_t(self, device=None) -> torch.Tensor:
        return torch.as_tensor(self.sigma_max, dtype=self.dtype, device=device)
