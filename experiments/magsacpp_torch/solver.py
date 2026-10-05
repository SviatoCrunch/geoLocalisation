"""Homography geometry in pure Torch: normalization, minimal/weighted DLT, forward residual.

Conventions (match the production reranker, patch_rerank/matcher.py + gpu_verify.py):
  * ``points1`` = source (map-crop) coords, ``points2`` = dest (query) coords.
  * The homography maps source -> dest (``H @ [x1,y1,1]^T ~ [x2,y2,1]^T``), i.e. map -> query.
  * The residual is the FORWARD reprojection error in dest (query patch-grid) units, SQUARED.

All heavy ops are batched Torch tensor ops (no NumPy/SciPy/OpenCV). float64 is the correctness
dtype. The minimal solver takes the TRUE 9-vector right nullspace via full-matrix SVD -- a reduced
SVD of the 8x9 design matrix omits the 9th right singular vector and is therefore WRONG here.
"""
from __future__ import annotations

from typing import Tuple

import torch


def normalize_points(pts: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hartley isotropic normalization per batch item, over VALID points only.

    ``pts`` (B, N, 2), ``valid`` (B, N) bool. Returns (pts_n (B,N,2), T (B,3,3)) with
    ``pts_n = (T @ [x,y,1]^T)`` and centroid at origin, mean valid-point distance sqrt(2).
    Padded points are transformed by the same T but ignored in the statistics.
    """
    dt = pts.dtype
    vf = valid.to(dt)                                   # (B, N)
    cnt = vf.sum(1).clamp_min(1.0)                      # (B,)
    centroid = (pts * vf.unsqueeze(-1)).sum(1) / cnt.unsqueeze(-1)   # (B, 2)
    centered = pts - centroid.unsqueeze(1)              # (B, N, 2)
    dist = torch.sqrt((centered ** 2).sum(-1) + 1e-300) * vf        # (B, N)
    mean_dist = dist.sum(1) / cnt                        # (B,)
    scale = torch.sqrt(torch.tensor(2.0, dtype=dt, device=pts.device)) / mean_dist.clamp_min(1e-300)
    B = pts.shape[0]
    T = torch.zeros(B, 3, 3, dtype=dt, device=pts.device)
    T[:, 0, 0] = scale
    T[:, 1, 1] = scale
    T[:, 0, 2] = -scale * centroid[:, 0]
    T[:, 1, 2] = -scale * centroid[:, 1]
    T[:, 2, 2] = 1.0
    ones = torch.ones(B, pts.shape[1], 1, dtype=dt, device=pts.device)
    homog = torch.cat([pts, ones], dim=-1)               # (B, N, 3)
    pts_n = torch.einsum("bij,bnj->bni", T, homog)[..., :2]
    return pts_n, T


def _dlt_rows(p1: torch.Tensor, p2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Build the weighted DLT design matrix A (B, 2M, 9) for the homography p1 -> p2.

    Each correspondence contributes two rows; both rows of correspondence i are scaled by weight
    ``w[...,i]`` so that solving ``min ||A h||`` is the weighted algebraic least squares. ``w`` can be
    a 0/1 validity mask (minimal/unweighted fit) or the sigma-consensus++ weights (IRLS refit).
    """
    B, M, _ = p1.shape
    x1, y1 = p1[..., 0], p1[..., 1]                      # (B, M)
    x2, y2 = p2[..., 0], p2[..., 1]
    z = torch.zeros_like(x1)
    o = torch.ones_like(x1)
    # row A: [-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2]
    rA = torch.stack([-x1, -y1, -o, z, z, z, x2 * x1, x2 * y1, x2], dim=-1)   # (B, M, 9)
    # row B: [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
    rB = torch.stack([z, z, z, -x1, -y1, -o, y2 * x1, y2 * y1, y2], dim=-1)
    ws = w.unsqueeze(-1)                                   # (B, M, 1)
    rA = rA * ws
    rB = rB * ws
    A = torch.cat([rA, rB], dim=1)                        # (B, 2M, 9)
    return A


def _nullspace_h(A: torch.Tensor, min_singular_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right nullspace (smallest right singular vector) of A (B, R, 9) -> (H (B,3,3), ok (B,)).

    Uses ``full_matrices=True`` so Vh is (B, 9, 9) and ``Vh[:, -1]`` is the genuine 9th right
    singular vector (the null vector). ``ok`` flags well-posed systems: enough singular values and a
    clear gap (s[-2] >> s[-1]) so the solution is not a near-degenerate ambiguity.
    """
    B = A.shape[0]
    # guard all-zero / non-finite design matrices before SVD (degenerate samples)
    finite = torch.isfinite(A.reshape(B, -1)).all(1)
    A_safe = torch.where(finite.view(B, 1, 1), A, torch.eye(A.shape[1], 9, dtype=A.dtype,
                                                            device=A.device).unsqueeze(0).expand_as(A))
    U, S, Vh = torch.linalg.svd(A_safe, full_matrices=True)     # Vh: (B, 9, 9)
    h = Vh[:, -1, :]                                            # (B, 9) smallest-sv right vector
    H = h.reshape(B, 3, 3)
    # conditioning: require the second-smallest singular value to dominate the smallest, and the
    # model to be finite and non-singular.
    s_small = S[:, -1]
    s_next = S[:, -2]
    well = finite & (s_next > min_singular_ratio * S[:, 0].clamp_min(1e-300)) \
        & (s_next > s_small)
    detH = torch.linalg.det(H)
    well = well & torch.isfinite(detH) & (detH.abs() > 1e-12)
    return H, well


def solve_minimal_h(p1: torch.Tensor, p2: torch.Tensor, min_singular_ratio: float = 1e-7,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched 4-point (or N>=4) homography DLT with Hartley normalization. p1,p2 (B, M, 2).

    Returns (H (B,3,3) mapping p1->p2 in the ORIGINAL frame, ok (B,) validity). All M points are
    used with unit weight; for the minimal sampler M==4.
    """
    B, M, _ = p1.shape
    valid = torch.ones(B, M, dtype=torch.bool, device=p1.device)
    p1n, T1 = normalize_points(p1, valid)
    p2n, T2 = normalize_points(p2, valid)
    A = _dlt_rows(p1n, p2n, valid.to(p1.dtype))
    Hn, ok = _nullspace_h(A, min_singular_ratio)
    # denormalize: H = T2^{-1} @ Hn @ T1
    T2inv = torch.linalg.inv(T2)
    H = T2inv @ Hn @ T1
    # scale so that H[2,2] = 1 when it is safe (|H22| not ~0); otherwise leave unnormalized and let
    # projection handle it via abs(w). Do NOT blindly divide by a ~0 h33.
    h22 = H[:, 2, 2]
    safe = h22.abs() > 1e-12
    scale = torch.where(safe, 1.0 / torch.where(safe, h22, torch.ones_like(h22)),
                        torch.ones_like(h22))
    H = H * scale.view(B, 1, 1)
    ok = ok & torch.isfinite(H.reshape(B, -1)).all(1)
    return H, ok


def _adj3(M: torch.Tensor) -> torch.Tensor:
    """Batched adjugate (classical adjoint) of 3x3 matrices: M @ adj(M) = det(M) I. Pure arithmetic."""
    m00, m01, m02 = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    m10, m11, m12 = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    m20, m21, m22 = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    r0 = torch.stack([m11 * m22 - m12 * m21, m02 * m21 - m01 * m22, m01 * m12 - m02 * m11], -1)
    r1 = torch.stack([m12 * m20 - m10 * m22, m00 * m22 - m02 * m20, m02 * m10 - m00 * m12], -1)
    r2 = torch.stack([m10 * m21 - m11 * m20, m01 * m20 - m00 * m21, m00 * m11 - m01 * m10], -1)
    return torch.stack([r0, r1, r2], -2)


def _det3(M: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    m10, m11, m12 = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    m20, m21, m22 = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    return (m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20))


def _inv_sim3(T: torch.Tensor) -> torch.Tensor:
    """Analytic inverse of a Hartley normalization matrix [[a,0,b],[0,a,c],[0,0,1]] (no linalg kernel)."""
    a = T[:, 0, 0]
    b = T[:, 0, 2]
    c = T[:, 1, 2]
    inv = torch.zeros_like(T)
    ia = 1.0 / a
    inv[:, 0, 0] = ia
    inv[:, 1, 1] = ia
    inv[:, 0, 2] = -b * ia
    inv[:, 1, 2] = -c * ia
    inv[:, 2, 2] = 1.0
    return inv


def _closed_form_4pt(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Closed-form 4-point homography p1->p2 (B,4,2)->(B,3,3), up to scale, NO SVD / NO division.

    Maps the canonical basis to each point set via A = [lambda_i x_i] with lambda = adj([x1 x2 x3]) x4,
    then H = B' @ adj(A') (adjugate replaces inverse). Pure tensor arithmetic -> batches perfectly on
    GPU, avoiding the tiny-matrix SVD that dominates the per-hypothesis cost.
    (Refs: Radially-Distorted-Homographies-Revisited adjugate 4-pt; SKS/ACA decomposition.)
    """
    B = p1.shape[0]
    one = torch.ones(B, 4, 1, dtype=p1.dtype, device=p1.device)
    X = torch.cat([p1, one], -1)                      # (B,4,3) homogeneous
    Y = torch.cat([p2, one], -1)

    def basis(P):
        M = P[:, :3, :].transpose(1, 2)               # (B,3,3): columns = first 3 points
        x4 = P[:, 3, :]                                # (B,3)
        lam = torch.einsum("bij,bj->bi", _adj3(M), x4)   # adj(M) x4  (no 1/det: overall scale)
        return M * lam.unsqueeze(1)                    # scale column j by lam_j

    return basis(Y) @ _adj3(basis(X))                 # B' @ adj(A') ~ B' A'^{-1}


def solve_minimal_h_closed_form(p1: torch.Tensor, p2: torch.Tensor,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form minimal 4-point homography (drop-in for :func:`solve_minimal_h`). p1,p2 (B,4,2).

    Hartley-normalizes first (float32 conditioning), solves in closed form, denormalizes. Returns
    (H (B,3,3) p1->p2, ok (B,)). Same contract/scale convention as the SVD solver; matches it up to
    the usual scale/sign ambiguity on well-posed samples (tests/test_solver_closed_form.py)."""
    B, M, _ = p1.shape
    valid = torch.ones(B, M, dtype=torch.bool, device=p1.device)
    p1n, T1 = normalize_points(p1, valid)
    p2n, T2 = normalize_points(p2, valid)
    Hn = _closed_form_4pt(p1n, p2n)
    H = _inv_sim3(T2) @ Hn @ T1
    h22 = H[:, 2, 2]
    safe = h22.abs() > 1e-12
    scale = torch.where(safe, 1.0 / torch.where(safe, h22, torch.ones_like(h22)),
                        torch.ones_like(h22))
    H = H * scale.view(B, 1, 1)
    detH = _det3(H)
    ok = torch.isfinite(H.reshape(B, -1)).all(1) & torch.isfinite(detH) & (detH.abs() > 1e-12)
    return H, ok


def solve_minimal(p1: torch.Tensor, p2: torch.Tensor, method: str = "svd",
                  min_singular_ratio: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal-solver dispatcher: ``svd`` (full-matrices nullspace) or ``closed_form`` (adjugate 4-pt,
    no SVD -- the GPU-fast path). Both return (H p1->p2, ok)."""
    if method == "closed_form":
        return solve_minimal_h_closed_form(p1, p2)
    return solve_minimal_h(p1, p2, min_singular_ratio)


def solve_weighted_h(p1: torch.Tensor, p2: torch.Tensor, weights: torch.Tensor,
                     valid: torch.Tensor, min_singular_ratio: float = 1e-7,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted non-minimal homography DLT (the IRLS refit). p1,p2 (B,N,2), weights/valid (B,N).

    Minimizes the WEIGHTED ALGEBRAIC error sum_i w_i^2 ||A_i h||^2 (homogeneous DLT via SVD), like
    danini's weighted least-squares refit -- but danini uses an inhomogeneous h33=1 system solved by
    column-pivoted QR (solver_homography_four_point.h::estimateNonMinimalModel). We use the
    homogeneous SVD form: it avoids the h33=1 limitation (degrades when the true h33~0) and is
    numerically robust; the difference from danini's QR refit is documented in REPORT.md and is a
    refinement-path divergence to be measured against the oracle, not a math-core divergence.
    """
    dt = p1.dtype
    vf = valid.to(dt)
    w = (weights.to(dt) * vf)                            # zero-weight padded/invalid points
    p1n, T1 = normalize_points(p1, valid)
    p2n, T2 = normalize_points(p2, valid)
    A = _dlt_rows(p1n, p2n, w)
    Hn, ok = _nullspace_h(A, min_singular_ratio)
    T2inv = torch.linalg.inv(T2)
    H = T2inv @ Hn @ T1
    h22 = H[:, 2, 2]
    safe = h22.abs() > 1e-12
    scale = torch.where(safe, 1.0 / torch.where(safe, h22, torch.ones_like(h22)),
                        torch.ones_like(h22))
    H = H * scale.view(H.shape[0], 1, 1)
    enough = (w > 0).sum(1) >= 4
    ok = ok & enough & torch.isfinite(H.reshape(H.shape[0], -1)).all(1)
    return H, ok


def forward_sq_residual(H: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
                        min_abs_denominator: float = 1e-12) -> torch.Tensor:
    """Squared forward reprojection residual ||pi(H p1) - p2||^2 in dest units.

    ``H`` (..., 3, 3), ``p1``/``p2`` (..., N, 2) broadcasting over the leading dims. Points whose
    homogeneous denominator ``w = h20 x + h21 y + h22`` is near zero (|w| < ``min_abs_denominator``)
    project near infinity -> residual +inf (so they count as outliers, never inliers, and never get
    silently clamped). A NEGATIVE w is a valid projection and is kept as-is. Non-finite results
    become +inf, never NaN.
    """
    x = p1[..., 0]
    y = p1[..., 1]
    h = H
    # unfold to broadcast H (..., 3, 3) against points (..., N)
    h00 = h[..., 0, 0].unsqueeze(-1); h01 = h[..., 0, 1].unsqueeze(-1); h02 = h[..., 0, 2].unsqueeze(-1)
    h10 = h[..., 1, 0].unsqueeze(-1); h11 = h[..., 1, 1].unsqueeze(-1); h12 = h[..., 1, 2].unsqueeze(-1)
    h20 = h[..., 2, 0].unsqueeze(-1); h21 = h[..., 2, 1].unsqueeze(-1); h22 = h[..., 2, 2].unsqueeze(-1)
    wdenom = h20 * x + h21 * y + h22
    near_inf = wdenom.abs() < min_abs_denominator
    wsafe = torch.where(near_inf, torch.ones_like(wdenom), wdenom)
    px = (h00 * x + h01 * y + h02) / wsafe
    py = (h10 * x + h11 * y + h12) / wsafe
    sq = (px - p2[..., 0]) ** 2 + (py - p2[..., 1]) ** 2
    big = torch.full_like(sq, float("inf"))
    sq = torch.where(near_inf | ~torch.isfinite(sq), big, sq)
    return sq
