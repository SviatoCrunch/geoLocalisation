"""Solver tests: Hartley normalization, minimal nullspace correctness, weighted fit, residual edges."""
from __future__ import annotations

import numpy as np
import torch

from magsacpp_torch.solver import (forward_sq_residual, normalize_points, solve_minimal_h,
                                    solve_weighted_h)
from magsacpp_torch.tests.conftest import apply_h, make_correspondences, random_homography

DT = torch.float64


def test_normalization_centroid_and_scale():
    rng = np.random.default_rng(0)
    pts = torch.tensor(rng.uniform(-50, 200, size=(1, 30, 2)), dtype=DT)
    valid = torch.ones(1, 30, dtype=torch.bool)
    pn, T = normalize_points(pts, valid)
    assert torch.allclose(pn.mean(1), torch.zeros(1, 2, dtype=DT), atol=1e-9)
    mean_dist = pn.norm(dim=-1).mean(1)
    assert torch.allclose(mean_dist, torch.full((1,), np.sqrt(2.0), dtype=DT), atol=1e-9)


def test_minimal_solver_recovers_known_H_exactly():
    rng = np.random.default_rng(1)
    for _ in range(20):
        H = random_homography(rng)
        p1 = torch.tensor(rng.uniform(0, 100, size=(4, 2)), dtype=DT)
        p2 = apply_h(H, p1)
        Hest, ok = solve_minimal_h(p1.unsqueeze(0), p2.unsqueeze(0))
        assert bool(ok[0])
        # compare via projection (removes scale/sign ambiguity)
        assert torch.allclose(apply_h(Hest[0], p1), p2, atol=1e-7)


def test_minimal_solver_needs_full_matrices_nullspace():
    # explicit guard that we take the TRUE 9th right singular vector: a reduced SVD would give a
    # wrong null vector and the reprojection would NOT be ~0.
    rng = np.random.default_rng(7)
    H = random_homography(rng)
    p1 = torch.tensor(rng.uniform(0, 100, size=(4, 2)), dtype=DT)
    p2 = apply_h(H, p1)
    Hest, ok = solve_minimal_h(p1.unsqueeze(0), p2.unsqueeze(0))
    assert float((apply_h(Hest[0], p1) - p2).norm()) < 1e-6


def test_residual_zero_for_exact_model():
    rng = np.random.default_rng(2)
    H = random_homography(rng)
    p1 = torch.tensor(rng.uniform(0, 100, size=(50, 2)), dtype=DT)
    p2 = apply_h(H, p1)
    sq = forward_sq_residual(H, p1, p2)
    assert torch.all(sq < 1e-12)


def test_residual_negative_denominator_is_valid():
    # a homography whose denominator is negative for some points must still give a finite residual
    # (we must NOT clamp negative denominators to +eps).
    H = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [-0.02, 0, 1.0]], dtype=DT)  # w<0 when x>50
    p1 = torch.tensor([[80.0, 10.0]], dtype=DT)                              # w = 1 - 0.02*80 = -0.6
    p2 = apply_h(H, p1)
    sq = forward_sq_residual(H, p1, p2)
    assert torch.isfinite(sq).all()
    assert float(sq[0]) < 1e-12


def test_residual_near_infinity_is_inf():
    H = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [-0.02, 0, 1.0]], dtype=DT)
    p1 = torch.tensor([[50.0, 0.0]], dtype=DT)        # w = 1 - 0.02*50 = 0 -> near infinity
    p2 = torch.tensor([[0.0, 0.0]], dtype=DT)
    sq = forward_sq_residual(H, p1, p2, min_abs_denominator=1e-9)
    assert torch.isinf(sq).all()


def test_weighted_solver_recovers_H_with_outlier_downweighted():
    rng = np.random.default_rng(3)
    p1, p2, H, inl = make_correspondences(rng, n_total=40, outlier_frac=0.25, noise=0.0)
    w = inl.to(DT)                                     # oracle weights: 1 for inliers, 0 for outliers
    Hest, ok = solve_weighted_h(p1.unsqueeze(0), p2.unsqueeze(0), w.unsqueeze(0),
                                torch.ones(1, 40, dtype=torch.bool))
    assert bool(ok[0])
    assert torch.allclose(apply_h(Hest[0], p1[inl]), p2[inl], atol=1e-6)
