"""Closed-form adjugate 4-point solver == SVD nullspace solver (up to scale), + end-to-end parity."""
from __future__ import annotations

import numpy as np
import torch

from magsacpp_torch import MagsacppConfig, estimate_homography_magsacpp
from magsacpp_torch.solver import solve_minimal_h, solve_minimal_h_closed_form
from magsacpp_torch.tests.conftest import apply_h, make_correspondences, random_homography

DT = torch.float64


def test_closed_form_recovers_known_H_exactly():
    rng = np.random.default_rng(40)
    for _ in range(30):
        H = random_homography(rng)
        p1 = torch.tensor(rng.uniform(0, 100, size=(4, 2)), dtype=DT)
        p2 = apply_h(H, p1)
        Hest, ok = solve_minimal_h_closed_form(p1.unsqueeze(0), p2.unsqueeze(0))
        assert bool(ok[0])
        assert torch.allclose(apply_h(Hest[0], p1), p2, atol=1e-7)


def test_closed_form_matches_svd_up_to_scale():
    rng = np.random.default_rng(41)
    for _ in range(30):
        H = random_homography(rng)
        p1 = torch.tensor(rng.uniform(0, 100, size=(4, 2)), dtype=DT)
        p2 = apply_h(H, p1)
        Hc, okc = solve_minimal_h_closed_form(p1.unsqueeze(0), p2.unsqueeze(0))
        Hs, oks = solve_minimal_h(p1.unsqueeze(0), p2.unsqueeze(0))
        assert bool(okc[0]) and bool(oks[0])
        # both H23-normalized already; compare by projecting control points (scale/sign invariant)
        assert float((apply_h(Hc[0], p1) - apply_h(Hs[0], p1)).norm()) < 1e-7


def test_closed_form_detects_collinear_degenerate():
    # 3 collinear source points -> degenerate; ok must be False (no bogus H)
    p1 = torch.tensor([[[0., 0.], [1., 1.], [2., 2.], [5., 0.]]], dtype=DT)  # first 3 on y=x
    p2 = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 5.]]], dtype=DT)
    H, ok = solve_minimal_h_closed_form(p1, p2)
    assert not bool(ok[0])


def test_estimator_closed_form_recovers_like_svd():
    rng = np.random.default_rng(42)
    p1, p2, H, inl = make_correspondences(rng, n_total=120, outlier_frac=0.3, noise=0.0)
    cfg_svd = MagsacppConfig(sigma_max=1.0, max_hypotheses=800, irls_iters=3, minimal_solver="svd")
    cfg_cf = MagsacppConfig(sigma_max=1.0, max_hypotheses=800, irls_iters=3, minimal_solver="closed_form")
    sched = torch.stack([torch.from_numpy(rng.choice(120, 4, replace=False)) for _ in range(300)])
    r_svd = estimate_homography_magsacpp(p1, p2, config=cfg_svd, hypothesis_indices=sched)
    r_cf = estimate_homography_magsacpp(p1, p2, config=cfg_cf, hypothesis_indices=sched)
    assert r_svd.success and r_cf.success
    # same schedule + same math (only the minimal-solve numerics differ) -> near-identical result
    assert float((apply_h(r_svd.H, p1) - apply_h(r_cf.H, p1)).norm(dim=-1).max()) < 1e-5
    assert abs(r_svd.inlier_count - r_cf.inlier_count) <= 1
