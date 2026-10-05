"""Estimator tests: known-H recovery, outliers, noise, edge cases, replay + chunk invariance."""
from __future__ import annotations

import numpy as np
import torch

from magsacpp_torch import MagsacppConfig, estimate_homography_magsacpp
from magsacpp_torch.tests.conftest import apply_h, gen, make_correspondences

DT = torch.float64


def _cfg(**kw):
    # sigma_max is matched to the synthetic coordinate/noise scale (coords ~100 px, sub-px noise):
    # tau = k*sigma_max ~ 3.6 px. The library DEFAULT is danini's raw 10.0, which must be tuned to
    # the problem -- see README "Choosing sigma_max". Oversized sigma_max lets sigma-consensus++
    # weight far outliers and drifts the IRLS fit.
    base = dict(max_hypotheses=400, irls_iters=3, sigma_max=1.0)
    base.update(kw)
    return MagsacppConfig(**base)


def test_recovers_clean_homography_all_inliers():
    rng = np.random.default_rng(10)
    p1, p2, H, inl = make_correspondences(rng, n_total=80, outlier_frac=0.0, noise=0.0)
    res = estimate_homography_magsacpp(p1, p2, config=_cfg(), generator=gen(0))
    assert res.success and res.status == "ok"
    assert res.inlier_count >= 78
    assert torch.allclose(apply_h(res.H, p1), p2, atol=1e-4)


def test_recovers_under_moderate_outliers():
    for seed, frac in [(11, 0.2), (12, 0.5)]:
        rng = np.random.default_rng(seed)
        p1, p2, H, inl = make_correspondences(rng, n_total=200, outlier_frac=frac, noise=0.0)
        res = estimate_homography_magsacpp(p1, p2, config=_cfg(max_hypotheses=2000),
                                           generator=gen(seed))
        assert res.success, f"frac={frac}"
        # recovered inliers should match the true inliers closely
        recovered = res.inlier_mask
        true_inl = inl
        agree = (recovered == true_inl).to(DT).mean()
        assert float(agree) > 0.95, f"frac={frac} agree={float(agree)}"


def test_high_outlier_ratio_with_big_budget():
    # 80% outliers: success is budget-dependent (clean-sample prob ~0.2^4). With a large budget we
    # expect recovery; assert the *conditional* quality if it succeeds + that it does succeed here.
    rng = np.random.default_rng(13)
    p1, p2, H, inl = make_correspondences(rng, n_total=250, outlier_frac=0.8, noise=0.0)
    res = estimate_homography_magsacpp(p1, p2, config=_cfg(max_hypotheses=8000), generator=gen(99))
    assert res.success
    # the 50 true inliers should be mostly recovered
    overlap = (res.inlier_mask & inl).sum().item() / int(inl.sum())
    assert overlap > 0.8


def test_small_noise_keeps_most_inliers():
    rng = np.random.default_rng(14)
    p1, p2, H, inl = make_correspondences(rng, n_total=150, outlier_frac=0.3, noise=0.5)
    res = estimate_homography_magsacpp(p1, p2, config=_cfg(max_hypotheses=3000, inlier_threshold=2.0),
                                       generator=gen(14))
    assert res.success
    n_true = int(inl.sum())
    assert res.inlier_count >= 0.7 * n_true


def test_edge_n_lt_4():
    p1 = torch.tensor([[0.0, 0], [1, 0], [0, 1]], dtype=DT)
    res = estimate_homography_magsacpp(p1, p1.clone(), config=_cfg(), generator=gen(0))
    assert (not res.success) and res.status == "n_lt_4"
    assert res.inlier_count == 0 and torch.isnan(res.H).all()


def test_edge_all_invalid():
    rng = np.random.default_rng(15)
    p1, p2, H, inl = make_correspondences(rng, n_total=20, outlier_frac=0.0)
    valid = torch.zeros(20, dtype=torch.bool)
    res = estimate_homography_magsacpp(p1, p2, valid, config=_cfg(), generator=gen(0))
    assert (not res.success) and res.status == "all_invalid"


def test_edge_degenerate_collinear():
    # all source points on a line -> no valid homography
    x = torch.linspace(0, 100, 30, dtype=DT)
    p1 = torch.stack([x, 2 * x + 1], 1)
    p2 = torch.stack([x, -x], 1)
    res = estimate_homography_magsacpp(p1, p2, config=_cfg(), generator=gen(0))
    assert not res.success  # degenerate or no_valid_hypothesis, never a bogus success
    assert res.status in ("degenerate", "no_valid_hypothesis")


def test_replay_schedule_is_deterministic():
    rng = np.random.default_rng(16)
    p1, p2, H, inl = make_correspondences(rng, n_total=60, outlier_frac=0.3, noise=0.0)
    S = 200
    sched = torch.stack([torch.randperm(60)[:4] for _ in range(S)])  # may hit outliers; that's fine
    r1 = estimate_homography_magsacpp(p1, p2, config=_cfg(), hypothesis_indices=sched)
    r2 = estimate_homography_magsacpp(p1, p2, config=_cfg(), hypothesis_indices=sched)
    assert torch.equal(r1.H, r2.H)
    assert r1.total_loss == r2.total_loss


def test_chunk_size_invariance():
    rng = np.random.default_rng(17)
    p1, p2, H, inl = make_correspondences(rng, n_total=120, outlier_frac=0.4, noise=0.0)
    S = 300
    sched = torch.stack([torch.from_numpy(rng.choice(120, 4, replace=False)) for _ in range(S)])
    big = estimate_homography_magsacpp(p1, p2, config=_cfg(points_budget=10 ** 9),
                                       hypothesis_indices=sched)
    tiny = estimate_homography_magsacpp(p1, p2, config=_cfg(points_budget=120 * 3),  # chunk=3 hyps
                                        hypothesis_indices=sched)
    assert torch.allclose(big.H, tiny.H, atol=1e-10)
    assert abs(big.total_loss - tiny.total_loss) < 1e-9


def test_padding_does_not_change_result():
    rng = np.random.default_rng(18)
    p1, p2, H, inl = make_correspondences(rng, n_total=60, outlier_frac=0.3, noise=0.0)
    S = 200
    sched = torch.stack([torch.from_numpy(rng.choice(60, 4, replace=False)) for _ in range(S)])
    base = estimate_homography_magsacpp(p1, p2, config=_cfg(), hypothesis_indices=sched)
    # append 40 padded (invalid) garbage points; schedule still indexes the first 60
    pad = torch.tensor(rng.uniform(-1e3, 1e3, size=(40, 2)), dtype=DT)
    p1p = torch.cat([p1, pad]); p2p = torch.cat([p2, pad])
    valid = torch.cat([torch.ones(60, dtype=torch.bool), torch.zeros(40, dtype=torch.bool)])
    padded = estimate_homography_magsacpp(p1p, p2p, valid, config=_cfg(), hypothesis_indices=sched)
    assert torch.allclose(base.H, padded.H, atol=1e-10)
    assert abs(base.total_loss - padded.total_loss) < 1e-9
    assert int(padded.inlier_mask[60:].sum()) == 0   # padded points never inliers


def test_batched_api_returns_list():
    rng = np.random.default_rng(19)
    pairs = [make_correspondences(rng, 50, 0.2, 0.0) for _ in range(3)]
    p1 = torch.stack([p[0] for p in pairs])
    p2 = torch.stack([p[1] for p in pairs])
    out = estimate_homography_magsacpp(p1, p2, config=_cfg(max_hypotheses=1500), generator=gen(1))
    assert isinstance(out, list) and len(out) == 3
    assert all(r.success for r in out)
