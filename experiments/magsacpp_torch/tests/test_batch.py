"""Cross-pair batch path == per-pair loop on a shared schedule (same math), + edge cases."""
from __future__ import annotations

import numpy as np
import torch

from magsacpp_torch import MagsacppConfig, estimate_homography_magsacpp
from magsacpp_torch.batch import estimate_homography_magsacpp_batch
from magsacpp_torch.tests.conftest import apply_h, make_correspondences

DT = torch.float64


def _cfg(**kw):
    base = dict(sigma_max=1.0, max_hypotheses=200, irls_iters=3, dtype=DT)
    base.update(kw)
    return MagsacppConfig(**base)


def _sched(ns, S, seed=0):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(np.stack([rng.choice(n, 4, replace=False) for _ in range(S)])) for n in ns]


def test_batch_matches_loop_on_shared_schedule():
    rng = np.random.default_rng(30)
    P, N, S = 6, 90, 150
    pairs = [make_correspondences(rng, N, 0.3, 0.0) for _ in range(P)]
    p1 = torch.stack([p[0] for p in pairs]); p2 = torch.stack([p[1] for p in pairs])
    valid = torch.ones(P, N, dtype=torch.bool)
    scheds = _sched([N] * P, S)
    sched_t = torch.stack(scheds)                                    # (P,S,4)
    cfg = _cfg()

    batch = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg, hypothesis_indices=sched_t)
    for i in range(P):
        loop = estimate_homography_magsacpp(p1[i], p2[i], config=cfg, hypothesis_indices=scheds[i])
        assert batch[i].status == loop.status == "ok"
        # projection-space geometry (scale/sign invariant) must match tightly
        pr_b = apply_h(batch[i].H, p1[i]); pr_l = apply_h(loop.H, p1[i])
        assert float((pr_b - pr_l).norm(dim=-1).max()) < 1e-8, i
        assert abs(batch[i].total_loss - loop.total_loss) < 1e-9, i
        assert batch[i].inlier_count == loop.inlier_count, i
        assert torch.equal(batch[i].inlier_mask, loop.inlier_mask), i


def test_batch_ragged_padding_matches_unpadded_loop():
    rng = np.random.default_rng(31)
    ns = [60, 90, 75]
    S = 120
    pairs = [make_correspondences(rng, n, 0.3, 0.0) for n in ns]
    Nmax = max(ns)
    P = len(ns)
    p1 = torch.zeros(P, Nmax, 2, dtype=DT); p2 = torch.zeros(P, Nmax, 2, dtype=DT)
    valid = torch.zeros(P, Nmax, dtype=torch.bool)
    for i, (a, b, _, _) in enumerate(pairs):
        p1[i, :ns[i]] = a; p2[i, :ns[i]] = b; valid[i, :ns[i]] = True
    scheds = _sched(ns, S, seed=1)
    sched_pad = torch.zeros(P, S, 4, dtype=torch.long)
    for i in range(P):
        sched_pad[i] = scheds[i]
    cfg = _cfg()
    batch = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg, hypothesis_indices=sched_pad)
    for i in range(P):
        a, b, _, _ = pairs[i]
        loop = estimate_homography_magsacpp(a, b, config=cfg, hypothesis_indices=scheds[i])
        assert batch[i].inlier_count == loop.inlier_count, i
        assert int(batch[i].inlier_mask[ns[i]:].sum()) == 0          # padded never inliers
        pr_b = apply_h(batch[i].H, a); pr_l = apply_h(loop.H, a)
        assert float((pr_b - pr_l).norm(dim=-1).max()) < 1e-8, i


def test_batch_edge_cases():
    rng = np.random.default_rng(32)
    good = make_correspondences(rng, 50, 0.2, 0.0)
    P, N = 3, 50
    p1 = torch.zeros(P, N, 2, dtype=DT); p2 = torch.zeros(P, N, 2, dtype=DT)
    valid = torch.zeros(P, N, dtype=torch.bool)
    p1[0], p2[0] = good[0], good[1]; valid[0] = True                 # ok
    p1[1, :3], p2[1, :3] = good[0][:3], good[1][:3]; valid[1, :3] = True   # N<4
    # pair 2 fully invalid (all mask False)
    cfg = _cfg(max_hypotheses=300)
    gen = torch.Generator().manual_seed(0)
    out = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg, generator=gen)
    assert out[0].success and out[0].status == "ok"
    assert (not out[1].success) and out[1].status == "n_lt_4"
    assert (not out[2].success) and out[2].status == "all_invalid"


def test_batch_recovers_clean_homography():
    rng = np.random.default_rng(33)
    pairs = [make_correspondences(rng, 80, 0.3, 0.0) for _ in range(4)]
    p1 = torch.stack([p[0] for p in pairs]); p2 = torch.stack([p[1] for p in pairs])
    valid = torch.ones(4, 80, dtype=torch.bool)
    gen = torch.Generator().manual_seed(3)
    out = estimate_homography_magsacpp_batch(p1, p2, valid, config=_cfg(max_hypotheses=800), generator=gen)
    for i, (_, _, H, inl) in enumerate(pairs):
        assert out[i].success
        agree = (out[i].inlier_mask == inl).to(DT).mean()
        assert float(agree) > 0.95, i
