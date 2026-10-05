"""GPU==CPU device parity. Skipped unless CUDA is present (runs on the server).

Uses a RECORDED schedule (replay mode) so CPU and CUDA evaluate the IDENTICAL minimal samples --
a shared RNG seed is NOT enough for cross-device determinism. float64 must match tightly; float32
is the fast mode with looser (documented) tolerances.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from magsacpp_torch import MagsacppConfig, estimate_homography_magsacpp
from magsacpp_torch.gamma import GammaMath
from magsacpp_torch.tests.conftest import make_correspondences

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
DT = torch.float64


def _project(H, p, eps=1e-12):
    H = H.double().cpu(); p = p.double().cpu()
    x, y = p[:, 0], p[:, 1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    near = w.abs() < eps
    ws = torch.where(near, torch.ones_like(w), w)
    px = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / ws
    py = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / ws
    out = torch.stack([px, py], -1)
    return out, near


def _schedule(n, S, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.stack([rng.choice(n, 4, replace=False) for _ in range(S)]))


@cuda
def test_gamma_device_parity_float64():
    gm_c = GammaMath(4, 3.64, 0.25, 1.0, dtype=torch.float64)
    sq_c = torch.linspace(0, 60, 4000, dtype=torch.float64)
    sq_g = sq_c.cuda()
    assert torch.allclose(gm_c.loss(sq_c), gm_c.loss(sq_g).cpu(), atol=1e-12, rtol=1e-10)
    assert torch.allclose(gm_c.weight(sq_c), gm_c.weight(sq_g).cpu(), atol=1e-12, rtol=1e-10)


@cuda
def test_estimator_device_parity_float64():
    rng = np.random.default_rng(20)
    p1, p2, H, inl = make_correspondences(rng, n_total=160, outlier_frac=0.4, noise=0.0)
    sched = _schedule(160, 400)
    cfg = MagsacppConfig(sigma_max=1.0, irls_iters=3, dtype=torch.float64)
    rc = estimate_homography_magsacpp(p1, p2, config=cfg, hypothesis_indices=sched)
    rg = estimate_homography_magsacpp(p1.cuda(), p2.cuda(), config=cfg,
                                      hypothesis_indices=sched.cuda())
    assert rc.success and rg.success and rc.status == rg.status
    # geometry: project the source points through both H (scale/sign invariant), mask near-infinity
    pc, nc = _project(rc.H, p1); pg, ng = _project(rg.H, p1)
    good = ~(nc | ng)
    assert float((pc[good] - pg[good]).norm(dim=-1).max()) < 1e-6
    assert abs(rc.total_loss - rg.total_loss) / max(1.0, abs(rc.total_loss)) < 1e-9
    assert torch.equal(rc.inlier_mask.cpu(), rg.inlier_mask.cpu())


@cuda
def test_estimator_device_parity_float32():
    rng = np.random.default_rng(21)
    p1, p2, H, inl = make_correspondences(rng, n_total=160, outlier_frac=0.4, noise=0.3)
    sched = _schedule(160, 400)
    cfg = MagsacppConfig(sigma_max=1.5, irls_iters=3, dtype=torch.float32)
    rc = estimate_homography_magsacpp(p1, p2, config=cfg, hypothesis_indices=sched)
    rg = estimate_homography_magsacpp(p1.cuda(), p2.cuda(), config=cfg,
                                      hypothesis_indices=sched.cuda())
    assert rc.success and rg.success
    pc, nc = _project(rc.H, p1); pg, ng = _project(rg.H, p1)
    good = ~(nc | ng)
    assert float((pc[good] - pg[good]).norm(dim=-1).max()) < 1e-2          # float32 projection tol
    # masks may differ only at the threshold boundary; require high agreement
    agree = (rc.inlier_mask.cpu() == rg.inlier_mask.cpu()).to(DT).mean()
    assert float(agree) >= 0.98
