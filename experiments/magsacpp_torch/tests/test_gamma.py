"""Gamma-math tests: analytic vs scipy, danini constant reproduction, LUT parity, shape behaviour."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special as sp
import torch

from magsacpp_torch.config import DANINI_C, DANINI_DOF, DANINI_QUANTILE_K
from magsacpp_torch.gamma import GammaLUT, GammaMath, complete_gamma

DT = torch.float64


def _gm(sigma=10.0):
    return GammaMath(DANINI_DOF, DANINI_QUANTILE_K, DANINI_C, sigma, dtype=DT)


def test_reproduces_danini_gamma_value_of_k():
    # danini estimators.h:59 getUpperIncompleteGammaOfK() = 0.0036572608340910764 for homography.
    # that is the unnormalized upper-incomplete gamma Gamma((n-1)/2, k^2/2), n=4 -> Gamma(1.5, 6.6248).
    a2 = (DANINI_DOF - 1) / 2.0
    tk = DANINI_QUANTILE_K ** 2 / 2.0
    val = sp.gammaincc(a2, tk) * sp.gamma(a2)
    assert abs(val - 0.0036572608340910764) < 1e-15
    gm = _gm()
    assert abs(float(gm._gu_a2_tk) - 0.0036572608340910764) < 1e-14


def test_analytic_matches_scipy_lower_upper():
    a = 1.5
    x = torch.tensor([0.0, 0.5, 2.0, 6.6248, 50.0], dtype=DT)
    ga = complete_gamma(a, DT)
    gl = torch.special.gammainc(torch.full_like(x, a), x) * ga
    gu = torch.special.gammaincc(torch.full_like(x, a), x) * ga
    gl_sp = sp.gammainc(a, x.numpy()) * sp.gamma(a)
    gu_sp = sp.gammaincc(a, x.numpy()) * sp.gamma(a)
    assert np.allclose(gl.numpy(), gl_sp, atol=1e-12, rtol=1e-10)
    assert np.allclose(gu.numpy(), gu_sp, atol=1e-12, rtol=1e-10)
    # lower + upper = complete
    assert torch.allclose(gl + gu, ga.expand_as(x), atol=1e-10)


def test_loss_zero_at_origin_and_saturates():
    gm = _gm(sigma=5.0)
    sq = torch.tensor([0.0], dtype=DT)
    assert float(gm.loss(sq)) == pytest.approx(0.0, abs=1e-12)
    # beyond the cutoff tau^2 = (k*sigma)^2 the loss is the constant rho_sat
    tau2 = (DANINI_QUANTILE_K * 5.0) ** 2
    beyond = torch.tensor([tau2 * 1.5, tau2 * 10.0, 1e9], dtype=DT)
    assert torch.allclose(gm.loss(beyond), torch.full_like(beyond, gm.rho_sat), atol=1e-9)
    # continuity at the cutoff: loss(tau2^-) ~ rho_sat
    near = torch.tensor([tau2 * (1 - 1e-9)], dtype=DT)
    assert float(gm.loss(near)) == pytest.approx(gm.rho_sat, rel=1e-5)


def test_loss_monotone_increasing_inside():
    gm = _gm()
    tau2 = (DANINI_QUANTILE_K * gm.sigma_max) ** 2
    sq = torch.linspace(0, tau2, 500, dtype=DT)
    loss = gm.loss(sq)
    assert torch.all(loss[1:] - loss[:-1] >= -1e-12)   # non-decreasing


def test_weight_positive_decreasing_zero_beyond_cutoff():
    gm = _gm()
    tau2 = (DANINI_QUANTILE_K * gm.sigma_max) ** 2
    sq = torch.linspace(0, tau2, 400, dtype=DT)
    w = gm.weight(sq)
    assert float(w[0]) > 0
    assert torch.all(w >= -1e-12)
    assert torch.all(w[1:] - w[:-1] <= 1e-12)          # non-increasing
    beyond = torch.tensor([tau2 * 1.001, tau2 * 5], dtype=DT)
    assert torch.allclose(gm.weight(beyond), torch.zeros_like(beyond), atol=1e-12)


def test_invalid_residual_is_outlier_not_nan():
    gm = _gm()
    sq = torch.tensor([float("inf")], dtype=DT)
    assert torch.isfinite(gm.loss(sq)).all()
    assert float(gm.loss(sq)) == pytest.approx(gm.rho_sat, rel=1e-6)
    assert float(gm.weight(sq)) == 0.0


def test_lut_matches_analytic():
    gm = _gm()
    lut = GammaLUT(gm, size=100_000, interp=True)
    tau2 = (DANINI_QUANTILE_K * gm.sigma_max) ** 2
    sq = torch.linspace(0, tau2 * 1.2, 2000, dtype=DT)
    assert torch.allclose(lut.loss(sq), gm.loss(sq), atol=1e-6, rtol=1e-5)
    assert torch.allclose(lut.weight(sq), gm.weight(sq), atol=1e-6, rtol=1e-5)


def test_total_loss_ignores_padding():
    gm = _gm()
    sq = torch.tensor([[0.0, 0.0, 1e9]], dtype=DT)          # 2 inliers + 1 "outlier"
    valid_all = torch.tensor([[True, True, True]])
    valid_pad = torch.tensor([[True, True, False]])         # mask the last
    assert float(gm.total_loss(sq, valid_all, 1)) == pytest.approx(gm.rho_sat, rel=1e-6)
    assert float(gm.total_loss(sq, valid_pad, 1)) == pytest.approx(0.0, abs=1e-9)
