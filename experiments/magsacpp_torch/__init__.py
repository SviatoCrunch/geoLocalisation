"""Torch MAGSAC++ homography core (experiment module).

Status: the marginalized loss, sigma-consensus++ weights and IRLS follow the paper + danini/magsac
closed form (see ``config.py`` / ``REPORT.md``), verified locally against the analytic gamma
functions and on synthetic data. They are NOT yet checked bit-for-bit against the C++ oracle, so a
run is not called "MAGSAC++-equivalent" until :mod:`oracle` parity passes on the GPU server.

Public API:
    from magsacpp_torch import estimate_homography_magsacpp, MagsacppConfig
"""
from __future__ import annotations

from .config import MagsacppConfig
from .estimator import HomographyResult, estimate_homography_magsacpp
from .gamma import GammaLUT, GammaMath

__all__ = [
    "estimate_homography_magsacpp",
    "HomographyResult",
    "MagsacppConfig",
    "GammaMath",
    "GammaLUT",
]
