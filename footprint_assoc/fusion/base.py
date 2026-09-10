"""Fusion contract + Registry + shared helpers.

A fusion turns per-method per-level scores into a :class:`FrameEstimate` — a soft
distribution over levels, a (possibly refused) best scale, and a confidence. Fusion
NEVER forces argmax when evidence is weak (returns SOFT/REFUSE instead).
"""
from __future__ import annotations

from typing import Callable, Protocol, Tuple, runtime_checkable

import numpy as np

from ..schemas import FrameEstimate


@runtime_checkable
class Fusion(Protocol):
    @property
    def name(self) -> str: ...

    def fuse(self, frame_id, center_lat, center_lon, scales, per_method, cfg) -> FrameEstimate: ...


def softmax_masked(vals: np.ndarray, tau: float, mask: np.ndarray) -> np.ndarray:
    """Softmax of ``vals/tau`` over entries where ``mask`` is True; others get 0."""
    v = np.asarray(vals, np.float64).copy()
    out = np.zeros_like(v)
    if not mask.any():
        return out
    z = v[mask] / max(tau, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    out[mask] = e / e.sum()
    return out


def method_scores(per_method: dict, name: str, scales: list) -> np.ndarray:
    """Aligned score vector for a method over ``scales`` (0 where missing)."""
    arr = np.zeros(len(scales), np.float64)
    lst = per_method.get(name)
    if lst:
        by = {round(ls.scale_m, 6): ls.score for ls in lst}
        for i, s in enumerate(scales):
            arr[i] = by.get(round(s, 6), 0.0)
    return arr


_REGISTRY: dict = {}


def register_fusion(name: str, factory: Callable) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("fusion name must be a non-empty string")
    if name in _REGISTRY:
        raise ValueError(f"fusion {name!r} already registered")
    _REGISTRY[name] = factory


def available_fusion() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_fusion(name: str):
    f = _REGISTRY.get(name)
    if f is None:
        raise KeyError(f"unknown fusion {name!r}; available: {list(available_fusion())}")
    return f()


def _register_builtins() -> None:
    from .cascade import CascadeFusion
    from .rank_vote import RankVoteFusion
    for cls in (CascadeFusion, RankVoteFusion):
        if cls.name not in _REGISTRY:
            register_fusion(cls.name, cls)


_register_builtins()
