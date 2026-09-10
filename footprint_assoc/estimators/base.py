"""Per-level estimator contract + Registry.

An estimator scores how well one satellite level matches the UAV query, producing a
``LevelScore``. Estimators are stateless (params come from ``FootprintConfig`` at call
time) and operate ONLY on numpy features — no backbone here.
"""
from __future__ import annotations

from typing import Callable, Protocol, Tuple, runtime_checkable

from ..schemas import LevelScore, LevelFeatures, QueryFeatures


@runtime_checkable
class LevelEstimator(Protocol):
    @property
    def name(self) -> str: ...

    def score_level(self, query: QueryFeatures, level: LevelFeatures, cfg) -> LevelScore: ...


_REGISTRY: dict = {}


def register_estimator(name: str, factory: Callable) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("estimator name must be a non-empty string")
    if name in _REGISTRY:
        raise ValueError(f"estimator {name!r} already registered")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTRY[name] = factory


def available_estimators() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_estimator(name: str):
    f = _REGISTRY.get(name)
    if f is None:
        raise KeyError(f"unknown estimator {name!r}; available: {list(available_estimators())}")
    return f()


def _register_builtins() -> None:
    from .global_vlad import GlobalVladEstimator
    from .patch_overlap import PatchOverlapEstimator
    for cls in (GlobalVladEstimator, PatchOverlapEstimator):
        if cls.name not in _REGISTRY:
            register_estimator(cls.name, cls)


_register_builtins()
