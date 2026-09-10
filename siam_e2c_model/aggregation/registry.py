"""Aggregation Registry: stable name -> strategy factory (explicit built-in registration).

Built-ins: ``supervlad`` (soft) and ``vlad`` (classic/residual) with ``residual`` as an
alias for ``vlad`` (identical construction). No eval / dynamic import; factories build
strategies lazily from the recast blob / centroids.
"""
from __future__ import annotations

from typing import Callable, Tuple

_REGISTRY: dict = {}


def register_aggregation(name: str, factory: Callable) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("aggregation name must be a non-empty string")
    if name in _REGISTRY:
        raise ValueError(f"aggregation {name!r} is already registered")
    if not callable(factory):
        raise TypeError(f"factory for {name!r} must be callable")
    _REGISTRY[name] = factory


def available_aggregations() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_aggregation(name: str, *, blob=None, centroids=None, cfg=None):
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"unknown aggregation {name!r}; available: {list(available_aggregations())}")
    return factory(blob=blob, centroids=centroids, cfg=cfg)


def _register_builtins() -> None:
    from .supervlad import SuperVLADAggregation
    from .vlad import VladAggregation
    if "supervlad" not in _REGISTRY:
        register_aggregation("supervlad", SuperVLADAggregation.build)
    if "vlad" not in _REGISTRY:
        register_aggregation("vlad", VladAggregation.build)
    if "residual" not in _REGISTRY:          # alias: classic VLAD == residual VLAD here
        register_aggregation("residual", VladAggregation.build)


_register_builtins()
