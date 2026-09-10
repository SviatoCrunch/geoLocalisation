"""Sky-masker contract + Registry + resolver assembly.

A masker returns a pixel keep-mask (True=ground) for a frame, or ``None`` if it can't
(e.g. precomputed miss) so the next backend in precedence can try. The resolver walks
``cfg.backend_order`` (precomputed → neural by default) and returns the first hit.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ..schemas import SkyMask


@runtime_checkable
class SkyMasker(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def mask(self, frame_id: str, image) -> Optional[np.ndarray]:
        """Pixel keep-mask (H,W) bool, or None if this backend has no mask for the frame."""
        ...


class CascadingMasker:
    """Try maskers in order; first non-None keep-mask wins (records its backend)."""

    def __init__(self, maskers):
        self._maskers = list(maskers)

    def resolve(self, frame_id: str, image=None) -> SkyMask:
        for m in self._maskers:
            keep = m.mask(frame_id, image)
            if keep is not None:
                return SkyMask(frame_id=frame_id, keep=np.asarray(keep, bool),
                               backend=m.name, version=m.version)
        raise RuntimeError(f"no sky backend produced a mask for frame {frame_id!r} "
                           f"(tried: {[m.name for m in self._maskers]})")


_REGISTRY: dict = {}


def register_masker(name: str, factory: Callable) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("masker name must be a non-empty string")
    if name in _REGISTRY:
        raise ValueError(f"masker {name!r} already registered")
    _REGISTRY[name] = factory


def available_maskers() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_resolver(cfg, store=None, neural_impl=None) -> CascadingMasker:
    """Assemble the resolver from ``cfg.backend_order``. ``neural_impl`` is an injected
    SkyMasker (so the heavy model stays optional/testable)."""
    cfg.validate()
    maskers = []
    for name in cfg.backend_order:
        f = _REGISTRY.get(name)
        if f is None:
            raise KeyError(f"unknown masker {name!r}; available: {list(available_maskers())}")
        maskers.append(f(cfg=cfg, store=store, neural_impl=neural_impl))
    return CascadingMasker(maskers)


def _register_builtins() -> None:
    from .precomputed import PrecomputedMasker
    from .neural import NeuralSkyMasker
    if "precomputed" not in _REGISTRY:
        register_masker("precomputed", lambda cfg, store, neural_impl: PrecomputedMasker(store))
    if "neural" not in _REGISTRY:
        register_masker("neural",
                        lambda cfg, store, neural_impl: neural_impl or NeuralSkyMasker(cfg))


_register_builtins()
