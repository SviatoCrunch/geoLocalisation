"""Central Registry mapping a stable rule name -> a strategy factory.

The ONLY place that decides which concrete strategy is used. Built-ins are registered
explicitly (no eval / dynamic import / repo scanning). Factories are registered, not
instances — nothing is constructed at import time.
"""
from __future__ import annotations

from typing import Callable, Mapping, Tuple, Union

from .config import PositiveSelectionConfig
from .protocol import PositiveSelector

# factory: params-mapping -> PositiveSelector
Factory = Callable[[Mapping], PositiveSelector]

_REGISTRY: dict = {}


def register_positive_selector(name: str, factory: Factory) -> None:
    """Register a factory under ``name``. Re-registering an existing name is an error."""
    if not isinstance(name, str) or not name:
        raise ValueError("strategy name must be a non-empty string")
    if name in _REGISTRY:
        raise ValueError(f"positive selector {name!r} is already registered")
    if not callable(factory):
        raise TypeError(f"factory for {name!r} must be callable")
    _REGISTRY[name] = factory


def available_positive_selectors() -> Tuple[str, ...]:
    """Registered names in deterministic (sorted) order."""
    return tuple(sorted(_REGISTRY))


def create_positive_selector(config: Union[PositiveSelectionConfig, Mapping]) -> PositiveSelector:
    """Instantiate the configured strategy. Validates params via the strategy factory."""
    if not isinstance(config, PositiveSelectionConfig):
        config = PositiveSelectionConfig.from_mapping(config)
    factory = _REGISTRY.get(config.strategy)
    if factory is None:
        raise KeyError(
            f"unknown positive selector {config.strategy!r}; "
            f"available: {list(available_positive_selectors())}")
    return factory(config.params)


def _register_builtins() -> None:
    from .strategies import (ContainsPointSelector, CurrentRuleSelector, NearestTileSelector,
                             PyramidTopIoU250Selector, TileIoU1000Selector)
    for cls in (CurrentRuleSelector, ContainsPointSelector, NearestTileSelector,
                TileIoU1000Selector, PyramidTopIoU250Selector):
        if cls.name not in _REGISTRY:
            register_positive_selector(cls.name, cls.from_params)


_register_builtins()
