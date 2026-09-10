"""The public Strategy contract.

A :class:`PositiveSelector` maps ONE point against the gallery to its positive tiles.
It must be pure and deterministic, know nothing about train/val/test, components or
optimisation, and must not mutate its inputs or touch the filesystem.
"""
from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from .models import GalleryIndex, GeoPoint, PositiveMatch


@runtime_checkable
class PositiveSelector(Protocol):
    @property
    def name(self) -> str:
        """Stable registry name of the rule (e.g. ``"current_rule"``)."""
        ...

    @property
    def version(self) -> str:
        """Rule version; part of the fingerprint so results are traceable."""
        ...

    def select(self, point: GeoPoint, gallery: GalleryIndex) -> Sequence[PositiveMatch]:
        """Return the positive matches for ``point`` (order-independent; the service
        applies a stable sort). Must not mutate ``point`` or ``gallery``."""
        ...

    def resolved_config(self) -> Mapping[str, object]:
        """Fully-resolved params INCLUDING defaults, version, expected CRS and the
        units of any thresholds. Recorded verbatim in the run artifacts."""
        ...
