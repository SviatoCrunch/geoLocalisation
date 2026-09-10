"""geo_split_no_overlap — leakage-free geographic train/val/test split.

Build train / val / test sets of query points such that NO gallery tile that is a
positive for a point in one split geometrically overlaps (area > ``area_epsilon``)
with a positive tile for a point in another split. Points whose positives conflict
are fused into *indivisible components* (union-find); every component is assigned
whole to one split or excluded whole. Balancing is by point count.

The positive rule is provided by the pluggable :mod:`geo_split_no_overlap.positive_selection`
subsystem (Strategy + Registry). The split layer consumes only its immutable
``MaterializedPositiveSets`` snapshot + ``GalleryIndex`` — never a concrete strategy.
Existing project definitions are reused inside that subsystem's project adapters
(``make_multicity_split`` helpers + the ``multicity_vlad_*.h5`` gallery).

Public entry points live in :mod:`geo_split_no_overlap.cli`.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
