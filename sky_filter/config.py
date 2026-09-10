"""Configuration for the sky-filter precompute stage."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class SkyFilterConfig:
    # ── where precomputed masks live (npz keyed by frame_id) ──────────────────
    mask_store: str = ""

    # ── backend precedence: use a ready mask if present, else run the model ────
    backend_order: list = field(default_factory=lambda: ["precomputed", "neural"])

    # ── token-grid reduction (DROP semantics) ─────────────────────────────────
    # a token cell is DROPPED if its sky fraction exceeds this (0.0 = drop on ANY sky)
    cell_sky_max: float = 0.5

    # ── neural backend (server-only) ──────────────────────────────────────────
    neural_model: str = "segformer_uavid"   # opaque id for the injected adapter
    device: str = "cuda"

    def validate(self) -> None:
        if not (0.0 <= self.cell_sky_max <= 1.0):
            raise ValueError("cell_sky_max must be in [0, 1]")
        for b in self.backend_order:
            if b not in ("precomputed", "neural", "heuristic"):
                raise ValueError(f"unknown backend {b!r}")
        if not self.backend_order:
            raise ValueError("backend_order is empty")

    def to_dict(self) -> dict:
        return asdict(self)
