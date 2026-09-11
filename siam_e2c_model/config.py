"""Model configuration for the e2c Stage-2 query-conditioned model.

Only architecture knobs live here (aggregation, K/vocab, pyramid, dims). Data concerns
(grid resampling, H5 IO) and training (loss/batch/optimizer) are elsewhere.

Two pyramid modes (run separately for A/B; never merged):
  * ``cell``       — regular n×n spatial splits per tile (``scales_cells``); the original.
  * ``concentric`` — nested central crops of physical size (``concentric_sizes_m``), apex 250 m.
``scales_cells`` is used ONLY in cell mode; ``concentric_sizes_m`` ONLY in concentric mode.
Physical sizes (metres) are NOT cell counts — they are different concepts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

CONCENTRIC_APEX_M = 250.0                    # smallest allowed central crop (pyramid apex)


@dataclass
class E2cModelConfig:
    # ── aggregation (switchable) + vocabulary size ────────────────────────────
    agg: str = "supervlad"                  # supervlad | vlad | residual
    k: int = 32                             # n_groups (SuperVLAD/VLAD clusters); "the size"
    assign_path: str = ""                   # recast k*.pt: assign_weight [+ centroids for vlad]

    # ── output pyramid (default = cell; switchable) ───────────────────────────
    pyramid_mode: str = "cell"              # cell | concentric
    scales_cells: tuple = (8, 4, 2, 1)      # CELL mode only: n×n spatial cell splits
    tile_size_m: float = 1000.0             # physical tile side (concentric ratios / metadata)
    concentric_sizes_m: tuple = (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0,
                                 350.0, 300.0, 250.0)   # CONCENTRIC mode only; apex = 250 m

    # ── dims / vocab ──────────────────────────────────────────────────────────
    d_token: int = 1536
    d_group: int = 32                       # PerGroupProjection output dim
    d_out: int = 256                        # projection head output
    d_hidden: int = 512                     # projection head hidden
    dropout: float = 0.1
    tau_min: float = 0.01
    tau_init: float = 0.1
    intra: bool = True
    freeze_assignment: bool = True

    def validate(self) -> None:
        if self.agg not in ("supervlad", "vlad", "residual"):
            raise ValueError(f"unknown agg {self.agg!r} (supervlad|vlad|residual)")
        if int(self.k) <= 0:
            raise ValueError("k (n_groups) must be > 0")
        if self.pyramid_mode not in ("cell", "concentric"):
            raise ValueError(f"pyramid_mode must be 'cell' or 'concentric', got {self.pyramid_mode!r}")
        if not (float(self.tile_size_m) > 0):
            raise ValueError(f"tile_size_m must be > 0, got {self.tile_size_m}")
        object.__setattr__(self, "tile_size_m", float(self.tile_size_m))

        # cell params are always validated (default mode; used only in cell mode)
        sc = tuple(int(n) for n in self.scales_cells)
        if not sc or any(n < 1 for n in sc):
            raise ValueError(f"scales_cells must be non-empty ints >= 1, got {self.scales_cells}")
        object.__setattr__(self, "scales_cells", sc)

        # concentric params validated + normalised to canonical descending order
        object.__setattr__(self, "_concentric_order_normalized", False)
        if self.pyramid_mode == "concentric":
            self._validate_concentric()

        for name in ("d_token", "d_group", "d_out", "d_hidden"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")

    def _validate_concentric(self) -> None:
        tile = float(self.tile_size_m)
        given = tuple(float(s) for s in self.concentric_sizes_m)
        if not given:
            raise ValueError("concentric_sizes_m must be non-empty in concentric mode")
        if len(set(given)) != len(given):
            raise ValueError(f"concentric_sizes_m has duplicates: {given}")
        for s in given:
            if not (0.0 < s <= tile):
                raise ValueError(f"concentric level {s} m out of range (0, tile_size_m={tile}]")
        # monotonic (asc or desc); we canonicalise to DESCENDING and record if reordered
        asc = all(a < b for a, b in zip(given, given[1:]))
        desc = all(a > b for a, b in zip(given, given[1:]))
        if not (asc or desc):
            raise ValueError(f"concentric_sizes_m must be monotonic (asc or desc), got {given}")
        norm = tuple(sorted(given, reverse=True))          # canonical: largest → apex
        if tile not in norm:
            raise ValueError(f"concentric_sizes_m must include the full tile ({tile} m), got {given}")
        if min(norm) != CONCENTRIC_APEX_M:                 # apex must be exactly 250 m
            raise ValueError(f"concentric apex (smallest level) must be exactly {CONCENTRIC_APEX_M} m, "
                             f"got {min(norm)}")
        if any(s < CONCENTRIC_APEX_M for s in norm):       # (redundant with apex==250, explicit)
            raise ValueError(f"concentric levels < {CONCENTRIC_APEX_M} m are forbidden: {given}")
        object.__setattr__(self, "concentric_sizes_m", norm)
        object.__setattr__(self, "_concentric_order_normalized", norm != given)
        object.__setattr__(self, "_concentric_input_order",
                           list(given) if norm != given else None)

    def concentric_levels(self) -> tuple:
        """Canonical descending physical levels (metres). Call after ``validate()``."""
        return tuple(float(s) for s in self.concentric_sizes_m)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scales_cells"] = list(self.scales_cells)
        d["concentric_sizes_m"] = list(self.concentric_sizes_m)
        return d
