"""E2cModel — the e2c Stage-2 query-conditioned model with a switchable aggregation arm
and a switchable output pyramid (``cell`` | ``concentric``).

Wraps the in-package vendored Stage-2 core (``siam_e2c_model.vendored``) — either the CELL core
(``Stage2QueryConditionedModel``) or the CONCENTRIC core (``ConcentricStage2Model``) — and a chosen
aggregation Strategy (SuperVLAD soft / classic-residual VLAD). Exposes a uniform, model-only
interface for a training loop + ``geo_train_batching`` to consume:

    build_V(grids)     encode_query(tokens)     score(Q, V)     score_with_details(Q, V)
    scales_cells       trainable_parameters()   state_dict / load_state_dict

No loss, no batch, no split, no data IO here. The two pyramid modes are run SEPARATELY (A/B);
they are never merged into a hybrid.
"""
from __future__ import annotations

from .aggregation import create_aggregation
from .adapters import stage3 as S3
from .config import E2cModelConfig
from .scoring import score_with_details as _score_with_details


class E2cModel:
    def __init__(self, core, agg, cfg: E2cModelConfig):
        self.core = core                    # Stage2QueryConditionedModel | ConcentricStage2Model
        self.agg = agg                      # AggregationStrategy
        self.cfg = cfg

    # ── uniform interface ─────────────────────────────────────────────────────
    @property
    def pyramid_mode(self) -> str:
        return getattr(self.core, "pyramid_mode", "cell")

    @property
    def scales_cells(self) -> tuple:
        return tuple(self.core.scales_cells)

    def level_values(self) -> tuple:
        """Per-level values: physical metres (concentric) or cell counts n² (cell)."""
        if self.pyramid_mode == "concentric":
            return tuple(float(s) for s in self.core.level_values)
        return tuple(int(n) * int(n) for n in self.core.scales_cells)

    def build_V(self, grids):
        """Map token grids (M,H,W,D) → V (cell: {n:(M,n²,d)}; concentric: {ℓ:(M,1,d)})."""
        return self.agg.build_V(self.core, grids)

    def encode_query(self, tokens):
        """Query tokens (N,D) → query embedding (d_out,). Mode-independent (global)."""
        return self.agg.encode_query(self.core, tokens)

    def score(self, Q, V, *, tile_chunk=None):
        """Query-conditioned scores: Q (B,d), V → (B,M)."""
        return self.core.score_queries_against_tiles(Q, V, tile_chunk=tile_chunk)

    def score_with_details(self, Q, V, *, tile_chunk=None) -> dict:
        """Diagnostics: tile_scores (B,M) + per-level scores (B,M,L) + best level, margin, entropy.

        ``best_level_value`` is a pseudo-footprint estimate (physical metres in concentric mode),
        NOT a metric-verified footprint; ``level_probabilities`` is a normalized level score, not a
        calibrated footprint probability. See README §"Footprint interpretation"."""
        return _score_with_details(self.core, Q, V, self.level_values(), self.pyramid_mode,
                                   tile_chunk=tile_chunk)

    # ── training plumbing (delegates to the core module) ──────────────────────
    def trainable_parameters(self):
        return [p for p in self.core.parameters() if p.requires_grad]

    def parameters(self):
        return self.core.parameters()

    def to(self, device):
        self.core.to(device)
        return self

    def train(self, mode: bool = True):
        self.core.train(mode)
        return self

    def eval(self):
        self.core.eval()
        return self

    def state_dict(self):
        return self.core.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        """Load into ``core``. On a shape mismatch (typically a pyramid-mode mismatch — cell has
        per-scale ρ + a scale_gate of width len(scales_cells); concentric has no ρ + width L) raise
        a clear error naming the current mode. Never silently falls back to strict=False."""
        try:
            return self.core.load_state_dict(sd, strict=strict)
        except RuntimeError as e:
            raise RuntimeError(
                f"load_state_dict failed for pyramid_mode={self.pyramid_mode!r} "
                f"(agg={self.cfg.agg!r}). This usually means the checkpoint was trained with a "
                f"different pyramid_mode / level count — build the model with the matching config. "
                f"Original error: {e}") from e

    def resolved_config(self) -> dict:
        rc = {**self.cfg.to_dict(), "aggregation": dict(self.agg.resolved_config()),
              "pyramid_mode": self.pyramid_mode,
              "tile_size_m": float(self.cfg.tile_size_m),
              "level_values": list(self.level_values()),
              "trainable_params": int(sum(p.numel() for p in self.trainable_parameters()))}
        if self.pyramid_mode == "concentric":
            rc["concentric_sizes_m"] = list(self.core.concentric_sizes_m)
            rc["n_levels"] = len(self.core.concentric_sizes_m)
            if getattr(self.cfg, "_concentric_order_normalized", False):
                rc["concentric_order_normalized"] = True
                rc["concentric_input_order"] = list(getattr(self.cfg, "_concentric_input_order", []) or [])
        else:
            rc["scales_cells"] = list(self.scales_cells)
        return rc


def build_e2c_model(cfg: E2cModelConfig, *, assign_weight=None, centroids=None,
                    device=None) -> E2cModel:
    """Build the e2c model in the configured ``pyramid_mode``. Loads ``assign_weight`` (+ ``centroids``
    for the vlad/residual arm) from ``cfg.assign_path`` when not passed directly."""
    cfg.validate()
    blob = None
    if assign_weight is None:
        if not cfg.assign_path:
            raise ValueError("assign_path required (or pass assign_weight=...)")
        assign_weight = S3.load_assign_weight(cfg.assign_path, k=cfg.k, d=cfg.d_token)
        blob = S3.load_blob(cfg.assign_path)
    # §10 fix: the vlad/residual arm needs centroids even when assign_weight was passed
    # explicitly; load them from assign_path independently instead of only inside the
    # `assign_weight is None` branch.
    if cfg.agg in ("vlad", "residual") and centroids is None and blob is None and cfg.assign_path:
        blob = S3.load_blob(cfg.assign_path)

    common = dict(d_token=cfg.d_token, n_groups=cfg.k, n_ghost=0,
                  group_projection_dim=cfg.d_group, d_out=cfg.d_out, head_hidden=cfg.d_hidden,
                  dropout=cfg.dropout, assign_weight=assign_weight,
                  freeze_assignment=cfg.freeze_assignment)
    if cfg.pyramid_mode == "concentric":
        core = S3.build_concentric_core(concentric_sizes_m=cfg.concentric_levels(),
                                        tile_size_m=cfg.tile_size_m, intra=bool(cfg.intra), **common)
    else:
        core = S3.build_core_model(scales_cells=tuple(cfg.scales_cells),
                                   tau_min=cfg.tau_min, tau_init=cfg.tau_init, **common)
        core.intra = bool(cfg.intra)

    agg = create_aggregation(cfg.agg, blob=blob, centroids=centroids, cfg=cfg)
    model = E2cModel(core, agg, cfg)
    if device is not None:
        model.to(device)
    return model
