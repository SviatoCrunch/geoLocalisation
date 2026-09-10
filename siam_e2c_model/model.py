"""E2cModel — the e2c Stage-2 query-conditioned model with a switchable aggregation arm.

Wraps the in-package vendored Stage-2 core (``siam_e2c_model.vendored``) and a chosen
aggregation Strategy (SuperVLAD soft / classic-residual VLAD). Exposes a uniform, model-only
interface for a training loop + ``geo_train_batching`` to consume:

    build_V(grids)     encode_query(tokens)     score(Q, V)
    scales_cells       trainable_parameters()   state_dict / load_state_dict

No loss, no batch, no split, no data IO here.
"""
from __future__ import annotations

from .aggregation import create_aggregation
from .adapters import stage3 as S3
from .config import E2cModelConfig


class E2cModel:
    def __init__(self, core, agg, cfg: E2cModelConfig):
        self.core = core                    # Stage2QueryConditionedModel (reused stage3)
        self.agg = agg                      # AggregationStrategy
        self.cfg = cfg

    # ── uniform interface ─────────────────────────────────────────────────────
    @property
    def scales_cells(self) -> tuple:
        return tuple(self.core.scales_cells)

    def build_V(self, grids):
        """Map token grids (M,H,W,D) → cell pyramid V {n:(M,n²,d_out)}."""
        return self.agg.build_V(self.core, grids)

    def encode_query(self, tokens):
        """Query tokens (N,D) → query embedding (d_out,)."""
        return self.agg.encode_query(self.core, tokens)

    def score(self, Q, V, *, tile_chunk=None):
        """Query-conditioned scores: Q (B,d), V {n:(M,n²,d)} → (B,M)."""
        return self.core.score_queries_against_tiles(Q, V, tile_chunk=tile_chunk)

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
        return self.core.load_state_dict(sd, strict=strict)

    def resolved_config(self) -> dict:
        return {**self.cfg.to_dict(), "aggregation": dict(self.agg.resolved_config()),
                "scales_cells": list(self.scales_cells),
                "trainable_params": int(sum(p.numel() for p in self.trainable_parameters()))}


def build_e2c_model(cfg: E2cModelConfig, *, assign_weight=None, centroids=None,
                    device=None) -> E2cModel:
    """Build the e2c model. Loads ``assign_weight`` (+ ``centroids`` for vlad) from
    ``cfg.assign_path`` unless they are passed directly (programmatic / tests)."""
    cfg.validate()
    blob = None
    if assign_weight is None:
        if not cfg.assign_path:
            raise ValueError("assign_path required (or pass assign_weight=...)")
        assign_weight = S3.load_assign_weight(cfg.assign_path, k=cfg.k, d=cfg.d_token)
        blob = S3.load_blob(cfg.assign_path)

    core = S3.build_core_model(
        d_token=cfg.d_token, n_groups=cfg.k, n_ghost=0, group_projection_dim=cfg.d_group,
        scales_cells=tuple(cfg.scales_cells), d_out=cfg.d_out, head_hidden=cfg.d_hidden,
        dropout=cfg.dropout, tau_min=cfg.tau_min, tau_init=cfg.tau_init,
        assign_weight=assign_weight, freeze_assignment=cfg.freeze_assignment)
    core.intra = bool(cfg.intra)

    agg = create_aggregation(cfg.agg, blob=blob, centroids=centroids, cfg=cfg)
    model = E2cModel(core, agg, cfg)
    if device is not None:
        model.to(device)
    return model
