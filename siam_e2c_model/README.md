# siam_e2c_model

The **e2c Stage-2 query-conditioned model**, packaged as **model-only** with a
switchable per-cell aggregation and a configurable output pyramid. Contains **no loss,
no batch, no split, no data IO** — those live in `geo_train_batching` /
`geo_split_no_overlap`. A training loop wires this model into them.

Reuses `siam_model_stage3` **unchanged** (never edited); the classic/residual VLAD arm
is **ported from `train_multicity_e2c.py`** and locked by a parity test.

## Model contract

```python
from siam_e2c_model import E2cModelConfig, build_e2c_model

cfg = E2cModelConfig(agg="supervlad", k=32, assign_path="cache/supervlad_multicity/k32.pt",
                     scales_cells=(8, 4, 2, 1))
model = build_e2c_model(cfg, device="cuda")

V = model.build_V(tile_grids)          # (M,H,W,D) -> {n:(M,n²,d_out)}  cell pyramid
q = model.encode_query(uav_tokens)     # (N,D) -> (d_out,)
S = model.score(Q, V, tile_chunk=256)  # Q (B,d_out), V -> (B,M)
model.scales_cells                     # e.g. (8,4,2,1)
model.trainable_parameters()           # group_proj + heads + scale_gate + rho (assignment frozen)
model.state_dict() / load_state_dict()
```

That interface is exactly what the DSS batcher + loss in `geo_train_batching` expect
(`build_V` for tiles, `encode_query` for queries + pair embeddings, `score` for the
B×B / full-gallery matrix).

## Switchable aggregation (Strategy + Registry)

Both arms share the SAME pyramid / `group_proj` / drone+map heads / `scale_gate` / `rho`
— only the per-cell aggregation differs (as in e2c):

| name | assignment | value | source |
|---|---|---|---|
| `supervlad` | soft `softmax(assign)` (frozen) | x̄ (soft mean) | delegates to `core.build_V` / `encode_query_from_tokens` |
| `vlad` (= `residual`) | hard `argmin` to centroid | residual `Σ(x̄−c_k)` | ported e2c residual arm |

Classic VLAD (Jégou 2010) **is** the residual-to-centroid construction, so `vlad` and
`residual` are the same strategy (both names registered). Switch by config:

```yaml
# supervlad ↔ vlad by one line
agg: supervlad      # or: vlad / residual
k: 32               # 32 / 64 / 128 ... the vocabulary size
```

- `supervlad` reads `assign_weight` from the recast `k*.pt`.
- `vlad`/`residual` also needs `centroids` (from the same blob, or passed in).

Register a new arm without touching the model:
```python
from siam_e2c_model import register_aggregation
register_aggregation("my_agg", MyAgg.build)   # build(*, blob, centroids, cfg) -> strategy
```

## Output pyramid (configurable; default = e2c)

`scales_cells` defaults to the e2c `(8, 4, 2, 1)` and is a plain config knob — pass e.g.
`(4, 2, 1)` or `(16, 8, 4, 2, 1)` to change it later (the token grid must be ≥ the finest
cell count). Also configurable: `k`, `d_group` (32), `d_out` (256), `d_hidden` (512),
`d_token` (1536), `intra`, `tau_min/tau_init`, `freeze_assignment`.

## Cell pyramid vs concentric physical pyramid

Two **separate** pyramid modes (`pyramid_mode`, default `cell`), run independently for A/B —
never merged into a hybrid:

```
cell (8,4,2,1):
  1000 m tile → 64 + 16 + 4 + 1 = 85 spatial cells (n×n splits)

concentric:
  1000 m tile → 9 nested CENTRAL crops
  1000, 840, 710, 600, 500, 420, 350, 300, 250 m
  250 m is the smallest central crop and the apex of the pyramid
```

- **cell** — regular `n×n` splits; built for **partial-overlap** retrieval anywhere in a tile
  (the per-scale softmax picks the best-matching cell). `V[n] = (M, n², d_out)`.
- **concentric** — nested central crops of a physical size; built to **select a central scale**
  under aligned centres. One region per level → `V[ℓ] = (M, 1, d_out)`, keyed by level index.
  Levels validated: `0 < size ≤ tile_size_m`, no duplicates, monotonic (canonicalised to
  descending, recorded in resolved config), must include `tile_size_m`, apex **exactly 250 m**,
  none `< 250`.

**Token mask (deterministic rounding).** For level `s` and a token axis of length `L`, the central
side is `round((s/tile_size_m)·L)` tokens (central block). For a **60×60** grid the sides are:

| physical | 1000 | 840 | 710 | 600 | 500 | 420 | 350 | 300 | 250 |
|---|---|---|---|---|---|---|---|---|---|
| token side | 60 | 50 | 43 | 36 | 30 | 25 | 21 | 18 | 15 |

This `round(ratio·L)` rule can differ by one token from a naive token-centre threshold (710 m →
**43** here vs 42 by the strict `abs(x−0.5) ≤ ratio/2` test); we use `round` to match the intended
physical side and keep symmetric nesting. Same rule for even/odd `H/W`; the centre never drifts by a
whole token between levels; H and W use the same physical fraction independently.

**Scoring formula (both modes)** — the SAME cross-scale scorer:
`score(q, tile) = Σ_ℓ β_ℓ(q) · S_ℓ`, `β = softmax(ScaleGate(q))`,
`S_ℓ = ⟨ L2( Σ_cells softmax(⟨V[ℓ],q⟩/τ_ℓ) · V[ℓ] ) , q ⟩`.
Cell: level ℓ has `n²` cells, `τ_ℓ = τ_min + softplus(ρ_ℓ)` (n>1) else 1. Concentric: **1 cell**
per level ⇒ the per-cell softmax is the identity, `τ≡1` (no `ρ`), so `S_ℓ = ⟨V[ℓ], q⟩`. Spatial
pooling (masked mean inside the crop) is separated from cross-scale pooling (β); a concentric region
is **not** treated as `n²` cells. `ScaleGate` width = number of levels (so it is created per config
and lands in `state_dict`); loading a checkpoint from the other mode raises a clear error.

### Footprint interpretation (important)

Retrieval loss alone does **not** make the selected level a true footprint. `score_with_details`
separates: (1) `tile_scores` = retrieval result; (2) `level_scores` = per-level contribution;
(3) `best_level_value` = **pseudo-footprint** estimate (physical metres in concentric mode);
(4) a calibrated footprint is only possible after validation on frames with known footprint.
`level_probabilities` is a **normalized level score** (softmax over retrieval scores), NOT a
calibrated footprint probability. No scale supervision is added; the level is latent.

### A/B configs

`configs/ab_cell.yaml` and `configs/ab_concentric.yaml` hold the two model configs + the
fair-comparison checklist. Note `cell` has 85 regional candidates vs `concentric` 9 — do not read
a Recall difference as a pure footprint effect (region count + pooling opportunities differ).

## Boundaries

- Reuses `siam_model_stage3` via the single module `adapters/stage3.py` (factory +
  `load_assign_weight` + centroid loading), **never modifying stage3**.
- Out of scope (as in e2c): tile/query **token-grid resampling** (`grid_size=60`), H5
  loading, optimizer, eval — those belong to the data/training-loop package.
- Frozen: the assignment (`assign_weight`). Trainable: `group_proj`, drone/map heads,
  `scale_gate`, per-scale `rho`.

## Tests

```bash
python -m pytest siam_e2c_model -q
```

Covers: build both arms (V/pyramid shapes, encode, score, grad through the trainable
tail), Registry (builtins, unknown, duplicate, vlad-needs-centroids), pyramid
configurability + config-only aggregation swap, and **parity** — the `vlad` arm equals
`train_multicity_e2c.residual_build_V` / `residual_encode_query` exactly, and `supervlad`
delegates to the stage3 core.
