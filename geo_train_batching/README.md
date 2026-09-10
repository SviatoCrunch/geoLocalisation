# geo_train_batching

Model-agnostic **loss + batch construction + relevance** for retrieval training. It
holds *only* the training-data and objective logic — **no model**. Losses take score
tensors or a `score_fn`; the batcher takes pair embeddings as plain arrays. The model
lives in a **separate** package and is wired in by the (future) training loop.

Designed to snap onto `geo_split_no_overlap`: the split adapter reuses that package's
positive-selection snapshot, so **training positives are identical to the split's**.

## Layout

```
geo_train_batching/
├── loss/         # multi-positive CE, full-gallery NLL (dense+streaming), symmetric InfoNCE
├── relevance/    # RelevanceTable (pos/safe/ignore): geometry (IoU) or explicit
├── batching/     # canonical pairs, neighbour cache, logical-batch planner, Y[B,B]
├── adapters/     # split integration (the ONLY cross-package dependency)
└── tests/
```

The proven math is **ported verbatim** from `train_multicity_e2c.py` +
`siam_model_stage4_full_gallery/{batching/dss.py, loss/full_gallery.py}` and locked by
parity tests (`tests/test_parity.py`) against those originals — this package does not
import `siam_model_stage4_full_gallery` at runtime.

## Losses (`geo_train_batching.loss`)

- `symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.10)` — the DSS B×B objective:
  `0.5·(query→tile + tile→query)` multi-positive CE. `R_cand ⊇ R_pos`; ignore = neither.
- `multipositive_ce(S, pos, cand, tau_loss)` — one direction (`= dense_full_gallery_nll` loss).
- `dense_full_gallery_nll` / `streaming_full_gallery_nll(score_fn, n_gallery, pos, cand, …)`
  — full-gallery NLL; streaming = exact (chunk-invariant) via `logaddexp`, denominator never
  truncated. `validate_masks` / preflight fail closed (no silent skips).
- `streaming_soft_relevance_ce(…)` — soft-target CE (`logZ − Σ y·z`).
- `symmetric_infonce(q, r, logit_scale, label_smoothing=0)` — Sample4Geo-style diagonal
  InfoNCE (1:1 pairing); pass `model.logit_scale.exp()` to keep its gradient.

`masked_logsumexp` is the shared primitive. `tau_loss` default `0.10`.

## Relevance (`geo_train_batching.relevance`)

`RelevanceTable` = `pos_of(i)`, `safe_of(i)` (tile-row arrays), `n_queries`, `query_ids`.
- `GeometryRelevanceTable` — IoU footprints: pos ≥ threshold, safe ≤ eps, else ignore.
- `ExplicitRelevanceTable` — pos/safe supplied directly (used by the split adapter).

## Batching (`geo_train_batching.batching`)

`build_pair_pool` (canonical query→nearest-positive-tile) · `build_neighbour_cache`
(cosine-kNN over caller-supplied pair embeddings) · `plan_logical_batch`
(1 seed + B/2−1 neighbours + B/2 random, no duplicate query/tile, frequency-balanced;
short rather than padded) · `build_cross_relevance` (Y[B,B] pos/cand from geometry) ·
`microbatch_ranges` (for a two-pass GradCache split done in the training loop).

## Split integration (`geo_train_batching.adapters`)

```python
from geo_train_batching.adapters import build_split_relevance, build_pairs_from_split

sr    = build_split_relevance("cfg.yaml", "run/split.json", which="train")
pairs = build_pairs_from_split(sr)      # canonical pairs
# sr.relevance / sr.q_xy / sr.tile_xy / sr.gallery / sr.snapshot / sr.fingerprint
```

`build_split_relevance` loads the same gallery + points + **positive-selection snapshot**
as `geo_split_no_overlap` (single-gallery or per-city merged), reads `split.json` to pick
the split, and returns a `RelevanceTable` where **positives = the split's positives**
(mapped to tile rows) and **safe = zero-overlap tiles** (ignore = the rest). Supports the
same strategies (`current_rule`, `contains_point`, `tile_iou_1000`, `pyramid_top_iou_250`).

Import note: `geo_train_batching` core does **not** import `geo_split_no_overlap`; only
`geo_train_batching.adapters` does. So the loss/batching layers stay dependency-free.

## How a training loop (in the model package) uses this

```python
sr    = build_split_relevance(cfg, split_json, "train")
pairs = build_pairs_from_split(sr)
pair_vecs = model.encode_pairs_eval(pairs)          # model package
nbr   = build_neighbour_cache(pair_vecs, epoch=e)   # refresh every N epochs
plan  = plan_logical_batch(pairs, nbr, B_log=32, seed_pair=s, rng=rng)
pib   = [pairs[i] for i in plan.pair_indices]
S     = model.score(Q_of(pib), V_of(pib))           # model package
R_pos, R_cand = build_cross_relevance(pib, sr.relevance)
loss  = symmetric_multipositive_ce(S, R_pos, R_cand)
```

The model, encoder and GradCache orchestration live in a different package; this one only
supplies the batch plan, the relevance masks and the loss.

## Tests

```bash
python -m pytest geo_train_batching -q
```

Covers each loss (incl. dense↔streaming parity, chunk-invariance, fail-closed masks,
InfoNCE), relevance (geometry + explicit), batching (canonical pairs, deterministic
neighbours/plan, no-dup, Y[B,B]), the split-adapter integration (positives == snapshot,
feeds batch+loss, reproducible, train/val disjoint), and **parity** vs the
`siam_model_stage4_full_gallery` originals.
