# retrieval_overfit_diagnostics

Read-only diagnostics for the **train-loss ↔ full-gallery-retrieval gap**. This package ONLY
imports/calls existing project code — it changes no model/loss/split math and no checkpoints.

## Integration points (found by reading the project)

| concern | file / function called |
|---|---|
| model, `build_V`/`encode_query`/`score`/`score_with_details`, trainable params | `siam_e2c_model.model.E2cModel`, `build_e2c_model`, `E2cModelConfig` |
| **train candidate set + loss** | `geo_e2c_train/train.py`: `S=model.score(Q,V)` over the **DSS logical-batch** tiles (`plan_logical_batch`→`build_V`), `symmetric_multipositive_ce(S, R_pos, R_cand)`, `R_cand=build_cross_relevance(batch_pairs, sr.relevance)` (pos∪safe within the ~`B_log` batch). NOT B×B in-batch, NOT full gallery. |
| negative mining | `geo_train_batching.build_neighbour_cache` — built **once** (`epoch=0`) in the trainer ⇒ static across epochs |
| full-gallery eval / gallery V | `geo_e2c_train.eval.build_gallery_V` (rebuilt from the loaded checkpoint), `score_against_gallery` |
| split + positives | `geo_train_batching.adapters.split_relevance.build_split_relevance` → `SplitRelevance` |
| checkpoint | trainer saves `{model, epoch, resolved_config}`; loaded via `resolved_config`→`build_e2c_model`→`load_state_dict(strict=True)` |

## One command

```bash
python -m retrieval_overfit_diagnostics.cli \
  --checkpoint …/run_residual_cell/best.pt \
  --split-config …/split_pyr250.yaml --split-json …/split_pyr250/split.json \
  --galleries kramatorsc=…/map_dinov2_kramatorsc_s250m_d1024.h5 kup=… liman_day=… \
  --queries   kramatorsc=…/query_kramatorsc_d1024.h5 kup=… liman_day=… \
  --assign …/dict/k32.pt --checks all \
  --output-dir retrieval_overfit_diagnostics/outputs/<run_id>
```

`--checks` = `all` or a comma list of: `train-full-gallery, inbatch-vs-full, cache, checkpoint,
labels, gate, ablations, negatives, split`. TRAIN+VAL only by default; `--include-test` analyses
TEST for **reporting only** (never for selection) with a logged warning.

## Outputs (`outputs/<run_id>/`)

`resolved_config.json`, `train_objective_metrics.json`, `train_full_gallery_metrics.json`,
`val_full_gallery_metrics.json`, `decisive_summary.json`, `cache_audit.json`,
`checkpoint_audit.json`, `labels_audit.json`, `scale_gate_summary.json`, `ablation_metrics.json`,
`negative_mining_audit.json`, `split_shift_audit.json`, `per_query.csv`, `hypotheses.json`,
`diagnostic_report.md` (hypothesis table), `run.log`. JSONs carry `schema_version` and keep both
`R@k` fractions **and** `hits/n`.

## The decisive test (§4)

`decisive_summary.json` reports **A) train-objective** metric (rank of a train query's positive
within its DSS candidate set) vs **B) train full-gallery** vs **val full-gallery**, and a coarse
`interpretation.primary_suspect` (objective/gallery mismatch · memorization/shift ·
eval/label/checkpoint bug · some-generalization). It is a suspicion, not a verdict — the hypothesis
table keeps `Insufficient` where evidence doesn't separate causes.

Level signals are pseudo-footprints; nothing here is a metric-verified footprint. Do not use TEST to
pick anything.

## Tests

`python -m pytest retrieval_overfit_diagnostics/tests` — synthetic only (no full training/eval):
metric consistency, aggregation ablations, false-negative detection, parameter-fingerprint staleness,
and a CLI smoke that runs every check on a tiny gallery and checks all output files appear.
