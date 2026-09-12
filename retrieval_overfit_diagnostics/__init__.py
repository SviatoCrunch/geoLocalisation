"""retrieval_overfit_diagnostics — diagnose the train-loss ↔ full-gallery-retrieval gap.

READ-ONLY diagnostics. This package ONLY imports and calls existing project code
(``siam_e2c_model`` model, ``geo_train_batching`` batching/loss/split adapter,
``geo_e2c_train`` loaders/eval). It does NOT change model/loss/split math or checkpoints;
every artifact of one run lands under ``outputs/<run_id>/``.

Goal (see README §completion): from ONE command answer —
  1. Does the model have high Recall on TRAIN queries over the FULL gallery?
  2. Does eval use descriptors built by the CURRENT checkpoint (not stale)?
  3. Is the gap memorization / objective-mismatch / label-or-eval-bug / insufficient-data?
"""
