"""geo_e2c_train — the training-loop GLUE that wires the isolated packages together.

Pulls three isolated pieces into one DSS training run (nothing new mathematically):
  * ``siam_e2c_model``      — the model (``build_e2c_model``: agg × pyramid_mode);
  * ``geo_train_batching``  — DSS pairs / neighbour cache / logical-batch planner / multipositive CE
                              + ``build_split_relevance`` (positives = the split's, from split.json);
  * ``geo_split_no_overlap``— the leakage-free split (via geo_train_batching's adapter).

Data (map tile grids + drone query tokens) comes from the ``map_extract`` H5s. Tile grids are
cached in RAM for speed; a tqdm bar tracks epochs. Eval = Recall@K + median rank on val/test.

This is the only place a model, a loss and a split meet — kept out of the three libraries so each
stays single-purpose. Run: ``python -m geo_e2c_train.train --help``.
"""
