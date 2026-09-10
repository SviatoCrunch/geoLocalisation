"""Vendored, self-contained copy of the Stage-2 query-conditioned model core.

``stage2_core.py`` is a byte-faithful copy of ``siam_model_stage3/model/stage2_core.py``
(itself a vendored copy of the proven Stage-2 model). ``assign_io.py`` is the standalone
``load_assign_weight`` from ``siam_model_stage3/model/factory.py``. Both depend only on
numpy/torch — vendored here so ``siam_e2c_model`` has NO external dependency on the
RevisitAnything repo (self-isolated subfolder).
"""
