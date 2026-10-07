"""Isolated local fine-reranker (no S3): local cell pyramids × query grids → ranked cells, GPU/CPU.

Modules:
  * :mod:`local_cell_store` — read local cell H5s (same schema as the S3 store), bounded handle LRU;
  * :mod:`rerank_local`    — the driver (cell-major/query-major, cpu_magsac verify, device choice);
  * :mod:`curve`           — top-K localisation curve from a run's ``--out`` + ``--dump-scores``.

Verify backend here is cv2 ``cpu_magsac`` only; the valid GPU MAGSAC++ reranker is
``magsacpp_torch.search`` (``--store-dir`` reads the same local store). Geometry primitives are reused
from the pure (no-IO) ``patch_rerank.matcher``; the scoring/aggregation is copied verbatim from
``patch_rerank.search_pyramid_s3`` for bit parity.
"""
