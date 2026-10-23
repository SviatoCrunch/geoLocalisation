"""Vendored packed (GPU) new-format store — reader + CellStore-compatible adapter.

Copied from geoLocalisation/pyramid_newformat (schema/s3io/reader) + a trimmed store adapter. Lets the
service fine-reranker read the packed ``/features`` gallery (the GPU-store the standalone locate eats)
in place of the per-cell CellStore, with zero change to reranker.rerank (duck-typed store interface).
"""
from __future__ import annotations

from .reader import NewFormatReader
from .store import NewFormatCellData, NewFormatCellStore

__all__ = ["NewFormatReader", "NewFormatCellStore", "NewFormatCellData"]
