"""Packed contiguous pyramid-embedding store — converter + reader (parallel to the per-cell S3 store).

Source (unchanged): ``patch_rerank.build_cell_store_s3`` per-cell H5 + ``_index.json``.
Target: one/few contiguous ``/features (N,P,L,H,W,D) float16`` files for fast batched cell reads and
direct GPU hand-off, in a sibling ``*_newformat`` S3 prefix. See README.md.

    from pyramid_newformat.convert import Converter
    from pyramid_newformat.reader import NewFormatReader
"""
from .reader import NewFormatReader
from .schema import AXES, SCHEMA_VERSION


def locate(*args, **kwargs):
    """Lazy proxy to :func:`pyramid_newformat.locate.locate` (keeps torch import out of package load)."""
    from .locate import locate as _locate
    return _locate(*args, **kwargs)


__all__ = ["NewFormatReader", "AXES", "SCHEMA_VERSION", "locate"]
