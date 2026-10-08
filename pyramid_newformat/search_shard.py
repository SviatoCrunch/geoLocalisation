"""Deprecated entry point — the minimum-reads shard search engine now lives in :mod:`locate`.

Kept so ``python -m pyramid_newformat.search_shard …`` and any existing import keep working; it simply
delegates to :func:`pyramid_newformat.locate.locate` / its CLI (one engine, no divergence). Prefer the
``locate`` interface in new code:

    from pyramid_newformat.locate import locate
"""
from __future__ import annotations

from .locate import locate, main

__all__ = ["locate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
