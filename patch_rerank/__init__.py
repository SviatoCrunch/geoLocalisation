"""Isolated training-free patch-RANSAC reranker for the coarse→fine geo-localization shortlist.

Import submodules directly (``patch_rerank.matcher``, ``.store``, ``.viz_search`` …). The package
``__init__`` is intentionally light (no torch) so the geometry/KMZ tools run without heavy deps.
"""
