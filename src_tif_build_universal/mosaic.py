"""Mosaic utilities and multi-zoom orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Union

logger = logging.getLogger(__name__)

from .config import MosaicConfig
from .coverage import CoverageProvider
from .downloader import download_tiles
from .geometry import geometry_to_aoi_parts, make_aoi_polygon
from .merge import merge_layers
from .sources.base import TileSource


def merge_tif_files(
    layer_paths: List[Union[str, Path]],
    out_path: Union[str, Path],
) -> Path:
    """Merge TIF layers into a single mosaic.

    Higher-resolution layers (smaller pixel size) fill first; lower-
    resolution layers only fill pixels that are still empty.

    Args:
        layer_paths: GeoTIF paths to merge.
        out_path: Output mosaic path.

    Returns:
        Path to the written mosaic.
    """
    out_path = Path(out_path)
    merge_layers([Path(p) for p in layer_paths], out_path)
    return out_path


def build_mosaic(
    source_factory: Callable[[int], TileSource],
    coverage: CoverageProvider,
    aoi_points: list,
    out_dir: Union[str, Path],
    config: Union[MosaicConfig, None] = None,
) -> Path:
    """Universal multi-zoom mosaic orchestrator.

    Steps
    -----
    1. Build AOI polygon.
    2. Query effective zoom coverage via ``coverage.query()`` — one call,
       source-specific (e.g. Google Viewport API or fixed zoom for Esri).
    3. For each zoom level (high → low): call ``download_tiles()`` for
       each geometry part using ``source_factory(zoom)``.
    4. Merge all layer TIFs into ``mosaic_final.tif``.

    Adding a new source
    -------------------
    Provide a ``source_factory`` and a compatible ``CoverageProvider``
    subclass.  Steps 3–4 are source-agnostic.

    Args:
        source_factory: Callable ``(zoom: int) -> TileSource``.  Called
            once per zoom level so each zoom gets its own source instance.
        coverage: CoverageProvider that returns ``{zoom: geometry}``.
        aoi_points: AOI vertices in ``(lat, lon)`` format.
        out_dir: Directory for layer TIFs and final mosaic.
        config: Mosaic parameters. Uses MosaicConfig defaults if None.

    Returns:
        Path to ``mosaic_final.tif`` if ``config.merge=True``, else ``out_dir``.
    """
    cfg = config if config is not None else MosaicConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi_polygon, ordered_points = make_aoi_polygon(aoi_points)

    effective_geoms = coverage.query(ordered_points, aoi_polygon)

    if not effective_geoms:
        raise RuntimeError("No effective zoom geometries found for this AOI.")

    zooms_sorted = sorted(effective_geoms.keys(), reverse=True)

    logger.info("========== Mosaic Plan ==========")
    for zoom in zooms_sorted:
        n_parts = len(geometry_to_aoi_parts(effective_geoms[zoom]))
        logger.info("  z%-3s: %d part(s)", zoom, n_parts)
    logger.info("=================================")

    layer_paths: List[Path] = []

    for zoom in zooms_sorted:
        geom = effective_geoms[zoom]
        aoi_parts = geometry_to_aoi_parts(geom)

        if not aoi_parts:
            logger.info("zoom=%d: no polygon parts in effective geometry, skipping", zoom)
            continue

        download_cfg = cfg.download_config_for_zoom(zoom)

        for part_idx, part_points in enumerate(aoi_parts):
            layer_path = out_dir / f"layer_z{zoom}_part_{part_idx}.tif"

            logger.info("--- zoom=%d  part=%d  overwrite=%s ---", zoom, part_idx, download_cfg.overwrite)

            source = source_factory(zoom)
            download_tiles(
                source=source,
                aoi_points=part_points,
                out_tif=layer_path,
                config=download_cfg,
            )

            layer_paths.append(layer_path)

    if not layer_paths:
        raise RuntimeError("No layers were downloaded — nothing to merge.")

    if not cfg.merge:
        logger.info("Merge skipped. Layer directory: %s", out_dir)
        return out_dir

    final_path = out_dir / "mosaic_final.tif"
    logger.info("Merging %d layer(s) → %s", len(layer_paths), final_path)
    merge_tif_files(layer_paths, final_path)

    logger.info("Done. Final mosaic: %s", final_path)
    return final_path
