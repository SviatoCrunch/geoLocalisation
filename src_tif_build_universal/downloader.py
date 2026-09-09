from __future__ import annotations

import itertools
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .buffer import TileEntry, WriteBuffer
from .config import DownloadConfig
from .sources.base import GridInfo, TileSource


_QUOTA_PHRASES = ("quota", "daily limit", "per day")

_TIF_PROFILE = {
    "driver": "GTiff",
    "count": 3,
    "dtype": "uint8",
    "compress": "deflate",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "interleave": "pixel",
    "BIGTIFF": "YES",
    "SPARSE_OK": "TRUE",
}

# How many futures to keep in flight at once: max_workers × _INFLIGHT_FACTOR.
# Large enough to keep the thread pool busy; small enough to cap RAM usage.
_INFLIGHT_FACTOR = 4


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _QUOTA_PHRASES)


def _fire(cb, *args) -> None:
    """Call a callback, swallowing any exception so the download never crashes."""
    if cb is not None:
        try:
            cb(*args)
        except Exception as e:
            logger.debug("Callback %r raised %r — ignored", cb, e)


def _validate_tile(
    job: tuple,
    result_job: tuple,
    arr: np.ndarray,
    window: Window,
    expected_shape: tuple,
    tif_width: int,
    tif_height: int,
) -> str | None:
    """Cheap mandatory checks on a tile before writing to disk.

    Returns an error string on failure, None if all checks pass.
    """
    if not isinstance(arr, np.ndarray):
        return f"not ndarray: {type(arr).__name__}"
    if arr.ndim != 3:
        return f"wrong ndim: {arr.ndim} (expected 3)"
    if arr.shape != expected_shape:
        return f"shape {arr.shape} != expected {expected_shape}"
    if arr.dtype != np.uint8:
        return f"dtype {arr.dtype} != uint8"
    if arr.size == 0:
        return "empty array"
    if result_job != job:
        return f"job mismatch: sent {job}, got {result_job}"
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    w = int(window.width)
    h = int(window.height)
    if w != expected_shape[2] or h != expected_shape[1]:
        return f"window size {w}×{h} != tile shape {expected_shape[2]}×{expected_shape[1]}"
    if col_off < 0 or row_off < 0 or col_off + w > tif_width or row_off + h > tif_height:
        return (
            f"window out of bounds: "
            f"col={col_off} row={row_off} w={w} h={h} "
            f"tif={tif_width}×{tif_height}"
        )
    return None


def _load_metadata(meta_path: Path) -> dict:
    bak = meta_path.with_suffix(".json.bak")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if bak.exists():
            try:
                return json.loads(bak.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return {}


def _save_metadata(meta_path: Path, payload: dict) -> None:
    tmp = meta_path.with_suffix(".json.tmp")
    bak = meta_path.with_suffix(".json.bak")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if meta_path.exists():
        meta_path.replace(bak)
    tmp.replace(meta_path)


def _open_or_create_tif(path: Path, grid: GridInfo):
    """Open existing result TIF for appending, or create a blank one."""
    profile = {
        **_TIF_PROFILE,
        "height": grid.height_px,
        "width": grid.width_px,
        "crs": grid.crs,
        "transform": grid.transform,
    }
    if not path.exists():
        with rasterio.open(path, "w", **profile):
            pass
    return rasterio.open(path, "r+")


def download_tiles(
    source: TileSource,
    aoi_points: list,
    out_tif: Union[str, Path],
    config: Union[DownloadConfig, None] = None,
) -> Path:
    """Download tiles from any TileSource into a GeoTIFF.

    Bounded submission: keeps at most max_workers × 4 futures alive at a
    time, so memory stays bounded regardless of grid size.

    Each tile is validated in RAM before being written to disk.

    Args:
        source: Any TileSource (EsriSource, GoogleSource, …).
        aoi_points: AOI vertices in (lat, lon) format.
        out_tif: Output GeoTIFF path.
        config: Download parameters. Uses DownloadConfig defaults if None.

    Returns:
        Path to the written GeoTIFF.
    """
    cfg = config if config is not None else DownloadConfig()
    cb = cfg.callbacks
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    meta_path = out_tif.with_suffix(".json")

    if cfg.overwrite:
        for p in (out_tif, meta_path):
            if p.exists():
                p.unlink()

    grid: GridInfo = source.build_grid(aoi_points)

    done_jobs: set = set()
    if not cfg.overwrite and meta_path.exists():
        saved = _load_metadata(meta_path)
        done_jobs = source.load_done_jobs(saved)
        logger.info("Resuming: %d / %d tiles done", len(done_jobs), len(grid.jobs))
        if done_jobs:
            _fire(cb.on_resume, len(done_jobs), len(grid.jobs))

    remaining = [j for j in grid.jobs if j not in done_jobs]

    logger.info(
        "Tiles   : total=%d  done=%d  remaining=%d",
        len(grid.jobs), len(done_jobs), len(remaining),
    )
    logger.info("Pixels  : %d × %d", grid.width_px, grid.height_px)

    if not remaining:
        logger.info("All tiles already downloaded: %s", out_tif)
        return out_tif

    meta_snapshot: dict = {
        **grid.meta,
        "done_tiles": source.serialise_done_jobs(done_jobs),
    }

    expected_shape = (3, grid.tile_px, grid.tile_px)
    max_in_flight = cfg.max_workers * _INFLIGHT_FACTOR
    buffer = WriteBuffer(flush_interval=cfg.flush_interval)
    tiles_since_meta_save = 0
    progress_count = 0
    total_remaining = len(remaining)

    dst = _open_or_create_tif(out_tif, grid)
    try:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            jobs_iter = iter(remaining)
            pending: dict = {}

            for job in itertools.islice(jobs_iter, max_in_flight):
                f = executor.submit(source.fetch_tile, job)
                pending[f] = job

            quota_hit = False

            with tqdm(total=total_remaining, desc="Downloading") as pbar:
                while pending:
                    done_futures, _ = futures_wait(pending, return_when=FIRST_COMPLETED)

                    for future in done_futures:
                        job = pending.pop(future)

                        # ── error handling ────────────────────────────────
                        try:
                            result_job, tile_arr = future.result()
                        except RuntimeError as exc:
                            if _is_quota_error(exc):
                                quota_hit = True
                                logger.warning("Daily quota limit reached — stopping gracefully.")
                                _fire(cb.on_error, job, exc)
                                progress_count += 1
                                _fire(cb.on_progress, progress_count, total_remaining)
                                pbar.update(1)
                                continue
                            logger.error("Job failed: job=%s  error=%r", job, exc)
                            _fire(cb.on_error, job, exc)
                            progress_count += 1
                            _fire(cb.on_progress, progress_count, total_remaining)
                            pbar.update(1)
                            raise
                        except Exception as exc:
                            logger.error("Job failed: job=%s  error=%r", job, exc)
                            _fire(cb.on_error, job, exc)
                            progress_count += 1
                            _fire(cb.on_progress, progress_count, total_remaining)
                            pbar.update(1)
                            raise

                        # ── validate in RAM before writing ────────────────
                        window = source.job_to_window(result_job)
                        err = _validate_tile(
                            job, result_job, tile_arr, window,
                            expected_shape, grid.width_px, grid.height_px,
                        )
                        entry = TileEntry(
                            job=result_job,
                            arr=tile_arr,
                            window=window,
                            validated=err is None,
                        )
                        if err:
                            logger.warning("Tile validation failed: job=%s — %s", job, err)
                            _fire(cb.on_error, job, ValueError(err))
                        else:
                            should_flush = buffer.add(entry)

                            if should_flush:
                                committed = buffer.flush_to(dst)
                                done_jobs.update(committed)
                                for j in committed:
                                    _fire(cb.on_download, j)
                                tiles_since_meta_save += len(committed)
                                if tiles_since_meta_save >= cfg.metadata_save_interval:
                                    meta_snapshot["done_tiles"] = source.serialise_done_jobs(done_jobs)
                                    _save_metadata(meta_path, meta_snapshot)
                                    tiles_since_meta_save = 0

                        # ── submit next job ───────────────────────────────
                        if not quota_hit:
                            next_job = next(jobs_iter, None)
                            if next_job is not None:
                                f = executor.submit(source.fetch_tile, next_job)
                                pending[f] = next_job

                        progress_count += 1
                        _fire(cb.on_progress, progress_count, total_remaining)
                        pbar.update(1)

                    if quota_hit:
                        for f in list(pending):
                            f.cancel()
                        break

    finally:
        committed = buffer.flush_to(dst)
        done_jobs.update(committed)
        for j in committed:
            _fire(cb.on_download, j)
        dst.close()
        meta_snapshot["done_tiles"] = source.serialise_done_jobs(done_jobs)
        _save_metadata(meta_path, meta_snapshot)
        logger.info("Progress saved: %d / %d tiles", len(done_jobs), len(grid.jobs))

    logger.info("Saved GeoTIFF: %s", out_tif)
    return out_tif
