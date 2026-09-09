"""Google Map Tiles API satellite source.

Self-contained — no dependency on src_tif_build or sys.path manipulation.
All Google tile logic lives here: rate limiting, session management,
coordinate math, tile grid building, HTTP download.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import box

from ..viewport import create_google_session
from .base import GridInfo, TileSource

logger = logging.getLogger(__name__)

_TILE_SIZE = 256
_EARTH_RADIUS = 6378137.0
_ORIGIN_SHIFT = math.pi * _EARTH_RADIUS
_TILE_URL = "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}"

TileIndex = Tuple[int, int]
LatLon = Tuple[float, float]


class _TokenBucketRateLimiter:
    """Thread-safe token bucket with adaptive recovery (AIMD).

    On 429: rate halved, all threads paused for ``pause`` seconds.
    Every 60 s of quiet: rate increased by 10 % up to the original value.
    """

    _THROTTLE_COOLDOWN = 30.0
    _RECOVERY_INTERVAL = 60.0
    _RECOVERY_FACTOR = 1.1

    def __init__(self, rate: float, capacity: float = 20.0) -> None:
        self._rate = rate
        self._initial_rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._blocked_until = 0.0
        self._last_throttle = 0.0
        self._lock = threading.Lock()
        threading.Thread(
            target=self._recovery_loop, daemon=True, name="RateLimiterRecovery"
        ).start()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now < self._blocked_until:
                    wait = self._blocked_until - now
                else:
                    self._refill()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)

    def throttle(self, factor: float = 0.5, pause: float = 60.0) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._last_throttle < self._THROTTLE_COOLDOWN:
                return
            self._last_throttle = now
            self._rate = max(1.0, self._rate * factor)
            self._tokens = 0.0
            self._blocked_until = now + pause
            new_rate = self._rate
        logger.warning("Rate limiter throttle → %.1f rps, pause %.0fs", new_rate, pause)

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(
            self._capacity, self._tokens + (now - self._last_refill) * self._rate
        )
        self._last_refill = now

    def _recovery_loop(self) -> None:
        while True:
            time.sleep(self._RECOVERY_INTERVAL)
            with self._lock:
                if time.monotonic() - self._last_throttle < self._RECOVERY_INTERVAL:
                    continue
                old = self._rate
                self._rate = min(self._initial_rate, self._rate * self._RECOVERY_FACTOR)
                new = self._rate
            if new != old:
                logger.info("Rate limiter recover → %.1f rps", new)


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> TileIndex:
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_mercator_origin(x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float, float]:
    resolution = 2 * math.pi * _EARTH_RADIUS / (_TILE_SIZE * 2 ** zoom)
    west = x_tile * _TILE_SIZE * resolution - _ORIGIN_SHIFT
    north = _ORIGIN_SHIFT - y_tile * _TILE_SIZE * resolution
    return west, north, resolution


def _tile_to_lonlat_box(x: int, y: int, zoom: int, to_latlon: Transformer):
    west_m, north_m, res = _tile_to_mercator_origin(x, y, zoom)
    size_m = _TILE_SIZE * res
    w_lon, n_lat = to_latlon.transform(west_m, north_m)
    e_lon, s_lat = to_latlon.transform(west_m + size_m, north_m - size_m)
    return box(w_lon, s_lat, e_lon, n_lat)


def _estimate_tile_grid(aoi_points: List[LatLon], zoom: int) -> dict:
    tiles = [_latlon_to_tile(lat, lon, zoom) for lat, lon in aoi_points]
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    return {
        "zoom": zoom,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width_tiles": w,
        "height_tiles": h,
        "total_requests": w * h,
        "width_px": w * _TILE_SIZE,
        "height_px": h * _TILE_SIZE,
    }


def _build_tile_jobs(info: dict, zoom: int, clip_polygon=None) -> List[TileIndex]:
    all_jobs = [
        (x, y)
        for y in range(info["min_y"], info["max_y"] + 1)
        for x in range(info["min_x"], info["max_x"] + 1)
    ]
    if clip_polygon is None:
        return all_jobs
    to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return [
        (x, y)
        for x, y in all_jobs
        if _tile_to_lonlat_box(x, y, zoom, to_latlon).intersects(clip_polygon)
    ]


class _GoogleTileDownloader:
    """Manages a Google Map Tiles API session and fetches individual tiles.

    Thread-safe: fetch_tile() may be called concurrently from multiple threads.
    """

    def __init__(
        self,
        api_key: str,
        zoom: int,
        max_retries: int = 5,
        max_workers: int = 32,
        requests_per_second: float = 80.0,
    ) -> None:
        self._api_key = api_key
        self._zoom = zoom
        self._max_retries = max_retries
        self._max_workers = max_workers
        self._session_lock = threading.Lock()
        self._session_token = create_google_session(api_key)
        self._thread_local = threading.local()
        self.quota_exhausted = threading.Event()
        self._rate_limiter = _TokenBucketRateLimiter(rate=requests_per_second)

    def refresh_session(self) -> str:
        with self._session_lock:
            self._session_token = create_google_session(self._api_key)
            return self._session_token

    def get_session_token(self) -> str:
        with self._session_lock:
            return self._session_token

    def get_thread_session(self) -> requests.Session:
        if not hasattr(self._thread_local, "session"):
            http = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self._max_workers,
                pool_maxsize=self._max_workers,
                max_retries=0,
            )
            http.mount("https://", adapter)
            self._thread_local.session = http
        return self._thread_local.session

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            return response.json()["error"]["message"]
        except Exception:
            return response.text[:500]

    def fetch_tile(self, x: int, y: int) -> Tuple[int, int, np.ndarray]:
        if self.quota_exhausted.is_set():
            raise RuntimeError("Google daily quota already exhausted")
        url = _TILE_URL.format(z=self._zoom, x=x, y=y)
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                http = self.get_thread_session()
                token = self.get_session_token()
                self._rate_limiter.acquire()
                response = http.get(
                    url,
                    params={"session": token, "key": self._api_key},
                    timeout=(10, 30),
                )
                if response.status_code == 429:
                    msg = self._extract_error_message(response)
                    if "per day" in msg:
                        if not self.quota_exhausted.is_set():
                            self.quota_exhausted.set()
                            logger.error(
                                "Google daily quota exceeded: z=%d x=%d y=%d attempt=%d msg=%s",
                                self._zoom, x, y, attempt, msg,
                            )
                        raise RuntimeError("Google daily quota already exhausted")
                    logger.warning(
                        "Google rate limited: z=%d x=%d y=%d attempt=%d msg=%s",
                        self._zoom, x, y, attempt, msg,
                    )
                    self._rate_limiter.throttle()
                    continue
                if response.status_code == 403:
                    msg = self._extract_error_message(response)
                    logger.warning(
                        "Google auth/forbidden: z=%d x=%d y=%d attempt=%d msg=%s",
                        self._zoom, x, y, attempt, msg,
                    )
                    self.refresh_session()
                    time.sleep(1)
                    continue
                if response.status_code >= 400:
                    logger.error(
                        "HTTP error: status=%d z=%d x=%d y=%d attempt=%d body=%s",
                        response.status_code, self._zoom, x, y, attempt,
                        response.text[:500],
                    )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "image" not in content_type:
                    raise ValueError(
                        f"Non-image response: {content_type} — {response.text[:500]}"
                    )
                encoded = np.frombuffer(response.content, dtype=np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Tile decode error")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return x, y, rgb.transpose(2, 0, 1).astype(np.uint8)
            except Exception as exc:
                last_error = exc
                txt = str(exc)
                if "per day" in txt or "Google daily quota already exhausted" in txt:
                    raise RuntimeError("Google daily quota already exhausted")
                logger.warning(
                    "Tile retry: z=%d x=%d y=%d attempt=%d/%d error=%r",
                    self._zoom, x, y, attempt, self._max_retries, exc,
                )
                time.sleep(min(2 ** attempt, 30) + np.random.uniform(0.0, 1.0))
        raise RuntimeError(f"Failed tile z={self._zoom}, x={x}, y={y}") from last_error


class GoogleSource(TileSource):
    """Google Map Tiles API satellite source.

    Args:
        api_key: Google Cloud API key with Map Tiles API enabled.
        zoom: Tile zoom level.
        max_workers: Parallel download threads.
        max_retries: Per-tile retry attempts.
        requests_per_second: Rate limit target.
        clip_polygon: Optional Shapely polygon (lon, lat) to filter tiles.
    """

    def __init__(
        self,
        api_key: str,
        zoom: int,
        max_workers: int = 32,
        max_retries: int = 5,
        requests_per_second: float = 80.0,
        clip_polygon=None,
    ) -> None:
        self._api_key = api_key
        self._zoom = zoom
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._requests_per_second = requests_per_second
        self._clip_polygon = clip_polygon
        self._downloader = None
        self._min_x = 0
        self._min_y = 0

    def build_grid(self, aoi_points: list) -> GridInfo:
        info = _estimate_tile_grid(aoi_points, self._zoom)
        jobs = _build_tile_jobs(info, self._zoom, self._clip_polygon)
        self._min_x = info["min_x"]
        self._min_y = info["min_y"]
        self._downloader = _GoogleTileDownloader(
            api_key=self._api_key,
            zoom=self._zoom,
            max_retries=self._max_retries,
            max_workers=self._max_workers,
            requests_per_second=self._requests_per_second,
        )
        west, north, resolution = _tile_to_mercator_origin(
            self._min_x, self._min_y, self._zoom
        )
        return GridInfo(
            width_px=info["width_px"],
            height_px=info["height_px"],
            transform=Affine(resolution, 0, west, 0, -resolution, north),
            crs="EPSG:3857",
            tile_px=_TILE_SIZE,
            jobs=jobs,
            meta={
                **info,
                "source": "google",
                "google_tile_size": _TILE_SIZE,
                "aoi_points": aoi_points,
                "clipped": self._clip_polygon is not None,
            },
        )

    def fetch_tile(self, job) -> tuple:
        if self._downloader is None:
            raise RuntimeError("build_grid() must be called before fetch_tile()")
        x, y = job
        _x, _y, arr = self._downloader.fetch_tile(x, y)
        return job, arr

    def job_to_window(self, job) -> Window:
        x, y = job
        return Window(
            col_off=(x - self._min_x) * _TILE_SIZE,
            row_off=(y - self._min_y) * _TILE_SIZE,
            width=_TILE_SIZE,
            height=_TILE_SIZE,
        )

    def load_done_jobs(self, meta: dict) -> set:
        return {tuple(t) for t in meta.get("done_tiles", [])}

    def serialise_done_jobs(self, done: set) -> list:
        return [list(t) for t in done]

    @property
    def quota_exhausted(self):
        if self._downloader:
            return self._downloader.quota_exhausted
        return None
