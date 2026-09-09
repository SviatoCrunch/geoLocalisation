"""Write buffer — smart tile accumulator.

TileEntry carries the full context of one tile (array, window, job,
validation result, timestamp, retry count).  WriteBuffer accumulates
entries in RAM and writes only validated ones on flush.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from rasterio.windows import Window

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class TileEntry:
    """Context for a single downloaded tile, ready for writing.

    Fields
    ------
    job         — source job identifier (tile coordinates, etc.)
    arr         — pixel data, shape (3, H, W) uint8
    window      — rasterio Window: position inside the output TIF
    validated   — True if all mandatory checks passed; only validated
                  entries are written to disk by WriteBuffer.flush_to()
    timestamp   — monotonic time of entry creation (seconds)
    retry_count — how many fetch retries were needed (0 = first attempt)
    """

    job: tuple
    arr: np.ndarray
    window: Window
    validated: bool = True
    timestamp: float = field(default_factory=time.monotonic)
    retry_count: int = 0


class WriteBuffer:
    """Thread-safe write buffer for TileEntry objects.

    Workers call add() concurrently; flush_to() is called from the main
    thread.  The lock is held only while mutating _pending — actual
    rasterio writes happen outside the lock so add() is never blocked
    by disk I/O.

    Only entries with validated=True are written; invalid entries are
    counted and logged but skipped.
    """

    def __init__(self, flush_interval: int = 100) -> None:
        self._flush_interval = flush_interval
        self._pending: List[TileEntry] = []
        self._lock = threading.Lock()

    def add(self, entry: TileEntry) -> bool:
        """Append a TileEntry.

        Returns True when the flush threshold is reached, signalling the
        caller to call flush_to().
        """
        with self._lock:
            self._pending.append(entry)
            return len(self._pending) >= self._flush_interval

    def flush_to(self, dst) -> list:
        """Write pending validated tiles to *dst* (open rasterio dataset).

        Steals the pending list under the lock, then writes outside it.
        Entries with validated=False are skipped and logged as warnings.

        Returns a list of job identifiers that were actually written to disk.
        The caller should use this list — not buffer.add() — as the source
        of truth for updating done_jobs.
        """
        with self._lock:
            items = self._pending[:]
            self._pending.clear()

        committed = []
        skipped = 0
        for entry in items:
            if not entry.validated:
                logger.warning("Skipping unvalidated tile: job=%s", entry.job)
                skipped += 1
                continue
            dst.write(entry.arr, window=entry.window)
            committed.append(entry.job)

        if committed:
            logger.debug("Flushed %d tiles to disk (%d skipped)", len(committed), skipped)
        return committed

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)
