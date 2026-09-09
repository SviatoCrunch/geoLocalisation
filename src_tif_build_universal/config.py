"""Download and mosaic configuration dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union


@dataclass
class EventCallbacks:
    """Optional hooks fired during download_tiles().

    All callbacks are optional (default None = no-op).
    Exceptions raised inside a callback are swallowed so they never
    interrupt the download loop.

    Signatures
    ----------
    on_resume(done, total)      — resuming from checkpoint; fired once
    on_progress(done, total)    — after every tile attempt (success or fail)
    on_download(job)            — tile successfully written to disk
    on_error(job, exc)          — tile failed (network, write, quota, …)
    """

    on_resume: Optional[Callable[[int, int], None]] = None
    on_progress: Optional[Callable[[int, int], None]] = None
    on_download: Optional[Callable[[tuple], None]] = None
    on_error: Optional[Callable[[tuple, Exception], None]] = None


@dataclass
class DownloadConfig:
    """Parameters for a single download_tiles() call."""

    max_workers: int = 16
    flush_interval: int = 100
    metadata_save_interval: int = 100
    overwrite: bool = False
    callbacks: EventCallbacks = field(default_factory=EventCallbacks)


@dataclass
class MosaicConfig:
    """Parameters for build_mosaic().

    ``download`` controls the inner download_tiles() call for each layer.
    ``overwrite_layers`` overrides ``download.overwrite`` per zoom level:
        False  — resume all layers
        True   — overwrite all layers
        [z, …] — overwrite only the listed zoom levels
    """

    download: DownloadConfig = field(default_factory=DownloadConfig)
    overwrite_layers: Union[bool, List[int]] = False
    merge: bool = True

    def download_config_for_zoom(self, zoom: int) -> DownloadConfig:
        """Return a DownloadConfig with overwrite set for the given zoom."""
        if isinstance(self.overwrite_layers, bool):
            overwrite = self.overwrite_layers
        else:
            overwrite = zoom in self.overwrite_layers
        return dataclasses.replace(self.download, overwrite=overwrite)
