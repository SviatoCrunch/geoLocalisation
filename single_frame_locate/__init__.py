"""single_frame_locate — faithful standalone of the prod service GPU localization path, for ONE frame.

See locate.py. The GPU compute (coarse e2c SuperVLAD + fine GPU MAGSAC++ over the packed pyramid
store) is vendored byte-for-byte from the live service branch server/e2c-live-snapshot; this package
only adds the single-frame orchestrator (the SiamEngine.localize slice without the service glue)."""
from __future__ import annotations

from .locate import SingleFrameLocator

__all__ = ["SingleFrameLocator"]
