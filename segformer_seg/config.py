"""Canonical class taxonomy + palette + name reconciliation for the SegFormer trainer.

ONE place that fixes the label space so every other file (manifest builder, label-map
composer, trainer, evaluator) agrees. The taxonomy is the **6-class union** the user chose:
the four classes the server GT masks actually ship (Railways/Road/Sky/Water) plus
``bridge`` and ``tower``, which exist ONLY in the newly hand-labelled COCO set. Server
images therefore carry no bridge/tower pixels -> those are background there, which is
correct (the class is simply unlabelled in that source).

Index 0 is an explicit ``background`` ("none of the above") class, NOT ignore: the drone
frames are region-labelled fairly exhaustively, so unlabelled pixels are a real negative.
Flip to ignore at train time via ``--background-mode ignore`` if you'd rather not supervise
them.
"""
from __future__ import annotations

# canonical id order — index == class id used in every label map
CLASSES: list[str] = ["background", "bridge", "railway", "road", "sky", "tower", "water"]
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
BACKGROUND_ID = 0
IGNORE_INDEX = 255  # used only when --background-mode ignore

# Map every raw annotation name (server PNG suffix, server Label-Studio, new COCO) to a
# canonical class — or None to DROP it. Server LS ships Building/Target, which are NOT in
# our target taxonomy (Building != tower, Target != bridge) and were never rasterised to
# PNG anyway; we drop them explicitly rather than silently mis-map.
NAME_ALIASES: dict[str, str | None] = {
    "railways": "railway", "railway": "railway",
    "road": "road",
    "sky": "sky",
    "water": "water",
    "bridge": "bridge",
    "tower": "tower",
    "building": None,   # server LS only — dropped
    "target": None,     # server LS only — dropped
}

# Paint order for overlap resolution: painted low -> high, so classes LATER in this list
# win where two masks cover the same pixel. Thin/structural classes (railway, bridge,
# tower) sit on top of large area classes (sky, water, road); background is the base.
PAINT_ORDER: list[str] = ["sky", "water", "road", "railway", "bridge", "tower"]

# RGB palette for QC visualisations (index == class id).
PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),        # background
    (255, 128, 0),    # bridge
    (128, 0, 128),    # railway
    (128, 64, 0),     # road
    (135, 206, 235),  # sky
    (255, 0, 0),      # tower
    (0, 0, 255),      # water
]


def canonical(name: str) -> str | None:
    """Raw annotation name -> canonical class name (or None to drop). Case-insensitive."""
    return NAME_ALIASES.get(name.strip().lower(), None)
