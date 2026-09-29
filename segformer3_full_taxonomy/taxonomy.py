"""Selectable taxonomy: full 22-class SkyScenes, or a coarse 5-class targets set.

Presets:
  * ``full22``   — SkyScenes.py `_CATEGORIES` minus `unlabeled` (void->ignore). 22 trainable.
  * ``targets5`` — collapse to what we actually validate on: ``other, road, railtrack, water,
    sky``. Every SkyScenes class except road/railtrack/water/sky folds into ``other``; void ->
    ignore. Coarser labels = more pixels/class + a train objective aligned with the external
    targets → better generalisation of road/railtrack/water to the real (analog-FPV) frames.

Call ``set_preset(name)`` once at startup (train.py does this) BEFORE building datasets/model;
all other modules read ``taxonomy.NUM_CLASSES`` etc. at runtime so the switch propagates.

External-validation targets stay road/railtrack/water (GT_flat_mask); sky ignored there.
"""
from __future__ import annotations

from pathlib import Path

IGNORE_INDEX = 255
_PALETTE_YAML = Path(__file__).parent / "configs" / "full_taxonomy" / "carla_palette.yaml"

_SKYSCENES_ORDER = [
    "unlabeled", "building", "fence", "other", "pedestrian", "pole", "roadline", "road",
    "sidewalk", "vegetation", "vehicles", "wall", "trafficsign", "sky", "ground", "bridge",
    "railtrack", "guardrail", "trafficlight", "static", "dynamic", "water", "terrain",
]
VOID = "unlabeled"
_KEEP5 = {"road", "railtrack", "water", "sky"}

EXTERNAL_TARGETS = ["road", "railtrack", "water"]
GT_FLAT_MASK_MAP = {"Road": "road", "Railways": "railtrack", "Water": "water", "Sky": None}

_PRESETS = {
    "full22": [c for c in _SKYSCENES_ORDER if c != VOID],
    "targets5": ["other", "road", "railtrack", "water", "sky"],
}

# active state — set by set_preset()
PRESET: str = ""
CLASSES: list[str] = []
CLASS_TO_ID: dict[str, int] = {}
NUM_CLASSES: int = 0
_GROUP: dict[str, str] = {}   # SkyScenes class name -> active class name


def _group_for(preset: str) -> dict[str, str]:
    names = [c for c in _SKYSCENES_ORDER if c != VOID]
    if preset == "full22":
        return {c: c for c in names}
    return {c: (c if c in _KEEP5 else "other") for c in names}   # targets5


def set_preset(name: str) -> None:
    global PRESET, CLASSES, CLASS_TO_ID, NUM_CLASSES, _GROUP
    if name not in _PRESETS:
        raise ValueError(f"unknown taxonomy preset {name!r} (have {list(_PRESETS)})")
    PRESET = name
    CLASSES = list(_PRESETS[name])
    CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
    NUM_CLASSES = len(CLASSES)
    _GROUP = _group_for(name)


set_preset("full22")   # default; train.py overrides via --classes


def load_palette(path=_PALETTE_YAML) -> dict[tuple, str]:
    """{(r,g,b): SkyScenes class_name} from the palette YAML (all 23)."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {tuple(c["rgb"]): c["name"] for c in data["classes"]}


def color_to_trainid(path=_PALETTE_YAML) -> dict[tuple, int]:
    """{(r,g,b): active train_id}; void/unknown -> IGNORE_INDEX. Honours the active preset."""
    out = {}
    for rgb, name in load_palette(path).items():
        out[rgb] = IGNORE_INDEX if name == VOID else CLASS_TO_ID.get(_GROUP.get(name), IGNORE_INDEX)
    return out


def id_palette(path=_PALETTE_YAML) -> list[tuple]:
    """RGB per active class for QC (grouped 'other' -> grey)."""
    pal = {name: rgb for rgb, name in load_palette(path).items()}
    return [pal.get(c, (80, 80, 80)) for c in CLASSES]
