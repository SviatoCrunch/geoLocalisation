"""Master taxonomy (22 trainable classes) + RGB palette -> train-id lookup.

Built from the verified SkyScenes.py `_CATEGORIES` order (23 classes); `unlabeled` (id 0,
colour 0,0,0) is VOID -> ignore_index=255, NOT a trainable class. So the SegFormer head has
22 logits (train ids 0..21 = building..terrain). Unknown colours also map to ignore.

External-validation targets (GT_flat_mask): road, railtrack, water. sky -> ignore there.
"""
from __future__ import annotations

from pathlib import Path

IGNORE_INDEX = 255
_PALETTE_YAML = Path(__file__).parent / "configs" / "full_taxonomy" / "carla_palette.yaml"

# SkyScenes.py order (23). Index here == original SkyScenes id.
_SKYSCENES_ORDER = [
    "unlabeled", "building", "fence", "other", "pedestrian", "pole", "roadline", "road",
    "sidewalk", "vegetation", "vehicles", "wall", "trafficsign", "sky", "ground", "bridge",
    "railtrack", "guardrail", "trafficlight", "static", "dynamic", "water", "terrain",
]
VOID = "unlabeled"

# trainable classes = everything except VOID, contiguous train ids 0..21
CLASSES: list[str] = [c for c in _SKYSCENES_ORDER if c != VOID]
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)                       # 22

# external-val target classes (scored on GT_flat_mask); everything else there = other/ignore
EXTERNAL_TARGETS = ["road", "railtrack", "water"]
# GT_flat_mask raw class name -> master name (Sky is dropped -> ignore)
GT_FLAT_MASK_MAP = {"Road": "road", "Railways": "railtrack", "Water": "water", "Sky": None}


def load_palette(path: str | Path = _PALETTE_YAML) -> dict[tuple, str]:
    """{(r,g,b): class_name} from the palette YAML (all 23, incl. unlabeled)."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {tuple(c["rgb"]): c["name"] for c in data["classes"]}


def color_to_trainid(path: str | Path = _PALETTE_YAML) -> dict[tuple, int]:
    """{(r,g,b): train_id}; VOID and any name not in CLASSES -> IGNORE_INDEX."""
    out = {}
    for rgb, name in load_palette(path).items():
        out[rgb] = CLASS_TO_ID.get(name, IGNORE_INDEX)
    return out


# RGB palette for QC colourisation (train id -> rgb); ignore -> black.
def id_palette(path: str | Path = _PALETTE_YAML) -> list[tuple]:
    pal = {name: rgb for rgb, name in load_palette(path).items()}
    return [pal[c] for c in CLASSES]
