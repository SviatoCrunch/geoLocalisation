"""OSM tag constants, band ordering, and Overpass query builder."""
from __future__ import annotations

BAND_ORDER: list[str] = [
    "waterway",
    "natural_water",
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "road",
    "track",
]

_HIGHWAY_VALUES: frozenset[str] = frozenset(
    [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "road",
        "track",
    ]
)


def classify_way(tags: dict) -> str | None:
    hw = tags.get("highway")
    if hw in _HIGHWAY_VALUES:
        return hw
    if "waterway" in tags:
        return "waterway"
    if tags.get("natural") == "water":
        return "natural_water"
    return None


def build_overpass_query(
    south: float, west: float, north: float, east: float, timeout: int = 120
) -> str:
    bbox = f"{south},{west},{north},{east}"
    hw = "|".join(sorted(_HIGHWAY_VALUES))
    return (
        f'[out:json][timeout:{timeout}];\n'
        f'(\n  way["highway"~"^({hw})$"]({bbox});\n'
        f'  way["waterway"]({bbox});\n'
        f'  way["natural"="water"]({bbox});\n'
        f');\nout body;\n>;\nout skel qt;\n'
    )
