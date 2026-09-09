"""Parse Overpass OSM JSON → {category: [Shapely geometry]}."""
from __future__ import annotations

import logging

from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from ._tags import classify_way

logger = logging.getLogger(__name__)


def parse_osm_response(data: dict) -> dict[str, list]:
    nodes = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    categories = {}
    skipped = 0
    for el in data.get("elements", []):
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        category = classify_way(tags)
        if category is None:
            continue
        refs = el.get("nodes", [])
        coords = [nodes[n] for n in refs if n in nodes]
        if len(coords) < 2:
            skipped += 1
            continue
        is_closed = coords[0] == coords[-1]
        if is_closed and category == "natural_water" and len(coords) >= 4:
            try:
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = make_valid(geom)
                if geom.is_empty:
                    skipped += 1
                    continue
            except Exception as exc:
                logger.debug("Polygon build failed for way %s: %r", el.get("id"), exc)
                skipped += 1
                continue
        else:
            geom = LineString(coords)
        categories.setdefault(category, []).append(geom)

    total = sum(len(v) for v in categories.values())
    logger.info(
        "Parsed %d geometries across %d categories (%d skipped)",
        total, len(categories), skipped,
    )
    return categories
