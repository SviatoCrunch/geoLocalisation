"""Native-crop geometry + cell ordering for the ``native_hierarchical`` map source.

All bounds are in the map's projected CRS (EPSG:3857), in CRS units (metres × 1/cos(lat)).
A base 1000 m tile is centred at ``(cx, cy)`` with CRS half-side ``half = 1000·true_scale/2``.

Conventions (pinned here and enforced by native_map_pyramid/tests/test_geometry_order.py):
  * A cell/crop at grid position ``(i, j)`` uses ``i`` = ROW from the NORTH (top, i=0 → maxy side)
    and ``j`` = COL from the WEST (left, j=0 → minx side). This matches the legacy token layout:
    image row 0 = north (top), col 0 = west (left); ``region_ids`` flattens row-major.
  * Flat index of an n×n grid cell is row-major: ``flat = i*n + j``.
  * n=8 comes from splitting each 250 m (n=4) parent DINO grid into 2×2 quadrants ``(dy, dx)``
    (dy=0 → north half, dx=0 → west half). The global n=8 index is
        global_row = 2*i + dy ,  global_col = 2*j + dx ,  flat = global_row*8 + global_col .
No approximate division of an already-resized RGB tile is used — every crop is a CRS bbox read
straight from the COG at its exact bounds.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropBounds:
    """One native crop: its grid position and exact EPSG:3857 bounds (CRS units)."""
    i: int              # row from north (0 = northmost)
    j: int              # col from west  (0 = westmost)
    flat: int           # row-major flat index within its n×n grid
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def center(self) -> tuple:
        return (0.5 * (self.minx + self.maxx), 0.5 * (self.miny + self.maxy))


def base_bounds(center_x: float, center_y: float, tile_true_m: float,
                true_scale: float) -> tuple:
    """Base tile bbox in CRS units from its centre + TRUE footprint (m) and ``true_scale``."""
    half = 0.5 * tile_true_m * true_scale
    return (center_x - half, center_y - half, center_x + half, center_y + half)


def split_bounds(base: tuple, n: int) -> list:
    """Split a base bbox into an n×n grid of non-overlapping crops (row-major, i=north/j=west).

    Returns ``n²`` :class:`CropBounds` in flat row-major order (flat == index in the list)."""
    minx, miny, maxx, maxy = base
    cw = (maxx - minx) / n          # CRS cell width  (east-west)
    ch = (maxy - miny) / n          # CRS cell height (north-south)
    out: list[CropBounds] = []
    for i in range(n):              # i = row from NORTH → y descends from maxy
        y1 = maxy - i * ch
        y0 = maxy - (i + 1) * ch
        for j in range(n):          # j = col from WEST → x ascends from minx
            x0 = minx + j * cw
            x1 = minx + (j + 1) * cw
            out.append(CropBounds(i=i, j=j, flat=i * n + j, minx=x0, miny=y0, maxx=x1, maxy=y1))
    return out


def n8_flat_index(parent_i: int, parent_j: int, dy: int, dx: int) -> int:
    """Global n=8 flat index of quadrant ``(dy,dx)`` of the 250 m parent ``(parent_i,parent_j)``.

    ``global_row = 2*parent_i + dy`` (dy=0 north), ``global_col = 2*parent_j + dx`` (dx=0 west);
    row-major flatten over the 8×8 grid."""
    if not (0 <= parent_i < 4 and 0 <= parent_j < 4):
        raise ValueError(f"parent (i,j) must be in 0..3, got ({parent_i},{parent_j})")
    if dy not in (0, 1) or dx not in (0, 1):
        raise ValueError(f"quadrant (dy,dx) must be in {{0,1}}, got ({dy},{dx})")
    global_row = 2 * parent_i + dy
    global_col = 2 * parent_j + dx
    return global_row * 8 + global_col


def quadrant_token_slices(grid_h: int, grid_w: int) -> list:
    """2×2 quadrant token slices of a (grid_h, grid_w) DINO grid → 4×(dy,dx,rows,cols).

    ``dy=0`` is the NORTH (top) half (rows [0:h/2]); ``dx=0`` is the WEST (left) half.
    Requires even grid dims so the 30×30 (from 60×60) split is exact."""
    if grid_h % 2 or grid_w % 2:
        raise ValueError(f"grid {grid_h}×{grid_w} must have even dims for a 2×2 quadrant split")
    hh, ww = grid_h // 2, grid_w // 2
    out = []
    for dy in (0, 1):
        for dx in (0, 1):
            out.append((dy, dx, slice(dy * hh, (dy + 1) * hh), slice(dx * ww, (dx + 1) * ww)))
    return out


# Canonical view plan for one base 1000 m tile: 1×1000 + 4×500 + 16×250 = 21 DINO views.
VIEW_SCALES_M = (1000.0, 500.0, 250.0)          # n=1, n=2, n=4 native crops
VIEW_GRID_N = {1000.0: 1, 500.0: 2, 250.0: 4}
N_VIEWS_PER_TILE = 1 + 4 + 16                     # = 21
CELL_COUNTS = {8: 64, 4: 16, 2: 4, 1: 1}          # per scale n
N_CELLS_TOTAL = 64 + 16 + 4 + 1                   # = 85


def native_view_plan(center_x: float, center_y: float, tile_true_m: float,
                     true_scale: float) -> dict:
    """Full crop plan for a base tile → {footprint_m: [CropBounds,...]} for the 21 DINO views.

    n=8 cells are NOT separate crops — they are quadrants of the n=4 (250 m) views, handled at
    aggregation time via :func:`quadrant_token_slices` + :func:`n8_flat_index`."""
    base = base_bounds(center_x, center_y, tile_true_m, true_scale)
    return {s: split_bounds(base, VIEW_GRID_N[s]) for s in VIEW_SCALES_M}
