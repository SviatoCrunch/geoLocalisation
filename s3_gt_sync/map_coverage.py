"""
map_coverage.py — check how many GT frames fall inside a raster map (GeoTIFF/COG).

Pure geometry, no S3. For every GT frame it parses ``lat``/``lon`` from the file
name (the ``..._<lat>_<lon>.<ext>`` convention produced by ``pull_from_s3``),
transforms WGS84 -> the raster's CRS and classifies the point:

  * ``inside``  — within the raster bbox (and, when --check-nodata, on a valid,
                  non-masked pixel)
  * ``nodata``  — inside the bbox but landing on the raster's nodata / masked
                  border (typical for an oblique or clipped COG)
  * ``outside`` — beyond the raster bbox (``dist`` = distance past the nearest
                  edge, in the CRS's units)
  * ``bad``     — file name without parseable coordinates

Distances/margins are reported in the raster CRS units (metres for a projected
UTM map; degrees for a geographic map).

Depends on ``rasterio`` and ``pyproj`` (imported lazily so the rest of the
package keeps working without them).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def parse_gt(filename: str):
    """Return ``(lat, lon)`` from ``..._<lat>_<lon>.<ext>``, or ``None``."""
    parts = Path(filename).stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return float(parts[-2]), float(parts[-1])
    except ValueError:
        return None


@dataclass
class FrameCoverage:
    """Per-frame classification against the raster."""
    filename: str
    lat: float | None
    lon: float | None
    x: float | None          # in raster CRS
    y: float | None
    status: str              # 'inside' | 'nodata' | 'outside' | 'bad'
    dist_out: float = 0.0    # distance past nearest edge (outside), CRS units
    margin: float = 0.0      # distance to nearest edge (inside/nodata), CRS units


@dataclass
class CoverageReport:
    """Whole-directory coverage summary."""
    tif: str
    frames_dir: str
    crs: str
    units: str
    bbox: tuple[float, float, float, float]   # left, bottom, right, top
    width: int
    height: int
    frames: list[FrameCoverage] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.frames)

    @property
    def n_parseable(self) -> int:
        return sum(1 for f in self.frames if f.status != "bad")

    def by_status(self, status: str) -> list[FrameCoverage]:
        return [f for f in self.frames if f.status == status]

    def counts(self) -> dict[str, int]:
        out = {"inside": 0, "nodata": 0, "outside": 0, "bad": 0}
        for f in self.frames:
            out[f.status] += 1
        return out

    def to_dict(self) -> dict:
        return {
            "tif": self.tif,
            "frames_dir": self.frames_dir,
            "crs": self.crs,
            "units": self.units,
            "bbox": list(self.bbox),
            "width": self.width,
            "height": self.height,
            "n_total": self.n_total,
            "n_parseable": self.n_parseable,
            "counts": self.counts(),
            "frames": [
                {
                    "filename": f.filename,
                    "lat": f.lat, "lon": f.lon,
                    "x": f.x, "y": f.y,
                    "status": f.status,
                    "dist_out": f.dist_out, "margin": f.margin,
                }
                for f in self.frames
            ],
        }


def coverage_report(
    tif_path: str,
    frames_dir: str,
    frame_glob: str = "*.jpg",
    check_nodata: bool = True,
) -> CoverageReport:
    """Classify every GT frame in ``frames_dir`` against the raster ``tif_path``.

    When ``check_nodata`` is True a point inside the bbox is additionally sampled
    against the raster's valid-data mask (``read_masks``, which folds in nodata
    and alpha) and demoted to ``nodata`` if it lands on a masked pixel.
    """
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    with rasterio.open(tif_path) as ds:
        b = ds.bounds
        units = ds.crs.linear_units if ds.crs.is_projected else "degrees"
        to_ds = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)

        report = CoverageReport(
            tif=str(tif_path), frames_dir=str(frames_dir),
            crs=str(ds.crs), units=units,
            bbox=(float(b.left), float(b.bottom), float(b.right), float(b.top)),
            width=int(ds.width), height=int(ds.height),
        )

        for f in sorted(Path(frames_dir).glob(frame_glob)):
            gt = parse_gt(f.name)
            if gt is None:
                report.frames.append(FrameCoverage(f.name, None, None, None, None, "bad"))
                continue
            lat, lon = gt
            x, y = to_ds.transform(lon, lat)

            if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                dx = max(b.left - x, 0.0, x - b.right)
                dy = max(b.bottom - y, 0.0, y - b.top)
                report.frames.append(FrameCoverage(
                    f.name, lat, lon, x, y, "outside",
                    dist_out=(dx * dx + dy * dy) ** 0.5,
                ))
                continue

            margin = min(x - b.left, b.right - x, y - b.bottom, b.top - y)
            status = "inside"
            if check_nodata:
                r, c = ds.index(x, y)
                if 0 <= r < ds.height and 0 <= c < ds.width:
                    m = ds.read_masks(1, window=Window(c, r, 1, 1))
                    if m.size and m[0, 0] == 0:
                        status = "nodata"
            report.frames.append(FrameCoverage(
                f.name, lat, lon, x, y, status, margin=margin,
            ))

    return report
