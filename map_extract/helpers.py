"""I/O + AOI helpers (vendored from RevisitAnything's ``tif_dino_extract``).

``_PREPROCESS`` (ImageNet normalise) needs torchvision; the rest lazily import
rasterio / pyproj / cv2 / shapely / rio-cogeo only when their feature is used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torchvision.transforms as T

_PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _is_remote_uri(s: str) -> bool:
    return bool(re.match(r"^(s3|gs|https?)://", s))


def _parse_image_stem(stem: str):
    """Parse ``{idx}_{lat}_{lon}`` stem → ``(idx, lat, lon)`` (lat/lon None if no match)."""
    parts = stem.split("_")
    try:
        return parts[0], float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return stem, None, None


def _ensure_cog(tif_path: Path) -> Path:
    """Return a valid COG path for *tif_path*, converting if needed (best-effort)."""
    try:
        from rio_cogeo.cogeo import cog_validate, cog_translate
        from rio_cogeo.profiles import cog_profiles
    except ImportError:
        print("[warn] rio-cogeo not installed — skipping COG check. "
              "Install with: pip install rio-cogeo")
        return tif_path

    is_valid, _, _ = cog_validate(str(tif_path))
    if is_valid:
        print(f"[cog] {tif_path.name} is already a valid COG.")
        return tif_path

    cog_path = tif_path.with_stem(tif_path.stem + "_cog")
    print(f"[cog] Converting to COG → {cog_path.name} ...")
    cog_translate(str(tif_path), str(cog_path), cog_profiles.get("deflate"),
                  overview_resampling="average", quiet=False)
    print(f"[cog] Done: {cog_path}")
    return cog_path


def _load_aoi_from_kmz(kmz_path, name_filter: str | None = None) -> np.ndarray:
    """Parse KMZ Point placemarks → convex-hull polygon as (N,2) [[lat,lon]] array."""
    import zipfile
    import xml.etree.ElementTree as ET
    from shapely.geometry import MultiPoint

    KML_NS = "{http://www.opengis.net/kml/2.2}"
    points: list[tuple[float, float]] = []

    with zipfile.ZipFile(kmz_path) as z:
        kml_files = [n for n in z.namelist() if n.endswith(".kml")]
        if not kml_files:
            raise ValueError(f"No KML file found in {kmz_path}")
        with z.open(kml_files[0]) as f:
            root = ET.parse(f).getroot()

    for placemark in root.iter(f"{KML_NS}Placemark"):
        if name_filter:
            name_el = placemark.find(f"{KML_NS}name")
            if name_el is None or name_filter.lower() not in (name_el.text or "").lower():
                continue
        pt = placemark.find(f".//{KML_NS}Point")
        if pt is None:
            continue
        coords_el = pt.find(f"{KML_NS}coordinates")
        if coords_el is None or not coords_el.text:
            continue
        for tok in coords_el.text.strip().split():
            parts = tok.split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                points.append((lon, lat))

    if len(points) < 3:
        raise ValueError(
            f"Need ≥3 Point placemarks for an AOI polygon, got {len(points)}"
            + (f" (filter: '{name_filter}')" if name_filter else ""))

    hull = MultiPoint(points).convex_hull
    return np.array([[lat, lon] for lon, lat in hull.exterior.coords])


def _save_aoi_preview(tif_path, aoi_points_latlon, out_path,
                      max_px: int = 1024, pad_frac: float = 0.15) -> None:
    """Save a downsampled TIF crop around the AOI with the hull drawn (best-effort)."""
    try:
        import cv2
        import rasterio
        import rasterio.windows
        from pyproj import Transformer
        from rasterio.enums import Resampling

        pts_ll = np.asarray(aoi_points_latlon, dtype=float)
        lats, lons = pts_ll[:, 0], pts_ll[:, 1]
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = tr.transform(lons, lats)
        xs, ys = np.asarray(xs), np.asarray(ys)
        padx = max((xs.max() - xs.min()) * pad_frac, 100.0)
        pady = max((ys.max() - ys.min()) * pad_frac, 100.0)

        with rasterio.open(tif_path) as src:
            b = src.bounds
            left = max(xs.min() - padx, b.left)
            right = min(xs.max() + padx, b.right)
            bottom = max(ys.min() - pady, b.bottom)
            top = min(ys.max() + pady, b.top)
            window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
            if window.width <= 0 or window.height <= 0:
                print("[aoi] preview skipped: AOI outside TIF bounds")
                return
            scale = max_px / max(window.width, window.height)
            out_w = max(1, int(window.width * scale))
            out_h = max(1, int(window.height * scale))
            img = src.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w),
                           resampling=Resampling.bilinear).transpose(1, 2, 0)

        bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        def to_px(x, y):
            return (int(round((x - left) / (right - left) * out_w)),
                    int(round((top - y) / (top - bottom) * out_h)))

        poly = np.array([to_px(x, y) for x, y in zip(xs, ys)], dtype=np.int32)
        overlay = bgr.copy()
        cv2.fillPoly(overlay, [poly], (0, 165, 255))
        cv2.addWeighted(overlay, 0.25, bgr, 0.75, 0, bgr)
        cv2.polylines(bgr, [poly], isClosed=True, color=(0, 140, 255),
                      thickness=2, lineType=cv2.LINE_AA)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)
        print(f"[aoi] preview saved → {out_path}  ({out_w}x{out_h})")
    except Exception as e:
        print(f"[aoi] preview skipped: {type(e).__name__}: {e}")
