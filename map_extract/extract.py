"""GeoTIFF/query-image → DINO features → HDF5 (GPU-efficient), with --true_meters.

Isolated port of RevisitAnything's ``tif_dino_extract_gpu.py`` (background tile prefetch,
batched on-device post-processing) wired to the vendored siblings in this package. The H5
schema is unchanged, so it is a drop-in replacement for the map/gallery build.

New: ``--true_meters`` makes ``--tile_size_m`` / ``--stride_m`` TRUE ground metres. The
sampler grid + read windows live in EPSG:3857 (unit = metre × 1/cos(lat)), so both sizes
are scaled up by ``1/cos(centre_lat)``; the stored ``window_size_m`` is divided back to
record the real footprint. Off by default (byte-identical legacy behaviour).

Run::

    python -m map_extract.extract --tif <tif|s3://…> --out map.h5 \
        --true_meters --tile_size_m 1000 --stride_m 250 --levels 1 --output_px 224
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .dino import build_dino_extractor, RandomProjector
from .geometry import mercator_true_scale, scaled_pyramid_config
from .helpers import (_PREPROCESS, _ensure_cog, _is_remote_uri, _load_aoi_from_kmz,
                      _parse_image_stem, _save_aoi_preview)
from .pyramid import PyramidContextSampler

_PREFETCH_FACTOR = 3


# ── Batching / feature extraction (GPU-efficient) ─────────────────────────────

def _tile_to_batch(images, device: str, patch_size: int = 14, preprocess=None):
    pp = preprocess or _PREPROCESS
    tensors = [pp(img) for img in images]
    _, h, w = tensors[0].shape
    h_r, w_r = h // patch_size, w // patch_size
    tensors = [TF.center_crop(t, [h_r * patch_size, w_r * patch_size]) for t in tensors]
    batch = torch.stack(tensors)
    if torch.device(device).type == "cuda":
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        batch = batch.to(device)
    return batch, h_r, w_r


def _extract_level_features(extractor, batch, h_r, w_r, projector=None):
    with torch.no_grad():
        feat = extractor(batch)
        if projector is not None:
            feat = projector(feat)
    B, desc_dim = feat.shape[0], feat.shape[-1]
    f = feat.reshape(B, h_r, w_r, desc_dim).permute(0, 3, 1, 2)
    f = F.normalize(f, dim=1)
    arr = f.cpu().numpy()
    return [arr[i:i + 1] for i in range(B)]


def _flush_buffer(buffer, dino, f, device, patch_size=14, projector=None,
                  preprocess=None, true_scale: float = 1.0):
    all_images, meta = [], []
    for tile in buffer:
        for lvl_idx in sorted(tile.levels.keys()):
            all_images.append(tile.levels[lvl_idx].image)
            meta.append((tile, lvl_idx, tile.levels[lvl_idx]))

    batch, h_r, w_r = _tile_to_batch(all_images, device, patch_size, preprocess)
    feats = _extract_level_features(dino, batch, h_r, w_r, projector)

    for feat, (tile, lvl_idx, level) in zip(feats, meta):
        grp = f.create_group(f"{tile.index}_lvl{lvl_idx}")
        grp.create_dataset("ift_dino", data=feat, chunks=True)
        grp.attrs["tile_index"]    = tile.index
        grp.attrs["level"]         = lvl_idx
        grp.attrs["lat"]           = tile.center_lat
        grp.attrs["lon"]           = tile.center_lon
        # TRUE ground metres: config windows are EPSG:3857 units under --true_meters;
        # divide back (true_scale == 1.0 -> legacy value unchanged).
        grp.attrs["window_size_m"] = level.window_size_m / true_scale

    return h_r, w_r


# ── Query images (batched post-processing) ────────────────────────────────────

def extract_images(image_paths, out_h5: Path, device: str = "cpu", batch_size: int = 8,
                   segment_sky: bool = False, sky_pixel_threshold: float = 0.5,
                   sky_patch_threshold: float = 0.5, backbone: str = "dinov2_vitg14",
                   dino_layer: int | None = None, dino_facet: str = "value",
                   dino_weights: str | None = None, desc_dim: int | None = None,
                   proj_seed: int = 0) -> None:
    """Extract DINO patch features from query images (stems ``{idx}_{lat}_{lon}``)."""
    from PIL import Image

    sky_seg = None
    if segment_sky:
        from .sky import SkySegmenter, sky_keep_mask
        sky_seg = SkySegmenter()

    dino = build_dino_extractor(backbone, layer=dino_layer, facet=dino_facet,
                                device=device, norm_descs=False, weights=dino_weights)
    patch_size = getattr(dino, "patch_size", 14)
    preprocess = getattr(dino, "preprocess", None)
    projector = RandomProjector(desc_dim, proj_seed)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as f:
        f.attrs["backbone"] = backbone
        f.attrs["patch_size"] = int(patch_size)
        for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Query batches"):
            batch_paths = image_paths[batch_start: batch_start + batch_size]
            images = [np.array(Image.open(p).convert("RGB")) for p in batch_paths]

            batch, h_r, w_r = _tile_to_batch(images, device, patch_size, preprocess)
            feats = _extract_level_features(dino, batch, h_r, w_r, projector)

            for feat, img_path, image_rgb in zip(feats, batch_paths, images):
                idx, lat, lon = _parse_image_stem(img_path.stem)
                if feat.ndim == 4 and feat.shape[0] == 1:
                    feat = feat[0]
                desc_dim_, patch_h, patch_w = feat.shape
                feat_flat = feat.reshape(desc_dim_, -1)

                if sky_seg is not None:
                    sky_mask = sky_seg.segment(image_rgb, threshold=sky_pixel_threshold)
                    keep = sky_keep_mask(sky_mask, (patch_h, patch_w), sky_patch_threshold)
                    feat_filtered = feat_flat[:, keep]
                    if feat_filtered.shape[1] == 0:
                        print(f"WARNING: all patches are sky for {img_path.name} — keeping all.")
                        feat_filtered = feat_flat
                        keep_indices = np.arange(patch_h * patch_w, dtype=np.int32)
                    else:
                        keep_indices = np.flatnonzero(keep).astype(np.int32)
                else:
                    feat_filtered = feat_flat
                    keep_indices = None

                grp = f.create_group(f"img{idx}")
                grp.create_dataset("ift_dino", data=feat_filtered, chunks=True)
                if keep_indices is not None:
                    grp.create_dataset("keep_indices", data=keep_indices, chunks=True)
                grp.attrs["lat"] = lat if lat is not None else float("nan")
                grp.attrs["lon"] = lon if lon is not None else float("nan")
                grp.attrs["filename"] = img_path.name
                grp.attrs["patch_grid_h"] = int(patch_h)
                grp.attrs["patch_grid_w"] = int(patch_w)
                grp.attrs["num_patches_original"] = int(patch_h * patch_w)
                grp.attrs["num_patches_kept"] = int(feat_filtered.shape[1])
                grp.attrs["segment_sky"] = segment_sky
                grp.attrs["sky_pixel_threshold"] = float(sky_pixel_threshold)
                grp.attrs["sky_patch_threshold"] = float(sky_patch_threshold)

        for k, v in projector.h5_attrs().items():
            f.attrs[k] = v


# ── GeoTIFF pyramid (prefetched tile reads) ───────────────────────────────────

def _tif_centre_lat(tif_path) -> float:
    """Centre latitude (EPSG:4326) of the raster, for the true-metre scale."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(tif_path) as src:
        b, crs = src.bounds, src.crs
    cx, cy = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
    _, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)
    return float(lat)


def extract(tif_path, out_h5: Path, tile_size_m: float, levels: int, scale_factor: float,
            stride_m, output_size_px: int, device: str, batch_tiles: int = 16,
            aoi_points=None, backbone: str = "dinov2_vitg14", dino_layer: int | None = None,
            dino_facet: str = "value", dino_weights: str | None = None,
            desc_dim: int | None = None, proj_seed: int = 0,
            true_meters: bool = False) -> None:
    """GeoTIFF → DINO features → HDF5. See module docstring for ``true_meters``."""
    true_scale = 1.0
    if true_meters:
        centre_lat = _tif_centre_lat(tif_path)
        true_scale = mercator_true_scale(centre_lat)
        print(f"[true_meters] centre lat {centre_lat:.4f} -> true_scale={true_scale:.4f} "
              f"(EPSG:3857 units per true metre); tile {tile_size_m}->{tile_size_m*true_scale:.1f}, "
              f"stride {stride_m}->{None if stride_m is None else round(stride_m*true_scale,1)}")

    config = scaled_pyramid_config(tile_size_m, stride_m, levels, scale_factor,
                                   output_size_px, true_scale)
    filter_pts = np.array(aoi_points) if aoi_points is not None else None
    sampler = PyramidContextSampler(tif_path, config, filter_points_latlon=filter_pts)
    dino = build_dino_extractor(backbone, layer=dino_layer, facet=dino_facet,
                                device=device, norm_descs=False, weights=dino_weights)
    patch_size = getattr(dino, "patch_size", 14)
    preprocess = getattr(dino, "preprocess", None)
    projector = RandomProjector(desc_dim, proj_seed)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    h_r = w_r = 0
    total = len(sampler)

    tile_q: queue.Queue = queue.Queue(maxsize=max(_PREFETCH_FACTOR * batch_tiles, 8))
    err: list[Exception] = []

    def _producer():
        try:
            for tile in sampler.iter_tiles():
                tile_q.put(tile)
        except Exception as e:
            err.append(e)
        finally:
            tile_q.put(None)

    reader = threading.Thread(target=_producer, daemon=True)
    reader.start()

    with h5py.File(out_h5, "w") as f:
        f.attrs["backbone"] = backbone
        f.attrs["patch_size"] = int(patch_size)
        buffer = []
        pbar = tqdm(total=total, desc="Tiles")
        while True:
            tile = tile_q.get()
            if tile is None:
                break
            buffer.append(tile)
            if len(buffer) >= batch_tiles:
                h_r, w_r = _flush_buffer(buffer, dino, f, device, patch_size,
                                         projector, preprocess, true_scale=true_scale)
                pbar.update(len(buffer))
                buffer.clear()
        if buffer:
            h_r, w_r = _flush_buffer(buffer, dino, f, device, patch_size,
                                     projector, preprocess, true_scale=true_scale)
            pbar.update(len(buffer))
        pbar.close()

        for k, v in projector.h5_attrs().items():
            f.attrs[k] = v

    reader.join()
    if err:
        raise err[0]

    n_groups = len(sampler) * levels
    desc_dim_out = desc_dim if projector.active else "native"
    print(f"\nSaved → {out_h5}")
    print(f"{len(sampler)} tiles × {levels} levels = {n_groups} groups")
    print(f"Feature shape per group: (1, {desc_dim_out}, {h_r}, {w_r})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract DINO features from a GeoTIFF pyramid or query images (GPU).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tif", help="Path to input GeoTIFF (EPSG:3857)")
    src.add_argument("--images", help="Directory with query images; stems: {idx}_{lat}_{lon}.ext")

    parser.add_argument("--out", required=True, help="Output .h5 file path")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument("--batch_tiles", type=int, default=16,
                        help="Images/tiles per forward pass (default: 16)")

    parser.add_argument("--backbone", default="dinov2_vitg14",
                        help="dinov2_* (patch 14, torch.hub) or dinov3_* (patch 16). "
                             "Default dinov2_vitg14.")
    parser.add_argument("--dino_layer", type=int, default=None,
                        help="Block index to read features from (default: v2->31, v3->last)")
    parser.add_argument("--dino_facet", default="value",
                        help="Attention facet for v2 (query/key/value/token); v3 ignores it.")
    parser.add_argument("--dino_weights", default=None,
                        help="v3 only: HF repo id / local checkpoint overriding --backbone.")
    parser.add_argument("--desc_dim", type=int, default=None,
                        help="Project descriptors down to this dim (must match map/query).")
    parser.add_argument("--proj_seed", type=int, default=0,
                        help="Seed for the --desc_dim projection matrix (default: 0)")

    parser.add_argument("--segment_sky", action="store_true",
                        help="Segment sky and drop sky-dominated patches (--images only)")
    parser.add_argument("--sky_pixel_threshold", type=float, default=0.5)
    parser.add_argument("--sky_patch_threshold", type=float, default=0.5)

    parser.add_argument("--tile_size_m", type=float, default=100.0,
                        help="Base tile size (level 0). TRUE metres with --true_meters, "
                             "else raw EPSG:3857 units. Default 100.")
    parser.add_argument("--levels", type=int, default=3, help="Pyramid levels (default 3)")
    parser.add_argument("--scale_factor", type=float, default=4.0,
                        help="Scale between pyramid levels (default 4.0)")
    parser.add_argument("--stride_m", type=float, default=None,
                        help="Grid stride (default: tile_size_m). TRUE metres with "
                             "--true_meters, else raw EPSG:3857 units.")
    parser.add_argument("--true_meters", action="store_true",
                        help="Interpret --tile_size_m/--stride_m as TRUE ground metres "
                             "(scale by 1/cos(centre_lat) into EPSG:3857 units). Without it, "
                             "--stride_m 250 gives ~162 m at lat 49, not 250 m.")
    parser.add_argument("--output_px", type=int, default=224,
                        help="Pixel size of each level image, multiple of 14 (default 224)")
    parser.add_argument("--to_cog", action="store_true",
                        help="Convert input TIF to COG before extraction if not already COG.")

    parser.add_argument("--aoi_kmz", default=None, help="KMZ with Point placemarks (AOI)")
    parser.add_argument("--aoi_name_filter", default=None,
                        help="Case-insensitive substring to filter placemark names")

    parser.add_argument("--aws_profile", default=None, help="AWS named profile")
    parser.add_argument("--aws_no_sign_request", action="store_true",
                        help="Access S3 anonymously (sets AWS_NO_SIGN_REQUEST=YES)")

    args = parser.parse_args(argv)

    if args.aws_profile:
        os.environ["AWS_PROFILE"] = args.aws_profile
    if args.aws_no_sign_request:
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.geotiff")

    patch_size = 16 if args.backbone.startswith("dinov3") else 14

    if args.images:
        _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_dir = Path(args.images)
        if not img_dir.is_dir():
            parser.error(f"--images: {img_dir} is not a directory")
        image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
        if not image_paths:
            parser.error(f"--images: no images found in {img_dir}")
        extract_images(
            image_paths=image_paths, out_h5=Path(args.out), device=args.device,
            batch_size=args.batch_tiles, segment_sky=args.segment_sky,
            sky_pixel_threshold=args.sky_pixel_threshold,
            sky_patch_threshold=args.sky_patch_threshold, backbone=args.backbone,
            dino_layer=args.dino_layer, dino_facet=args.dino_facet,
            dino_weights=args.dino_weights, desc_dim=args.desc_dim, proj_seed=args.proj_seed)
        return 0

    if args.output_px % patch_size != 0:
        parser.error(f"--output_px must be a multiple of {patch_size} for {args.backbone}, "
                     f"got {args.output_px}")

    if _is_remote_uri(args.tif):
        tif_path = args.tif
        if args.to_cog:
            parser.error("--to_cog is not supported for remote URIs (s3:// / https://)")
    else:
        tif_path = Path(args.tif)
        if args.to_cog:
            tif_path = _ensure_cog(tif_path)

    aoi_points = None
    if args.aoi_kmz:
        aoi_points = _load_aoi_from_kmz(args.aoi_kmz, args.aoi_name_filter)
        print(f"[aoi] {len(aoi_points)} hull vertices from {Path(args.aoi_kmz).name}"
              + (f" (filter: '{args.aoi_name_filter}')" if args.aoi_name_filter else ""))
        preview_path = Path(args.out).parent / f"aoi_preview_{Path(args.aoi_kmz).stem}.jpg"
        _save_aoi_preview(tif_path, aoi_points, preview_path, max_px=1024)

    extract(
        tif_path=tif_path, out_h5=Path(args.out), tile_size_m=args.tile_size_m,
        levels=args.levels, scale_factor=args.scale_factor, stride_m=args.stride_m,
        output_size_px=args.output_px, device=args.device, batch_tiles=args.batch_tiles,
        aoi_points=aoi_points, backbone=args.backbone, dino_layer=args.dino_layer,
        dino_facet=args.dino_facet, dino_weights=args.dino_weights, desc_dim=args.desc_dim,
        proj_seed=args.proj_seed, true_meters=args.true_meters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
