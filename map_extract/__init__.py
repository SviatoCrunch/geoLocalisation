"""map_extract — isolated GeoTIFF → DINOv2/v3 features → HDF5 map/gallery builder.

Self-contained port of RevisitAnything's ``tif_dino_extract_gpu.py`` + its local deps
(``utilities.build_dino_extractor``/``RandomProjector``, ``src_tif_data.torchgeo_tif``,
``src_tif_data.fishnet``, the ``tif_dino_extract`` helpers, and the ONNX sky segmenter).
No imports from RevisitAnything — this folder is standalone.

Key difference from the original: ``--true_meters`` interprets ``--tile_size_m`` /
``--stride_m`` as TRUE ground metres (scaled by 1/cos(centre_lat) into EPSG:3857 units),
so a ``--stride_m 250`` grid is exactly 250 m on the ground (the raw EPSG:3857 stride is
~cos(lat)× smaller: ~162 m at lat 49). See :mod:`map_extract.geometry`.

Heavy deps (torch, rasterio, onnxruntime) are imported lazily inside the modules that
need them, so ``map_extract.geometry`` / ``fishnet`` / ``pyramid`` (config) import with
only numpy + stdlib — the unit tests exercise the geometry without a GPU/geo stack.

Run::

    python -m map_extract.extract --tif <tif|s3://…> --out map.h5 \
        --true_meters --tile_size_m 1000 --stride_m 250 --levels 1 --output_px 224
"""
