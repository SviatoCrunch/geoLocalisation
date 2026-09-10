"""Unit tests for s3_gt_sync.masks — COCO result.json -> per-class binary masks.
No network; pure local rasterisation with PIL/numpy."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from s3_gt_sync import masks


def _write_coco(tmp_path, images, annotations, categories):
    p = tmp_path / "result.json"
    p.write_text(json.dumps({
        "images": images, "annotations": annotations, "categories": categories,
    }), encoding="utf-8")
    return str(p)


# ── decoding primitives ───────────────────────────────────────────────────────

def test_decode_polygon_square():
    # 10x10 canvas, a 4x4 filled square from (2,2) to (6,6)
    poly = [2, 2, 6, 2, 6, 6, 2, 6]
    m = masks._decode_polygons([poly], 10, 10)
    assert m.dtype == bool
    assert m[3, 3] and not m[0, 0]
    assert m.sum() >= 12  # ~16 px, minus polygon edge rounding


def test_decode_bbox_rectangle():
    m = masks._decode_bbox([1, 2, 3, 4], 10, 10)  # x,y,w,h
    assert m[2, 1] and m[5, 3]
    assert not m[6, 1]
    assert m.sum() == 12


def test_decode_uncompressed_rle_roundtrip():
    # column-major runs starting with 0: full first column of a 3x3
    # counts [0,3,6] -> first 3 flat px True (col 0), rest False
    m = masks._rle_decode_uncompressed([0, 3, 6], 3, 3)
    assert m[:, 0].all()
    assert not m[:, 1].any()


# ── masks_from_coco ───────────────────────────────────────────────────────────

def test_masks_from_coco_writes_per_class_png(tmp_path):
    images = [{"id": 1, "file_name": "5_48.1_37.2.jpg", "height": 10, "width": 10}]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1,
         "segmentation": [[2, 2, 6, 2, 6, 6, 2, 6]]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 3, 3]},
    ]
    categories = [{"id": 1, "name": "building"}, {"id": 2, "name": "road"}]
    coco = _write_coco(tmp_path, images, annotations, categories)
    dest = tmp_path / "GT_flat_mask"

    stats = masks.masks_from_coco(coco, str(dest))
    assert stats["written"] == 2 and stats["empty"] == 0
    assert set(stats["classes"]) == {"building", "road"}

    # naming keys off the lat_lon coordinate pair, dropping the frame index
    b = dest / "48.1_37.2__building.png"
    r = dest / "48.1_37.2__road.png"
    assert b.exists() and r.exists()
    arr = np.asarray(Image.open(b))
    assert set(np.unique(arr).tolist()) <= {0, 255}  # strictly binary


def test_masks_unions_same_class_annotations(tmp_path):
    images = [{"id": 1, "file_name": "1_0_0.jpg", "height": 10, "width": 10}]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 5, 2, 2]},
    ]
    categories = [{"id": 1, "name": "car"}]
    coco = _write_coco(tmp_path, images, annotations, categories)
    dest = tmp_path / "m"

    stats = masks.masks_from_coco(coco, str(dest))
    assert stats["written"] == 1  # one merged mask, not two
    arr = np.asarray(Image.open(dest / "1_0_0__car.png"))
    assert arr[0, 0] == 255 and arr[6, 6] == 255 and arr[3, 3] == 0


def test_masks_skip_existing_and_dry_run(tmp_path):
    images = [{"id": 1, "file_name": "1_0_0.jpg", "height": 8, "width": 8}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]}]
    categories = [{"id": 1, "name": "x"}]
    coco = _write_coco(tmp_path, images, annotations, categories)
    dest = tmp_path / "m"

    # dry-run writes nothing
    s0 = masks.masks_from_coco(coco, str(dest), dry_run=True)
    assert s0["written"] == 1 and not dest.exists()

    s1 = masks.masks_from_coco(coco, str(dest))
    assert s1["written"] == 1
    s2 = masks.masks_from_coco(coco, str(dest))
    assert s2["written"] == 0 and s2["skipped"] == 1


def test_class_name_sanitized_for_filename(tmp_path):
    images = [{"id": 1, "file_name": "1_0_0.jpg", "height": 6, "width": 6}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 3, 3]}]
    categories = [{"id": 1, "name": "traffic light/sign"}]
    coco = _write_coco(tmp_path, images, annotations, categories)
    dest = tmp_path / "m"
    masks.masks_from_coco(coco, str(dest))
    assert (dest / "1_0_0__traffic_light_sign.png").exists()


def test_label_studio_windows_path_stem(tmp_path):
    # COCO from Label Studio: full ..\..\ Windows path + hash-prefixed name.
    # Mask name must reduce to lat_lon__class only.
    fn = (r"..\..\label-studio\label-studio\media\upload\3"
          r"\0d0aeb21-102_48.5975981021_37.5907872557.jpg")
    images = [{"id": 1, "file_name": fn, "height": 8, "width": 8}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]}]
    categories = [{"id": 1, "name": "Road"}]
    coco = _write_coco(tmp_path, images, annotations, categories)
    dest = tmp_path / "m"
    masks.masks_from_coco(coco, str(dest))
    assert (dest / "48.5975981021_37.5907872557__Road.png").exists()
    assert list(dest.iterdir())  # nothing with backslashes / hash prefix


def test_frame_stem_helper():
    assert masks._frame_stem(r"..\a\0d-102_48.59_37.59.jpg") == "48.59_37.59"
    assert masks._frame_stem("102_48.59,37.59.jpg") == "48.59_37.59"  # comma variant
    assert masks._frame_stem("plain_name.png") == "plain_name"        # no coords


def test_non_coco_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        masks.masks_from_coco(str(p), str(tmp_path / "m"))
