"""Audit tool: single index/rgb masks + per-class-binary external-val layout."""
import numpy as np
from PIL import Image

from segformer3_full_taxonomy.tools.audit_segmentation_data import (
    audit_per_class_binary, audit_single,
)


def _save(arr, path):
    Image.fromarray(arr).save(path)


def test_single_index_masks_report_unique_values(tmp_path):
    imgd, mkd = tmp_path / "img", tmp_path / "msk"
    imgd.mkdir(); mkd.mkdir()
    for i in range(3):
        _save((np.random.rand(8, 8, 3) * 255).astype(np.uint8), imgd / f"{i}.png")
        m = np.zeros((8, 8), np.uint8); m[:4] = 2; m[4:] = 7   # ids 2 and 7
        _save(m, mkd / f"{i}.png")
    imgs = sorted(imgd.glob("*.png")); msks = sorted(mkd.glob("*.png"))
    rep = audit_single(imgs, msks, "auto", do_hist=True)
    assert rep["mask_format"] == "index"
    assert rep["paired"] == 3
    assert set(rep["class_pixel_counts"]) == {"2", "7"}
    assert rep["unmatched_images"]["count"] == 0


def test_single_rgb_palette_reports_colors(tmp_path):
    imgd, mkd = tmp_path / "i", tmp_path / "m"
    imgd.mkdir(); mkd.mkdir()
    _save((np.random.rand(6, 6, 3) * 255).astype(np.uint8), imgd / "a.png")
    rgb = np.zeros((6, 6, 3), np.uint8); rgb[:, :3] = (128, 64, 0); rgb[:, 3:] = (0, 0, 255)
    _save(rgb, mkd / "a.png")
    rep = audit_single(sorted(imgd.glob("*.png")), sorted(mkd.glob("*.png")), "rgb", do_hist=True)
    assert rep["mask_format"] == "rgb"
    assert any("128" in k for k in rep["class_pixel_counts"])   # colour keys like "rgb(128, 64, 0)"


def test_unmatched_image_recorded(tmp_path):
    imgd, mkd = tmp_path / "i", tmp_path / "m"
    imgd.mkdir(); mkd.mkdir()
    _save(np.zeros((4, 4, 3), np.uint8), imgd / "lonely.png")
    _save(np.zeros((4, 4), np.uint8), mkd / "other.png")
    rep = audit_single(sorted(imgd.glob("*.png")), sorted(mkd.glob("*.png")), "index", do_hist=True)
    assert rep["unmatched_images"]["count"] == 1
    assert "lonely.png" in rep["unmatched_images"]["sample"]


def test_per_class_binary_latlon_keying(tmp_path):
    imgd, mkd = tmp_path / "GT_flat", tmp_path / "GT_flat_mask"
    imgd.mkdir(); mkd.mkdir()
    _save(np.zeros((5, 5, 3), np.uint8), imgd / "1_48.5_37.6.jpg")
    _save((np.ones((5, 5), np.uint8) * 255), mkd / "48.5_37.6__Road.png")
    _save((np.ones((5, 5), np.uint8) * 255), mkd / "48.5_37.6__Water.png")
    _save((np.ones((5, 5), np.uint8) * 255), mkd / "99.9_11.1__Sky.png")   # orphan
    rep = audit_per_class_binary(sorted(imgd.glob("*.jpg")), str(mkd), "latlon", do_hist=True)
    assert rep["images_with_masks"] == 1
    assert set(rep["classes_images"]) == {"Road", "Water"}
    assert rep["classes_fg_pixels"]["Road"] == 25
    assert rep["orphan_masks_no_image"]["count"] == 1
