"""PyramidConfig pyramid math (imports need only numpy — rasterio/pyproj are lazy)."""
import pytest

from map_extract.pyramid import PyramidConfig


def test_window_sizes_scale_geometrically():
    cfg = PyramidConfig(tile_size_m=100.0, levels=3, scale_factor=4.0, output_size_px=224)
    assert cfg.window_size_m(0) == pytest.approx(100.0)
    assert cfg.window_size_m(1) == pytest.approx(400.0)
    assert cfg.window_size_m(2) == pytest.approx(1600.0)
    assert cfg.all_window_sizes() == pytest.approx([100.0, 400.0, 1600.0])


def test_safe_margin_is_half_outermost():
    cfg = PyramidConfig(tile_size_m=100.0, levels=3, scale_factor=4.0)
    assert cfg.safe_margin_m == pytest.approx(1600.0 / 2.0)


def test_single_level_config():
    cfg = PyramidConfig(tile_size_m=1000.0, levels=1, scale_factor=4.0, output_size_px=224)
    assert cfg.all_window_sizes() == pytest.approx([1000.0])
    assert cfg.safe_margin_m == pytest.approx(500.0)


def test_from_base_backcomputes_tile_size():
    cfg = PyramidConfig.from_base(base_window_m=1600.0, levels=3, scale_factor=4.0)
    assert cfg.window_size_m(cfg.levels - 1) == pytest.approx(1600.0)
    assert cfg.tile_size_m == pytest.approx(100.0)


def test_from_apex_sets_level0():
    cfg = PyramidConfig.from_apex(apex_window_m=250.0, levels=2, scale_factor=2.0)
    assert cfg.window_size_m(0) == pytest.approx(250.0)
    assert cfg.window_size_m(1) == pytest.approx(500.0)
