"""The core of the --true_meters fix: stride/tile in TRUE ground metres.

EPSG:3857 inflates true distance by 1/cos(lat); the sampler steps in EPSG:3857 units, so a
true-metre footprint must be scaled UP by 1/cos(lat) into config units, and the stored
window_size_m divided back. These tests need only math + numpy (no torch/rasterio).
"""
import math

import pytest

from map_extract.geometry import mercator_true_scale, scaled_pyramid_config

LAT = 48.5
COS = math.cos(math.radians(LAT))


def test_scale_is_inverse_cos():
    assert mercator_true_scale(0.0) == pytest.approx(1.0)          # equator: no-op
    assert mercator_true_scale(LAT) == pytest.approx(1.0 / COS)
    assert mercator_true_scale(49.68) == pytest.approx(1.0 / math.cos(math.radians(49.68)))


def test_scale_rejects_bad_latitude():
    for bad in (None, float("nan"), 95.0, -90.0):
        with pytest.raises(ValueError):
            mercator_true_scale(bad)


def test_scaled_config_true_metres_maps_to_3857_units():
    ts = mercator_true_scale(LAT)
    cfg = scaled_pyramid_config(tile_size_m=1000.0, stride_m=250.0, levels=1,
                                scale_factor=4.0, output_size_px=224, true_scale=ts)
    # config sizes are in EPSG:3857 units (inflated): 1000 true -> ~1512, 250 -> ~378
    assert cfg.tile_size_m == pytest.approx(1000.0 / COS)
    assert cfg.stride_m == pytest.approx(250.0 / COS)


def test_stored_window_size_round_trips_to_true_metres():
    # _flush_buffer stores level.window_size_m / true_scale -> the real footprint.
    ts = mercator_true_scale(LAT)
    cfg = scaled_pyramid_config(1000.0, 250.0, 1, 4.0, 224, true_scale=ts)
    stored = cfg.window_size_m(0) / ts
    assert stored == pytest.approx(1000.0)


def test_true_scale_one_is_legacy_behaviour():
    cfg = scaled_pyramid_config(1000.0, 250.0, 2, 4.0, 224, true_scale=1.0)
    assert cfg.tile_size_m == pytest.approx(1000.0)
    assert cfg.stride_m == pytest.approx(250.0)
    assert cfg.window_size_m(1) == pytest.approx(4000.0)          # unchanged pyramid math


def test_scaled_config_none_stride_stays_none():
    cfg = scaled_pyramid_config(1000.0, None, 1, 4.0, 224, true_scale=mercator_true_scale(LAT))
    assert cfg.stride_m is None
