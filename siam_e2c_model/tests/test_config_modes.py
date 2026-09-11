"""Config validation is mode-gated: an inactive mode's params must not reject a valid config."""
import pytest

from siam_e2c_model.config import E2cModelConfig


def test_concentric_config_ignores_scales_cells():
    # nonsense scales_cells is fine in concentric mode (unused)
    cfg = E2cModelConfig(pyramid_mode="concentric", scales_cells=(0,),
                         concentric_sizes_m=(1000.0, 500.0, 250.0))
    cfg.validate()                                   # must not raise
    assert cfg.concentric_levels() == (1000.0, 500.0, 250.0)


def test_cell_config_ignores_concentric_sizes():
    # a concentric list that would be invalid (no 250 apex) is fine in cell mode (unused)
    cfg = E2cModelConfig(pyramid_mode="cell", scales_cells=(8, 4, 2, 1),
                         concentric_sizes_m=(1000.0, 500.0))
    cfg.validate()                                   # must not raise
    assert cfg.scales_cells == (8, 4, 2, 1)


def test_concentric_validation_still_enforced_in_concentric_mode():
    with pytest.raises(ValueError):                  # apex must be exactly 250
        E2cModelConfig(pyramid_mode="concentric",
                       concentric_sizes_m=(1000.0, 500.0, 300.0)).validate()
    with pytest.raises(ValueError):                  # must include the full tile
        E2cModelConfig(pyramid_mode="concentric",
                       concentric_sizes_m=(900.0, 500.0, 250.0)).validate()
    with pytest.raises(ValueError):                  # no level < 250
        E2cModelConfig(pyramid_mode="concentric",
                       concentric_sizes_m=(1000.0, 250.0, 100.0)).validate()


def test_cell_validation_still_enforced_in_cell_mode():
    with pytest.raises(ValueError):
        E2cModelConfig(pyramid_mode="cell", scales_cells=(4, 0)).validate()


def test_ascending_concentric_is_normalized_descending_and_recorded():
    cfg = E2cModelConfig(pyramid_mode="concentric", concentric_sizes_m=(250.0, 500.0, 1000.0))
    cfg.validate()
    assert cfg.concentric_sizes_m == (1000.0, 500.0, 250.0)
    assert cfg._concentric_order_normalized is True
