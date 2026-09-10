import pytest

from geo_split_no_overlap.positive_selection import (
    PositiveSelectionConfig, available_positive_selectors, create_positive_selector,
    register_positive_selector)
from geo_split_no_overlap.positive_selection.models import PositiveMatch


def test_create_by_name_returns_right_strategy():
    sel = create_positive_selector({"strategy": "current_rule", "params": {"query_size_m": 5.0}})
    assert sel.name == "current_rule"
    assert sel.resolved_config()["query_size_m"] == 5.0


def test_unknown_name_lists_available():
    with pytest.raises(KeyError) as e:
        create_positive_selector({"strategy": "does_not_exist"})
    msg = str(e.value)
    assert "current_rule" in msg          # available strategies are shown


def test_duplicate_registration_forbidden():
    with pytest.raises(ValueError):
        register_positive_selector("current_rule", lambda p: None)


def test_available_is_sorted_and_contains_builtins():
    names = available_positive_selectors()
    assert names == tuple(sorted(names))
    assert "current_rule" in names and "contains_point" in names


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        create_positive_selector({"strategy": "current_rule", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        create_positive_selector({"strategy": "current_rule", "params": {"query_size_m": -1}})


def test_register_and_create_custom_strategy():
    class _Stub:
        name, version = "stub_test_only", "9"

        @classmethod
        def from_params(cls, params):
            return cls()

        def select(self, point, gallery):
            return [PositiveMatch(tile_id=t.tile_id) for t in gallery.all_tiles()[:1]]

        def resolved_config(self):
            return {"version": self.version}

    register_positive_selector(_Stub.name, _Stub.from_params)
    assert "stub_test_only" in available_positive_selectors()
    sel = create_positive_selector({"strategy": "stub_test_only"})
    assert sel.name == "stub_test_only"


def test_config_shape_validation():
    with pytest.raises(ValueError):
        PositiveSelectionConfig.from_mapping({"strategy": "", "params": {}})
    with pytest.raises(ValueError):
        PositiveSelectionConfig.from_mapping({"strategy": "x", "extra": 1})
