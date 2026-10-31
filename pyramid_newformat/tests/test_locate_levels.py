"""Unit tests for locate._cell_record — per-level dump is opt-in and math is unchanged."""
from __future__ import annotations

import numpy as np

from pyramid_newformat.locate import _cell_record

# P=3 positions, L=2 levels. pyr(sum) = [0.3, 1.2, 0.0] → winning position bi=1;
# s[1] = [0.9, 0.3] → winning level index 0 → levels[0] = 1000.
_S = np.array([[0.1, 0.2], [0.9, 0.3], [0.0, 0.0]])
_LEVELS = [1000, 500]
_PLAT = np.array([10.0, 11.0, 12.0])
_PLON = np.array([20.0, 21.0, 22.0])


def test_cell_record_default_keys_and_math():
    r = _cell_record("c1", _S, _LEVELS, "sum", "mean", _PLAT, _PLON)
    assert set(r) == {"cell_id", "cell_score", "lat", "lon", "level_m"}   # no dump fields
    assert r["cell_id"] == "c1"
    assert r["level_m"] == 1000                                           # winning level
    assert r["lat"] == 11.0 and r["lon"] == 21.0                         # winning position bi=1
    assert abs(r["cell_score"] - float(np.mean([0.3, 1.2, 0.0]))) < 1e-12


def test_cell_record_dump_levels():
    r = _cell_record("c1", _S, _LEVELS, "sum", "mean", _PLAT, _PLON, dump_levels=True)
    assert r["winning_position_idx"] == 1
    assert r["level_scores"] == [0.9, 0.3]                                # s[bi], all L levels
    assert len(r["position_scores"]) == 3 and r["position_scores"][1] == 1.2
    assert r["level_m"] == 1000                                           # unchanged vs default


def test_cell_record_level_agg_max():
    # level_agg="max": pyr = row-max = [0.2, 0.9, 0.0] → bi=1 still; cell_agg="max" → 0.9
    r = _cell_record("c", _S, _LEVELS, "max", "max", _PLAT, _PLON, dump_levels=True)
    assert r["winning_position_idx"] == 1
    assert abs(r["cell_score"] - 0.9) < 1e-12
    assert r["position_scores"] == [0.2, 0.9, 0.0]
