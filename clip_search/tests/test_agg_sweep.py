"""Tests for the offline small->big aggregation sweep (pure numpy; no store/torch)."""
from __future__ import annotations

import json

import numpy as np

from clip_search.agg_sweep import _reduce, evaluate, load_dump, score_cell


def test_reduce_lse_between_mean_and_max():
    x = np.array([0.0, 1.0, 5.0])
    assert _reduce(x, "mean", 5.0) <= _reduce(x, "lse", 5.0) <= _reduce(x, "max", 5.0)
    # large beta -> ~max (LogMeanExp has a -log(n)/beta term, so needs a big beta), small beta -> ~mean
    assert abs(_reduce(x, "lse", 2000.0) - x.max()) < 1e-3
    assert abs(_reduce(x, "lse", 1e-3) - x.mean()) < 1e-2


def test_reduce_axis():
    S = np.array([[1.0, 2.0], [3.0, 0.0]])
    assert np.allclose(_reduce(S, "sum", 5.0, axis=1), [3.0, 3.0])
    assert np.allclose(_reduce(S, "max", 5.0, axis=1), [2.0, 3.0])


def test_score_cell_argmax_position():
    # position 1 has the strong level → argmax picks its lat/lon
    cell = {"S": np.array([[0.1, 0.1], [0.1, 0.9]]), "lat": np.array([10.0, 20.0]),
            "lon": np.array([30.0, 40.0]), "levels": [1000, 500]}
    cs, la, lo = score_cell(cell, "max", "max", "argmax", 5.0)
    assert (la, lo) == (20.0, 40.0) and cs == 0.9


def _write_dump(tmp):
    # 2 queries; GT near cell C1 position p0 (level 500 strong). C2 is far.
    agg = tmp / "aggregation.jsonl"; summ = tmp / "summary.jsonl"
    rows = []
    for qid in ("kram:q1", "kram:q2"):
        rows.append({"query_id": qid, "cell_id": "kram:1_lvl0", "positions": [
            {"position_id": 0, "pos_lat": 48.0001, "pos_lon": 37.0001,
             "level_scores": {"1000": 0.05, "500": 0.8}},
            {"position_id": 1, "pos_lat": 48.004, "pos_lon": 37.004,
             "level_scores": {"1000": 0.02, "500": 0.03}}]})
        rows.append({"query_id": qid, "cell_id": "kram:2_lvl0", "positions": [
            {"position_id": 0, "pos_lat": 49.0, "pos_lon": 38.0,
             "level_scores": {"1000": 0.1, "500": 0.1}}]})
    agg.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    summ.write_text("\n".join(json.dumps({"query_id": q, "gt": [48.0, 37.0]})
                              for q in ("kram:q1", "kram:q2")), encoding="utf-8")


def test_load_and_evaluate(tmp_path):
    _write_dump(tmp_path)
    q = load_dump(str(tmp_path))
    assert set(q) == {"kram:q1", "kram:q2"}
    c = q["kram:q1"]["cells"]["kram:1_lvl0"]
    assert c["S"].shape == (2, 2) and c["levels"] == [1000, 500]
    # max/max/argmax: C1 (0.8) beats C2 (0.1); predicted pos p0 ~ GT → within 250 m
    r = evaluate(q, "max", "max", "argmax", 5.0, topk=5)
    assert r["n"] == 2
    assert r["top1"][250] == 1.0 and r["top1"][1000] == 1.0
