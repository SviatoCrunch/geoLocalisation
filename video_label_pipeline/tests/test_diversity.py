import numpy as np

from video_label_pipeline.diversity import farthest_point_select


def test_returns_all_when_k_ge_n():
    X = np.eye(3)
    assert farthest_point_select(X, 5) == [0, 1, 2]


def test_empty_and_nonpositive_k():
    assert farthest_point_select(np.zeros((0, 4)), 3) == []
    assert farthest_point_select(np.eye(3), 0) == []


def test_picks_distinct_and_spread():
    X = np.array([[1, 0], [0.98, 0.1], [0.97, -0.1], [-1, 0], [0, 1]], dtype=float)
    sel = farthest_point_select(X, 3)
    assert len(sel) == 3
    assert len(set(sel)) == 3
    # the two most opposite directions must be chosen
    assert 3 in sel and (0 in sel or 4 in sel)


def test_deterministic():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 8)
    assert farthest_point_select(X, 5) == farthest_point_select(X, 5)
