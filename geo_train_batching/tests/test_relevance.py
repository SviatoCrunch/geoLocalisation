import numpy as np

from geo_train_batching.relevance import (GeometryRelevanceTable, ExplicitRelevanceTable,
                                          RelevanceTable)


def test_geometry_pos_and_safe():
    # tiles at 0, 500, 5000 (grid metres); query at 0 -> tile0 pos, tile2 safe
    tile_xy = np.array([[0.0, 0.0], [500.0, 0.0], [5000.0, 0.0]])
    q_xy = np.array([[0.0, 0.0]])
    rel = GeometryRelevanceTable(q_xy, tile_xy, ["q0"], tile_size_m=1000.0,
                                 query_size_m=1000.0, pos_iou=0.25)
    assert 0 in rel.pos_of(0)                 # identical footprint -> IoU 1.0
    assert 2 in rel.safe_of(0)                # far tile -> no overlap
    assert 2 not in rel.pos_of(0)


def test_geometry_satisfies_protocol():
    rel = GeometryRelevanceTable(np.zeros((1, 2)), np.zeros((1, 2)), ["q"])
    assert isinstance(rel, RelevanceTable)
    assert rel.n_queries == 1 and rel.query_ids == ["q"]


def test_explicit_roundtrip():
    rel = ExplicitRelevanceTable(["a", "b"], [[0, 1], [2]], [[3], [0, 4]])
    assert rel.n_queries == 2
    assert list(rel.pos_of(0)) == [0, 1]
    assert list(rel.safe_of(1)) == [0, 4]
    assert isinstance(rel, RelevanceTable)


def test_touching_tile_is_safe_not_pos():
    tile_xy = np.array([[0.0, 0.0], [1000.0, 0.0]])   # tile1 shares an edge -> IoU 0
    rel = GeometryRelevanceTable(np.array([[0.0, 0.0]]), tile_xy, ["q"],
                                 tile_size_m=1000.0, query_size_m=1000.0)
    assert 1 in rel.safe_of(0) and 1 not in rel.pos_of(0)
