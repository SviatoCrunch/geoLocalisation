import numpy as np
import pytest

from geo_train_batching.batching import (build_pair_pool, build_neighbour_cache,
                                         plan_logical_batch, build_cross_relevance,
                                         microbatch_ranges)
from geo_train_batching.relevance import ExplicitRelevanceTable


def _pairs(n=8):
    # n queries, query i canonical-nearest to tile i
    rel = ExplicitRelevanceTable([f"q{i}" for i in range(n)],
                                 pos_rows=[[i] for i in range(n)],
                                 safe_rows=[[(i + 3) % n] for i in range(n)])
    q_xy = np.array([[float(i), 0.0] for i in range(n)])
    tile_xy = np.array([[float(i), 0.0] for i in range(n)])
    return build_pair_pool(rel, q_xy, tile_xy), rel


def test_canonical_is_nearest_positive():
    rel = ExplicitRelevanceTable(["q"], pos_rows=[[5, 2, 9]], safe_rows=[[0]])
    q_xy = np.array([[2.0, 0.0]])
    tile_xy = np.array([[float(i), 0.0] for i in range(10)])
    pairs = build_pair_pool(rel, q_xy, tile_xy)
    assert pairs[0].canonical_tile_row == 2       # nearest of {5,2,9} to x=2


def test_neighbour_cache_deterministic_and_self_excluded():
    X = np.random.RandomState(0).randn(6, 4)
    a = build_neighbour_cache(X, epoch=0, top_k=3)
    b = build_neighbour_cache(X, epoch=0, top_k=3)
    assert a.version_hash == b.version_hash
    for i in range(6):
        assert i not in a.of(i)                   # self excluded
        assert len(a.of(i)) == 3


def test_plan_no_duplicate_query_or_tile():
    pairs, _ = _pairs(8)
    nbr = build_neighbour_cache(np.random.RandomState(1).randn(8, 5), epoch=0)
    rng = np.random.RandomState(2)
    plan = plan_logical_batch(pairs, nbr, B_log=6, seed_pair=0, rng=rng)
    idx = plan.pair_indices
    qids = [pairs[i].query_id for i in idx]
    tiles = [pairs[i].canonical_tile_row for i in idx]
    assert len(set(qids)) == len(qids)            # no duplicate query
    assert len(set(tiles)) == len(tiles)          # no duplicate tile
    assert plan.B_log_actual <= 6


def test_plan_deterministic_same_seed():
    pairs, _ = _pairs(10)
    nbr = build_neighbour_cache(np.random.RandomState(3).randn(10, 5), epoch=0)
    p1 = plan_logical_batch(pairs, nbr, B_log=6, seed_pair=1,
                            rng=np.random.RandomState(9), freq_counter={})
    p2 = plan_logical_batch(pairs, nbr, B_log=6, seed_pair=1,
                            rng=np.random.RandomState(9), freq_counter={})
    assert p1.pair_indices == p2.pair_indices


def test_cross_relevance_diagonal_positive():
    pairs, rel = _pairs(6)
    sub = pairs[:4]
    R_pos, R_cand = build_cross_relevance(sub, rel)
    assert R_pos.shape == (4, 4)
    assert bool(R_pos.diagonal().all())           # each pair's own tile is its positive
    assert bool((R_cand | ~R_pos).all())          # cand superset of pos


def test_microbatch_ranges():
    r = microbatch_ranges(10, 4)
    assert [list(x) for x in r] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
