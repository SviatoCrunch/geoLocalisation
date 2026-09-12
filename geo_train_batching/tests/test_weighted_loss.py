"""Weighted (soft-label) multi-positive CE: one toggleable mode, off == the hard loss.

Covers: weights=None parity, single-positive no-op, equivalence to streaming_soft_relevance_ce,
the k sharpening knob, and the batch weight-matrix / relevance-table plumbing.
"""
import numpy as np
import pytest
import torch

from geo_train_batching import (ExplicitRelevanceTable, CanonicalPair,
                                build_cross_relevance, build_cross_weights)
from geo_train_batching.loss import multipositive_ce, symmetric_multipositive_ce
from geo_train_batching.loss.full_gallery import streaming_soft_relevance_ce
from ._synth import rand_bb_problem


def test_weights_none_is_identical_to_hard():
    S, R_pos, R_cand = rand_bb_problem(seed=5)
    a = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    b = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.1, weights=None)
    assert torch.equal(a, b)                                   # off == byte-identical hard loss


def test_single_positive_weighted_equals_hard():
    S = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    pos = torch.zeros(4, 8, dtype=torch.bool)
    cand = torch.zeros(4, 8, dtype=torch.bool)
    for i in range(4):
        pos[i, i] = True
        cand[i, i] = cand[i, (i + 1) % 8] = cand[i, (i + 2) % 8] = True
    W = pos.float() * 0.37                                     # arbitrary positive weight
    hard = multipositive_ce(S, pos, cand, tau_loss=0.1)
    soft = multipositive_ce(S, pos, cand, tau_loss=0.1, weights=W)
    assert torch.allclose(hard, soft, atol=1e-6)              # one positive ⇒ weighting is a no-op


def test_weighted_matches_streaming_soft_relevance_ce():
    torch.manual_seed(0)
    B, M = 3, 10
    S = torch.randn(B, M)
    pos = torch.zeros(B, M, dtype=torch.bool)
    cand = torch.zeros(B, M, dtype=torch.bool)
    W = torch.zeros(B, M)
    for i in range(B):
        p0, p1 = i, (i + 3) % M
        pos[i, p0] = pos[i, p1] = True
        cand[i, p0] = cand[i, p1] = cand[i, (i + 5) % M] = True
        W[i, p0], W[i, p1] = 0.7, 0.3
    y = W / W.sum(dim=1, keepdim=True)                         # normalized target (Σ=1 over P)
    lhs = multipositive_ce(S, pos, cand, tau_loss=0.1, weights=W)
    rhs = streaming_soft_relevance_ce(lambda rows: S[:, rows], M, cand, y,
                                      tau_loss=0.1, chunk_size=4)["loss"]
    assert torch.allclose(lhs, rhs, atol=1e-6)                # our B×B soft-CE == the streaming form


def test_sharpening_power_approaches_hard_nearest():
    S = torch.tensor([[3.0, 1.0, -5.0]])
    pos = torch.tensor([[True, True, False]])
    cand = torch.tensor([[True, True, True]])
    W = torch.tensor([[0.9, 0.1, 0.0]])
    lo = multipositive_ce(S, pos, cand, tau_loss=0.1, weights=W, weight_power=1.0)
    hi = multipositive_ce(S, pos, cand, tau_loss=0.1, weights=W, weight_power=50.0)
    hard0 = multipositive_ce(S, torch.tensor([[True, False, False]]), cand, tau_loss=0.1)
    assert abs(float(hi) - float(hard0)) < abs(float(lo) - float(hard0))   # large k → hard nearest


def test_zero_total_weight_row_raises():
    S = torch.randn(1, 3)
    pos = torch.tensor([[True, False, False]])
    cand = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError):
        multipositive_ce(S, pos, cand, tau_loss=0.1, weights=torch.zeros(1, 3))


def test_table_and_build_cross_weights():
    rel = ExplicitRelevanceTable(["q0", "q1"], [np.array([0]), np.array([1, 2])],
                                 [np.array([1]), np.array([0])],
                                 pos_weights=[np.array([1.0]), np.array([0.6, 0.4])])
    assert rel.has_weights and rel.weight_lookup(1) == {1: 0.6, 2: 0.4}
    pairs = [CanonicalPair(0, 0, "q0", 0), CanonicalPair(1, 1, "q1", 1)]
    R_pos, _ = build_cross_relevance(pairs, rel)
    W = build_cross_weights(pairs, rel, R_pos)
    assert W[0, 0].item() == 1.0 and W[0, 1].item() == 0.0     # pair0 canon∈q0 pos; pair1 not
    assert abs(W[1, 1].item() - 0.6) < 1e-9 and W[1, 0].item() == 0.0


def test_build_cross_weights_requires_weighted_table():
    rel = ExplicitRelevanceTable(["q0"], [np.array([0])], [np.array([1])])   # no weights
    pairs = [CanonicalPair(0, 0, "q0", 0)]
    R_pos, _ = build_cross_relevance(pairs, rel)
    assert not rel.has_weights
    with pytest.raises(ValueError):
        build_cross_weights(pairs, rel, R_pos)


def test_split_adapter_carries_overlap_weights(tmp_path):
    pytest.importorskip("h5py"); pytest.importorskip("shapely"); pytest.importorskip("yaml")
    from geo_train_batching.adapters.split_relevance import build_split_relevance
    from ._synth import make_split_dataset
    cfg, sj = make_split_dataset(tmp_path, strategy="overlap_weighted", params={"min_overlap": 0.0})
    sr = build_split_relevance(str(cfg), str(sj), "train", query_size_m=1000.0)
    assert sr.relevance.has_weights
    for i in range(sr.relevance.n_queries):
        w = sr.relevance.weight_of(i)
        assert w.shape == sr.relevance.pos_of(i).shape
        assert (w > 0).all() and (w <= 1.0 + 1e-6).all()      # valid intersection ratios
