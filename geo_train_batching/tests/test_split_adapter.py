"""Integration: build training relevance/batches directly from a geo_split_no_overlap run."""
import numpy as np
import torch

from geo_train_batching.adapters import build_split_relevance, build_pairs_from_split
from geo_train_batching.batching import (build_neighbour_cache, plan_logical_batch,
                                         build_cross_relevance)
from geo_train_batching.loss import symmetric_multipositive_ce
from ._synth import make_split_dataset


def test_positives_match_the_split_snapshot(tmp_path):
    cfgp, split = make_split_dataset(tmp_path, n=30)
    sr = build_split_relevance(cfgp, split, which="train")
    assert len(sr.query_ids) > 0
    row_of = {t: i for i, t in enumerate(sr.tile_ids)}
    for i, qid in enumerate(sr.query_ids):
        snap_rows = {row_of[t] for t in sr.snapshot.point_to_tile_ids[qid]}
        pos = set(int(x) for x in sr.relevance.pos_of(i))
        safe = set(int(x) for x in sr.relevance.safe_of(i))
        assert pos == snap_rows                      # positives == the split's positives
        assert not (pos & safe)                      # pos and safe are disjoint


def test_feeds_batch_and_loss(tmp_path):
    cfgp, split = make_split_dataset(tmp_path, n=30)
    sr = build_split_relevance(cfgp, split, which="train")
    pairs = build_pairs_from_split(sr)
    assert len(pairs) == len(sr.query_ids)           # isolated queries -> one pair each

    pair_vecs = np.random.RandomState(0).randn(len(pairs), 8)   # stand-in for model embeddings
    nbr = build_neighbour_cache(pair_vecs, epoch=0)
    plan = plan_logical_batch(pairs, nbr, B_log=min(8, len(pairs)), seed_pair=0,
                              rng=np.random.RandomState(1))
    pib = [pairs[i] for i in plan.pair_indices]
    R_pos, R_cand = build_cross_relevance(pib, sr.relevance)

    S = torch.randn(len(pib), len(pib), requires_grad=True)
    loss = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    loss.backward()
    assert torch.isfinite(loss) and S.grad is not None


def test_train_val_query_sets_are_disjoint(tmp_path):
    cfgp, split = make_split_dataset(tmp_path, n=30)
    tr = set(build_split_relevance(cfgp, split, "train").query_ids)
    va = set(build_split_relevance(cfgp, split, "val").query_ids)
    assert tr and va and not (tr & va)


def test_reproducible_positives(tmp_path):
    cfgp, split = make_split_dataset(tmp_path, n=30)
    a = build_split_relevance(cfgp, split, "train")
    b = build_split_relevance(cfgp, split, "train")
    assert a.fingerprint == b.fingerprint
    assert a.query_ids == b.query_ids
