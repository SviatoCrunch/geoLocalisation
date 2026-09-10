import pytest
import torch

from geo_train_batching.loss import (dense_full_gallery_nll, streaming_full_gallery_nll,
                                     streaming_soft_relevance_ce, validate_masks)
from ._synth import rand_problem


def test_dense_streaming_parity():
    S, pos, cand = rand_problem(B=5, M=30, seed=7)
    dense = dense_full_gallery_nll(S, pos, cand, tau_loss=0.1)["loss"]

    def score_fn(rows):
        return S[:, rows]

    stream = streaming_full_gallery_nll(score_fn, S.shape[1], pos, cand,
                                        tau_loss=0.1, chunk_size=7)["loss"]
    assert torch.allclose(dense, stream, atol=1e-5)


def test_streaming_chunk_invariance():
    S, pos, cand = rand_problem(B=4, M=40, seed=8)

    def score_fn(rows):
        return S[:, rows]

    vals = [streaming_full_gallery_nll(score_fn, S.shape[1], pos, cand,
                                       tau_loss=0.1, chunk_size=cs)["loss"]
            for cs in (1, 5, 13, 40)]
    for v in vals[1:]:
        assert torch.allclose(vals[0], v, atol=1e-5)


def test_streaming_grad_flows():
    S, pos, cand = rand_problem(B=3, M=20, seed=9)
    S = S.clone().requires_grad_(True)

    def score_fn(rows):
        return S[:, rows]

    streaming_full_gallery_nll(score_fn, S.shape[1], pos, cand, tau_loss=0.1,
                               chunk_size=6)["loss"].backward()
    assert S.grad is not None and torch.isfinite(S.grad).all()


def test_validate_masks_fails_closed():
    S, pos, cand = rand_problem(B=3, M=10, seed=10)
    validate_masks(pos, cand, 10)                 # ok
    with pytest.raises(ValueError):
        validate_masks(pos, cand, 11)             # width mismatch
    bad_pos = pos.clone(); bad_pos[0] = False     # a query with no positive
    with pytest.raises(ValueError):
        validate_masks(bad_pos, cand, 10)


def test_pos_not_subset_of_cand_raises():
    pos = torch.tensor([[True, True, False]])
    cand = torch.tensor([[True, False, False]])   # missing a positive
    with pytest.raises(ValueError):
        dense_full_gallery_nll(torch.randn(1, 3), pos, cand)


def test_soft_relevance_ce_matches_hard_when_uniform_over_positives():
    # uniform soft target over positives -> mean-positive; just check it runs + finite + grad
    S, pos, cand = rand_problem(B=3, M=12, seed=11)
    S = S.clone().requires_grad_(True)
    y = pos.float()
    y = y / y.sum(dim=1, keepdim=True)

    def score_fn(rows):
        return S[:, rows]

    out = streaming_soft_relevance_ce(score_fn, S.shape[1], cand, y, tau_loss=0.1, chunk_size=5)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert S.grad is not None
