"""Parity vs the original siam_model_stage4_full_gallery implementations (ported math).

Skipped automatically if that package is not importable.
"""
import pytest
import torch

from geo_train_batching.loss import (symmetric_multipositive_ce, dense_full_gallery_nll,
                                     streaming_full_gallery_nll)
from ._synth import rand_problem, rand_bb_problem


def test_parity_symmetric_multipositive_ce():
    dss = pytest.importorskip("siam_model_stage4_full_gallery.batching.dss")
    S, R_pos, R_cand = rand_bb_problem(seed=5)
    ours = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    theirs = dss.symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    assert torch.allclose(ours, theirs)


def test_parity_dense_full_gallery_nll():
    fg = pytest.importorskip("siam_model_stage4_full_gallery.loss.full_gallery")
    S, pos, cand = rand_problem(B=5, M=25, seed=6)
    ours = dense_full_gallery_nll(S, pos, cand, tau_loss=0.1)["loss"]
    theirs = fg.dense_full_gallery_nll(S, pos, cand, tau_loss=0.1)["loss"]
    assert torch.allclose(ours, theirs)


def test_parity_streaming_full_gallery_nll():
    fg = pytest.importorskip("siam_model_stage4_full_gallery.loss.full_gallery")
    S, pos, cand = rand_problem(B=4, M=30, seed=7)

    def score_fn(rows):
        return S[:, rows]

    ours = streaming_full_gallery_nll(score_fn, S.shape[1], pos, cand,
                                      tau_loss=0.1, chunk_size=8)["loss"]
    theirs = fg.streaming_full_gallery_nll(score_fn, S.shape[1], pos, cand,
                                           tau_loss=0.1, chunk_size=8)["loss"]
    assert torch.allclose(ours, theirs, atol=1e-6)
