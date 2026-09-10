import torch

from geo_train_batching.loss import (masked_logsumexp, multipositive_ce,
                                     symmetric_multipositive_ce, dense_full_gallery_nll)
from ._synth import rand_problem, rand_bb_problem


def test_masked_logsumexp_matches_manual():
    z = torch.tensor([[1.0, 2.0, 3.0]])
    m = torch.tensor([[True, False, True]])
    got = masked_logsumexp(z, m, dim=1)
    exp = torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0)
    assert torch.allclose(got.squeeze(), exp)


def test_empty_row_is_neg_inf():
    z = torch.zeros(1, 3)
    m = torch.zeros(1, 3, dtype=torch.bool)
    assert masked_logsumexp(z, m, dim=1).item() == float("-inf")


def test_multipositive_ce_equals_dense_full_gallery():
    S, pos, cand = rand_problem(seed=1)
    a = multipositive_ce(S, pos, cand, tau_loss=0.1)
    b = dense_full_gallery_nll(S, pos, cand, tau_loss=0.1)["loss"]
    assert torch.allclose(a, b)


def test_symmetric_is_mean_of_both_directions():
    S, R_pos, R_cand = rand_bb_problem(seed=2)
    sym = symmetric_multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    row = multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    col = multipositive_ce(S.t().contiguous(), R_pos.t().contiguous(),
                           R_cand.t().contiguous(), tau_loss=0.1)
    assert torch.allclose(sym, 0.5 * (row + col))


def test_symmetric_symmetric_input():
    S, R_pos, R_cand = rand_bb_problem(seed=3)
    S = S + S.t()
    R_pos = R_pos | R_pos.t()
    R_cand = R_cand | R_cand.t() | R_pos
    row = multipositive_ce(S, R_pos, R_cand, tau_loss=0.1)
    col = multipositive_ce(S.t().contiguous(), R_pos.t().contiguous(),
                           R_cand.t().contiguous(), tau_loss=0.1)
    assert torch.allclose(row, col)


def test_gradient_flows():
    S, pos, cand = rand_problem(seed=4)
    S = S.clone().requires_grad_(True)
    multipositive_ce(S, pos, cand, tau_loss=0.1).backward()
    assert S.grad is not None and torch.isfinite(S.grad).all()


def test_perfect_scores_give_low_loss():
    # positives get huge score -> loss ~ 0
    pos = torch.tensor([[True, False, False, False]])
    cand = torch.tensor([[True, True, True, True]])
    S = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    assert multipositive_ce(S, pos, cand, tau_loss=0.1).item() < 1e-3
