import pytest
import torch
import torch.nn.functional as F

from geo_train_batching.loss import symmetric_infonce


def test_perfect_alignment_low_loss():
    q = torch.eye(4)                       # orthonormal rows
    r = q.clone()
    loss = symmetric_infonce(q, r, logit_scale=20.0)
    assert loss.item() < 0.1


def test_permuted_pairs_higher_loss():
    q = torch.eye(4)
    aligned = symmetric_infonce(q, q.clone(), logit_scale=20.0)
    permuted = symmetric_infonce(q, q[[1, 2, 3, 0]], logit_scale=20.0)
    assert permuted.item() > aligned.item()


def test_matches_manual_symmetric_ce():
    torch.manual_seed(0)
    q = torch.randn(5, 8)
    r = torch.randn(5, 8)
    ls = 14.0
    got = symmetric_infonce(q, r, logit_scale=ls)
    qn, rn = F.normalize(q, dim=-1), F.normalize(r, dim=-1)
    logits = ls * (qn @ rn.t())
    labels = torch.arange(5)
    exp = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    assert torch.allclose(got, exp)


def test_requires_two_rows():
    with pytest.raises(ValueError):
        symmetric_infonce(torch.randn(1, 4), torch.randn(1, 4))


def test_grad_flows_to_logit_scale():
    q = torch.randn(4, 6)
    r = torch.randn(4, 6)
    logit_scale = torch.tensor(2.0, requires_grad=True)
    symmetric_infonce(q, r, logit_scale=logit_scale.exp()).backward()
    assert logit_scale.grad is not None and torch.isfinite(logit_scale.grad)
