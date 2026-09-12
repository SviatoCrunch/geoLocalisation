"""Unit tests for the pure diagnostic helpers (no model / no full pipeline)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geo_train_batching import ExplicitRelevanceTable
from retrieval_overfit_diagnostics.common import rank_matrix


def test_rank_matrix_known_ranks():
    rel = ExplicitRelevanceTable(["a", "b"], [np.array([2]), np.array([0])],
                                 [np.array([], int), np.array([], int)])
    sr = SimpleNamespace(relevance=rel)
    qids, qid_row = ["a", "b"], {"a": 0, "b": 1}
    S = np.array([[0.1, 0.2, 0.9, 0.3], [0.9, 0.1, 0.2, 0.3]])          # both positives rank 1
    m = rank_matrix(S, qids, qid_row, sr)
    assert m["R@1"] == 1.0 and m["median_rank"] == 1.0 and m["hits"]["R@1"] == 2


def test_aggregations_learned_uniform_max():
    ls = torch.tensor([[[0.9, 0.0], [0.4, 0.8], [0.1, 0.1]]])           # (1 query, 3 tiles, 2 levels)
    beta = torch.tensor([[1.0, 0.0]])                                  # gate favours level 0
    learned = (ls * beta[:, None, :]).sum(2)
    uniform = ls.mean(2)
    mx = ls.max(2).values
    assert torch.allclose(learned, torch.tensor([[0.9, 0.4, 0.1]]))
    assert torch.allclose(uniform, torch.tensor([[0.45, 0.60, 0.1]]))
    assert torch.allclose(mx, torch.tensor([[0.9, 0.8, 0.1]]))
    rel = ExplicitRelevanceTable(["q"], [np.array([1])], [np.array([], int)])   # GT = tile 1
    sr = SimpleNamespace(relevance=rel)
    # gate (level 0) ranks tile0 first ⇒ tile1 not top ⇒ learned R@1=0; uniform favours tile1 ⇒ R@1=1
    assert rank_matrix(learned.numpy(), ["q"], {"q": 0}, sr)["R@1"] == 0.0
    assert rank_matrix(uniform.numpy(), ["q"], {"q": 0}, sr)["R@1"] == 1.0


def test_false_negative_suspect_detection(tmp_path):
    from retrieval_overfit_diagnostics import audit_positive_labels
    rel = ExplicitRelevanceTable(["a:0"], [np.array([0])], [np.array([1, 2])])
    sr = SimpleNamespace(query_ids=["a:0"], q_xy=np.array([[0.0, 0.0]]),
                         tile_xy=np.array([[0.0, 0.0], [100.0, 100.0], [5000.0, 5000.0]]),
                         relevance=rel)
    ctx = SimpleNamespace(sr={"train": sr}, args=SimpleNamespace(output_dir=str(tmp_path)))
    res = audit_positive_labels.run(ctx)
    # tile1 ~141 m *0.657 ≈ 93 m ≤ 250 and NOT positive → exactly one suspect; tile2 is far
    assert res["total_false_negative_suspects"] == 1


def test_param_fingerprint_tracks_map_params():
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from retrieval_overfit_diagnostics.common import param_fingerprints
    aw, c = torch.randn(6, 32), torch.randn(6, 32)
    m = build_e2c_model(E2cModelConfig(agg="residual", k=6, d_token=32, scales_cells=(2, 1),
                                       d_group=8, d_out=16, d_hidden=32),
                        assign_weight=aw, centroids=c)
    f0 = param_fingerprints(m)
    with torch.no_grad():
        for n, p in m.core.named_parameters():
            if n.startswith("map_head") and p.requires_grad:
                p.add_(1.0); break
    f1 = param_fingerprints(m)
    assert f0["map_branch"] != f1["map_branch"]                        # V-affecting change detected
    assert f0["query_branch"] == f1["query_branch"]                    # query branch untouched
