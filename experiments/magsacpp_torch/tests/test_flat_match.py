"""Flat vectorized mutual-NN (search._mutual_nn_padded) == per-crop matched_coords_batch."""
from __future__ import annotations

import numpy as np
import torch

from magsacpp_torch.search import _mutual_nn_padded
from patch_rerank.matcher import grid_keypoints, matched_coords_batch


def _sets(qm, rm, valid, c):
    return set(map(tuple, np.c_[qm[c][valid[c]].numpy(), rm[c][valid[c]].numpy()].tolist()))


def test_flat_match_equals_per_crop():
    rng = np.random.default_rng(0)
    Nq, Nr, D, C = 40, 64, 16, 7
    q = torch.tensor(rng.standard_normal((Nq, D)), dtype=torch.float32)
    R = torch.tensor(rng.standard_normal((C, Nr, D)), dtype=torch.float32)
    qxy, rxy = grid_keypoints(5, 8), grid_keypoints(8, 8)
    mc = matched_coords_batch(q, R, qxy, rxy)
    qm, rm, valid = _mutual_nn_padded(q, R, qxy, rxy, "cpu")
    for c in range(C):
        ref = set(map(tuple, np.c_[mc[c][0], mc[c][1]].tolist()))
        assert _sets(qm, rm, valid, c) == ref, c


def test_flat_match_chunk_invariant():
    # chunking the sims tensor by match_budget must not change the matches
    rng = np.random.default_rng(1)
    q = torch.tensor(rng.standard_normal((30, 16)), dtype=torch.float32)
    R = torch.tensor(rng.standard_normal((9, 64, 16)), dtype=torch.float32)
    qxy, rxy = grid_keypoints(5, 6), grid_keypoints(8, 8)
    big = _mutual_nn_padded(q, R, qxy, rxy, "cpu", match_budget=10 ** 9)
    tiny = _mutual_nn_padded(q, R, qxy, rxy, "cpu", match_budget=30 * 64)  # ~1 crop/chunk
    for c in range(9):
        assert _sets(*big, c) == _sets(*tiny, c), c


def test_flat_match_self_is_full():
    # a crop whose tokens ARE the query -> every token mutually matches itself
    rng = np.random.default_rng(2)
    q = torch.tensor(rng.standard_normal((64, 16)), dtype=torch.float32)
    R = q.unsqueeze(0).clone()                              # 1 crop == query
    xy = grid_keypoints(8, 8)
    qm, rm, valid = _mutual_nn_padded(q, R, xy, xy, "cpu")
    assert int(valid[0].sum()) == 64
    assert torch.allclose(qm[0][valid[0]], rm[0][valid[0]])   # self-match: qxy == rxy
