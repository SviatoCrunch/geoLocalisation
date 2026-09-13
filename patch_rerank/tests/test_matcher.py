"""Patch-RANSAC matcher: mutual-NN + homography inlier scoring."""
import numpy as np
import pytest
import torch

from patch_rerank.matcher import grid_keypoints, mutual_nn, ransac_match


def test_grid_keypoints_layout():
    kp = grid_keypoints(2, 3)                       # rows=2, cols=3 → 6 points, row-major (x,y)
    assert kp.shape == (6, 3 - 1)                   # (6,2)
    assert kp[0].tolist() == [0.0, 0.0] and kp[1].tolist() == [1.0, 0.0]
    assert kp[3].tolist() == [0.0, 1.0]             # start of row 1


def test_mutual_nn_identity():
    g = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
    qi, ri, s = mutual_nn(g, g)                     # a grid matched to itself → identity pairs
    assert len(qi) == 10
    assert np.array_equal(qi, ri) and np.allclose(s, 1.0, atol=1e-5)


def test_mutual_nn_permutation():
    g = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
    perm = torch.tensor([2, 0, 1, 5, 3, 4])
    qi, ri, _ = mutual_nn(g, g[perm])
    # query i's NN is the ref position holding row i = index of i in perm
    inv = {int(perm[j]): j for j in range(len(perm))}
    order = np.argsort(qi)
    assert [int(ri[o]) for o in order] == [inv[i] for i in range(6)]


def test_ransac_identical_grid_scores_high():
    pytest.importorskip("cv2")
    feat = torch.nn.functional.normalize(torch.randn(25, 16), dim=1)
    xy = grid_keypoints(5, 5)
    r = ransac_match(feat, feat.clone(), xy, xy.copy())
    assert r.n_inliers >= 20 and r.score > 0.7      # same grid → homography ≈ identity, most inliers


def test_ransac_random_pair_scores_low():
    pytest.importorskip("cv2")
    a = torch.nn.functional.normalize(torch.randn(25, 16), dim=1)
    b = torch.nn.functional.normalize(torch.randn(25, 16), dim=1)
    xy = grid_keypoints(5, 5)
    r = ransac_match(a, b, xy, xy.copy())
    assert r.score < 0.7                             # unrelated features → few/no consistent inliers
