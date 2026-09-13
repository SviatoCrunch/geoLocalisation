"""Patch-RANSAC matcher: mutual-NN + homography inlier scoring."""
import numpy as np
import pytest
import torch

from patch_rerank.matcher import (central_mask, grid_keypoints, grid_to_latlon, mutual_nn,
                                   ransac_match)


def test_central_mask_levels():
    assert int(central_mask(10, 10, 1000, 1000).sum()) == 100      # full tile = all tokens
    assert int(central_mask(10, 10, 500, 1000).sum()) == 36        # central 50% → 6x6
    assert int(central_mask(10, 10, 300, 1000).sum()) == 16        # central 30% → 4x4


def test_grid_to_latlon_centre_and_signs():
    lat0, lon0 = 48.9, 37.6
    la, lo = grid_to_latlon((10 - 1) / 2, (10 - 1) / 2, 10, 10, lat0, lon0, 1000.0)
    assert abs(la - lat0) < 1e-9 and abs(lo - lon0) < 1e-9          # grid centre = tile centre
    top_left, _ = grid_to_latlon(0, 0, 10, 10, lat0, lon0, 1000.0)
    assert top_left > lat0                                          # row 0 = north → higher lat
    _, east = grid_to_latlon(9, 0, 10, 10, lat0, lon0, 1000.0)
    assert east > lon0                                             # last col = east → higher lon


def test_ransac_returns_inlier_coords():
    pytest.importorskip("cv2")
    feat = torch.nn.functional.normalize(torch.randn(25, 16), dim=1)
    xy = grid_keypoints(5, 5)
    r = ransac_match(feat, feat.clone(), xy, xy.copy())
    assert r.inlier_r_xy.shape[0] == r.n_inliers and r.inlier_r_xy.shape[1] == 2


def test_all_models_and_estimators_verify_identity():
    pytest.importorskip("cv2")
    from patch_rerank.matcher import GEOM_MODELS, ESTIMATORS, matched_coords, verify_inliers
    feat = torch.nn.functional.normalize(torch.randn(25, 16), dim=1)
    xy = grid_keypoints(5, 5)
    qm, rm, n = matched_coords(feat, feat.clone(), xy, xy.copy())
    assert n == 25                                             # identity → all mutual
    for model in GEOM_MODELS:                                  # every model×estimator runs + fits
        for est in ESTIMATORS:
            inl = verify_inliers(qm, rm, model=model, estimator=est)
            assert inl.shape[1] == 2 and inl.shape[0] >= 20    # identity → mostly inliers


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
