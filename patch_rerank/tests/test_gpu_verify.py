"""GPU homography verification backends (device='cpu' here so it runs without CUDA).

Covers: matched_coords_batch_gpu parity with the numpy path, planted-homography recovery, edge cases
(<4 matches, collinear/degenerate, variable M in one batch), the kornia single-pair backend, and an
end-to-end map_rerank run with --verify-backend gpu_batch.
"""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("kornia")

from patch_rerank.gpu_verify import ransac_homography_batch, verify_inliers_kornia
from patch_rerank.matcher import (grid_keypoints, matched_coords_batch, matched_coords_batch_gpu)


def _norm(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape), dim=1)


def test_matched_coords_batch_gpu_equals_numpy():
    """The GPU-resident mutual-NN selection must be bit-identical to matched_coords_batch."""
    torch.manual_seed(1)
    Nq, D, P = 36, 16, 5
    q = _norm(Nq, D)
    qxy, rxy = grid_keypoints(6, 6), grid_keypoints(6, 6)
    grids = [_norm(36, D) for _ in range(P)]
    grids[0] = q.clone()                                            # an identity (all-mutual) case
    cpu = matched_coords_batch(q, torch.stack(grids), qxy, rxy.copy())
    gpu = matched_coords_batch_gpu(q, torch.stack(grids), qxy, rxy.copy(), "cpu")
    for i in range(P):
        assert gpu[i][2] == cpu[i][2]
        assert np.array_equal(gpu[i][0].cpu().numpy(), cpu[i][0])
        assert np.array_equal(gpu[i][1].cpu().numpy(), cpu[i][1])


def test_batch_recovers_planted_homography():
    """rm→qm through a known H with planted outliers: MSAC recovers the inliers, rejects outliers."""
    from kornia.geometry.linalg import transform_points
    torch.manual_seed(0)
    N, n_out = 80, 20
    rm = torch.rand(N, 2, dtype=torch.float64) * 50 + 5
    H0 = torch.tensor([[1.02, 0.03, 1.5], [-0.02, 0.98, -2.0], [1e-4, -1e-4, 1.0]], dtype=torch.float64)
    qm = transform_points(H0[None], rm[None])[0].clone()
    qm[-n_out:] = torch.rand(n_out, 2, dtype=torch.float64) * 50 + 5    # outliers
    inl = ransac_homography_batch([(qm.numpy(), rm.numpy())], reproj_thresh=2.0, n_hyp=512,
                                  device="cpu", seed=0)[0]
    n_in = inl.shape[0]
    assert inl.shape[1] == 2
    assert 55 <= n_in <= 66                                         # ~60 planted inliers recovered


def test_batch_identity_all_inliers():
    xy = grid_keypoints(6, 6).astype(np.float64)                   # 36 non-collinear grid points
    inl = ransac_homography_batch([(xy.copy(), xy.copy())], reproj_thresh=2.0, n_hyp=64,
                                  device="cpu", seed=0)[0]
    assert inl.shape[0] >= 34                                      # identity → (almost) all inliers


def test_batch_edge_cases_variable_M():
    """A single batch mixing <4, identity, random and degenerate pairs — each handled independently."""
    xy = grid_keypoints(6, 6).astype(np.float64)
    x = np.linspace(0, 50, 20); collinear = np.stack([x, 2 * x + 1], 1)   # rank-deficient source
    rng = np.random.default_rng(0)
    pairs = [
        (np.zeros((2, 2)), np.zeros((2, 2))),                      # <4 → empty
        (xy.copy(), xy.copy()),                                    # identity → many
        (rng.random((30, 2)) * 50, rng.random((30, 2)) * 50),     # random → few
        (rng.random((20, 2)) * 50, collinear.copy()),             # degenerate src vs unrelated dst
    ]
    inl = ransac_homography_batch(pairs, reproj_thresh=2.0, n_hyp=128, device="cpu", seed=0)
    assert inl[0].shape == (0, 2)                                 # too few matches → empty
    assert inl[1].shape[0] >= 34                                  # identity → (almost) all inliers
    assert all(x.shape[1] == 2 for x in inl)                      # every result a valid (n,2), no crash
    assert all(np.isfinite(x).all() for x in inl)                 # never NaN/inf coords
    assert inl[2].shape[0] <= 15                                  # unrelated random → few consistent
    assert inl[3].shape[0] <= 15                                  # degenerate/unrelated → few


def test_batch_no_refine_matches_shape():
    xy = grid_keypoints(6, 6).astype(np.float64)
    inl = ransac_homography_batch([(xy.copy(), xy.copy())], reproj_thresh=2.0, n_hyp=64,
                                  refine_iter=0, device="cpu", seed=0)[0]
    assert inl.shape[0] >= 34 and inl.shape[1] == 2


def test_kornia_identity_random_and_too_few():
    xy = grid_keypoints(6, 6).astype(np.float64)
    inl = verify_inliers_kornia(xy, xy, reproj_thresh=2.0, device="cpu", seed=0)
    assert inl.shape[1] == 2 and inl.shape[0] >= 20
    rng = np.random.default_rng(0)
    inl2 = verify_inliers_kornia(rng.random((20, 2)) * 50, rng.random((20, 2)) * 50,
                                 reproj_thresh=2.0, device="cpu", seed=0)
    assert inl2.shape[1] == 2
    assert verify_inliers_kornia(np.zeros((3, 2)), np.zeros((3, 2)), device="cpu").shape == (0, 2)


def test_map_rerank_gpu_batch_end_to_end(tmp_path):
    """The whole query-major pipeline runs with --verify-backend gpu_batch and records the backend."""
    pytest.importorskip("h5py")
    from patch_rerank.map_rerank import main
    from patch_rerank.tests.test_crop_major_parity import _build
    sl, di, qh5, st = _build(tmp_path)
    out = tmp_path / "gpu.json"
    rc = main(["--shortlist", sl, "--dense-index", di, "--queries", f"kup={qh5}", "--store", st,
               "--k", "70", "--topk", "5", "--only-city", "kup", "--device", "cpu",
               "--levels-m", "1000", "800", "600", "400",
               "--verify-backend", "gpu_batch", "--gpu-n-hyp", "64", "--out", str(out)])
    assert rc == 0
    j = json.loads(out.read_text())
    assert j["meta"]["verify_backend"] == "gpu_batch"
    assert j["meta"]["gpu_verify"]["n_hyp"] == 64
    assert j["per_query"]                                          # produced scored queries
    for rec in j["per_query"].values():
        assert rec["topk"] and all("cell_score" in t for t in rec["topk"])
