"""GPU/CPU K-means converges to well-separated blobs, deterministically."""
import torch

from vlad_vocab.kmeans import kmeans


def _three_blobs(n=300, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[5.0] + [0.0] * (d - 1),
                            [0.0, 5.0] + [0.0] * (d - 2),
                            [0.0, 0.0, 5.0] + [0.0] * (d - 3)])
    X = torch.cat([c + 0.2 * torch.randn(n, d, generator=g) for c in centres], 0)
    return X, centres


def test_kmeans_recovers_blob_centres():
    X, centres = _three_blobs()
    C, inertia, n_it = kmeans(X, K=3, iters=50, seed=0, device="cpu")
    assert tuple(C.shape) == (3, X.shape[1])
    # each true centre has a fitted centre within 0.5
    for tc in centres:
        assert torch.cdist(tc[None], C).min().item() < 0.5
    assert inertia > 0 and 1 <= n_it <= 50


def test_kmeans_is_deterministic():
    X, _ = _three_blobs()
    C1, i1, _ = kmeans(X, K=3, iters=50, seed=7, device="cpu")
    C2, i2, _ = kmeans(X, K=3, iters=50, seed=7, device="cpu")
    assert torch.allclose(C1, C2)
    assert i1 == i2


def test_kmeans_handles_more_clusters_than_needed():
    X, _ = _three_blobs()
    C, _, _ = kmeans(X, K=8, iters=30, seed=1, device="cpu")   # empty clusters reseeded
    assert tuple(C.shape) == (8, X.shape[1])
    assert torch.isfinite(C).all()
