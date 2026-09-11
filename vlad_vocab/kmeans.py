"""Self-contained GPU K-means (Lloyd) — vendored from RevisitAnything's ``fit_vlad_vocab``.

No faiss. Deterministic given ``seed``. Empty clusters are reseeded from random points.
Returns (centers (K,D) cpu float, inertia, iters_run).
"""
from __future__ import annotations

import torch


def _assign_chunked(X: torch.Tensor, C: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        e = min(X.shape[0], s + chunk)
        out[s:e] = torch.cdist(X[s:e], C).argmin(dim=1)
    return out


def kmeans(X: torch.Tensor, K: int, iters: int, seed: int, device: str, tol: float = 1e-4):
    X = X.to(device)
    gcpu = torch.Generator().manual_seed(int(seed))
    C = X[torch.randperm(X.shape[0], generator=gcpu)[:K].to(device)].clone()
    labels = None
    it = 0
    for it in range(iters):
        labels = _assign_chunked(X, C)
        newC = torch.zeros_like(C)
        newC.index_add_(0, labels, X)
        counts = torch.zeros(K, device=device).index_add_(
            0, labels, torch.ones(X.shape[0], device=device))
        empty = counts == 0
        newC = newC / counts.clamp_min(1.0).unsqueeze(1)
        if bool(empty.any()):
            ridx = torch.randint(0, X.shape[0], (int(empty.sum().item()),), device=device)
            newC[empty] = X[ridx]
        shift = (newC - C).norm(dim=1).max().item()
        C = newC
        if shift < tol:
            break
    d = torch.cdist(X, C)
    inertia = float((d.gather(1, labels.unsqueeze(1)) ** 2).sum().item())
    return C.cpu(), inertia, it + 1
