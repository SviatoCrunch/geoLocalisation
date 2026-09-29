"""GPU homography verification backends for the patch-RANSAC reranker.

Drop-in replacements for the CPU ``matcher.verify_inliers`` (cv2 ``USAC_MAGSAC``) that keep the
matched-point coordinates, hypotheses, reprojection errors and inlier masks on the GPU. Two backends:

* ``gpu_kornia`` — per-pair :class:`kornia.geometry.ransac.RANSAC` (``model_type="homography"``,
  ``score_type="msac"``). This is the FIRST WORKING variant. Kornia's ``batch_size`` batches
  *hypotheses of one pair*, NOT several independent pairs — so calling it once per crop is a serial
  loop of GPU launches. Use it as a correctness reference / small-batch backend.
* ``gpu_batch`` — a custom PyTorch MSAC that batches minimal-sample hypotheses across MANY independent
  (crop ↔ query) pairs at once (one ``find_homography_dlt`` solve + one vectorised reprojection over
  the whole batch), which is where the throughput win over the per-pair cv2/kornia calls comes from.

Direction & units MATCH the CPU code (``matcher.verify_inliers`` → ``cv2.findHomography(rm, qm)``):
the homography maps **map-crop coords → query coords** (``kp1=rm`` → ``kp2=qm``); the reprojection
error and ``reproj_thresh`` live in **query patch-grid units** (the same qm/rm that feed cv2). The
returned value is the tile-side inlier coordinates ``rm[mask]`` — identical contract to
``verify_inliers`` — so ``map_rerank`` reads ``inl.shape[0]`` (the score numerator) unchanged.

MSAC (truncated-L2 cost, ``score_type="msac"``) is NOT the same objective as MAGSAC++ (marginalisation
over noise scale). Inlier masks can therefore differ from cv2; the reranker consequence is measured, not
forced to match. The CPU cv2 MAGSAC path stays the default and the fallback (see ``matcher`` / CLI).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

_MIN = 4  # homography DLT minimal sample


def _to_pts(a, device) -> torch.Tensor:
    """(M,2) float64 tensor on ``device`` (kornia DLT is stable in double)."""
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.asarray(a, np.float64), device=device, dtype=torch.float64)


def verify_inliers_kornia(qm, rm, *, reproj_thresh: float = 2.0, max_iter: int = 10,
                          seed: int = 0, device: str = "cuda") -> np.ndarray:
    """Per-pair kornia RANSAC (first working GPU variant). H maps rm→qm, same as cv2 code.
    Returns tile-side inlier coords (n_in, 2) numpy, empty on <4 matches / degenerate / no consensus.

    Uses kornia's RANSAC (inlier-count) scorer: verified to match cv2 exactly on planted data. Kornia's
    ``score_type="msac"`` is BROKEN in the installed kornia (0.8.3) — it collapses to ~0 inliers on
    noisy correspondences — so it is deliberately NOT used here; the MSAC objective lives in the
    (correct, batched) ``ransac_homography_batch`` backend instead."""
    from kornia.geometry.ransac import RANSAC
    rm_t, qm_t = _to_pts(rm, device), _to_pts(qm, device)
    if rm_t.shape[0] < _MIN:
        return np.empty((0, 2))
    model = RANSAC(model_type="homography", inl_th=float(reproj_thresh), max_iter=int(max_iter),
                   score_type="ransac", seed=seed).to(device)
    try:
        H, mask = model(rm_t.float(), qm_t.float())     # kp1=rm → kp2=qm (map→query)
    except Exception:
        return np.empty((0, 2))
    if H is None or not torch.isfinite(H).all():
        return np.empty((0, 2))
    mask = mask.reshape(-1).bool()
    return rm_t[mask].cpu().numpy()


def _chunks_by_budget(counts: Sequence[int], n_hyp: int, budget: int):
    """Greedy chunks over pairs (given already-sorted-by-M order) so that
    ``len(chunk) * n_hyp * max_M_in_chunk <= budget`` — bounds the (P·S, M_max) work tensors."""
    chunk, mmax = [], 0
    for i, m in enumerate(counts):
        nm = max(mmax, m)
        if chunk and (len(chunk) + 1) * n_hyp * nm > budget:
            yield chunk
            chunk, mmax = [i], m
        else:
            chunk.append(i); mmax = nm
    if chunk:
        yield chunk


def _pad_batch(pairs: List[Tuple[torch.Tensor, torch.Tensor]], device):
    """Pad a list of (qm (M,2), rm (M,2)) to (P, M_max, 2) + a (P, M_max) validity mask."""
    P = len(pairs)
    mmax = max(p[0].shape[0] for p in pairs)
    qm = torch.zeros(P, mmax, 2, dtype=torch.float64, device=device)
    rm = torch.zeros(P, mmax, 2, dtype=torch.float64, device=device)
    valid = torch.zeros(P, mmax, dtype=torch.bool, device=device)
    for i, (q, r) in enumerate(pairs):
        m = q.shape[0]
        qm[i, :m] = q; rm[i, :m] = r; valid[i, :m] = True
    return qm, rm, valid


def _msac_batch(qm, rm, valid, *, reproj_thresh, n_hyp, score_type, refine, refine_iter, generator):
    """Core batched MSAC over a padded (P, M_max, 2) batch. Returns a (P, M_max) bool inlier mask.

    H maps rm→qm (matches cv2 code). Hypotheses: n_hyp minimal 4-point samples per pair, solved with
    kornia normalised DLT in ONE batched call; scored by truncated-L2 (msac) or inlier count (ransac)
    over ALL that pair's valid points; best hypothesis's inliers optionally refined via IRLS DLT."""
    from kornia.geometry.homography import (find_homography_dlt, find_homography_dlt_iterated,
                                            sample_is_valid_for_homography)
    from kornia.geometry.linalg import transform_points
    P, M = valid.shape
    S = n_hyp
    thr2 = float(reproj_thresh) ** 2
    ar = torch.arange(P, device=qm.device)

    # --- sample 4 DISTINCT correspondences per (pair, hypothesis) via weighted draw over valid pts ---
    w = valid.double().repeat_interleave(S, 0)                       # (P*S, M)
    idx4 = torch.multinomial(w, _MIN, replacement=False, generator=generator)  # (P*S, 4)
    pair_of = ar.repeat_interleave(S)                                # (P*S,)
    gather4 = idx4.unsqueeze(-1).expand(-1, -1, 2)                   # (P*S, 4, 2)
    rm_s = rm[pair_of].gather(1, gather4)                            # (P*S, 4, 2)
    qm_s = qm[pair_of].gather(1, gather4)
    valid_quad = sample_is_valid_for_homography(rm_s.float(), qm_s.float())    # (P*S,) collinearity guard

    Hs = find_homography_dlt(rm_s, qm_s)                             # (P*S, 3, 3)  rm→qm
    finite = torch.isfinite(Hs.reshape(P * S, -1)).all(1) & valid_quad

    # --- score every hypothesis against ALL its pair's points ---
    rm_e, qm_e = rm[pair_of], qm[pair_of]                           # (P*S, M, 2)
    vmask = valid[pair_of]
    proj = transform_points(Hs, rm_e)                               # (P*S, M, 2)  homogeneous divide inside
    err2 = ((proj - qm_e) ** 2).sum(-1)                             # (P*S, M)
    err2 = torch.where(torch.isfinite(err2), err2, torch.full_like(err2, 1e18))
    inl = (err2 <= thr2) & vmask
    if score_type == "msac":
        cost = torch.where(inl, err2, torch.full_like(err2, thr2))
        cost = torch.where(vmask, cost, torch.zeros_like(cost)).sum(1)          # ignore padding
        cost = torch.where(finite, cost, torch.full_like(cost, float("inf")))
        score_ps = -cost
    else:                                                            # classic RANSAC: max inliers
        score_ps = torch.where(finite, inl.sum(1).double(), torch.full((P * S,), -1.0, device=qm.device))
    best = score_ps.view(P, S).argmax(1)                            # (P,)
    bestH = Hs.view(P, S, 3, 3)[ar, best]                           # (P, 3, 3)
    best_finite = finite.view(P, S)[ar, best]                       # (P,)

    # --- recompute inlier mask with the winning hypothesis on all points ---
    proj_b = transform_points(bestH, rm)                            # (P, M, 2)
    err2_b = ((proj_b - qm) ** 2).sum(-1)
    err2_b = torch.where(torch.isfinite(err2_b), err2_b, torch.full_like(err2_b, 1e18))
    mask = (err2_b <= thr2) & valid

    if refine:
        enough = mask.sum(1) >= _MIN
        Href = find_homography_dlt_iterated(rm, qm, weights=mask.double(),
                                            soft_inl_th=float(reproj_thresh), n_iter=int(refine_iter))
        ok = torch.isfinite(Href.reshape(P, -1)).all(1) & enough
        Hf = torch.where(ok.view(P, 1, 1), Href, bestH)
        proj_f = transform_points(Hf, rm)
        err2_f = ((proj_f - qm) ** 2).sum(-1)
        err2_f = torch.where(torch.isfinite(err2_f), err2_f, torch.full_like(err2_f, 1e18))
        mask = (err2_f <= thr2) & valid

    return mask & best_finite.view(P, 1)


def ransac_homography_batch(pairs: Sequence[Tuple], *, reproj_thresh: float = 2.0, n_hyp: int = 256,
                            score_type: str = "msac", refine: bool = True, refine_iter: int = 5,
                            device: str = "cuda", seed: int = 0,
                            points_budget: int = 24_000_000) -> List[np.ndarray]:
    """Batched multi-pair homography MSAC on the GPU. ``pairs`` = list of ``(qm, rm)`` matched-coord
    tensors/arrays (any per-pair count). Returns a list (aligned to ``pairs``) of tile-side inlier
    coords ``(n_in, 2)`` numpy — empty for pairs with <4 matches / degenerate / no consensus.

    All pairs are padded and processed in memory-bounded chunks; every heavy step (DLT solve,
    reprojection, thresholding, refinement) is one batched CUDA op over the chunk."""
    n = len(pairs)
    out: List[np.ndarray] = [np.empty((0, 2)) for _ in range(n)]
    qms = [_to_pts(q, device) for (q, _) in pairs]
    rms = [_to_pts(r, device) for (_, r) in pairs]
    counts = [q.shape[0] for q in qms]
    todo = [i for i in range(n) if counts[i] >= _MIN]                # <4 matches → empty (edge case)
    if not todo:
        return out
    todo.sort(key=lambda i: counts[i])                              # sort by M for tight padding
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))
    for chunk in _chunks_by_budget([counts[i] for i in todo], n_hyp, points_budget):
        idxs = [todo[c] for c in chunk]
        qm, rm, valid = _pad_batch([(qms[i], rms[i]) for i in idxs], device)
        mask = _msac_batch(qm, rm, valid, reproj_thresh=reproj_thresh, n_hyp=n_hyp,
                           score_type=score_type, refine=refine, refine_iter=refine_iter, generator=gen)
        for j, i in enumerate(idxs):
            m = counts[i]
            sel = mask[j, :m]
            out[i] = rm[j, :m][sel].cpu().numpy()
    return out
