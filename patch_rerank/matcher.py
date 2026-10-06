"""Training-free patch-RANSAC matcher (isolated port of the patch_retrieval / stage5 algorithm).

Given two DINO token grids (query frame, map tile) as L2-normalised feature matrices + their 2D
patch coordinates, it (1) finds mutual nearest neighbours in cosine space, (2) fits a homography
with RANSAC on the matched patch coordinates, (3) scores the pair by the inlier ratio. Reranking a
shortlist by this score is the geometric second stage after coarse retrieval.

Pure algorithm — no project imports. cv2 is imported lazily so the mutual-NN half is usable (and
testable) without OpenCV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def grid_keypoints(h: int, w: int) -> np.ndarray:
    """(h*w, 2) patch-centre coordinates as (x=col, y=row), row-major to match a (H,W,·) grid."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def central_mask(h: int, w: int, level_m: float, tile_m: float) -> np.ndarray:
    """Row-major boolean mask (h*w,) selecting the CENTRAL ``level_m``×``level_m`` tokens of a
    ``tile_m`` grid — the concentric-pyramid crop matched against the query at that scale."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    f = float(level_m) / float(tile_m)
    m = (np.abs(xs - (w - 1) / 2.0) <= f * w / 2.0) & (np.abs(ys - (h - 1) / 2.0) <= f * h / 2.0)
    return m.ravel()


def grid_to_latlon(x: float, y: float, h: int, w: int, tile_lat: float, tile_lon: float,
                   tile_m: float):
    """Map a token coordinate (x=col, y=row) in a ``tile_m`` grid centred at (tile_lat, tile_lon)
    to lat/lon (row increases south, col increases east) — the sub-tile position estimate."""
    east = (x - (w - 1) / 2.0) / w * tile_m
    north = -((y - (h - 1) / 2.0) / h * tile_m)
    lat = tile_lat + north / 110540.0
    lon = tile_lon + east / (math.cos(math.radians(tile_lat)) * 111320.0)
    return lat, lon


def mutual_nn(q: torch.Tensor, r: torch.Tensor):
    """Mutual nearest neighbours by cosine similarity. q (Nq,D), r (Nr,D) → (q_idx, r_idx, sims)."""
    sims = q @ r.t()                                  # (Nq, Nr) cosine (inputs assumed L2-normed)
    q2r = sims.argmax(dim=1)                          # best ref for each query patch
    r2q = sims.argmax(dim=0)                          # best query for each ref patch
    qi = torch.arange(q.shape[0], device=q.device)
    mutual = r2q[q2r] == qi
    q_idx = qi[mutual]
    r_idx = q2r[mutual]
    return q_idx.cpu().numpy(), r_idx.cpu().numpy(), sims[q_idx, r_idx].cpu().numpy()


@dataclass
class MatchResult:
    score: float                                      # inliers / n_query_patches
    n_mutual: int
    n_inliers: int
    inlier_r_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))  # tile-side inlier coords


GEOM_MODELS = ("homography", "affine", "similarity")   # 8- / 6- / 4-DOF
ESTIMATORS = ("ransac", "magsac")                       # classic RANSAC vs MAGSAC++ (USAC)
_MIN_MATCHES = {"homography": 4, "affine": 3, "similarity": 2}


def matched_coords(q_feat: torch.Tensor, r_feat: torch.Tensor, q_xy: np.ndarray, r_xy: np.ndarray):
    """Mutual-NN matched coordinate pairs. → (qm (M,2), rm (M,2), n_mutual). Compute ONCE, then run
    any number of geometric verifiers on the same pairs."""
    q = F.normalize(q_feat.float(), dim=1)
    r = F.normalize(r_feat.float(), dim=1)
    q_idx, r_idx, _ = mutual_nn(q, r)
    return np.asarray(q_xy, np.float64)[q_idx], np.asarray(r_xy, np.float64)[r_idx], int(len(q_idx))


def matched_coords_batch(q_feat: torch.Tensor, r_feats: torch.Tensor, q_xy: np.ndarray,
                         r_xy: np.ndarray):
    """Batched mutual-NN for P tiles that share the query and a common token layout.

    ``r_feats`` = (P, Nr, D) stacked tile grids (all Nr identical). Returns a list of P
    ``(qm (M,2), rm (M,2), n_mutual)`` tuples — **identical** to calling :func:`matched_coords`
    per tile, but as ONE batched matmul + a single host transfer instead of P matmuls each with
    its own ``.cpu()`` sync (which was the serial GPU bottleneck of the reranker)."""
    q = F.normalize(q_feat.float(), dim=1)                    # (Nq, D)
    r = F.normalize(r_feats.float(), dim=2)                   # (P, Nr, D)
    sims = torch.matmul(q, r.transpose(1, 2))                 # (P, Nq, Nr): q broadcast over P
    q2r = sims.argmax(dim=2)                                  # (P, Nq) best ref per query patch
    r2q = sims.argmax(dim=1)                                  # (P, Nr) best query per ref patch
    back = torch.gather(r2q, 1, q2r)                          # (P, Nq): r2q[p, q2r[p, i]]
    mutual = back == torch.arange(q.shape[0], device=q.device)   # (P, Nq)
    mutual_c = mutual.cpu().numpy()                           # ONE transfer for the whole batch
    q2r_c = q2r.cpu().numpy()
    qxy = np.asarray(q_xy, np.float64)
    rxy = np.asarray(r_xy, np.float64)
    out = []
    for p in range(mutual_c.shape[0]):
        qidx = np.nonzero(mutual_c[p])[0]
        ridx = q2r_c[p][qidx]
        out.append((qxy[qidx], rxy[ridx], int(qidx.size)))
    return out


def matched_coords_batch_gpu(q_feat: torch.Tensor, r_feats: torch.Tensor, q_xy, r_xy, device):
    """Like :func:`matched_coords_batch` but returns the matched coordinates as float64 tensors ON
    ``device`` (no ``.cpu().numpy()`` round-trip) so a GPU verifier can consume them directly. Returns
    a list of P ``(qm (M,2), rm (M,2), n_mutual)`` with qm/rm CUDA tensors. Selection is bit-identical
    to :func:`matched_coords_batch`; only the host transfer is dropped."""
    q = F.normalize(q_feat.float(), dim=1)                    # (Nq, D)
    r = F.normalize(r_feats.float(), dim=2)                   # (P, Nr, D)
    sims = torch.matmul(q, r.transpose(1, 2))                 # (P, Nq, Nr)
    q2r = sims.argmax(dim=2)                                  # (P, Nq)
    r2q = sims.argmax(dim=1)                                  # (P, Nr)
    back = torch.gather(r2q, 1, q2r)                          # (P, Nq)
    mutual = back == torch.arange(q.shape[0], device=q.device)   # (P, Nq)
    qxy_t = torch.as_tensor(np.asarray(q_xy, np.float64), device=device)
    rxy_t = torch.as_tensor(np.asarray(r_xy, np.float64), device=device)
    out = []
    for p in range(mutual.shape[0]):
        qidx = mutual[p].nonzero(as_tuple=True)[0]
        ridx = q2r[p][qidx]
        out.append((qxy_t[qidx], rxy_t[ridx], int(qidx.numel())))
    return out


def matched_coords_batch_idx(q_feat: torch.Tensor, r_feats: torch.Tensor, q_xy: np.ndarray,
                             r_xy: np.ndarray):
    """Diagnostics variant of :func:`matched_coords_batch` that ALSO returns the patch indices of
    every mutual-NN correspondence. Returns a list of P
    ``(qm (M,2), rm (M,2), n_mutual, q_idx (M,), r_idx (M,))`` tuples. Selection logic is identical
    to :func:`matched_coords_batch` (same deterministic argmax), so ``qm``/``rm`` — and therefore any
    score computed from them — are **bit-identical**; this only additionally surfaces the indices for
    the archive (which grid token each matched coordinate came from)."""
    q = F.normalize(q_feat.float(), dim=1)                    # (Nq, D)
    r = F.normalize(r_feats.float(), dim=2)                   # (P, Nr, D)
    sims = torch.matmul(q, r.transpose(1, 2))                 # (P, Nq, Nr)
    q2r = sims.argmax(dim=2)                                  # (P, Nq)
    r2q = sims.argmax(dim=1)                                  # (P, Nr)
    back = torch.gather(r2q, 1, q2r)                          # (P, Nq)
    mutual = back == torch.arange(q.shape[0], device=q.device)   # (P, Nq)
    mutual_c = mutual.cpu().numpy()
    q2r_c = q2r.cpu().numpy()
    qxy = np.asarray(q_xy, np.float64)
    rxy = np.asarray(r_xy, np.float64)
    out = []
    for p in range(mutual_c.shape[0]):
        qidx = np.nonzero(mutual_c[p])[0]
        ridx = q2r_c[p][qidx]
        out.append((qxy[qidx], rxy[ridx], int(qidx.size),
                    qidx.astype(np.int64), ridx.astype(np.int64)))
    return out


@dataclass
class HomographyResult:
    """Full output of a single cv2 homography fit — everything the diagnostics archive needs from one
    cv2 call. ``H`` maps ``rm`` (map crop, tile-side) → ``qm`` (query frame). ``mask`` is in the exact
    order the correspondences were passed to cv2. ``n_inliers`` equals ``mask.sum()`` and is the number
    the service score ``inliers / n_query_patches`` is built from — identical to the non-diag path."""
    status: str                                       # "ok" | "no_consensus"
    n_inliers: int
    mask: np.ndarray                                  # (M,) uint8, input order (empty if no H)
    H: np.ndarray | None                              # (3,3) float64, direction rm→qm (None if no H)
    residuals: np.ndarray                             # (M,) float64 reprojection residual per corr.
    inlier_r_xy: np.ndarray                           # (n_inliers, 2) tile-side inlier coords
    verify_s: float


def _reproj_residuals(H: np.ndarray, rm: np.ndarray, qm: np.ndarray) -> np.ndarray:
    """Per-correspondence Euclidean reprojection residual ‖π(H·rm) − qm‖ (diagnostic only; does NOT
    affect the inlier mask, which is cv2's). NaN where the projective denominator vanishes."""
    if H is None or len(rm) == 0:
        return np.full((len(rm),), np.nan, np.float64)
    pts = np.concatenate([rm, np.ones((len(rm), 1))], axis=1)          # (M,3)
    proj = pts @ np.asarray(H, np.float64).T                            # (M,3)
    w = proj[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = proj[:, :2] / w
    return np.linalg.norm(xy - qm, axis=1)


def verify_homography_full(qm: np.ndarray, rm: np.ndarray, reproj_thresh: float,
                           method_int: int) -> HomographyResult:
    """Fit a homography ``rm → qm`` with the given cv2 ``method_int`` in ONE call and return the full
    result (H, inlier mask in input order, inlier count, per-correspondence reprojection residuals).

    This is the diagnostics counterpart of :func:`verify_inliers`: it performs the identical single
    ``cv2.findHomography(rm, qm, method_int, ransacReprojThreshold=thr)`` call, so ``n_inliers`` (and
    any score derived from it) matches that function exactly — but it additionally surfaces ``H`` and
    the residuals for the archive. **No silent RANSAC fallback**: a ``cv2.error`` propagates (the
    caller is expected to have verified the estimator works), so the archive can never conflate
    MAGSAC++ with a fallback estimator."""
    import time

    import cv2
    t0 = time.perf_counter()
    H, mask = cv2.findHomography(rm, qm, method_int, ransacReprojThreshold=float(reproj_thresh))
    dt = time.perf_counter() - t0
    if mask is None:
        return HomographyResult("no_consensus", 0, np.empty((0,), np.uint8),
                                np.asarray(H, np.float64) if H is not None else None,
                                _reproj_residuals(H, rm, qm), np.empty((0, 2)), dt)
    m = mask.ravel().astype(np.uint8)
    Hf = np.asarray(H, np.float64) if H is not None else None
    return HomographyResult("ok", int(m.sum()), m, Hf, _reproj_residuals(Hf, rm, qm),
                            rm[m.astype(bool)], dt)


def _cv2_method(estimator: str):
    import cv2
    return {"ransac": cv2.RANSAC, "magsac": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            "usac": getattr(cv2, "USAC_DEFAULT", cv2.RANSAC), "lmeds": cv2.LMEDS}[estimator]


def magsac_method_int() -> int:
    """cv2.USAC_MAGSAC integer, or raise if this OpenCV build lacks it (so the diagnostics run fails
    loudly instead of silently reranking with RANSAC)."""
    import cv2
    if not hasattr(cv2, "USAC_MAGSAC"):
        raise RuntimeError("this OpenCV build has no cv2.USAC_MAGSAC — refusing to fall back to RANSAC "
                           f"for a MAGSAC++ diagnostics run (cv2 {cv2.__version__})")
    return int(cv2.USAC_MAGSAC)


def verify_inliers(qm: np.ndarray, rm: np.ndarray, model: str = "homography",
                   estimator: str = "ransac", reproj_thresh: float = 2.0):
    """Fit ``model`` (homography/affine/similarity) with ``estimator`` on matched coords → the tile-
    side inlier coordinates. Falls back to plain RANSAC if a cv2 build rejects the estimator flag."""
    import cv2
    if len(qm) < _MIN_MATCHES[model]:
        return np.empty((0, 2))
    method = _cv2_method(estimator)
    thr = float(reproj_thresh)

    def _fit(m):
        if model == "homography":
            return cv2.findHomography(rm, qm, m, ransacReprojThreshold=thr)[1]
        if model == "affine":
            return cv2.estimateAffine2D(rm, qm, method=m, ransacReprojThreshold=thr)[1]
        return cv2.estimateAffinePartial2D(rm, qm, method=m, ransacReprojThreshold=thr)[1]

    try:
        mask = _fit(method)
    except cv2.error:
        mask = _fit(cv2.RANSAC)
    if mask is None:
        return np.empty((0, 2))
    return rm[mask.ravel().astype(bool)]


def ransac_match(q_feat: torch.Tensor, r_feat: torch.Tensor, q_xy: np.ndarray, r_xy: np.ndarray,
                 *, reproj_thresh: float = 2.0, model: str = "homography",
                 estimator: str = "ransac") -> MatchResult:
    """Mutual-NN + geometric verification. Score = inliers / #query patches; ``inlier_r_xy`` = the
    tile-side inlier coordinates (for sub-tile position)."""
    qm, rm, n_mutual = matched_coords(q_feat, r_feat, q_xy, r_xy)
    n_q = int(q_feat.shape[0])
    if n_mutual < _MIN_MATCHES[model]:
        return MatchResult(0.0, n_mutual, 0)
    inl = verify_inliers(qm, rm, model=model, estimator=estimator, reproj_thresh=reproj_thresh)
    n_in = int(inl.shape[0])
    return MatchResult(score=(n_in / n_q if n_q else 0.0), n_mutual=n_mutual, n_inliers=n_in,
                       inlier_r_xy=inl)
