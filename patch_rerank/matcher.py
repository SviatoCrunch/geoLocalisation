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


def _cv2_method(estimator: str):
    import cv2
    return {"ransac": cv2.RANSAC, "magsac": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            "usac": getattr(cv2, "USAC_DEFAULT", cv2.RANSAC), "lmeds": cv2.LMEDS}[estimator]


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
