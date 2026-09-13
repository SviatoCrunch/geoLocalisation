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


def ransac_match(q_feat: torch.Tensor, r_feat: torch.Tensor, q_xy: np.ndarray, r_xy: np.ndarray,
                 *, reproj_thresh: float = 2.0, min_matches: int = 4) -> MatchResult:
    """Mutual-NN + homography RANSAC on patch coordinates. Score = inliers / #query patches.
    ``inlier_r_xy`` = the tile-side coordinates of the inliers (for sub-tile position)."""
    q = F.normalize(q_feat.float(), dim=1)
    r = F.normalize(r_feat.float(), dim=1)
    q_idx, r_idx, _ = mutual_nn(q, r)
    n_mutual = int(len(q_idx))
    n_q = int(q_feat.shape[0])
    if n_mutual < min_matches:
        return MatchResult(0.0, n_mutual, 0)
    import cv2
    qm = np.asarray(q_xy, np.float64)[q_idx]
    rm = np.asarray(r_xy, np.float64)[r_idx]
    _, mask = cv2.findHomography(rm, qm, cv2.RANSAC, ransacReprojThreshold=float(reproj_thresh))
    if mask is None:
        return MatchResult(0.0, n_mutual, 0)
    inl = mask.ravel().astype(bool)
    n_in = int(inl.sum())
    return MatchResult(score=(n_in / n_q if n_q else 0.0), n_mutual=n_mutual, n_inliers=n_in,
                       inlier_r_xy=rm[inl])
