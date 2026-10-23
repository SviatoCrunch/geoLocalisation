"""Training-free patch-RANSAC matcher — vendored from geoLocalisation/patch_rerank/matcher.py.

Given two DINO token grids (query frame, map-cell crop) as feature matrices + their 2D patch
coordinates: (1) mutual nearest neighbours in cosine space, (2) cv2 homography with MAGSAC on the
matched patch coordinates, (3) score = inliers / #query patches. Pure numpy/torch/cv2 — no project
imports. cv2 is imported lazily so the mutual-NN half works without OpenCV.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_MIN_MATCHES = {"homography": 4, "affine": 3, "similarity": 2}


def grid_keypoints(h: int, w: int) -> np.ndarray:
    """(h*w, 2) patch-centre coords as (x=col, y=row), row-major to match a (H,W,·) grid."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def grid_to_latlon(x: float, y: float, h: int, w: int, tile_lat: float, tile_lon: float,
                   tile_m: float):
    """Token coord (x=col, y=row) in a tile_m grid centred at (tile_lat, tile_lon) → lat/lon."""
    east = (x - (w - 1) / 2.0) / w * tile_m
    north = -((y - (h - 1) / 2.0) / h * tile_m)
    lat = tile_lat + north / 110540.0
    lon = tile_lon + east / (math.cos(math.radians(tile_lat)) * 111320.0)
    return lat, lon


def mutual_nn(q: torch.Tensor, r: torch.Tensor):
    """Mutual nearest neighbours by cosine (inputs L2-normed). → (q_idx, r_idx) numpy."""
    sims = q @ r.t()                                  # (Nq, Nr)
    q2r = sims.argmax(dim=1)
    r2q = sims.argmax(dim=0)
    qi = torch.arange(q.shape[0], device=q.device)
    mutual = r2q[q2r] == qi
    return qi[mutual].cpu().numpy(), q2r[mutual].cpu().numpy()


def matched_coords(q_feat: torch.Tensor, r_feat: torch.Tensor, q_xy: np.ndarray, r_xy: np.ndarray):
    """Mutual-NN matched coordinate pairs → (qm (M,2), rm (M,2), n_mutual)."""
    q = F.normalize(q_feat.float(), dim=1)
    r = F.normalize(r_feat.float(), dim=1)
    q_idx, r_idx = mutual_nn(q, r)
    return np.asarray(q_xy, np.float64)[q_idx], np.asarray(r_xy, np.float64)[r_idx], int(len(q_idx))


def _cv2_method(estimator: str):
    import cv2
    return {"ransac": cv2.RANSAC, "magsac": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            "lmeds": cv2.LMEDS}[estimator]


def verify_inliers(qm: np.ndarray, rm: np.ndarray, model: str = "homography",
                   estimator: str = "magsac", reproj_thresh: float = 2.0) -> np.ndarray:
    """Fit ``model`` with ``estimator`` (cv2) mapping rm→qm → the tile-side inlier coords. Falls back
    to plain RANSAC if a cv2 build rejects the estimator flag. Empty on <min matches / no consensus."""
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
