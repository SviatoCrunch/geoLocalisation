"""Faithful port of the original SegProp single-pass vote (Marcu et al., ACCV 2020).

Mirrors the per-frame body of ``segprop.vote`` from github.com/vlicaret/segprop as
closely as our stack allows, so calibration measures the SAME propagation the paper
reports. Verified numerically against the authors' own ``flow.py`` / ``map2d.py`` /
``classmap.py`` in tests/test_segprop_parity.py.

The SegProp rule for one interior frame between two annotated keyframes (``pre`` at
the block start, ``nxt`` at the block end) accumulates FOUR label sources into a
per-class vote map and takes the argmax:

    key projection  (forward-warp keyframe GT to the current frame, via flow-invert):
        pre_fw = end_map(trace(start->cur), pre_gt)     weight pre_w * key_weight
        nxt_bk = end_map(trace(end->cur),   nxt_gt)     weight nxt_w * key_weight
    cur projection  (sample the end GT where current pixels land):
        cur_fw = apply(nxt_gt, trace(cur->end))         weight nxt_w * cur_weight
        cur_bk = apply(pre_gt, trace(cur->start))       weight pre_w * cur_weight
    optional homography key votes (per-connected-component, weight key_homography_weight)

    vote = [(pre_fw*pre_w + nxt_bk*nxt_w)*key_weight
            + (cur_fw*nxt_w + cur_bk*pre_w)*cur_weight
            + (pre_fw_h*pre_w + nxt_bk_h*nxt_w)*key_homography_weight] / vote_weight

with the SegProp temporal weighting over the NORMALISED position delta in [0,1]:

    pre_w = exp(-beta*delta), nxt_w = exp(-beta*(1-delta)), then normalised to sum 1.

Conventions match the reference: flow fields are (nsteps, H, W, 2) with the last dim
ordered (dy, dx); pixel locations are (y, x); GT / vote maps are (H, W, C) with one
channel per class. beta == ``dist_weighting_beta``; beta=0 gives equal (linear)
weighting. Homography uses cv2.LMEDS by default — that is what the ORIGINAL CODE
uses (the paper text says RANSAC); pass ``hom_estimator='magsac'`` to override.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ------------------------- ported primitives (parity-checked) -------------------------

def flow_trace(flow_data: torch.Tensor, vector_mult: float = 1.0) -> torch.Tensor:
    """Port of ``flow.flow`` (rounded, bounds-clamped, no edge tracking).

    ``flow_data`` (nsteps, H, W, 2) last dim (dy, dx): follow each pixel through the
    chain, adding the step displacement at its current rounded location and clamping
    to the frame. Returns the final landing coords (H, W, 2) as (y, x). Empty chain
    (0 steps) returns the identity grid."""
    nof, noy, nox, _ = flow_data.shape
    dev = flow_data.device
    ys, xs = torch.meshgrid(torch.arange(noy, device=dev), torch.arange(nox, device=dev),
                            indexing="ij")
    loc = torch.stack([ys.reshape(-1), xs.reshape(-1)], 1).float()   # (H*W, 2) (y,x)
    for i in range(nof):
        yi = loc[:, 0].round().long()
        xi = loc[:, 1].round().long()
        loc = loc + flow_data[i, yi, xi, :] * vector_mult
        loc[:, 0] = loc[:, 0].clamp(0, noy - 1)
        loc[:, 1] = loc[:, 1].clamp(0, nox - 1)
    return loc.reshape(noy, nox, 2)


def map2d_apply(source: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Port of ``map2d.apply`` (rounded). Sample ``source`` (H,W,C) at ``mapping``
    (H,W,2)=(y,x); out-of-range / NaN coords yield NaN in the output."""
    iy = (mapping[..., 0] < 0) | (mapping[..., 0] > source.shape[0] - 1) | torch.isnan(mapping[..., 0])
    ix = (mapping[..., 1] < 0) | (mapping[..., 1] > source.shape[1] - 1) | torch.isnan(mapping[..., 1])
    ignore = iy | ix
    m = mapping.clone()
    if ignore.any():
        m[ignore] = 0
    m = m.round().long()
    out = source[m[..., 0], m[..., 1]]
    if ignore.any():
        out = out.float()
        out[ignore] = float("nan")
    return out


def map2d_invert(mapping: torch.Tensor) -> torch.Tensor:
    """Port of ``map2d.invert``. Scatter identity indices to build the inverse map;
    pixels with no source stay NaN (holes)."""
    noy, nox, _ = mapping.shape
    dev = mapping.device
    iy = (mapping[..., 0] < 0) | (mapping[..., 0] > noy - 1) | torch.isnan(mapping[..., 0])
    ix = (mapping[..., 1] < 0) | (mapping[..., 1] > nox - 1) | torch.isnan(mapping[..., 1])
    ignore = iy | ix
    ys, xs = torch.meshgrid(torch.arange(noy, device=dev), torch.arange(nox, device=dev),
                            indexing="ij")
    idx = torch.stack([ys, xs], 2).float()
    inverted = torch.full((noy, nox, 2), float("nan"), device=dev)
    m = mapping.round().long()
    if not ignore.any():
        inverted[m[..., 0], m[..., 1], :] = idx
    else:
        inverted[m[~ignore][:, 0], m[~ignore][:, 1], :] = idx[~ignore]
    return inverted


def classmap_logical(indexed: torch.Tensor, no_classes: int) -> torch.Tensor:
    """Port of ``classmap.logical``: (H,W) index map -> (H,W,C) one-hot uint8."""
    noy, nox = indexed.shape
    out = torch.zeros((noy, nox, no_classes), device=indexed.device, dtype=torch.uint8)
    for c in range(no_classes):
        out[..., c] = indexed == c
    return out


# ------------------------- the SegProp vote (single pass) -------------------------

def _end_map(flowed: torch.Tensor, class_map: torch.Tensor) -> torch.Tensor:
    """Key projection = ``segprop._end_map`` (no homography): forward-warp ``class_map``
    into the current frame by inverting the traced flow, holes -> 0."""
    mapped = map2d_apply(class_map, map2d_invert(flowed))
    mapped = torch.nan_to_num(mapped, nan=0.0)
    return mapped.float()


def _end_map_homography(flowed: torch.Tensor, class_map: torch.Tensor, estimator: str = "lmeds",
                        reproj_thresh: float = 1.5) -> torch.Tensor:
    """Port of ``segprop._end_map_homography``: per-class, per-connected-component
    homography (>=4 px) from source pixels to their flow landing points, then warp the
    label region. cv2.LMEDS by default (as in the original code)."""
    import cv2
    from scipy import ndimage

    noy, nox, noc = class_map.shape
    cm = class_map.cpu().numpy()
    fl = flowed.cpu().numpy()
    mapped = np.zeros((noy, nox, noc), np.float32)
    method = {"lmeds": cv2.LMEDS, "ransac": cv2.RANSAC,
              "magsac": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)}.get(estimator, cv2.LMEDS)
    for c in range(noc):
        label, nob = ndimage.label(cm[:, :, c], np.ones((3, 3)))
        for o in range(1, nob + 1):
            src = np.nonzero(label == o)
            if len(src[0]) < 4:
                continue
            dst = fl[src[0], src[1]]
            src_pts = np.stack([src[0], src[1]], 1).astype(np.float64)
            try:
                h, _ = cv2.findHomography(src_pts, dst.astype(np.float64), method, reproj_thresh)
            except cv2.error:
                h = None
            if h is None:
                continue
            ones = np.ones((len(src[0]), 1))
            wrp = np.concatenate([src[0][:, None], src[1][:, None], ones], 1) @ h.T
            with np.errstate(divide="ignore", invalid="ignore"):
                wrp = wrp[:, :2] / wrp[:, 2:3]
            if not np.isfinite(wrp).all():
                continue
            wy = np.clip(np.round(wrp[:, 0]), 0, noy - 1).astype(np.int64)
            wx = np.clip(np.round(wrp[:, 1]), 0, nox - 1).astype(np.int64)
            mapped[wy, wx, c] = cm[src[0], src[1], c]
    return torch.from_numpy(mapped).to(class_map.device)


def temporal_weights(delta: float, beta: float = 1.0):
    """SegProp (pre_w, nxt_w) for normalised position ``delta`` in [0,1]:
    exp(-beta*delta) / exp(-beta*(1-delta)), normalised to sum 1. beta<=0 -> (0.5, 0.5)."""
    pre = math.exp(-beta * delta)
    nxt = math.exp(-beta * (1.0 - delta))
    s = pre + nxt
    return pre / s, nxt / s


def vote_frame(pre_gt: torch.Tensor, nxt_gt: torch.Tensor, fwd_chain: torch.Tensor,
               bkw_chain: torch.Tensor, cur_fwd_chain: torch.Tensor, cur_bkw_chain: torch.Tensor,
               *, delta: float, beta: float = 1.0, key_weight: float = 1.0,
               cur_weight: float = 1.0, key_homography_weight: float = 0.0,
               hom_estimator: str = "lmeds") -> torch.Tensor:
    """One interior frame's (H,W,C) vote map — the exact SegProp fusion (see module
    docstring). GTs are (H,W,C); the four chains are (nsteps,H,W,2)=(dy,dx) traces
    start->cur, end->cur, cur->end, cur->start. Caller argmaxes the result."""
    pre_gt = pre_gt.float()
    nxt_gt = nxt_gt.float()
    pre_flowed = flow_trace(fwd_chain)
    nxt_flowed = flow_trace(bkw_chain)
    pre_fw = _end_map(pre_flowed, pre_gt)
    nxt_bk = _end_map(nxt_flowed, nxt_gt)
    cur_fw = torch.nan_to_num(map2d_apply(nxt_gt, flow_trace(cur_fwd_chain)), nan=0.0).float()
    cur_bk = torch.nan_to_num(map2d_apply(pre_gt, flow_trace(cur_bkw_chain)), nan=0.0).float()

    pre_w, nxt_w = temporal_weights(delta, beta)
    vote_weight = key_weight + cur_weight + key_homography_weight
    vote = ((pre_fw * pre_w + nxt_bk * nxt_w) * key_weight
            + (cur_fw * nxt_w + cur_bk * pre_w) * cur_weight)
    if key_homography_weight != 0:
        pre_fw_h = _end_map_homography(pre_flowed, pre_gt, hom_estimator)
        nxt_bk_h = _end_map_homography(nxt_flowed, nxt_gt, hom_estimator)
        vote = vote + (pre_fw_h * pre_w + nxt_bk_h * nxt_w) * key_homography_weight
    return vote / vote_weight
