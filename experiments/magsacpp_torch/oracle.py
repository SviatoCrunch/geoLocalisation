"""Oracle / replay harness for numerical parity against the C++ MAGSAC++ reference.

Two LEVELS of verification (task brief sec.8):

A. Math parity vs an INDEPENDENT oracle -- the real danini/magsac C++ kernel (or OpenCV USAC),
   instrumented to dump normalized coords, minimal H, residuals, loss, weights, one weighted refit,
   the IRLS sequence and the final H/mask for a FIXED, recorded minimal-sample schedule. A Python
   re-implementation of the same formulas is NOT an independent oracle, so :class:`PythonReference`
   below is explicitly labelled a cross-check, not the oracle.

B. End-to-end quality vs the production CPU baseline (cv2 USAC_MAGSAC) on cached correspondences --
   not bit-exact; compares the applied ranker outcomes.

The C++ oracle must be built ON THE SERVER (no toolchain here). This file defines the record/replay
data contract and a thin loader so that, once the oracle dumps JSON traces, parity can be computed
with :func:`compare_trace` without re-running C++.

Recorded schedule format (JSON, one object per pair)::

    {
      "pair_id": "kup:q1|cellA|lvl1000",
      "points1": [[x,y], ...],          # source (map-crop), same array the Torch core receives
      "points2": [[x,y], ...],          # dest (query)
      "valid":   [true, ...],           # optional; defaults all-true
      "schedule": [[i,j,k,l], ...],     # S minimal samples (indices into points)
      "config":  {"dof":4,"quantile_k":3.64,"normalizer_c":0.25,"sigma_max":10.0, ...},
      "oracle": {                        # filled by the instrumented C++/cv2 run
         "minimal_H":   [[...],...],     # per-schedule-entry 3x3 (optional, for per-sample parity)
         "per_hyp_loss":[...],
         "best_hypothesis": int,
         "final_H":     [[...]],
         "final_inlier_mask": [bool,...],
         "irls_H":      [[[...]], ...]   # H after each IRLS step (optional)
      }
    }

See ``README.md`` -> "Building the C++ oracle" for the exact danini/magsac patch points.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from .config import MagsacppConfig
from .estimator import estimate_homography_magsacpp
from .solver import forward_sq_residual


def dump_schedule(pair_id: str, points1, points2, schedule, cfg: MagsacppConfig,
                  valid=None) -> dict:
    """Build a record (sans ``oracle`` block) to hand to the instrumented C++/cv2 run."""
    p1 = torch.as_tensor(points1, dtype=torch.float64)
    p2 = torch.as_tensor(points2, dtype=torch.float64)
    rec = {
        "pair_id": pair_id,
        "points1": p1.tolist(),
        "points2": p2.tolist(),
        "schedule": torch.as_tensor(schedule, dtype=torch.long).tolist(),
        "config": {
            "dof": cfg.dof, "quantile_k": cfg.quantile_k, "normalizer_c": cfg.normalizer_c,
            "sigma_max": cfg.sigma_max, "inlier_threshold": cfg.inlier_threshold,
            "min_sample": cfg.min_sample, "irls_iters": cfg.irls_iters,
        },
    }
    if valid is not None:
        rec["valid"] = torch.as_tensor(valid, dtype=torch.bool).tolist()
    return rec


def _cfg_from_record(rec: dict) -> MagsacppConfig:
    c = rec.get("config", {})
    return MagsacppConfig(
        dof=int(c.get("dof", 4)), quantile_k=float(c.get("quantile_k", 3.64)),
        normalizer_c=float(c.get("normalizer_c", 0.25)), sigma_max=float(c.get("sigma_max", 10.0)),
        inlier_threshold=float(c.get("inlier_threshold", 2.0)),
        irls_iters=int(c.get("irls_iters", 1)), dtype=torch.float64,
    )


def run_torch_replay(rec: dict, return_trace: bool = True):
    """Run the Torch core on a recorded pair+schedule (replay mode). Returns HomographyResult."""
    cfg = _cfg_from_record(rec)
    p1 = torch.as_tensor(rec["points1"], dtype=torch.float64)
    p2 = torch.as_tensor(rec["points2"], dtype=torch.float64)
    valid = torch.as_tensor(rec.get("valid", [True] * p1.shape[0]), dtype=torch.bool)
    sched = torch.as_tensor(rec["schedule"], dtype=torch.long)
    return estimate_homography_magsacpp(p1, p2, valid, config=cfg,
                                        hypothesis_indices=sched, return_trace=return_trace)


def _norm_H(H: torch.Tensor) -> torch.Tensor:
    """Remove the arbitrary scale/sign so two homographies are comparable (unit Frobenius, fixed
    sign). Does NOT divide by h33 (safe when h33 ~ 0)."""
    H = H / (H.norm() + 1e-300)
    # fix sign by the first significantly non-zero entry
    flat = H.reshape(-1)
    nz = flat[flat.abs() > 1e-9]
    if nz.numel() and float(nz[0]) < 0:
        H = -H
    return H


def compare_trace(rec: dict) -> Dict[str, float]:
    """Compare the Torch replay against the ``oracle`` block of a record. Returns error metrics.

    Geometry is compared by projecting the source points through both final H (masking near-infinity
    points) rather than comparing raw matrix entries, and by the sign/scale-normalized matrix
    difference. Loss/best-hypothesis agreement is reported when present.
    """
    oracle = rec.get("oracle")
    if not oracle:
        raise ValueError("record has no 'oracle' block; run the C++/cv2 oracle first")
    res = run_torch_replay(rec, return_trace=True)
    p1 = torch.as_tensor(rec["points1"], dtype=torch.float64)
    p2 = torch.as_tensor(rec["points2"], dtype=torch.float64)
    out: Dict[str, float] = {}

    if "final_H" in oracle:
        Ho = torch.as_tensor(oracle["final_H"], dtype=torch.float64)
        out["H_matrix_absdiff"] = float((_norm_H(res.H) - _norm_H(Ho)).abs().max())
        # projection-space geometric error (the primary check)
        pr_t = _project(res.H, p1)
        pr_o = _project(Ho, p1)
        finite = torch.isfinite(pr_t).all(-1) & torch.isfinite(pr_o).all(-1)
        if bool(finite.any()):
            out["proj_px_max"] = float((pr_t[finite] - pr_o[finite]).norm(dim=-1).max())
    if "per_hyp_loss" in oracle and res.trace is not None:
        lo = torch.as_tensor(oracle["per_hyp_loss"], dtype=torch.float64)
        lt = res.trace["per_hyp_loss"]
        m = min(lo.numel(), lt.numel())
        fin = torch.isfinite(lo[:m]) & torch.isfinite(lt[:m])
        if bool(fin.any()):
            out["loss_absmax"] = float((lo[:m][fin] - lt[:m][fin]).abs().max())
    if "best_hypothesis" in oracle and res.trace is not None:
        out["best_hyp_agree"] = float(int(oracle["best_hypothesis"]) == res.trace["best_hypothesis"])
    if "final_inlier_mask" in oracle:
        mo = torch.as_tensor(oracle["final_inlier_mask"], dtype=torch.bool)
        out["mask_agree_frac"] = float((mo == res.inlier_mask.cpu()).to(torch.float64).mean())
    return out


def _project(H: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x, y = p[..., 0], p[..., 1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    near = w.abs() < eps
    ws = torch.where(near, torch.ones_like(w), w)
    px = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / ws
    py = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / ws
    out = torch.stack([px, py], -1)
    return torch.where(near.unsqueeze(-1), torch.full_like(out, float("nan")), out)


class PythonReference:
    """A pure-Python/Torch recomputation of the formulas for CROSS-CHECK ONLY.

    NOT an independent oracle (it shares this repo's math). Use it to sanity-check the record/replay
    plumbing and to localize discrepancies, never as the parity ground truth. The real oracle is the
    instrumented danini/magsac C++ build (see README).
    """

    def __init__(self, cfg: Optional[MagsacppConfig] = None):
        self.cfg = cfg or MagsacppConfig()

    def residuals(self, H, points1, points2) -> torch.Tensor:
        H = torch.as_tensor(H, dtype=self.cfg.dtype)
        p1 = torch.as_tensor(points1, dtype=self.cfg.dtype)
        p2 = torch.as_tensor(points2, dtype=self.cfg.dtype)
        return forward_sq_residual(H, p1, p2, self.cfg.min_abs_denominator)


def load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]
