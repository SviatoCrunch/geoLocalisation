"""Numerical parity of our SegProp port vs the authors' ORIGINAL code.

Downloads the four pure reference modules into the git-ignored _segprop_ref/ (skips
the whole module if offline / unavailable), imports them under their bare names, and
asserts our ported primitives + full single-pass vote match bit-for-bit on random
inputs. This is the "maximum similarity" guarantee.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from keyframe_calib import segprop
from keyframe_calib._segprop_ref import fetch

_REF_DIR = Path(fetch.__file__).parent


@pytest.fixture(scope="module")
def ref():
    if not fetch.available(_REF_DIR):
        if not fetch.fetch(_REF_DIR):
            pytest.skip("SegProp reference not available (offline)")
    sys.path.insert(0, str(_REF_DIR))
    try:
        warnings.filterwarnings("ignore")
        import classmap as ref_classmap
        import flow as ref_flow
        import map2d as ref_map2d
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot import SegProp reference: {e}")
    return {"flow": ref_flow, "map2d": ref_map2d, "classmap": ref_classmap}


def _rand_flow(nsteps, h, w, scale=1.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((nsteps, h, w, 2), generator=g) * 2 - 1) * scale


def test_flow_trace_parity(ref):
    fd = _rand_flow(3, 6, 7, seed=1)
    ours = segprop.flow_trace(fd)
    theirs = ref["flow"].flow(fd)                     # full_path=False -> (H,W,2)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_map2d_apply_parity(ref):
    g = torch.Generator().manual_seed(2)
    source = torch.rand((6, 7, 4), generator=g)
    mapping = torch.rand((6, 7, 2), generator=g) * 8 - 1   # some coords out of range -> NaN path
    ours = segprop.map2d_apply(source, mapping.clone())
    theirs = ref["map2d"].apply(source, mapping.clone())
    assert torch.allclose(torch.nan_to_num(ours, nan=-9.0),
                          torch.nan_to_num(theirs, nan=-9.0), atol=1e-5)


def test_map2d_invert_parity(ref):
    fd = _rand_flow(2, 6, 7, seed=3)
    flowed = segprop.flow_trace(fd)
    ours = segprop.map2d_invert(flowed.clone())
    theirs = ref["map2d"].invert(flowed.clone())
    assert torch.allclose(torch.nan_to_num(ours, nan=-9.0),
                          torch.nan_to_num(theirs, nan=-9.0), atol=1e-5)


def test_classmap_logical_parity(ref):
    g = torch.Generator().manual_seed(4)
    idx = (torch.rand((6, 7), generator=g) * 3).long()
    assert torch.equal(segprop.classmap_logical(idx, 3), ref["classmap"].logical(idx, 3))


def test_vote_frame_parity(ref):
    """End-to-end: our vote_frame == the SegProp vote body rebuilt from ORIGINAL
    primitives (no homography, which is the deterministic core)."""
    F, M = ref["flow"], ref["map2d"]
    h, w, noc = 6, 7, 3
    g = torch.Generator().manual_seed(5)
    pre_gt = segprop.classmap_logical((torch.rand((h, w), generator=g) * noc).long(), noc).float()
    nxt_gt = segprop.classmap_logical((torch.rand((h, w), generator=g) * noc).long(), noc).float()
    fwd = _rand_flow(2, h, w, seed=6)      # start->cur
    bkw = _rand_flow(3, h, w, seed=7)      # end->cur
    curf = _rand_flow(3, h, w, seed=8)     # cur->end
    curb = _rand_flow(2, h, w, seed=9)     # cur->start
    delta, beta, kw, cw = 0.4, 1.0, 1.0, 1.0

    def endmap(fd, gt):
        mp = M.apply(gt, M.invert(F.flow(fd)))
        mp[torch.isnan(mp)] = 0
        return mp.float()

    def curmap(fd, gt):
        return torch.nan_to_num(M.apply(gt, F.flow(fd)), nan=0.0).float()

    pre_w = math.exp(-beta * delta)
    nxt_w = math.exp(-beta * (1 - delta))
    s = pre_w + nxt_w
    pre_w, nxt_w = pre_w / s, nxt_w / s
    ref_vote = ((endmap(fwd, pre_gt) * pre_w + endmap(bkw, nxt_gt) * nxt_w) * kw
                + (curmap(curf, nxt_gt) * nxt_w + curmap(curb, pre_gt) * pre_w) * cw) / (kw + cw)

    ours = segprop.vote_frame(pre_gt, nxt_gt, fwd, bkw, curf, curb,
                              delta=delta, beta=beta, key_weight=kw, cur_weight=cw)
    assert torch.allclose(ours, ref_vote, atol=1e-5)
    # and the final decisions agree
    assert torch.equal(ours.argmax(2), ref_vote.argmax(2))
