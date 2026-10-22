"""Torch-free tests for the clip-search glue (window logic, clip building, aggregation, KMZ)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from clip_search.aggregate import aggregate_clip, haversine_m, run
from clip_search.sample import build_clips, window_bounds


# ----------------------------- window_bounds -----------------------------

def test_window_centered_full():
    # GT far from edges: 32 frames, GT at offset 16 (16 before + GT + 15 after)
    a, b = window_bounds(100, 0, 1000, window=32)
    assert (a, b) == (84, 115)
    assert b - a + 1 == 32
    assert 100 - a == 16


def test_window_clipped_left():
    a, b = window_bounds(5, 0, 1000, window=32)
    assert a == 0 and b == 20            # 5-16 -> clipped to 0; right still 5+15=20
    assert b - a + 1 == 21 < 32          # shorter, not re-expanded ("може бути менше")


def test_window_clipped_both():
    a, b = window_bounds(10, 3, 18, window=32)
    assert (a, b) == (3, 18)             # whole sub-video, < window


def test_window_odd():
    a, b = window_bounds(50, 0, 1000, window=5)
    assert b - a + 1 == 5 and 50 - a == 2


# ----------------------------- build_clips -----------------------------

def _synth_index(tmp: Path) -> Path:
    rec = {
        "kmz": "s3://x/kram/2.4/2.4.kmz", "city": "kram", "version": "2.4",
        "videos": [
            {"video": "s3://b/chunkA.mp4", "fps": 30.0, "n_frames": 500, "sub_videos": [
                {"sub_index": 0, "start_frame": 10, "end_frame": 400, "gt_frames": [
                    {"n": 57, "lat": 48.10, "lon": 37.50, "frame_index_in_chunk": 100},
                    {"n": 58, "lat": 48.11, "lon": 37.51, "frame_index_in_chunk": 15},  # near left edge
                ]},
            ]},
            {"video": "s3://b/bad.mp4", "error": "download failed"},  # skipped
        ],
    }
    p = tmp / "kram_2.4.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def test_build_clips(tmp_path):
    clips = build_clips([_synth_index(tmp_path)], window=32)
    assert len(clips) == 2                                   # errored video skipped
    c57 = next(c for c in clips if c["gt_n"] == 57)
    assert c57["clip_id"] == "kram_2.4_n57"
    assert c57["frame_indices"] == list(range(84, 116))      # centred, 32 frames
    assert c57["gt_lat"] == 48.10
    c58 = next(c for c in clips if c["gt_n"] == 58)
    assert c58["start_frame"] == 10                           # clipped to sub-video start
    assert len(c58["frame_indices"]) < 32


# ----------------------------- aggregate_clip -----------------------------

def test_aggregate_sum_vs_votes():
    # 3 frames. cell "A" appears in all 3 with small scores; "B" in 1 frame with a huge score.
    frames = [
        [{"cell_id": "A", "cell_score": 1.0, "lat": 0.0, "lon": 0.0, "level_m": 300},
         {"cell_id": "B", "cell_score": 0.1, "lat": 1.0, "lon": 1.0, "level_m": 300}],
        [{"cell_id": "A", "cell_score": 1.0, "lat": 0.0, "lon": 0.0, "level_m": 300}],
        [{"cell_id": "A", "cell_score": 1.0, "lat": 0.0, "lon": 0.0, "level_m": 300},
         {"cell_id": "B", "cell_score": 9.0, "lat": 1.0, "lon": 1.0, "level_m": 400}],
    ]
    agg = aggregate_clip(frames, top_n=50)
    # votes: A (3) beats B (2)
    assert agg["votes"][0]["cell_id"] == "A" and agg["votes"][0]["votes"] == 3
    # sum: B (9.1) beats A (3.0)
    assert agg["sum"][0]["cell_id"] == "B"
    b = next(c for c in agg["sum"] if c["cell_id"] == "B")
    assert b["score_sum"] == 9.1
    assert b["lat"] == 1.0 and b["level_m"] == 400          # position from the strongest frame (score 9.0)


def test_methods_wsum_max_gated():
    # frame A: strong (conf 9) sees cell X; frames B,C: weak (conf 1) see cell Y in all
    frames = [
        [{"cell_id": "X", "cell_score": 9.0, "lat": 0.0, "lon": 0.0}],
        [{"cell_id": "Y", "cell_score": 1.0, "lat": 1.0, "lon": 1.0}],
        [{"cell_id": "Y", "cell_score": 1.0, "lat": 1.0, "lon": 1.0}],
    ]
    agg = aggregate_clip(frames, top_n=50, gate_frac=0.9)
    # votes favours Y (appears twice); score-based (sum/wsum/max) favour X (one strong frame, 9 > 1+1)
    assert agg["sum"][0]["cell_id"] == "X"
    assert agg["votes"][0]["cell_id"] == "Y"
    assert agg["wsum"][0]["cell_id"] == "X"          # 9*9 >> 1*1+1*1
    assert agg["max"][0]["cell_id"] == "X"           # max 9 > 1
    # gated keeps only the high-confidence frame (A) → only X survives
    gated_ids = {c["cell_id"] for c in agg["gated"]}
    assert gated_ids == {"X"}


def test_aggregate_topn_cut():
    # top_n=1 keeps only each frame's #1 cell → B (only in frame where it ranks below A) excluded
    frames = [[{"cell_id": "A", "cell_score": 2.0, "lat": 0, "lon": 0},
               {"cell_id": "B", "cell_score": 1.0, "lat": 1, "lon": 1}]]
    agg = aggregate_clip(frames, top_n=1)
    ids = {c["cell_id"] for c in agg["sum"]}
    assert ids == {"A"}


# ----------------------------- run() end-to-end (no torch/cv2) -----------------------------

def test_run_end_to_end(tmp_path):
    gt = (48.0, 37.0)
    # cell NEAR gt (~0 m) and a FAR cell (~1 deg away, ~100 km)
    near = {"cell_id": "kram:1_lvl0", "cell_score": 5.0, "lat": 48.0001, "lon": 37.0001, "level_m": 300}
    far = {"cell_id": "kram:2_lvl0", "cell_score": 0.2, "lat": 49.0, "lon": 38.0, "level_m": 300}
    locate = {"per_query": {
        "kram:0001": {"gt": list(gt), "topk": [dict(near), dict(far)]},
        "kram:0002": {"gt": list(gt), "topk": [dict(near), dict(far)]},
    }}
    manifest = {"meta": {}, "clips": [{
        "clip_id": "kram_2.4_n57", "gt_lat": gt[0], "gt_lon": gt[1],
        "frames": [{"qid": "kram:0001"}, {"qid": "kram:0002"}, {"qid": "kram:missing"}],
    }]}
    lp = tmp_path / "loc.json"; lp.write_text(json.dumps(locate))
    mp = tmp_path / "man.json"; mp.write_text(json.dumps(manifest))
    res = run(str(lp), str(mp), str(tmp_path / "out"), topn_sweep=(10, 20), topk=10, log=lambda *a: None)

    clip = res["per_clip"]["kram_2.4_n57"]
    assert clip["n_frames_scored"] == 2                      # the "missing" qid is dropped
    # near cell wins both sum and votes → top1 within 250 m
    assert res["summary"]["sum_top10"]["top1"]["distR@250m"] == 1.0
    assert res["summary"]["votes_top10"]["top1"]["distR@250m"] == 1.0
    # KMZ written per (agg, N)
    kdir = tmp_path / "out" / "kmz"
    assert (kdir / "sum_top10.kmz").exists() and (kdir / "votes_top20.kmz").exists()
    with zipfile.ZipFile(kdir / "sum_top10.kmz") as z:
        assert "doc.kml" in z.namelist()


def test_haversine_zero():
    assert haversine_m(48.0, 37.0, 48.0, 37.0) == 0.0
