"""subvideo_index continuity-aware split: structure annotation + overlap linking."""
import cv2
import numpy as np

from s3_gt_sync.subvideo_index import (rle_grades, bad_frame_ranges, link_runs,
                                        overlap_ratio)


# ---------------- structure annotation ----------------

def test_rle_grades():
    g = ["good", "good", "bad", "usable", "usable", "bad"]
    assert rle_grades(g) == [
        {"label": "good", "start": 0, "end": 1},
        {"label": "bad", "start": 2, "end": 2},
        {"label": "usable", "start": 3, "end": 4},
        {"label": "bad", "start": 5, "end": 5},
    ]


def test_bad_frame_ranges():
    g = ["good", "bad", "bad", "good", "bad"]
    assert bad_frame_ranges(g) == [[1, 2], [4, 4]]


# ---------------- run linking (the core decision) ----------------

def _se(runs):
    return [(r["start"], r["end"]) for r in runs]


def test_continuous_single_run():
    runs = link_runs([0, 1, 2, 3], lambda a, b: 1.0, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert _se(runs) == [(0, 3)]
    assert runs[0]["boundary"] == "first"


def test_cut_splits_two_runs():
    ratio = {(0, 1): 0.9, (1, 2): 0.02, (2, 3): 0.9}         # splice between 1 and 2
    runs = link_runs([0, 1, 2, 3], lambda a, b: ratio[(a, b)], bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert _se(runs) == [(0, 1), (2, 3)]
    assert runs[1]["boundary"] == "overlap_cut" and runs[1]["overlap_before"] == 0.02
    assert runs[1]["gap_before"] == 0                         # adjacent frames, hard cut


def test_bridge_short_bad_gap():
    runs = link_runs([0, 1, 5, 6], lambda a, b: 0.8, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert _se(runs) == [(0, 6)]                              # bad 2,3,4 bridged


def test_long_dropout_not_bridged():
    runs = link_runs([0, 1, 20, 21], lambda a, b: 0.9, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert _se(runs) == [(0, 1), (20, 21)]
    assert runs[1]["boundary"] == "long_dropout" and runs[1]["gap_before"] == 18


def test_min_scene_len_flags_short_not_dropped():
    ratio = {(0, 10): 0.02, (10, 11): 0.9, (11, 12): 0.9}    # 0 isolated, 10..12 continuous
    runs = link_runs([0, 10, 11, 12], lambda a, b: ratio[(a, b)], bridge_max_gap=100,
                     min_inlier_ratio=0.15, min_scene_len=2)
    assert _se(runs) == [(0, 0), (10, 12)]                    # (0,0) KEPT, not dropped
    assert runs[0]["too_short"] is True                      # 1 frame < min_scene_len 2
    assert runs[1]["too_short"] is False                     # 3 frames >= 2


def test_too_short_flag_all_false_when_min_scene_len_1():
    runs = link_runs([0, 1, 2, 3], lambda a, b: 1.0, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert all(r["too_short"] is False for r in runs)


# ---------------- overlap_ratio smoke ----------------

def test_overlap_ratio_identical_high_cut_low():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320), np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    orb = cv2.ORB_create(nfeatures=800)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp, desc = orb.detectAndCompute(img, None)
    same = overlap_ratio(desc, kp, desc, kp, matcher, 4.0)
    assert same > 0.8                                        # identical scene -> high inlier ratio

    other = cv2.GaussianBlur(rng.integers(0, 255, (240, 320), np.uint8), (0, 0), 1.0)
    kp2, desc2 = orb.detectAndCompute(other, None)
    cut = overlap_ratio(desc, kp, desc2, kp2, matcher, 4.0)
    assert cut < same                                        # unrelated scene -> lower
