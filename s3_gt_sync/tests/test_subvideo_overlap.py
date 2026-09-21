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

def test_continuous_single_run():
    keep = [0, 1, 2, 3]
    runs = link_runs(keep, lambda a, b: 1.0, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert runs == [(0, 3)]


def test_cut_splits_two_runs():
    keep = [0, 1, 2, 3]
    # overlap collapses between frame 1 and 2 (a splice)
    ratio = {(0, 1): 0.9, (1, 2): 0.02, (2, 3): 0.9}
    runs = link_runs(keep, lambda a, b: ratio[(a, b)], bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert runs == [(0, 1), (2, 3)]


def test_bridge_short_bad_gap():
    # frames 2,3,4 are bad -> keep jumps 1 -> 5; high overlap bridges it
    keep = [0, 1, 5, 6]
    runs = link_runs(keep, lambda a, b: 0.8, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert runs == [(0, 6)]


def test_long_dropout_not_bridged():
    keep = [0, 1, 20, 21]  # 18 bad frames between -> exceeds bridge_max_gap
    runs = link_runs(keep, lambda a, b: 0.9, bridge_max_gap=8,
                     min_inlier_ratio=0.15, min_scene_len=1)
    assert runs == [(0, 1), (20, 21)]


def test_min_scene_len_drops_short():
    keep = [0, 10, 11, 12]
    ratio = {(0, 10): 0.02, (10, 11): 0.9, (11, 12): 0.9}  # 0 isolated, 10..12 continuous
    runs = link_runs(keep, lambda a, b: ratio[(a, b)], bridge_max_gap=100,
                     min_inlier_ratio=0.15, min_scene_len=2)
    assert runs == [(10, 12)]                                 # (0,0) dropped


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
