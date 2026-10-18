import numpy as np

from video_label_pipeline.segment import (
    boundary_context,
    merge_short,
    segment_frames,
    segments_from_cuts,
    to_feature,
)


def _hsv(h, s, v, size=4):
    return np.full((size, size, 3), (h, s, v), dtype=np.float32)


def test_single_cut_between_two_uniform_blocks():
    feats = [_hsv(0, 0, 50) for _ in range(10)] + [_hsv(0, 0, 200) for _ in range(10)]
    segs, deltas, cuts = segment_frames(feats, adaptive_thr=3.0, min_content=15.0,
                                        window=2, min_seg=2)
    assert len(segs) == 2
    assert segs[0] == list(range(10)) and segs[1] == list(range(10, 20))
    assert 10 in cuts
    assert deltas[10] >= 15.0


def test_no_cut_for_constant_sequence():
    feats = [_hsv(10, 20, 100) for _ in range(12)]
    segs, _, cuts = segment_frames(feats)
    assert cuts == set()
    assert len(segs) == 1 and segs[0] == list(range(12))


def test_merge_short_absorbs_tiny_segment():
    assert merge_short([[0, 1, 2], [3], [4, 5]], 2) == [[0, 1, 2], [3, 4, 5]]
    assert merge_short([[0], [1, 2, 3]], 2) == [[0, 1, 2, 3]]


def test_segments_from_cuts_partition():
    assert segments_from_cuts(5, {2}) == [[0, 1], [2, 3, 4]]
    assert segments_from_cuts(3, set()) == [[0, 1, 2]]


def test_boundary_context_explains_cut():
    # a clear spike at index 3 over near-zero neighbours
    deltas = [0.0, 1.0, 1.0, 40.0, 1.0, 1.0]
    ctx = boundary_context(deltas, 3, window=2, min_content=15.0, adaptive_thr=3.0)
    assert ctx["content_delta"] == 40.0
    assert ctx["local_avg"] == 1.0                   # neighbours 1,2,4,5 all == 1.0
    assert ctx["ratio"] == round(40.0 / (1.0 + 1e-6), 3)
    assert ctx["min_content"] == 15.0 and ctx["adaptive_thr"] == 3.0


def test_to_feature_reduces_and_is_hsv():
    bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    bgr[..., 0] = 255                      # pure blue
    f = to_feature(bgr, width=480, reduce=4)
    assert f.shape == (16, 16, 3)          # 64/4
    assert f.dtype == np.float32
