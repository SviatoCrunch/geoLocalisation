"""fpv_video_qc.core — line-noise metric + segment grouping (pure numpy/stdlib, no cv2)."""
import numpy as np

from fpv_video_qc.core import bad_segments, combine_score, line_noise_from_gray


def test_line_noise_clean_vs_garbage():
    rng = np.random.default_rng(0)
    # clean: smooth vertical gradient (rows differ from each other but each row is horizontally smooth)
    clean = np.tile(np.linspace(0, 255, 200, dtype=np.float32)[:, None], (1, 256))
    assert line_noise_from_gray(clean) < 0.02

    # corrupted: same clean frame but ~20 rows replaced by full-range RF garbage
    bad = clean.copy()
    rows = rng.choice(200, size=20, replace=False)
    bad[rows] = rng.integers(0, 256, size=(20, 256)).astype(np.float32)
    ln = line_noise_from_gray(bad)
    assert ln > 0.05                                          # garbage rows detected
    assert ln > line_noise_from_gray(clean)


def test_line_noise_ignores_tiny_frames():
    assert line_noise_from_gray(np.zeros((2, 2), np.float32)) == 0.0


def test_combine_score_monotonic():
    assert combine_score(0, 0, 0) == 0.0
    assert combine_score(255, 0, 0) > combine_score(0, 0, 0)
    assert combine_score(0, 0.5, 0) > combine_score(0, 0.1, 0)


def test_bad_segments_grouping_merge_and_minlen():
    # frames:            0 1 2 3 4 5 6 7 8 9
    flags = [False, True, True, False, True, False, False, False, True, False]
    # runs: [1-2], [4-4], [8-8]. merge_gap=1 merges [1-2]+[4-4] (gap 1). [8-8] is 3 gap away -> separate.
    segs = bad_segments(flags, min_len=1, merge_gap=1)
    assert segs == [(1, 4), (8, 8)]
    # min_len=2 drops the isolated [8-8]
    assert bad_segments(flags, min_len=2, merge_gap=1) == [(1, 4)]
    # no merge (gap 0): [1-2],[4-4] stay separate; min_len=2 keeps only [1-2]
    assert bad_segments(flags, min_len=2, merge_gap=0) == [(1, 2)]


def test_bad_segments_empty():
    assert bad_segments([False, False, False]) == []
    assert bad_segments([]) == []
