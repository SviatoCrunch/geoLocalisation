"""keyframe_interval: greedy keyframe placement by overlap decay (pure logic)."""
from s3_gt_sync.keyframe_interval import place_keyframes, _sample_indices, _bad_set


def test_high_overlap_one_keyframe():
    keys = place_keyframes([0, 1, 2, 3, 4], lambda k, t: 0.9, 0.5)
    assert keys == [0]                                    # never drops -> single mask


def test_overlap_decay_opens_new_keyframes():
    # overlap to the current key decays with distance; resets after a new key
    def ratio(k, t):
        return 1.0 - 0.2 * (t - k)                        # 0.8,0.6,0.4,... from key
    keys = place_keyframes([0, 1, 2, 3, 4, 5, 6], ratio, 0.5)
    # from 0: t1=0.8,t2=0.6,t3=0.4(<0.5)->key3; from 3: t4=0.8,t5=0.6,t6=0.4->key6
    assert keys == [0, 3, 6]


def test_empty():
    assert place_keyframes([], lambda k, t: 1.0, 0.5) == []


def test_sample_indices_skips_bad_and_strides():
    v = {"bad_frames": [[2, 4]]}
    bad = _bad_set(v)
    sub = {"start_frame": 0, "end_frame": 10}
    assert _sample_indices(sub, bad, 2) == [0, 6, 8, 10]  # 2,4 bad; stride 2 skips odd


def test_bad_set():
    assert _bad_set({"bad_frames": [[1, 2], [5, 5]]}) == {1, 2, 5}
