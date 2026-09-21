"""gt_stats: anchor-pair distances + extractable-between counts (pure math)."""
from s3_gt_sync.gt_stats import haversine_m, anchor_pairs, summarize


def _rec(videos):
    return {"city": "t", "version": "9.9", "videos": videos}


def _sub(sub_index, pairs):
    return {"sub_index": sub_index,
            "gt_frames": [{"n": n, "frame_index_in_subvideo": si,
                           "frame_index_in_chunk": ci, "lat": lat, "lon": lon}
                          for (n, si, ci, lat, lon) in pairs]}


def test_haversine_known_distance():
    # 0.001 deg latitude ~ 111.19 m
    assert abs(haversine_m((0.0, 0.0), (0.001, 0.0)) - 111.19) < 1.0


def test_pair_n_between_and_distance():
    rec = _rec([{"video": "s3://x/chunk_a.mp4", "fps": 10.0,
                 "sub_videos": [_sub(0, [(1, 0, 0, 48.0, 37.0),
                                         (2, 10, 10, 48.001, 37.0)])]}])
    pairs = anchor_pairs(rec)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["frame_gap"] == 10 and p["n_between"] == 9      # 9 frames extractable
    assert p["sec_gap"] == 1.0 and abs(p["speed_mps"] - 111.19) < 1.0
    assert abs(p["dist_m"] - 111.2) < 1.0


def test_single_anchor_yields_no_pair():
    rec = _rec([{"video": "s3://x/c.mp4", "fps": 30.0,
                 "sub_videos": [_sub(0, [(1, 5, 5, 48.0, 37.0)])]}])
    s = summarize(rec)
    assert s["n_anchors"] == 1 and s["n_interpolatable_pairs"] == 0
    assert s["n_between_total_extractable"] == 0
    assert s["videos_single_anchor"] == 1


def test_summary_totals_over_two_videos():
    rec = _rec([
        {"video": "s3://x/a.mp4", "fps": 30.0,
         "sub_videos": [_sub(0, [(1, 0, 0, 48.0, 37.0), (2, 4, 4, 48.0, 37.001)])]},
        {"video": "s3://x/b.mp4", "fps": 30.0,
         "sub_videos": [_sub(0, [(3, 2, 2, 48.0, 37.0)])]},  # single anchor
    ])
    s = summarize(rec)
    assert s["n_anchors"] == 3
    assert s["n_interpolatable_pairs"] == 1
    assert s["n_between_total_extractable"] == 3            # span 4 -> 3 between
    assert s["extractable_total_with_anchors"] == 6
    assert s["videos_single_anchor"] == 1


def test_no_extrapolation_only_between_anchors():
    # three anchors -> two pairs, counts add up; nothing outside the anchor span
    rec = _rec([{"video": "s3://x/c.mp4", "fps": None,
                 "sub_videos": [_sub(0, [(1, 0, 0, 0.0, 0.0), (2, 3, 3, 0.0, 0.001),
                                         (3, 6, 6, 0.001, 0.001)])]}])
    pairs = anchor_pairs(rec)
    assert [p["n_between"] for p in pairs] == [2, 2]
    assert all(p["sec_gap"] is None for p in pairs)          # fps None -> null time
