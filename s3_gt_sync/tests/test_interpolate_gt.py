"""interpolate_gt: linear lat/lon densification within a sub-video (pure math)."""
from s3_gt_sync.interpolate_gt import interpolate_subvideo


def _sub(pairs):
    return {"gt_frames": [{"n": n, "frame_index_in_subvideo": si, "frame_index_in_chunk": ci,
                           "lat": lat, "lon": lon} for (n, si, ci, lat, lon) in pairs]}


def test_linear_interpolation_every_frame():
    sub = _sub([(10, 0, 100, 48.0, 37.0), (11, 4, 104, 48.4, 37.4)])
    out = interpolate_subvideo(sub, fps=10.0)
    assert [f["frame_index_in_subvideo"] for f in out] == [1, 2, 3]        # every frame between
    assert [f["frame_index_in_chunk"] for f in out] == [101, 102, 103]     # chunk idx tracks 1:1
    assert [round(f["lat"], 4) for f in out] == [48.1, 48.2, 48.3]
    assert [round(f["lon"], 4) for f in out] == [37.1, 37.2, 37.3]
    assert all(f["interpolated"] and f["between"] == [10, 11] for f in out)
    assert out[1]["t"] == 0.5 and out[1]["time_s"] == 10.2                 # chunk 102 / 10 fps


def test_adjacent_points_no_gap():
    sub = _sub([(1, 0, 0, 48.0, 37.0), (2, 1, 1, 48.1, 37.1)])
    assert interpolate_subvideo(sub, 30.0) == []                          # span==1 -> nothing between


def test_single_or_empty():
    assert interpolate_subvideo(_sub([(1, 0, 0, 48.0, 37.0)]), 30.0) == []
    assert interpolate_subvideo({"gt_frames": []}, 30.0) == []


def test_three_points_two_segments():
    sub = _sub([(1, 0, 0, 0.0, 0.0), (2, 2, 2, 2.0, 0.0), (3, 4, 4, 2.0, 2.0)])
    out = interpolate_subvideo(sub, fps=None)
    # segment 1: idx1 (lat1,lon0); segment 2: idx3 (lat2,lon1)
    assert [(f["frame_index_in_subvideo"], round(f["lat"], 3), round(f["lon"], 3)) for f in out] == \
        [(1, 1.0, 0.0), (3, 2.0, 1.0)]
    assert out[0]["time_s"] is None                                       # fps None -> null time
