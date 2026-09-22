"""relocate_gt: find a mis-associated GT still's true video+frame (pure logic)."""
from s3_gt_sync.relocate_gt import (candidate_videos, misplaced_anchors,
                                     find_true_home, place_anchor_in_rec, remove_anchor)


def _rec():
    return {"videos": [
        {"video": "s3://b/A.mp4", "sub_videos": [
            {"sub_index": 0, "start_frame": 0, "end_frame": 100,
             "gt_frames": [{"n": 1, "ncc": 0.999, "frame_index_in_chunk": 10},
                           {"n": 2, "ncc": 0.51, "frame_index_in_chunk": 20,
                            "lat": 48.0, "lon": 37.0, "file": "2_x.jpg"}]}]},
        {"video": "s3://b/B.mp4", "sub_videos": [
            {"sub_index": 0, "start_frame": 0, "end_frame": 300, "gt_frames": []}]},
    ]}


def test_candidate_videos_distinct():
    assert candidate_videos(_rec()) == ["s3://b/A.mp4", "s3://b/B.mp4"]


def test_misplaced_anchors_by_ncc():
    mp = misplaced_anchors(_rec(), 0.9)
    assert [a["n"] for a in mp] == [2]                 # only the ncc=0.51 anchor
    assert mp[0]["from_video"] == "s3://b/A.mp4" and mp[0]["from_ncc"] == 0.51


def test_misplaced_includes_gt_low_ncc():
    rec = _rec()
    rec["videos"][0]["gt_low_ncc"] = [{"n": 5, "ncc": 0.4, "frame_index_in_chunk": 7}]
    ns = sorted(a["n"] for a in misplaced_anchors(rec, 0.9))
    assert ns == [2, 5]


def test_find_true_home_picks_best_and_skips_source():
    # B matches best; A is the (wrong) source and must be skipped
    ncc = {"s3://b/A.mp4": 0.51, "s3://b/B.mp4": 0.9998, "s3://b/C.mp4": 0.7}
    get = lambda u: u                                   # fake: url is its own "path"
    match = lambda still, vp: (123, ncc[vp], 30.0)
    best = find_true_home("still.jpg", list(ncc), get, match, skip_url="s3://b/A.mp4")
    assert best[0] == "s3://b/B.mp4" and best[1][0] == 123 and best[1][1] == 0.9998


def test_find_true_home_handles_missing_video():
    get = lambda u: None if u == "s3://b/B.mp4" else u  # B undownloadable
    match = lambda still, vp: (1, 0.95, 30.0)
    best = find_true_home("s.jpg", ["s3://b/B.mp4", "s3://b/C.mp4"], get, match)
    assert best[0] == "s3://b/C.mp4"


def test_place_and_remove_roundtrip():
    rec = _rec()
    remove_anchor(rec, "s3://b/A.mp4", 2)
    assert all(f["n"] != 2 for f in rec["videos"][0]["sub_videos"][0]["gt_frames"])
    status = place_anchor_in_rec(rec, "s3://b/B.mp4",
                                 {"n": 2, "file": "2_x.jpg", "lat": 48.0, "lon": 37.0,
                                  "frame_index_in_chunk": 150, "time_s": 5.0, "ncc": 0.9998})
    gf = rec["videos"][1]["sub_videos"][0]["gt_frames"]
    assert status == "placed in sub0" and gf[-1]["n"] == 2
    assert gf[-1]["frame_index_in_subvideo"] == 150 and gf[-1]["relocated"] is True


def test_place_on_bad_frame_when_outside_subvideo():
    rec = _rec()
    status = place_anchor_in_rec(rec, "s3://b/B.mp4",
                                 {"n": 9, "frame_index_in_chunk": 999})
    assert "bad" in status and 9 in rec["videos"][1]["relocated_on_bad_frame"]
