"""subvideo_index: sub-video splitting + kmz path parsing (pure logic; no cv2/quality-cls/aws)."""
from s3_gt_sync.subvideo_index import _city_ver, gt_index, split_subvideos


def test_split_good_usable_runs():
    # good/usable = keep; bad breaks a sub-video. runs are inclusive (start, end).
    g = ["good", "usable", "good", "bad", "bad", "usable", "good", "bad", "good"]
    assert split_subvideos(g) == [(0, 2), (5, 6), (8, 8)]


def test_split_all_good_single_run():
    assert split_subvideos(["good"] * 5) == [(0, 4)]


def test_split_leading_trailing_bad():
    assert split_subvideos(["bad", "good", "good", "bad"]) == [(1, 2)]


def test_split_empty_and_all_bad():
    assert split_subvideos([]) == []
    assert split_subvideos(["bad", "bad"]) == []


def test_city_ver_from_kmz_url():
    assert _city_ver("s3://geo-reference/gt/raw/kram/2.7/2.7.kmz") == ("kram", "2.7")
    assert _city_ver("s3://b/gt/raw/lyman/3.1/3.1.kmz") == ("lyman", "3.1")


def test_gt_index(tmp_path):
    for name in ("232_48.5_37.7.jpg", "42_49.6_37.6.jpg", "noindex.jpg"):
        (tmp_path / name).write_bytes(b"x")
    idx = gt_index(tmp_path)
    assert set(idx) == {232, 42}
    assert idx[232].name.startswith("232_")
