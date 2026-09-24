"""upscale_resrgan: video selection + frame ordering + speed math (pure logic)."""
from s3_gt_sync.upscale_resrgan import is_sub_clip, want, out_frame_order, fps_of


def test_is_sub_clip():
    assert is_sub_clip("chunk_104_c01__sub00_f0-118_first.mp4") is True
    assert is_sub_clip("chunk_104_c01.mp4") is False


def test_want_full_sub_all():
    full = "d/chunk_104_c01.mp4"
    sub = "d/chunk_104_c01__sub01_f130-260_overlap_cut.mp4"
    assert want(full, "full") and not want(sub, "full")   # full = the big videos
    assert want(sub, "sub") and not want(full, "sub")
    assert want(full, "all") and want(sub, "all")


def test_out_frame_order_numeric_not_lexical():
    names = ["000010_out.png", "000002_out.png", "000001_out.png"]
    assert out_frame_order(names) == ["000001_out.png", "000002_out.png", "000010_out.png"]


def test_fps_of_and_zero_guard():
    assert fps_of(180, 9.0) == 20.0
    assert fps_of(100, 0.0) == 0.0                         # no div-by-zero
