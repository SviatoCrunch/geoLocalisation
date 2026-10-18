import numpy as np

from video_label_pipeline.daynight import chroma, day_night


def test_gray_frame_is_night():
    gray = np.full((32, 32, 3), 80, dtype=np.uint8)   # equal channels -> chroma 0
    label, c = day_night(gray, thr=12.0)
    assert c is not None and c < 1.0
    assert label == "night"


def test_colored_frame_is_day():
    blue = np.zeros((32, 32, 3), dtype=np.uint8)
    blue[..., 0] = 200                                 # high chroma
    label, c = day_night(blue, thr=12.0)
    assert c > 12.0
    assert label == "day"


def test_none_is_unknown():
    label, c = day_night(None)
    assert label == "unknown" and c is None


def test_chroma_downscales_long_side():
    big = np.zeros((1000, 500, 3), dtype=np.uint8)
    assert chroma(big, long_side=256) is not None     # must not error on large input
