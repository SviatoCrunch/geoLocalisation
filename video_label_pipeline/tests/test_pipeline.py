import numpy as np

from video_label_pipeline.pipeline import SplitRecipe, process_decoded
from video_label_pipeline.source import VideoRef


class FakeBackend:
    """No-torch backend: everything is 'good'; embedding = per-frame mean BGR."""

    def score(self, frames_bgr):
        return [("good", 0.9) for _ in frames_bgr]

    def embed(self, frames_bgr):
        return np.stack([f.reshape(-1, 3).mean(axis=0) for f in frames_bgr])


def _frame(bgr_tuple, size=16):
    return np.full((size, size, 3), bgr_tuple, dtype=np.uint8)


def _make_frames():
    frames = []
    idx = 0
    # Block A: grayscale ramp (night), slight variation so embeddings differ.
    for g in (40, 48, 56, 64, 72, 80):
        frames.append((idx, idx / 10.0, _frame((g, g, g))))
        idx += 1
    # Block B: blue ramp (day).
    for b in (120, 140, 160, 180, 200, 220):
        frames.append((idx, idx / 10.0, _frame((b, 0, 0))))
        idx += 1
    return frames


def test_process_decoded_splits_scores_selects_tags():
    frames = _make_frames()
    ref = VideoRef(station="104", date="10.10.2026",
                   chunk="chunk_104_2026-10-10_08-07-07_20s_c02")
    rec = process_decoded(ref, fps=10.0, n_frames=len(frames), frames=frames,
                          backend=FakeBackend(), recipe=SplitRecipe(), k=5)

    assert rec["n_subvideos"] == 2
    a, b = rec["subvideos"]
    assert a["n_frames"] == 6 and b["n_frames"] == 6
    assert a["boundary"] is None                    # first sub: no cut before it
    assert b["boundary"]["cut_before_frame"] == 6
    assert b["boundary"]["content_delta"] >= 15.0

    # k good frames selected per sub, all distinct, ranked, with day/night.
    for sub, expect in ((a, "night"), (b, "day")):
        sel = sub["selected"]
        assert len(sel) == 5
        assert len({s["frame_idx"] for s in sel}) == 5
        assert [s["rank"] for s in sel] == [1, 2, 3, 4, 5]
        assert all(s["label"] == "good" for s in sel)
        assert all(s["day_night"] == expect for s in sel)
        assert all("confidence" in s and "timestamp_s" in s for s in sel)
        assert sub["quality"]["good"] == 6


def test_manifest_fields_present():
    frames = _make_frames()
    ref = VideoRef(station="40", date="11.10.2026", chunk="chunk_40_2026-10-11_x",
                   s3_uri="s3://b/live/40/filtered/11.10.2026/chunk_40_2026-10-11_x.mp4",
                   bucket="b", key="live/40/filtered/11.10.2026/chunk_40_2026-10-11_x.mp4")
    rec = process_decoded(ref, 10.0, len(frames), frames, FakeBackend(), SplitRecipe(), k=3)
    assert rec["station"] == "40" and rec["date"] == "11.10.2026"
    assert rec["bucket"] == "b" and rec["key"].endswith(".mp4")
    assert rec["s3_uri"].startswith("s3://")
    assert all(len(s["selected"]) <= 3 for s in rec["subvideos"])
