import numpy as np

from video_label_pipeline.frames_tree import iter_chunk_groups
from video_label_pipeline.pipeline import process_frames_chunk


class FakeBackend:
    def score(self, frames_bgr):
        return [("good", 0.9) for _ in frames_bgr]

    def embed(self, frames_bgr):
        return np.stack([f.reshape(-1, 3).mean(axis=0) for f in frames_bgr])


def _make_tree(root):
    import cv2
    chunk = "chunk_104_2026-10-10_08-07-07_20s_c02"
    d = root / "104" / "10.10.2026" / chunk
    for sub, idxs in (("sub_01", range(0, 6)), ("sub_02", range(6, 12))):
        (d / sub).mkdir(parents=True, exist_ok=True)
        for i in idxs:
            img = np.full((32, 32, 3), (40 + i * 10, 0, 0), dtype=np.uint8)
            cv2.imwrite(str(d / sub / f"{chunk}_f{i:06d}.jpg"), img)
    return chunk


def test_iter_chunk_groups_discovers_subs(tmp_path):
    chunk = _make_tree(tmp_path)
    groups = list(iter_chunk_groups(tmp_path))
    assert len(groups) == 1
    g = groups[0]
    assert g.station == "104" and g.date == "10.10.2026" and g.chunk == chunk
    assert sorted(g.subs) == ["sub_01", "sub_02"]
    assert [i for i, _ in g.subs["sub_01"]] == [0, 1, 2, 3, 4, 5]


def test_process_frames_chunk_selects_and_tags(tmp_path):
    _make_tree(tmp_path)
    g = next(iter_chunk_groups(tmp_path))
    rec = process_frames_chunk(g, FakeBackend(), k=3)
    assert rec["source"] == "frames" and rec["n_subvideos"] == 2
    for sub in rec["subvideos"]:
        assert sub["boundary"] is None
        assert sub["n_frames"] == 6
        assert sub["quality"]["good"] == 6
        sel = sub["selected"]
        assert len(sel) == 3 and len({s["frame_idx"] for s in sel}) == 3
        assert all("path" in s and "file" in s and s["timestamp_s"] is None for s in sel)
        assert all(s["day_night"] in ("day", "night") for s in sel)
