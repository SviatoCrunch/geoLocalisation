import numpy as np

from video_label_pipeline.pipeline import build_frames_manifest


class FakeBackend:
    def score(self, frames_bgr):
        return [("good", 0.9) for _ in frames_bgr]

    def embed(self, frames_bgr):
        return np.stack([f.reshape(-1, 3).mean(axis=0) for f in frames_bgr])


def _make_two_chunks(root):
    import cv2
    for chunk in ("chunk_104_2026-10-10_08-07-07_20s_c02",
                  "chunk_104_2026-10-10_08-08-35_13s_c03"):
        d = root / "104" / "10.10.2026" / chunk / "sub_01"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            cv2.imwrite(str(d / f"{chunk}_f{i:06d}.jpg"),
                        np.full((16, 16, 3), (40 + i * 20, 0, 0), dtype=np.uint8))


def test_resume_appends_and_skips_done(tmp_path):
    _make_two_chunks(tmp_path)
    part = str(tmp_path / "m.json.partial.jsonl")

    m1 = build_frames_manifest(tmp_path, FakeBackend(), k=2, limit=1, partial_path=part)
    assert m1["new_this_run"] == 1 and m1["resumed_chunks"] == 0 and m1["n_videos"] == 1

    m2 = build_frames_manifest(tmp_path, FakeBackend(), k=2, partial_path=part)
    assert m2["resumed_chunks"] == 1          # first chunk read back from partial
    assert m2["new_this_run"] == 1            # only the second chunk processed now
    assert m2["n_videos"] == 2
    chunks = {v["chunk"] for v in m2["videos"]}
    assert len(chunks) == 2                   # no duplicates

    # partial JSONL holds exactly the two chunk records
    with open(part, encoding="utf-8") as f:
        assert len([ln for ln in f if ln.strip()]) == 2
