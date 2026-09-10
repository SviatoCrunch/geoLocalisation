import numpy as np

from sky_filter.config import SkyFilterConfig
from sky_filter.store import MaskStore
from sky_filter.ingest import ingest_sky_masks
from sky_filter.maskers import build_resolver
from ._synth import FakeNeuralMasker, keep_top_sky


def _touch(dir_, name):
    (dir_ / name).write_bytes(b"")          # empty placeholder; fake reader ignores content


def _reader_factory(mapping):
    def _read(path):
        from pathlib import Path
        return mapping[Path(path).name]     # (H,W) array: non-zero = sky
    return _read


def test_ingest_inverts_sky_to_keep(tmp_path):
    md = tmp_path / "GT_flat_mask"
    md.mkdir()
    _touch(md, "48.59_37.59__Sky.png")
    sky = np.zeros((8, 8), np.uint8); sky[:4] = 255       # top half sky
    store = MaskStore(tmp_path / "store")
    summ = ingest_sky_masks(md, "cramatorsc", store,
                            image_reader=_reader_factory({"48.59_37.59__Sky.png": sky}))
    assert summ["ingested"] == 1
    keep = store.load("cramatorsc:48.59_37.59")
    assert keep is not None
    assert not keep[:4].any() and keep[4:].all()          # sky dropped, ground kept


def test_ingest_skips_corrupt_latitude(tmp_path):
    md = tmp_path / "GT_flat_mask"; md.mkdir()
    _touch(md, "8.5916493467_37.597090358__Sky.png")      # bad lat (missing '4')
    _touch(md, "48.59_37.59__Sky.png")
    arr = np.zeros((4, 4), np.uint8)
    store = MaskStore(tmp_path / "store")
    summ = ingest_sky_masks(md, "cramatorsc", store, image_reader=_reader_factory({
        "8.5916493467_37.597090358__Sky.png": arr, "48.59_37.59__Sky.png": arr}))
    assert summ["ingested"] == 1
    assert summ["skipped_bad"] == ["8.5916493467_37.597090358"]


def test_ingested_frames_use_precomputed_others_neural(tmp_path):
    md = tmp_path / "GT_flat_mask"; md.mkdir()
    _touch(md, "48.60_37.60__Sky.png")
    store = MaskStore(tmp_path / "store")
    ingest_sky_masks(md, "cramatorsc", store,
                     image_reader=_reader_factory({"48.60_37.60__Sky.png": np.zeros((8, 8), np.uint8)}))
    res = build_resolver(SkyFilterConfig(), store=store, neural_impl=FakeNeuralMasker())
    # frame with a GT mask -> precomputed
    assert res.resolve("cramatorsc:48.60_37.60", image=None).backend == "precomputed"
    # frame without one -> neural fallback
    assert res.resolve("cramatorsc:99.99_99.99",
                       image=np.zeros((8, 8, 3), np.uint8)).backend == "neural"
