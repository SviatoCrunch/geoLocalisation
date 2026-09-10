import numpy as np

from sky_filter.config import SkyFilterConfig
from sky_filter.store import MaskStore
from sky_filter.maskers import build_resolver
from sky_filter.precompute import precompute
from ._synth import FakeNeuralMasker


def _loader(_fid):
    return np.zeros((8, 8, 3), np.uint8)


def test_precompute_writes_and_is_reproducible(tmp_path):
    store = MaskStore(tmp_path)
    res = build_resolver(SkyFilterConfig(), store=store, neural_impl=FakeNeuralMasker())
    frames = ["a", "b", "c"]
    summ = precompute(frames, _loader, res, store)
    assert summ["n_written"] == 3 and summ["by_backend"]["neural"] == 3
    assert all(store.has(f) for f in frames)
    fp1 = store.fingerprint()

    # rerun: everything already present -> skipped, fingerprint unchanged
    summ2 = precompute(frames, _loader, res, store)
    assert summ2["n_skipped"] == 3 and summ2["n_written"] == 0
    assert store.fingerprint() == fp1


def test_precompute_prefers_existing_precomputed(tmp_path):
    store = MaskStore(tmp_path)
    # pre-seed one frame as a "precomputed" mask
    from sky_filter.schemas import SkyMask
    from ._synth import keep_top_sky
    store.save(SkyMask("a", keep_top_sky(8, 8, 2), backend="precomputed"))
    res = build_resolver(SkyFilterConfig(), store=store, neural_impl=FakeNeuralMasker())
    summ = precompute(["a", "b"], _loader, res, store)   # 'a' skipped (exists), 'b' via neural
    assert summ["n_skipped"] == 1 and summ["by_backend"].get("neural") == 1
