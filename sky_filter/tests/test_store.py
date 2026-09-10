import numpy as np

from sky_filter.store import MaskStore
from sky_filter.schemas import SkyMask
from ._synth import keep_top_sky


def test_roundtrip_and_manifest(tmp_path):
    store = MaskStore(tmp_path)
    keep = keep_top_sky(6, 6, 3)
    store.save(SkyMask("kram:017_x", keep, backend="precomputed"))
    got = store.load("kram:017_x")
    assert got is not None and np.array_equal(got, keep)
    assert store.has("kram:017_x")
    assert store.frame_ids() == ["kram:017_x"]
    assert store.load("missing") is None


def test_fingerprint_changes_with_content(tmp_path):
    store = MaskStore(tmp_path)
    store.save(SkyMask("a", keep_top_sky(4, 4, 2), backend="neural"))
    fp1 = store.fingerprint()
    store.save(SkyMask("b", keep_top_sky(4, 4, 1), backend="neural"))
    fp2 = store.fingerprint()
    assert fp1 != fp2 and fp1.startswith("sky:")


def test_fingerprint_stable(tmp_path):
    s1, s2 = MaskStore(tmp_path / "a"), MaskStore(tmp_path / "b")
    for s in (s1, s2):
        s.save(SkyMask("a", keep_top_sky(4, 4, 2), backend="neural"))
    assert s1.fingerprint() == s2.fingerprint()
