import numpy as np
import pytest

from sky_filter.config import SkyFilterConfig
from sky_filter.store import MaskStore
from sky_filter.schemas import SkyMask
from sky_filter.maskers import build_resolver, available_maskers, CascadingMasker, PrecomputedMasker
from ._synth import keep_top_sky, FakeNeuralMasker


def test_builtins_registered():
    assert "precomputed" in available_maskers() and "neural" in available_maskers()


def test_precomputed_hit_wins(tmp_path):
    store = MaskStore(tmp_path)
    store.save(SkyMask("a", keep_top_sky(8, 8, 4), backend="precomputed"))
    res = build_resolver(SkyFilterConfig(), store=store, neural_impl=FakeNeuralMasker())
    sm = res.resolve("a", image=None)
    assert sm.backend == "precomputed"


def test_neural_fallback_on_miss(tmp_path):
    store = MaskStore(tmp_path)                      # empty -> precomputed miss
    res = build_resolver(SkyFilterConfig(), store=store, neural_impl=FakeNeuralMasker())
    sm = res.resolve("b", image=np.zeros((8, 8, 3), np.uint8))
    assert sm.backend == "neural" and sm.keep.shape == (8, 8)


def test_all_backends_miss_raises():
    # only precomputed, no store -> always None -> resolver raises
    res = CascadingMasker([PrecomputedMasker(None)])
    with pytest.raises(RuntimeError):
        res.resolve("x", image=None)
