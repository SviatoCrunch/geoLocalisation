"""8.8 cache round-trip + strict fingerprint incompatibility.

Run: cd geoLocalisation && python -m pytest native_map_pyramid/tests/test_cache_roundtrip.py -q
"""
import numpy as np
import pytest
import torch

from native_map_pyramid import cache as C


def _ident(**over):
    base = dict(backbone="dinov2_vitg14", dino_layer=None, dino_facet="value", output_px=840,
                tile_size_m=1000.0, patch_size=14, projection="random_gaussian_jl",
                projection_seed=0, projection_in_dim=1536, projection_out_dim=1024,
                n_groups=32, d_value=1024, vlad_dict_id="deadbeef")
    base.update(over)
    return C.build_fingerprint(**base)


def _synth_S(M=3, K=32, Dv=1024, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(M, C.N_CELLS_TOTAL, K, Dv).astype(np.float32)


def test_roundtrip_numeric_and_metadata(tmp_path):
    S = _synth_S()
    ids = ["kup:0", "kup:1", "kramatorsc:7"]
    ident = _ident()
    path = tmp_path / "S.h5"
    C.write_cache(path, S, ids, ident, extra_meta={"n_tiles": 3, "note": "unit"})

    cache = C.NativeCellCache(path)
    assert cache.tile_ids == ids
    assert cache.fingerprint == ident["fingerprint"]
    assert cache.identity["vlad_dict_id"] == "deadbeef"
    assert cache.meta["note"] == "unit"

    loaded = cache.load(ids)                             # {n:(M,n²,K,Dv)}
    assert set(loaded) == {8, 4, 2, 1}
    recon = torch.cat([loaded[n] for n in (8, 4, 2, 1)], dim=1).numpy()   # canonical order
    # fp16 storage → compare at fp16 tolerance
    assert np.allclose(recon, S.astype(np.float16).astype(np.float32), atol=1e-3, rtol=1e-2)
    cache.close()


def test_subset_and_reordered_load(tmp_path):
    S = _synth_S(M=4)
    ids = ["c:a", "c:b", "c:c", "c:d"]
    path = tmp_path / "S.h5"
    C.write_cache(path, S, ids, _ident())
    cache = C.NativeCellCache(path)
    want = ["c:c", "c:a"]                                 # reordered subset
    loaded = cache.load(want)
    recon = torch.cat([loaded[n] for n in (8, 4, 2, 1)], dim=1).numpy()
    expect = S[[2, 0]].astype(np.float16).astype(np.float32)
    assert np.allclose(recon, expect, atol=1e-3, rtol=1e-2)
    with pytest.raises(KeyError):
        cache.load(["c:a", "c:missing"])
    cache.close()


def test_compatible_passes(tmp_path):
    path = tmp_path / "S.h5"
    C.write_cache(path, _synth_S(), ["c:a", "c:b", "c:c"], _ident())
    cache = C.NativeCellCache(path)
    cache.assert_compatible(_ident())                    # identical → no raise
    cache.close()


@pytest.mark.parametrize("field,val", [
    ("backbone", "dinov3_vitl16"),
    ("vlad_dict_id", "cafebabe"),
    ("output_px", 1008),
    ("dino_facet", "key"),
    ("tile_size_m", 500.0),
    ("projection_out_dim", 512),
])
def test_incompatible_raises_naming_field(tmp_path, field, val):
    path = tmp_path / "S.h5"
    C.write_cache(path, _synth_S(), ["c:a", "c:b", "c:c"], _ident())
    cache = C.NativeCellCache(path)
    with pytest.raises(C.CacheIncompatibleError) as ei:
        cache.assert_compatible(_ident(**{field: val}))
    assert field in str(ei.value)                        # error names the offending field
    cache.close()


def test_scale_order_change_is_detected():
    # A cache whose token_order/scale layout differs must not validate against the canonical run.
    a = _ident()
    b = dict(a); b["token_order"] = "different-order"; b["fingerprint"] = C._digest(b)
    with pytest.raises(C.CacheIncompatibleError):
        C.assert_compatible(b, a)
