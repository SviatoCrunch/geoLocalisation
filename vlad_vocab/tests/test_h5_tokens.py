"""Token reader + sampler on a synthetic group-per-tile H5 (skips if no h5py)."""
import numpy as np
import pytest

from vlad_vocab.h5_tokens import city_from_stem, sample_tokens, tile_groups, tokens_from_ift


def test_city_from_stem_strips_new_and_old_tails():
    assert city_from_stem("map_dinov2_kramatorsc_s250m_d1024") == "kramatorsc"
    assert city_from_stem("map_dinov2_1000_kup") == "kup"
    assert city_from_stem("map_dinov2_1000_liman_day_fp16") == "liman_day"


def test_tokens_from_ift_shapes():
    arr = np.random.rand(1, 8, 4, 5).astype(np.float32)     # (1, D, H, W)
    toks, D = tokens_from_ift(arr)
    assert D == 8 and tuple(toks.shape) == (20, 8)


def _write_h5(path, n_tiles=3, D=8, H=4, W=4):
    import h5py
    with h5py.File(path, "w") as f:
        for i in range(n_tiles):
            g = f.create_group(f"{i}_lvl0")
            g.create_dataset("ift_dino", data=np.random.rand(1, D, H, W).astype(np.float32))
            g.attrs["tile_index"] = i
            g.attrs["lat"] = 48.5
            g.attrs["lon"] = 37.8


def test_tile_groups_and_sampling(tmp_path):
    pytest.importorskip("h5py")
    import torch  # noqa: F401
    p = tmp_path / "map_dinov2_kramatorsc_s250m_d1024.h5"
    _write_h5(p, n_tiles=3, D=8, H=4, W=4)
    import h5py
    with h5py.File(p, "r") as f:
        assert tile_groups(f) == ["0_lvl0", "1_lvl0", "2_lvl0"]
    X, n_tiles, D = sample_tokens([p], per_tile=5, max_tokens=1000, seed=0)
    assert n_tiles == 3 and D == 8
    assert X.shape[1] == 8 and X.shape[0] == 3 * 5          # 5 sampled per tile (16 available)
    import torch
    assert torch.allclose(X.norm(dim=1), torch.ones(X.shape[0]), atol=1e-5)   # L2-normalised
