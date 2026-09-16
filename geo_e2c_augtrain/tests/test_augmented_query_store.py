"""AugmentedQueryStore plumbing (indexing, per-epoch augment, batching, sky-filter, output shape) with
DINO/extractor monkeypatched (no weight downloads) + vendored uav_augment sanity."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("PIL")


def _ref_h5(tmp_path):
    import h5py
    p = tmp_path / "ref.h5"
    with h5py.File(p, "w") as f:
        f.attrs["backbone"] = "dinov2_vitg14"
        f.attrs["projection_out_dim"] = 16
        f.attrs["projection_seed"] = 0
        f.attrs["dino_facet"] = "value"
    return str(p)


def _patch(monkeypatch):
    """Stub the DINO extractor + feature extraction so no weights are downloaded."""
    class _Ext:
        patch_size = 14
        preprocess = None
    monkeypatch.setattr("map_extract.dino.build_dino_extractor", lambda *a, **k: _Ext())
    monkeypatch.setattr("map_extract.dino.RandomProjector", lambda *a, **k: (lambda x: x))
    monkeypatch.setattr("map_extract.extract._tile_to_batch",
                        lambda imgs, dev, ps, pp: (imgs, 2, 2))            # batch = the (augmented) images
    # feat DEPENDS on image content (like real DINO) so different augmentation -> different tokens
    monkeypatch.setattr("map_extract.extract._extract_level_features",
                        lambda dino, batch, h, w, proj, amp=False:
                        [np.random.default_rng(int(np.asarray(im).astype(np.int64).sum()) & 0xFFFFFFFF)
                         .standard_normal((1, 16, h, w)).astype(np.float32) for im in batch])


def test_store_reencode_shape_and_determinism(tmp_path, monkeypatch):
    from PIL import Image
    _patch(monkeypatch)
    gt = tmp_path / "gt_kup"
    gt.mkdir()
    for stem in ("101_48.9_37.6", "102_48.91_37.61"):
        Image.fromarray((np.random.rand(56, 56, 3) * 255).astype(np.uint8)).save(gt / f"{stem}.jpg")

    from geo_e2c_augtrain.augmented_query_store import AugmentedQueryStore
    st = AugmentedQueryStore({"kup": str(gt)}, _ref_h5(tmp_path), device="cpu",
                             preset="stage3_strong", segment_sky=False, base_seed=0, batch=8)
    qids = sorted(st.ids())
    assert qids == ["kup:101_48.9_37.6", "kup:102_48.91_37.61"]
    assert all(st.has(q) for q in qids) and not st.has("kup:nope")

    st.set_epoch(0, qids=qids)
    t0 = {q: st.tokens(q).clone() for q in qids}
    for q in qids:
        assert t0[q].shape == (4, 16)                     # 2x2 patch grid, D=16, no sky drop

    st.set_epoch(0, qids=qids)                            # same epoch -> identical (deterministic)
    for q in qids:
        assert np.allclose(st.tokens(q).numpy(), t0[q].numpy())

    st.set_epoch(1, qids=qids)                            # different epoch -> different augmentation
    assert any(not np.allclose(st.tokens(q).numpy(), t0[q].numpy()) for q in qids)


def test_vendored_uav_augment_shape_and_seed():
    from geo_e2c_augtrain.uav_augment import AugConfig, augment_frame, make_aug_rng
    cfg = AugConfig.from_cfg({"enabled": True, "preset": "stage3_strong"})
    rgb = (np.random.default_rng(1).random((48, 64, 3)) * 255).astype(np.uint8)
    a1, m1, _, _ = augment_frame(rgb, None, cfg, make_aug_rng(0, 5, 7))
    a2, _, _, _ = augment_frame(rgb, None, cfg, make_aug_rng(0, 5, 7))
    assert a1.shape == rgb.shape and a1.dtype == np.uint8
    assert np.array_equal(a1, a2)                         # same (seed,epoch,worker) -> identical
    a3, _, _, _ = augment_frame(rgb, None, cfg, make_aug_rng(0, 6, 7))
    assert not np.array_equal(a1, a3)                     # different epoch -> different
