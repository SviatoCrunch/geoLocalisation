"""Analog-FPV augmentation: runs, keeps geometry image<->mask, doesn't invent class ids."""
import numpy as np

from segformer3_full_taxonomy.aug import _chroma_bleed, _rf_lines


def test_custom_ops_keep_shape_dtype():
    img = (np.random.rand(40, 60, 3) * 255).astype(np.uint8)
    for fn in (_chroma_bleed, _rf_lines):
        out = fn(image=img)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_analog_dataset_valid_and_ids_preserved(tmp_path):
    import os
    from PIL import Image
    from segformer3_full_taxonomy.dataset import SkyScenesDataset
    from segformer3_full_taxonomy import taxonomy as tx

    img = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    m = np.zeros((300, 400, 3), np.uint8); m[:150] = (128, 64, 128); m[150:] = (45, 60, 150)
    Image.fromarray(img).save(tmp_path / "001_clrnoon.png")
    Image.fromarray(m).save(tmp_path / "001_semsegCarla_clrnoon.png")
    pairs = [{"image": str(tmp_path / "001_clrnoon.png"),
              "mask": str(tmp_path / "001_semsegCarla_clrnoon.png"), "town": "T"}]
    s = SkyScenesDataset(pairs, train=True, crop=256, aug="analog")[0]
    assert s["pixel_values"].shape == (3, 256, 256) and s["labels"].shape == (256, 256)
    ids = set(map(int, s["labels"].unique().tolist()))
    # analog aug degrades the IMAGE only; labels stay valid road/water ids (mask untouched)
    assert ids <= {tx.CLASS_TO_ID["road"], tx.CLASS_TO_ID["water"]}
