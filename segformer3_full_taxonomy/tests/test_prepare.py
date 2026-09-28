"""prepare_skyscenes: un-nest the double-tar and flatten real PNGs (nested tar.gz ignored)."""
import io
import tarfile
from pathlib import Path

from segformer3_full_taxonomy.tools.prepare_skyscenes import prepare_town


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    """Build an in-memory tar from {arcname: content}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_double_tar_flattened(tmp_path):
    deep = "srv/hoffman-lab/x/y/z/HF/SkyScenes/Images/H_35_P_0/ClearNoon/Town01"
    inner = _tar_bytes({
        f"{deep}/007761_clrnoon.png": b"PNGDATA-1",
        f"{deep}/007771_clrnoon.png": b"PNGDATA-2",
        f"{deep}/Town01.tar.gz": b"REDUNDANT-NESTED-TAR",   # must be ignored
    })
    # outer town tar: members are nested tars named <id>.png
    outer = tmp_path / "Town01.tar.gz"
    outer.write_bytes(_tar_bytes({"007751_clrnoon.png": inner,
                                  "007752_clrnoon.png": inner}))
    out = tmp_path / "out"
    n = prepare_town(outer, out)
    assert n == 2                                            # only the 2 real pngs
    got = sorted(p.name for p in out.glob("*.png"))
    assert got == ["007761_clrnoon.png", "007771_clrnoon.png"]
    assert (out / "007761_clrnoon.png").read_bytes() == b"PNGDATA-1"
    assert not (out / "Town01.tar.gz").exists()             # nested tar.gz skipped


def test_empty_outer_returns_zero(tmp_path):
    outer = tmp_path / "Empty.tar.gz"
    outer.write_bytes(_tar_bytes({"readme.txt": b"no pngs here"}))
    assert prepare_town(outer, tmp_path / "o") == 0
