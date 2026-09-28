"""prepare_skyscenes: extract a town tar and keep only REAL PNGs (drop the fake tar-.png)."""
import io
import tarfile

from segformer3_full_taxonomy.tools.prepare_skyscenes import prepare_town

_PNG = b"\x89PNG\r\n\x1a\n"


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_keeps_real_pngs_drops_fake_tar(tmp_path):
    fake_tar = _tar_bytes({"junk.txt": b"redundant nested tar"})   # a tar, but named *.png below
    outer = tmp_path / "Town01.tar.gz"
    outer.write_bytes(_tar_bytes({
        "007761_clrnoon.png": _PNG + b"realimg1",
        "007771_clrnoon.png": _PNG + b"realimg2",
        "007781_clrnoon.png": fake_tar,                            # png-named tar -> dropped
    }))
    n = prepare_town(outer, tmp_path / "out")
    assert n == 2
    got = sorted(p.name for p in (tmp_path / "out").glob("*.png"))
    assert got == ["007761_clrnoon.png", "007771_clrnoon.png"]
    assert (tmp_path / "out" / "007761_clrnoon.png").read_bytes() == _PNG + b"realimg1"


def test_empty_returns_zero(tmp_path):
    outer = tmp_path / "Empty.tar.gz"
    outer.write_bytes(_tar_bytes({"readme.txt": b"no pngs"}))
    assert prepare_town(outer, tmp_path / "o") == 0
