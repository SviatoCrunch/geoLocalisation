"""KMZ writer: valid zip with a doc.kml containing GT/pred coordinates (lon,lat order)."""
import zipfile

from patch_rerank.kmz import build_kml, write_kmz


def test_build_kml_has_coords_lon_lat():
    kml = build_kml([{"name": "q0", "gt": (48.9, 37.6), "pred": (48.91, 37.61), "dist_m": 730.0}])
    assert "<kml" in kml and "q0 (730 m)" in kml
    assert "37.6000000,48.9000000" in kml            # KML uses lon,lat order
    assert "37.6100000,48.9100000" in kml


def test_write_kmz_is_valid_zip(tmp_path):
    p = tmp_path / "out.kmz"
    write_kmz(p, [{"name": "a", "gt": (48.0, 37.0), "pred": (48.0, 37.0), "dist_m": 0.0}])
    with zipfile.ZipFile(p) as z:
        assert "doc.kml" in z.namelist()
        assert b"<kml" in z.read("doc.kml")
