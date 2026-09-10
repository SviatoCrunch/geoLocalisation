import zipfile

from geo_split_no_overlap import kmz
from geo_split_no_overlap.schemas import TRAIN, VAL, TEST, EXCLUDED
from ._synth import tile, make_gallery, geo_point, latlon_to_xy


def _fixture():
    x0, y0 = latlon_to_xy(48.5, 37.8)
    tiles = [tile(f"t{i}", x0 + i * 2000.0, y0) for i in range(4)]
    pts = [geo_point(f"p{i}", x0 + i * 2000.0, y0) for i in range(4)]
    return make_gallery(tiles), pts


_PS = {"p0": TRAIN, "p1": VAL, "p2": TEST, "p3": EXCLUDED}


def test_kmz_is_valid_zip_with_doc_kml(tmp_path):
    g, pts = _fixture()
    p = kmz.write_kmz(tmp_path / "s.kmz", pts, g, _PS)
    assert zipfile.is_zipfile(p)
    with zipfile.ZipFile(p) as z:
        assert z.namelist() == ["doc.kml"]
        doc = z.read("doc.kml").decode()
    assert doc.count("<Placemark>") == 4
    for pid in _PS:
        assert pid in doc


def test_kml_has_per_split_styles_and_folders():
    g, pts = _fixture()
    doc = kmz.build_kml(pts, g, _PS)
    for sp in (TRAIN, VAL, TEST, EXCLUDED):
        assert f"pt_{sp}" in doc
        assert f"points: {sp}" in doc


def test_kml_draws_tile_footprints_when_requested():
    g, pts = _fixture()
    p2t = {"p0": ["t0"], "p1": ["t1"], "p2": ["t2"]}   # p3 excluded -> its tile not drawn
    doc = kmz.build_kml(pts, g, _PS, point_to_tiles=p2t, draw_tiles=True)
    assert doc.count("<Polygon>") == 3
    assert "tile_train" in doc and "tile_val" in doc and "tile_test" in doc


def test_tile_polygon_coordinates_are_lonlat():
    g, pts = _fixture()
    doc = kmz.build_kml(pts, g, _PS, point_to_tiles={"p0": ["t0"]}, draw_tiles=True)
    # lon ~37.8, lat ~48.5 for the first tile — sanity range check on emitted coords
    assert ",48." in doc and "37." in doc


def test_kml_is_deterministic():
    g, pts = _fixture()
    assert kmz.build_kml(pts, g, _PS) == kmz.build_kml(pts, g, _PS)
