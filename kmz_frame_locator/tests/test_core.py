"""kmz_frame_locator.core — chunk-URL resolution + placemark parsing (pure logic, no cv2)."""
from kmz_frame_locator.core import parse_placemarks, resolve_chunk_url

_NS = 'xmlns="http://www.opengis.net/kml/2.2"'


def test_resolve_full_s3_url():
    u = "s3://mediamtx-recordings-crunch/live/104/filtered/29.07.2026/chunk_104_2026-07-29_12-44-07_9s_c01.mp4"
    assert resolve_chunk_url(u) == u
    assert resolve_chunk_url(f"{u} \n") == u                      # trailing whitespace tolerated


def test_resolve_bare_chunk_name_with_suffix():
    # a bare name with a trailing processing suffix -> reconstruct the s3 key (suffix dropped)
    desc = "chunk_104_2026-07-29_12-42-06_6s_c02_20260730_150733"
    assert resolve_chunk_url(desc) == (
        "s3://mediamtx-recordings-crunch/live/104/filtered/29.07.2026/"
        "chunk_104_2026-07-29_12-42-06_6s_c02.mp4")


def test_resolve_none_on_junk():
    assert resolve_chunk_url("") is None
    assert resolve_chunk_url("just a note, no chunk here") is None


def test_parse_placemarks_extracts_n_coords_and_video():
    kml = (f'<kml {_NS}><Document><Folder>'
           '<Placemark><name>232_48.5121768781,37.7232587583</name>'
           '<description>s3://mediamtx-recordings-crunch/live/104/filtered/29.07.2026/'
           'chunk_104_2026-07-29_12-44-07_9s_c01.mp4</description>'
           '<Point><coordinates>37.72,48.51,0</coordinates></Point></Placemark>'
           '<Placemark><name>230_48.5722876733,37.6222078526</name>'
           '<description>chunk_104_2026-07-29_12-42-06_6s_c02_20260730_150733</description></Placemark>'
           '<Placemark><name>ignore me</name></Placemark>'
           '</Folder></Document></kml>').encode()
    pms = parse_placemarks(kml)
    assert len(pms) == 2                                          # the nameless/garbled one is skipped
    a, b = pms
    assert a.n == 232 and abs(a.lat - 48.5121768781) < 1e-9 and abs(a.lon - 37.7232587583) < 1e-9
    assert a.video_url.endswith("chunk_104_2026-07-29_12-44-07_9s_c01.mp4")
    assert b.n == 230 and b.video_url.endswith("chunk_104_2026-07-29_12-42-06_6s_c02.mp4")
