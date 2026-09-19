"""kmz_video_audit.core — classify video references from KML/KMZ bytes across schema locations."""
import io
import zipfile

from kmz_video_audit.core import audit_kml_bytes, audit_kmz_bytes, extract_kml_bytes

_NS = 'xmlns="http://www.opengis.net/kml/2.2"'


def _kml(placemarks: str) -> bytes:
    return f'<kml {_NS}><Document>{placemarks}</Document></kml>'.encode()


def _zip_kmz(kml_bytes: bytes, name: str = "doc.kml") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, kml_bytes)
    return buf.getvalue()


def test_video_in_description_cdata():
    kml = _kml('<Placemark><name>Pt1</name>'
               '<description>see &lt;a href="https://youtu.be/abc123"&gt;clip&lt;/a&gt;</description>'
               '<Point><coordinates>37.6,48.5,0</coordinates></Point></Placemark>')
    a = audit_kml_bytes(kml, "d.kml")
    assert a.has_video and a.n_placemarks == 1
    hit = a.video_links[0]
    assert hit.reason == "host:youtu.be" and hit.where == "description" and hit.placemark == "Pt1"


def test_video_in_extended_data_mp4():
    kml = _kml('<Placemark><name>Pt2</name><ExtendedData>'
               '<Data name="video"><value>https://cdn.example.com/flight/clip.mp4?t=1</value></Data>'
               '</ExtendedData></Placemark>')
    a = audit_kml_bytes(kml, "e.kml")
    assert a.has_video and a.ext_data_keys == ["video"]
    hit = a.video_links[0]
    assert hit.reason == "ext:.mp4" and hit.where == "ExtendedData:video" and hit.placemark == "Pt2"


def test_no_video_link():
    kml = _kml('<Placemark><name>Pt3</name>'
               '<description>just a map screenshot, https://example.com/page.html</description>'
               '<Point><coordinates>37.6,48.5,0</coordinates></Point></Placemark>')
    a = audit_kml_bytes(kml, "n.kml")
    assert not a.has_video and a.urls == ["https://example.com/page.html"]


def test_kmz_zip_roundtrip_and_href_element():
    kml = _kml('<Placemark><name>Pt4</name><Link><href>https://host/v/movie.mov</href></Link></Placemark>')
    audit = audit_kmz_bytes(_zip_kmz(kml), "a.kmz")
    assert audit.has_video
    assert audit.video_links[0].reason == "ext:.mov" and audit.video_links[0].where == "href"
    # extract path also works
    assert b"movie.mov" in extract_kml_bytes(_zip_kmz(kml))


def test_bare_kml_bytes_via_kmz_entry():
    # audit_kmz_bytes must sniff a non-zip payload and treat it as KML
    kml = _kml('<Placemark><name>Pt5</name></Placemark>')
    a = audit_kmz_bytes(kml, "bare.kml")
    assert a.error is None and not a.has_video and a.n_placemarks == 1


def test_malformed_xml_reports_error_not_crash():
    a = audit_kmz_bytes(b"<kml><oops", "bad.kml")
    assert a.error is not None and not a.has_video
