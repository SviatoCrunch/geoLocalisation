"""Parse a KMZ/KML document and classify any video references it carries.

S3-agnostic: everything here works on raw bytes or local paths, so it is trivially testable.
KMZ reading mirrors the repo idiom (``src_tif_data.kmz_locator``): a KMZ is a zip whose payload
is ``doc.kml`` (or the first ``*.kml``). KML is namespaced XML; we match by LOCAL tag name so the
code is namespace-agnostic (KML 2.2, gx: extensions, etc. all parse the same).
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

# A URL inside element text, a description CDATA (HTML <a href>), or an attribute value.
# Matches http(s), s3:// URIs, and rtsp/rtmp streams — a "video link on S3" is typically an
# ``s3://bucket/key.mp4`` (or a presigned https S3 URL), which a http-only pattern would miss.
_URL_RE = re.compile(r"(?:https?|s3|rtsp|rtmp)://[^\s<>\"'\)\]]+", re.IGNORECASE)

# Classification signals (broad on purpose — this is an investigation tool).
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv",
               ".mpg", ".mpeg", ".ts", ".m3u8")
_VIDEO_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "rutube.ru", "dailymotion.com",
                "streamable.com", "twitch.tv", "t.me",
                # this project's own video/recording buckets (S3 links point here)
                "mediamtx-recordings-crunch", "drone-detections-map")


def _localname(tag: str) -> str:
    """Strip the ``{namespace}`` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _classify_url(url: str) -> str | None:
    """Return a short reason string if ``url`` looks like a video, else None."""
    low = url.lower()
    if low.startswith(("rtsp://", "rtmp://")):           # a live stream is a video by definition
        return f"scheme:{low.split(':', 1)[0]}"
    # strip a query string for the extension test (…/clip.mp4?token=… still counts)
    path = low.split("?", 1)[0].split("#", 1)[0]
    for ext in _VIDEO_EXTS:
        if path.endswith(ext):
            return f"ext:{ext}"
    for host in _VIDEO_HOSTS:
        if host in low:
            return f"host:{host}"
    if "video" in low:
        return "keyword:video"
    return None


@dataclass
class VideoHit:
    """One link classified as a video reference, with provenance for schema discovery."""
    url: str
    reason: str                 # ext:.mp4 | host:youtube.com | keyword:video
    where: str                  # KML location: element localname / description / ExtendedData:<name> / attr:<name>
    placemark: str | None = None  # nearest enclosing <name>, if any


@dataclass
class KmzAudit:
    """Per-file audit result. ``has_video`` is the headline; the rest is evidence/schema."""
    source: str
    n_placemarks: int = 0
    urls: list[str] = field(default_factory=list)          # every http(s) URL found (deduped, in order)
    video_links: list[VideoHit] = field(default_factory=list)
    ext_data_keys: list[str] = field(default_factory=list)  # ExtendedData <Data name="…"> keys seen (schema hint)
    error: str | None = None

    @property
    def has_video(self) -> bool:
        return bool(self.video_links)

    def summary(self) -> str:
        if self.error:
            return f"[ERR ] {self.source}: {self.error}"
        flag = "VIDEO" if self.has_video else "  -  "
        wheres = sorted({v.where for v in self.video_links})
        tail = f" via {wheres}" if wheres else ""
        return (f"[{flag}] {self.source}: placemarks={self.n_placemarks} "
                f"urls={len(self.urls)} video={len(self.video_links)}{tail}")


def extract_kml_bytes(kmz_bytes: bytes) -> bytes:
    """Return the KML payload of a KMZ (zip) — prefers ``doc.kml``, else the first ``*.kml``."""
    with zipfile.ZipFile(BytesIO(kmz_bytes), "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise ValueError("no .kml inside KMZ archive")
        preferred = [n for n in names if Path(n).name.lower() == "doc.kml"]
        return zf.read(preferred[0] if preferred else names[0])


def _iter_urls_with_context(root: ET.Element):
    """Yield ``(url, where, placemark_name)`` for every http(s) URL anywhere in the tree.

    Scans each element's text/tail and every attribute value. Tracks the nearest enclosing
    Placemark ``<name>`` and flags ExtendedData ``<Data name>`` / ``<value>`` locations, so a
    hit's provenance reveals where the schema actually stores links.
    """
    # Map child→parent and the placemark name in scope, via one DFS with an explicit stack.
    stack = [(root, None, None)]                              # (elem, parent_placemark_name, data_key)
    # Precompute placemark names is awkward; instead track scope during recursion.

    def _walk(elem, pm_name, data_key):
        tag = _localname(elem.tag)
        if tag == "Placemark":
            nm = elem.find("./{*}name")
            pm_name = (nm.text or "").strip() if nm is not None and nm.text else pm_name
        if tag == "Data":                                    # ExtendedData: <Data name="video">
            data_key = elem.get("name") or data_key
        # location label for any hit found directly on this element
        if tag == "description":
            where = "description"
        elif data_key and tag in ("value", "Data"):
            where = f"ExtendedData:{data_key}"
        else:
            where = tag
        for txt in (elem.text, elem.tail):
            if txt:
                for m in _URL_RE.finditer(txt):
                    yield m.group(0), where, pm_name
        for aname, aval in elem.attrib.items():
            for m in _URL_RE.finditer(aval):
                yield m.group(0), f"attr:{_localname(aname)}", pm_name
        for child in list(elem):
            yield from _walk(child, pm_name, data_key)

    yield from _walk(root, None, None)


def audit_kml_bytes(kml_bytes: bytes, source: str = "<kml>") -> KmzAudit:
    """Audit raw KML bytes for video references."""
    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError as e:
        return KmzAudit(source=source, error=f"XML parse error: {e}")

    n_placemarks = sum(1 for el in root.iter() if _localname(el.tag) == "Placemark")
    ext_keys = sorted({el.get("name") for el in root.iter()
                       if _localname(el.tag) == "Data" and el.get("name")})

    seen: set[str] = set()
    urls: list[str] = []
    hits: list[VideoHit] = []
    for url, where, pm in _iter_urls_with_context(root):
        if url not in seen:
            seen.add(url)
            urls.append(url)
        reason = _classify_url(url)
        if reason:
            hits.append(VideoHit(url=url, reason=reason, where=where, placemark=pm))
    # de-dup identical video hits (same url+where)
    uniq, seen_hit = [], set()
    for h in hits:
        k = (h.url, h.where)
        if k not in seen_hit:
            seen_hit.add(k)
            uniq.append(h)
    return KmzAudit(source=source, n_placemarks=n_placemarks, urls=urls,
                    video_links=uniq, ext_data_keys=list(ext_keys))


def audit_kmz_bytes(kmz_bytes: bytes, source: str = "<kmz>") -> KmzAudit:
    """Audit a KMZ (zip) or a bare KML given as bytes — sniffs the zip magic."""
    if kmz_bytes[:2] == b"PK":                                # zip local-file header
        try:
            kml = extract_kml_bytes(kmz_bytes)
        except Exception as e:                                # noqa: BLE001 — report, don't crash a batch
            return KmzAudit(source=source, error=f"KMZ read error: {e}")
        return audit_kml_bytes(kml, source=source)
    return audit_kml_bytes(kmz_bytes, source=source)


def audit_path(path) -> KmzAudit:
    """Audit a local ``.kmz`` or ``.kml`` file."""
    p = Path(path)
    data = p.read_bytes()
    src = p.name
    if p.suffix.lower() == ".kml":
        return audit_kml_bytes(data, source=src)
    return audit_kmz_bytes(data, source=src)
