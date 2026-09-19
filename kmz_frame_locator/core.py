"""KMZ placemark parsing, chunk-URL resolution, and still→video frame matching."""
from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

VIDEO_BUCKET = "mediamtx-recordings-crunch"

# chunk_<drone>_<YYYY-MM-DD>_<HH-MM-SS>_<dur>s_c<cam>  (a trailing _<suffix> is ignored)
_CHUNK_RE = re.compile(r"chunk_(\d+)_(\d{4})-(\d{2})-(\d{2})_(\d{2}-\d{2}-\d{2})_(\d+)s_c(\d+)")
# Placemark name "<N>_<lat>,<lon>"
_NAME_RE = re.compile(r"^\s*(\d+)_(-?\d+\.\d+),(-?\d+\.\d+)")


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def resolve_chunk_url(desc: str | None) -> str | None:
    """Placemark ``<description>`` → full ``s3://…mp4`` URL of its video chunk, or None.

    Accepts a full ``s3://…/chunk_….mp4`` verbatim, or a bare chunk name
    ``chunk_<drone>_<YYYY-MM-DD>_<HH-MM-SS>_<dur>s_c<cam>[_<extra>]`` which is reconstructed to
    ``s3://<bucket>/live/<drone>/filtered/<DD.MM.YYYY>/<chunk>.mp4`` (the date folder is
    ``DD.MM.YYYY`` from the chunk's own date; any trailing processing suffix is dropped).
    """
    if not desc:
        return None
    desc = desc.strip()
    if desc.startswith("s3://"):
        return desc.split()[0]
    m = _CHUNK_RE.search(desc)
    if not m:
        return None
    drone, y, mo, d, hms, dur, cam = m.groups()
    fname = f"chunk_{drone}_{y}-{mo}-{d}_{hms}_{dur}s_c{cam}.mp4"
    return f"s3://{VIDEO_BUCKET}/live/{drone}/filtered/{d}.{mo}.{y}/{fname}"


@dataclass
class Placemark:
    n: int
    lat: float
    lon: float
    video_url: str | None
    raw_desc: str


def parse_placemarks(kml_bytes: bytes) -> list[Placemark]:
    """Parse every ``<N>_<lat>,<lon>`` Placemark → :class:`Placemark` (with resolved video URL)."""
    root = ET.fromstring(kml_bytes)
    out: list[Placemark] = []
    for pm in root.iter():
        if _localname(pm.tag) != "Placemark":
            continue
        name = desc = ""
        for ch in pm.iter():                                   # name/description may be nested
            ln = _localname(ch.tag)
            if ln == "name" and ch.text and not name:
                name = ch.text.strip()
            elif ln == "description" and ch.text and not desc:
                desc = ch.text.strip()
        m = _NAME_RE.match(name)
        if not m:
            continue
        out.append(Placemark(int(m.group(1)), float(m.group(2)), float(m.group(3)),
                             resolve_chunk_url(desc), desc))
    return out


# ── still ↔ video frame matching (cv2, lazy) ──────────────────────────────────

def _prep(bgr):
    """BGR frame/image → zero-mean, L2-normalised 64×64 grayscale vector (for NCC via dot)."""
    import cv2
    import numpy as np
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
    g -= g.mean()
    nrm = float(np.linalg.norm(g))
    return g / nrm if nrm > 0 else g


def match_still(still_path, video_path):
    """Find the video frame most similar to the still.

    Returns ``(frame_index, fps, time_s, ncc, best_frame_bgr, n_frames)`` or None if the still or
    video can't be read. NCC ∈ [-1, 1]; ≈1.0 means the still IS that frame (validated ≈0.9998).
    """
    import cv2
    import numpy as np
    img = cv2.imread(str(still_path))
    if img is None:
        return None
    still = _prep(img)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    best_s, best_i, best_frame = -2.0, -1, None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        s = float((_prep(fr) * still).sum())
        if s > best_s:
            best_s, best_i, best_frame = s, i, fr
        i += 1
    cap.release()
    if best_i < 0:
        return None
    return best_i, fps, (best_i / fps if fps else 0.0), best_s, best_frame, i
