"""kmz_video_audit — investigate KMZ/KML files (on S3 or local) for video references.

Some GT/annotation KMZs link to a source video, some don't. This package reads each KMZ,
pulls out every URL / href / ExtendedData value, and classifies it — so we can see which
files carry a video link and, crucially, WHERE in the KML that link lives (description,
ExtendedData, a <href>, an attribute…). The detector is deliberately broad and dumps all
links so the real schema can be discovered before it is hardened.

`core` is S3-agnostic (bytes/paths only); `cli` walks an ``s3://bucket/prefix`` or a local dir.
"""
from .core import KmzAudit, VideoHit, audit_kml_bytes, audit_kmz_bytes, extract_kml_bytes

__all__ = ["KmzAudit", "VideoHit", "audit_kml_bytes", "audit_kmz_bytes", "extract_kml_bytes"]
