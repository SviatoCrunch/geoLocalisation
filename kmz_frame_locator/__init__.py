"""kmz_frame_locator — find each GT still's exact frame index inside its source video chunk.

The GT KMZ (gt/raw/<city>/<ver>/<ver>.kmz) has one Placemark per GT point, named ``<N>_<lat>,<lon>``
with a ``<description>`` naming the source video chunk (a full ``s3://…mp4`` or a bare
``chunk_<drone>_<date>_<time>_<dur>s_c<cam>[_<suffix>]``). The flat GT frames live in
``GT_flat/<N>_<lat>_<lon>.jpg``. This module resolves each frame's chunk, matches the still against
the decoded video frames (zero-mean L2-normalised 64×64 grayscale NCC — validated ≈0.9998 on real
data), and records the matched VIDEO frame index (the "ordinal", which is NOT the N in the filename).

Reuses :func:`kmz_video_audit.core.extract_kml_bytes`; S3 via the ``aws`` CLI (no boto3), cv2 for
decode — run with ``uv run --with opencv-python-headless --with numpy``.
"""
from .core import Placemark, parse_placemarks, resolve_chunk_url

__all__ = ["Placemark", "parse_placemarks", "resolve_chunk_url"]
