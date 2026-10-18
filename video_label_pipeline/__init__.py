"""video_label_pipeline — date-range videos -> annotation manifest (JSON).

Self-contained geoLocalisation module for **video annotation**. Given a date
range, it reads videos (S3 or local), splits each into sub-videos with the
content-adaptive shot-boundary recipe that produced ``frames_good``, scores
every frame's quality (good/usable/bad), selects a few (default 5) *good*
frames per sub-video by **maximum visual diversity** (farthest-point on DINO
embeddings — not random), tags each selected frame day/night (chroma rule), and
writes **one detailed JSON** describing the whole split. No frames are written
to disk — everything lives in the JSON.

Heavy dependencies (torch / quality_cls / DINOv2) are lazy-imported inside the
default backend, so the split / diversity / day-night / manifest logic is pure
and unit-testable without a GPU or the model.
"""
from __future__ import annotations

from .daynight import chroma, day_night
from .diversity import farthest_point_select
from .pipeline import SplitRecipe, build_manifest, process_decoded, process_video
from .segment import (
    boundary_context,
    content_deltas,
    cut_indices,
    feature_diff,
    merge_short,
    segment_frames,
    segments_from_cuts,
    to_feature,
)

__all__ = [
    "SplitRecipe",
    "build_manifest",
    "process_decoded",
    "process_video",
    "segment_frames",
    "boundary_context",
    "content_deltas",
    "cut_indices",
    "segments_from_cuts",
    "merge_short",
    "feature_diff",
    "to_feature",
    "farthest_point_select",
    "chroma",
    "day_night",
]
