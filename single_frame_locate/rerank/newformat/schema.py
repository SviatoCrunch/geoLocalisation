"""Target-format schema constants for the packed pyramid-embedding store.

One contiguous ``/features`` tensor per file/shard, axis order fixed below. The source per-cell H5
store (``patch_rerank.build_cell_store_s3`` output) is NEVER modified — this is a parallel format
written to a sibling S3 prefix for fast batched cell reads + direct GPU hand-off.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"

# /features axis order (6D). A cell is one contiguous block: position-major, then level-major.
AXES = ("cell", "position", "level", "token_y", "token_x", "feature")

FEATURES_DTYPE = "float16"          # stored verbatim from source (no renorm / recompute)

POS_GROUP_RE = r"^p\d+$"            # ONLY these HDF5 groups are positions (reject p_meta, pxy, ...)

# dataset names in each file/shard
DS_FEATURES = "features"
DS_CELL_IDS = "cell_ids"
DS_CELL_LAT = "cell_lat"
DS_CELL_LON = "cell_lon"
DS_POS_LAT = "position_lat"
DS_POS_LON = "position_lon"
DS_LEVELS = "levels_m"

MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "_convert_checkpoint.json"
