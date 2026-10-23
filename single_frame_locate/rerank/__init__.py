"""Fine geo-reranking (vendored from geoLocalisation/patch_rerank).

Self-contained, service-side fine stage: coarse (e2c supervlad) top-N candidate cells + the live
query's DINO token grid → top-K reranked map cells by patch-RANSAC (cv2 MAGSAC) over an S3-built,
locally-synced per-cell pyramid store. No cross-repo imports; deps = numpy, torch, cv2, h5py
(boto3 only for the S3 store fallback).

    from service.rerank import CellStore, rerank
    store = CellStore("/home/ubuntu/work/kup-files/pyramid/kup")   # local dir (prod)
    top5 = rerank(q_feat, q_xy, coarse_top100_cell_ids, store, topk=5)
"""
from .cell_store import CellData, CellStore
from .matcher import grid_keypoints, grid_to_latlon, matched_coords, verify_inliers
from .reranker import rerank, rerank_multiframe

__all__ = ["CellStore", "CellData", "rerank", "rerank_multiframe", "matched_coords",
           "verify_inliers", "grid_keypoints", "grid_to_latlon"]
