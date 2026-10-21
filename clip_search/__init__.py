"""Clip-level geo-localization: ≤32-frame windows around each GT, searched on the GPU scheme.

Two torch-free glue modules around the existing GPU pipeline (nothing in the search path is
re-implemented):

  * ``clip_search.sample``     — ``subvideo_index`` JSON → per-GT clip windows (≤32 frames, GT centred,
                                  clipped to the sub-video) → extract those frames from the S3 video
                                  → ``frames/{idx}_{lat}_{lon}.jpg`` + ``clips_manifest.json``.
  * ``clip_search.aggregate``  — the per-frame FINE result from ``pyramid_newformat.locate`` (GPU
                                  MAGSAC++) grouped by clip → sweep top-N ∈ {10,20,30,40,50} →
                                  per-cell SUM-of-scores and VOTES → top-10 cells → KMZ + distR eval.

Between the two the EXISTING tools run unchanged: ``map_extract.extract --images`` (embed frames),
``geo_e2c_train.coarse_rank_gallery`` (coarse top-100 shortlist, GPU), ``pyramid_newformat.locate``
(GPU fine rerank over the packed ``*_newformat`` store). See ``README.md`` for the full recipe.
"""
