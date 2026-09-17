"""native_map_pyramid — experimental map-cell feature source (isolated feature package).

Adds ONE new map-pyramid source, ``native_hierarchical``, that builds the n=[8,4,2,1] cell
DINOv2 features from independent NATIVE COG crops (1×1000 m + 4×500 m + 16×250 m; n=8 = 2×2
quadrants of each 250 m grid) instead of partitioning a single 1000 m token grid
(``legacy_token_partition``). The MODEL TAIL is unchanged — the same frozen VLAD assignment and
the same trainable PerGroupProjection / ProjectionHead / ScaleGate / τ(n) are reused. The only
difference is the SOURCE of the per-cell per-group residual sums fed to the trainable tail.

Nothing here modifies the legacy pipeline; legacy remains the default.
"""
