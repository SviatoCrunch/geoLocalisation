# geo_split_no_overlap

Build **train / val / test** sets of query points with **zero geometric leakage**
between splits: no gallery tile that is a *positive* for a point in one split may
overlap (true area `> area_epsilon`) with a positive tile for a point in another
split. Sharing the exact same tile counts as full overlap; a bare boundary touch
(zero area) does **not**.

This is the leak-tight successor to `make_multicity_split.py`. That script only
de-duplicated *identical* `tile_id`s across splits — but with the stride-250 /
1000 m gallery two *different* tiles overlap ~94 %, so its "disjoint" guarantee
leaked. (It also has a latent `NameError` and never ran.) This module reuses its
**positive-rule semantics and helpers** but closes the geometric gap.

## What it does

1. **Freeze positives** once — `P(q)` per point, hashed into a `fingerprint` so the
   audit can prove one immutable version was used. Points with no positive are
   handled by an explicit policy (`exclude` records a reason; `error` aborts) —
   never dropped silently.
2. **Detect conflicts** — union-find over the *unique positive tiles* (STRtree
   candidates + exact area, converted to true m² via the `cos²(lat)` factor since
   EPSG:3857 is not equal-area), then lift tile-clusters to points.
3. **Build indivisible components** — points fused by shared/overlapping positives.
   A component goes **whole** to one split or is excluded **whole**; transitivity is
   respected (q1–q2, q2–q3 ⇒ one component).
4. **Assign, two-phase** — balancing by **point count**, deterministic:
   - **Phase A** — no deletion; accept iff every split is within `ratio_tolerance`.
   - **Phase B** (only if A fails) — exclude *whole* components, lexicographically:
     minimise removed points → largest removed component → Σ(size²) → imbalance →
     SHA-256 canonical tie-break. Balance + non-empty splits stay **hard**.
   If no valid non-empty split exists even with deletion → **infeasible**.
5. **Independent audit** — re-derives components + conflicts from geometry and
   re-checks everything, *without trusting the optimizer*.

## Install / requirements

All declared project deps except OR-Tools: `numpy`, `h5py`, `shapely>=2`, `pyproj`,
`pyyaml` (+ `scipy`, `pandas`). The exact optimizer uses **OR-Tools CP-SAT**
(`pip install ortools`); if it is absent, `solver: auto` falls back to a
deterministic greedy heuristic (`optimality_proven=false`). Run from the **repo
root** so the reused `make_multicity_split` module is importable.

## Commands

```bash
# validate inputs + report counts / unique tiles / components, write nothing
python -m geo_split_no_overlap.cli build --config geo_split_no_overlap/config.example.yaml --dry-run

# build the split (writes the run directory)
python -m geo_split_no_overlap.cli build --config geo_split_no_overlap/config.example.yaml

# independently audit an existing split against freshly recomputed inputs
python -m geo_split_no_overlap.cli audit \
    --config geo_split_no_overlap/config.example.yaml \
    --split ~/work/out/gallery_h5/geo_split_run/split.json

# write a KMZ visualization of an existing split (points; --tiles adds footprints)
python -m geo_split_no_overlap.cli export-kmz --config cfg.yaml --split run/split.json --out run/split.kmz --tiles

# list the registered positive-selection strategies
python -m geo_split_no_overlap.cli list-positive-strategies
```

`build` also accepts `--kmz` (and `--kmz-tiles`) to drop a `split.kmz` next to the run.

## Two build modes

Chosen by the config, same positive rule / optimizer / audit in both:

- **single-gallery** — `gt` (one or more `city=dir`) + one `tiles_h5`. One split
  balanced over the **combined** point pool (a city may end up unevenly distributed).
- **per-city** — `cities: [{city, gt, tiles_h5}]`, one `(points, H5)` per city. Each
  city is split **independently** to the target ratios, then the per-city splits are
  **unioned**. Because the galleries don't overlap, components never cross a city, so
  the union is leakage-free; a single **combined audit** re-verifies cross-city
  disjointness and a per-city ratio check runs for each city. Extra outputs:
  `split.json` gains `"mode":"per_city"` + a `per_city` block, and each city gets
  `cities/<city>/split.json`. If any city can't be split (e.g. < 3 components) the whole
  run is `infeasible` (naming the city).

```yaml
# per-city config
cities:
  - {city: kramatorsc, gt: ~/work/gt_cramatorsc, tiles_h5: ~/work/out/gallery_h5/kramatorsc.h5}
  - {city: kup,        gt: ~/work/gt_kup,        tiles_h5: ~/work/out/gallery_h5/kup.h5}
  - {city: liman,      gt: ~/work/gt_liman,      tiles_h5: ~/work/out/gallery_h5/liman.h5}
positive_selection: {strategy: current_rule, params: {}}
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
ratio_tolerance: 0.05
out_dir: ~/work/out/gallery_h5/geo_split_percity
```

## KMZ visualization

`export-kmz` (or `build --kmz`) writes a KMZ (zipped KML) with points coloured by
split (train=green, val=blue, test=red, excluded=grey), one folder per split. With
`--tiles` / `--kmz-tiles` it also draws the **positive-tile footprints** per split
(polygons, inverse-Mercator to lon/lat) — a quick visual check that the per-split tile
coverage is geographically disjoint. Pure stdlib (no shapely/pyproj at the KMZ layer);
open the file in Google Earth.

CLI overrides: `--out --seed --ratios t,v,te --tolerance --area-epsilon --solver`.

**Exit codes:** `0` valid · `1` invalid (leakage / out of tolerance) · `2`
infeasible or error. `audit` exits non-zero if it finds *any* cross-split conflict.

## Inputs

- `tiles_h5` — a `multicity_vlad_*.h5` (from `build_multicity_vlad.py`) with datasets
  `lat`, `lon`, `city`, `tile_id`, `window_size_m`. No new data is produced.
- `gt` — `"<city>=<dir>"` (or plain dirs; `gt_` prefix stripped). Frames are
  `"<id>_<lat>_<lon>.<ext>"`; point id = `"<city>:<stem>"`.

The positive rule is chosen by the **positive-selection subsystem** (below), not
hard-coded — see `positive_selection: {strategy, params}` in the config.

## Positive-selection subsystem (`positive_selection/`)

The rule that decides *which tiles are positive for a point* is an isolated,
pluggable subsystem. The split algorithm never knows whether positives come from
IoU, area, distance or containment — it consumes only an immutable materialized
snapshot `dict[PointId, frozenset[TileId]]`.

**Public API** (`from geo_split_no_overlap.positive_selection import ...`):

```
GeoPoint, GalleryTile, GalleryIndex, PositiveMatch          # domain models
MaterializedPositiveSets, PositiveSelector                  # snapshot + Strategy contract
PositiveSelectionConfig                                     # {strategy, params}
register_positive_selector, create_positive_selector, available_positive_selectors   # Registry
materialize_positive_sets                                   # Strategy -> frozen snapshot
build_gallery_index, build_geo_points                       # loaders (reuse existing artifacts)
```

The rest of the module imports **only** from here; nothing reaches into
`positive_selection/strategies/*` or `positive_selection/adapters/*`. The subsystem
never imports the split components / optimizer / audit (enforced by an architecture
test). CLI wiring is exactly:

```python
selector       = create_positive_selector(config.positive_selection)
positive_sets  = materialize_positive_sets(points, gallery, selector)   # frozen
result         = build_split(points, gallery, positive_sets, split_config)
```

**Built-in strategies:** `current_rule` (production box rule, the parity adapter),
`contains_point` (point strictly inside the tile square), and two IoU rules
`tile_iou_1000` / `pyramid_top_iou_250` (below). All work in a metric CRS (EPSG:3857)
and validate it.

### IoU strategies

Both decide positivity purely by a threshold on the standard IoU

```
intersection = area(A ∩ B)
union        = area(A) + area(B) − intersection
IoU          = intersection / union            # dimensionless; positive iff IoU >= threshold
```

`threshold` ∈ (0, 1] (NaN/∞/out-of-range/unknown params are rejected at creation).
IoU is returned in `PositiveMatch.score`, a short machine-readable tag in `reason`.
The fixed square sizes are part of each rule's semantics and go into
`resolved_config()` + the fingerprint (even though the user only sets `threshold`).
Not IoU-substitutes: neither uses intersection-over-one-square nor absolute
intersection area.

**`tile_iou_1000`** — the gallery item is a plain **1000×1000 m** tile; a **1000×1000 m**
reference square is built around the point:

```
query_box = centered_square(center=q, side_m=1000)
score     = IoU(tile_1000, query_box)          # FULL tile vs full query box
positive iff score >= threshold
```
No pyramid, no sub-levels, no centre crop. Tiles that are not ~1000 m squares (within
`geometry_tolerance_m`) are rejected, never silently rescaled.

```yaml
positive_selection:
  strategy: tile_iou_1000
  params: { threshold: 0.70 }
```

**`pyramid_top_iou_250`** — one gallery item is a whole concentric-square pyramid
(base 1000, top 250, shared centre). Positivity is decided **only by the 250×250 m
top** vs a **250×250 m** query square; on a hit the id of the **whole pyramid** is
returned (the top is just a probe geometry):

```
query_box   = centered_square(center=q,             side_m=250)
pyramid_top = centered_square(center=pyramid.centre, side_m=250)
score       = IoU(pyramid_top, query_box)
positive iff score >= threshold
```
IoU is **never** taken against the 1000 m base, any middle level, base-vs-250, or as a
max over levels — this is *not* a multi-level/pyramid-level search. Intermediate levels
are not generated and do not affect positivity. The top is derived concentric with the
validated base centre (the model stores only base geometry; a malformed base is
rejected, not corrected).

```yaml
positive_selection:
  strategy: pyramid_top_iou_250
  params: { threshold: 0.70 }
```

**Why 1000-vs-1000 differs from 250-vs-250:** the same point can be positive under
`tile_iou_1000` (large boxes tolerate larger offsets) yet negative under
`pyramid_top_iou_250` (a 250 m top needs the point within ~top/2 of the tile centre) —
the pyramid rule is far stricter. Both require a metric CRS; IoU being a ratio, the
EPSG:3857 area distortion cancels.

**Add a new rule** — one class + one registration, no split-code change:

```python
# geo_split_no_overlap/positive_selection/strategies/my_rule.py
class MyRuleSelector:
    name, version = "my_rule", "1.0"
    @classmethod
    def from_params(cls, params): ...          # validate params, return instance
    def select(self, point, gallery): ...      # -> Sequence[PositiveMatch]
    def resolved_config(self): ...             # params + version + crs + units
# register in registry._register_builtins(): register_positive_selector(MyRuleSelector.name, MyRuleSelector.from_params)
```

Then select it purely from YAML:

```yaml
positive_selection:
  strategy: my_rule
  params: { threshold: 0.7 }
```

The **materialized snapshot** is frozen and stably sorted; it also carries the
strategy name+version, resolved params, per-point positive stats, the list of points
with no positive, and a `fingerprint` that changes if the rule, a parameter, or the
inputs change. Components / optimizer / audit read this snapshot and never re-run a
strategy, so `P(q)` cannot change mid-run.

Positive-rule details of the built-ins: a query is positive for a tile iff
`|dx|,|dy| ≤ (window_size_m + query_size_m)/2` in EPSG:3857 (`current_rule`;
`query_size_m=0` ⇒ point inside tile square = `contains_point`). Matching is per-city
when `same_city_only`.

## Outputs (in `out_dir`)

| file | contents |
|---|---|
| `split.json` | `train` / `val` / `test` point-id lists, `excluded` (with reason + component), `seed`, target & actual ratios, counts, `fingerprint`, `status`, meta |
| `components.json` | every component: `component_id`, `point_ids`, `size`, `assigned`, `exclusion_reason`, `n_positive_tiles` |
| `excluded_components.json` | excluded components with sizes + reason |
| `audit.json` | full independent-audit report (checks, counts, CRS, `area_epsilon`, cross-split conflicts, size histogram, largest/excluded components, fingerprint, status) |
| `summary.md` | human-readable summary |
| `resolved_config.yaml` | the fully-resolved config used |

Per-city runs additionally write `cities/<city>/split.json` (per-city lists + ratios)
and add `"mode":"per_city"` + a `per_city` block to `split.json`. `build --kmz` (or
`export-kmz`) writes `split.kmz`.

## Status meanings

- **valid** — a split exists, every hard check passes: components indivisible & not
  split, no shared positive `tile_id` across splits, no cross-split tile overlap
  `> area_epsilon`, all splits non-empty and within `ratio_tolerance`.
- **invalid** — a split was produced but the **independent audit** found a violation
  (leakage or out-of-tolerance). Investigate before use; exit code `1`.
- **infeasible** — no valid non-empty split exists within the constraints, even after
  excluding whole components (e.g. one component larger than a split's target band).
  No misleading "successful" split file is written; exit code `2`.

## Design notes / non-obvious choices

- **Why not reuse `siam_model_stage2/graph_split.py`?** It is a *vertex separator*:
  its constraint is satisfied by deleting **individual boundary points**, which this
  task forbids (components are indivisible). We instead solve a component-level
  weighted assignment (`x[c,s]` + `d[c]`, balanced by point count) but mirror its
  proven conventions (OR-Tools→greedy cascade, staged lexicographic objective,
  SHA-256 deterministic tie-break).
- **CRS.** The positive rule stays in EPSG:3857 (the grid the tiles were cut on) so
  its semantics are unchanged. Overlap **areas** are converted to true m² via the
  `cos²(lat)` conformal factor, so `area_epsilon` is a real-metre knob. Both the CRS
  and the conversion are recorded in `audit.json`.
- **Determinism.** No Python `hash()` anywhere (salted). Component ids, orderings and
  the canonical tie-break are stable, so identical inputs + seed give an identical
  split (covered by tests).

## Isolation

All logic lives in this folder. The only reach into existing code is
`positive_selection/adapters/project_rule.py`, which imports `make_multicity_split`'s
helpers (`_merc`, `_parse_stem`, `_collect_frames`, `_city_and_dir`) and reads the
existing gallery H5 — no existing file is modified. Dependency flow is one-way:
`split orchestration → positive_selection public API → strategies + project adapters`
(enforced by `positive_selection/tests/test_boundaries.py`).

## Tests

```bash
python -m pytest geo_split_no_overlap -q
```

Split layer: shared tile, overlapping different tiles, boundary-touch (zero area),
transitive chains, independent components, no-deletion distribution, deletion-only-
when-needed (smallest removed), large-component infeasibility, determinism, an
independent audit catching a corrupted split, and a strategy swap via YAML only.

Split layer also: KMZ writer (valid zip, per-split styles/folders, tile footprints,
lon/lat coords, determinism) and per-city mode (each city balanced, combined = union,
cross-city audit clean, determinism, small-city→infeasible, KMZ export).

Subsystem (`positive_selection/tests`): Registry create/unknown/duplicate/invalid
params, Protocol conformance + purity + CRS check, two strategies giving different
`P(q)` but the same downstream type, snapshot immutability + stats, fingerprint
stability/sensitivity, `current_rule` **parity** vs an independent computation of the
production formula, the two **IoU** rules (identical=1.0, disjoint/touching=0.0,
offset-500→⅓ and offset-125→⅓, exact-threshold `>=`, wrong-size rejection, pyramid
top-only decision + whole-pyramid id) + IoU integration (Registry swap, no
service branches, fingerprint differs by rule/threshold, reproducibility), and the
architecture boundary test.
