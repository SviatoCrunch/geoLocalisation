# pyramid_newformat — packed contiguous pyramid-embedding store

Repacks the per-cell DINO pyramid store (`patch_rerank.build_cell_store_s3` output + `_index.json`)
into **one contiguous `/features` tensor** per file/shard for fast batched cell reads and direct GPU
hand-off. The **source is never modified**; output goes to a sibling `*_newformat` S3 prefix.

```
/features  (N_cells, 25, 8, 32, 32, 1024)  float16   # axes: cell, position, level, token_y, token_x, feature
/cell_ids /cell_lat /cell_lon /position_lat /position_lon /levels_m
```
A cell is one contiguous block (position-major, then the 8 levels `1000,900,…,300`). `float16` stored
**verbatim** (no renorm/recompute). `manifest.json` (written LAST) holds schema, config, axes, per-cell
`cell_id→file/local_idx/source_key`, per-file `shape/bytes/sha256/features_byte_offset/bytes_per_cell`
for direct S3 Range GET.

## Prefixes (confirmed from `_index.json`)
| | source | destination |
|---|---|---|
| kup  | `s3://geo-reference/embeddings/kup/pyramid_dinov2vitg14_p448/kup`   | `s3://geo-reference/embeddings/kup_newformat/pyramid_dinov2vitg14_p448/kup` |
| kram | `s3://geo-reference/embeddings/kram/pyramid_dinov2vitg14_p448/kram` | `s3://geo-reference/embeddings/kram_newformat/pyramid_dinov2vitg14_p448/kram` |

kup: 221 cells, `/features` ≈ **86.33 GiB** → use `--cells-per-shard` to bound shard/disk size.

Run from `geoLocalisation/`. Deps: `--with h5py --with numpy --with boto3`.

## Smoke (few cells → TEST destination; never the real prefix)
```bash
cd ~/work/geoLocalisation
uv run --python 3.11 --with h5py --with numpy --with boto3 python -m pyramid_newformat.convert \
  --source-prefix s3://geo-reference/embeddings/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://geo-reference/embeddings/_smoke_newformat/kup \
  --work-dir /home/ubuntu/work/_nf_work --cells-per-shard 2 --max-cells 3 --resume
uv run --python 3.11 --with h5py --with numpy --with boto3 python -m pyramid_newformat.verify \
  --source-prefix s3://geo-reference/embeddings/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://geo-reference/embeddings/_smoke_newformat/kup \
  --work-dir /home/ubuntu/work/_nf_work --max-cells 3
```

## Dry-run (size + disk estimate, no writes)
```bash
uv run … python -m pyramid_newformat.convert --dry-run \
  --source-prefix s3://geo-reference/embeddings/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://geo-reference/embeddings/kup_newformat/pyramid_dinov2vitg14_p448/kup \
  --work-dir /home/ubuntu/work/_nf_work --cells-per-shard 40
```

## Full convert (kup, then kram) — resumable
```bash
uv run --python 3.11 --with h5py --with numpy --with boto3 python -m pyramid_newformat.convert \
  --source-prefix s3://geo-reference/embeddings/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://geo-reference/embeddings/kup_newformat/pyramid_dinov2vitg14_p448/kup \
  --work-dir /home/ubuntu/work/_nf_work --cells-per-shard 40 --resume
# kram: swap kup→kram and kup→kram in both prefixes
```
`--cells-per-shard 40` ⇒ shard ≈ 15.6 GiB, peak disk ≈ shard + one source cell (~0.4 GiB). Re-running
the same command resumes (checkpoint in `--work-dir`); it refuses a param/order/source-fingerprint change.

## Verify a finished convert (streaming, bitwise, bounded RAM)
```bash
uv run … python -m pyramid_newformat.verify \
  --source-prefix s3://…/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://…/kup_newformat/pyramid_dinov2vitg14_p448/kup \
  --work-dir /home/ubuntu/work/_nf_work        # (add --max-cells N to spot-check)
```

## Read
```python
from pyramid_newformat.reader import NewFormatReader
r = NewFormatReader("s3://geo-reference/embeddings/kup_newformat/pyramid_dinov2vitg14_p448/kup")  # or a local dir
blk   = r.read_cell("kup:0_lvl0")             # (25, 8, 32, 32, 1024) float16  (S3: one Range GET)
lvl   = r.read_cell_level("kup:0_lvl0", 1000) # (25, 32, 32, 1024)
batch = r.read_cells(["kup:0_lvl0","kup:1_lvl0"])  # request order; duplicates share one read; runs coalesced
# No-copy / no-stack: fill a caller-owned buffer in place (returns the (file,local_idx) order it wrote)
import numpy as np
buf = np.empty((len(ids),) + r.block_shape, np.dtype("<f2"))
out, order = r.read_cells_into(ids, buf)      # out is buf; order aligns rows to cell_ids
plat, plon = r.read_cell_coords("kup:0_lvl0") # TRUE per-position coords (from manifest, one read)
```
Local cache mode: point `NewFormatReader` at a local dir (an `aws s3 sync` of the dest). GPU transfer is
a separate step (`torch.from_numpy(blk).to("cuda")`).

## Read-path design — why *few, large, sequential* reads (documented facts vs our measurements)
**Documented properties** (each decision traces to an official doc):
- **Contiguous, uncompressed `/features` ⇒ a cell / a run of cells is one physical span.** HDF5 lays a
  contiguous dataset out in C order with no chunk index, so a slice maps to a single byte range — there
  is no chunk cache to tune. *HDF5 "Dataset Chunking Issues" & "Chunking in HDF5"
  (https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_dset_layout.html).*
- **`read_direct` fills a pre-allocated array**, avoiding the `list → np.stack` second big copy that a
  plain `ds[sel]` + stack incurs. *h5py `Dataset.read_direct`
  (https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.read_direct).*
- **One S3 GET returns one contiguous range** (`Range: bytes=o-o+n-1`); there is no multi-range GET.
  `VersionId` pins an immutable object; for unversioned buckets `If-Match: <etag>` makes a replaced
  object fail loudly instead of serving bytes that no longer match the manifest offset. *AWS S3
  `GetObject` (https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html).*
- **EBS I/O is split into ≤256 KiB units and gp2 throughput is burst-credit limited** — many small
  random reads burn the bucket and collapse to the baseline rate, so coalescing into few large
  sequential reads matters. *AWS EBS volume types
  (https://docs.aws.amazon.com/ebs/latest/userguide/ebs-volume-types.html).*
- **Pinned host memory + `non_blocking=True` lets H2D overlap compute**; the pinned source buffer must
  stay alive until the copy completes. *PyTorch CUDA semantics
  (https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers).*

**Our measurements** (this repo, not vendor claims): a parallel-read A/B on the kup EBS volume was
*slower* than one sequential big read (≈129 vs ≈204 MB/s) — consistent with burst-credit/256-KiB
behaviour above — so the readers default to sequential. A full-kup `search_shard` run measured
`wall≈487 s, read≈347 s (≈65 %), gpu≈140 s`, i.e. the path is **read-bound**, which is why the
refactor targets read count/bytes/copies. Reproduce with `bench.py` / `search_shard.py` before quoting.

## Shared read scheduler (`scheduler.plan_reads`)
Separates **I/O planning** from GPU order: dedup → drop cached → group by file → sort by
`local_cell_idx` → contiguous runs → optionally merge across gaps (bounded) → cap each read by bytes.
Reports `n_reads / n_cells / useful_bytes / read_bytes / overread_bytes / overread_ratio /
peak_read_bytes`. Knobs (also exposed by `bench.py`): `max_read_bytes`, `max_gap_cells`,
`max_overread_ratio`, and `cached`. Modes follow from the knobs — `max_gap_cells=0` = adjacent-only,
`>0` + `max_overread_ratio` = merge-near, huge `max_gap_cells` = one read per shard. **A gap costs a
whole 400 MiB cell of overread**, so merging is gated on the overread budget, never unbounded.

## Bounded cross-query cache (`cache.CellCache`)
LRU bounded by a byte budget; key = `(file, version_or_etag, local_cell_idx)` so a reuploaded shard
can never serve stale bytes under the same offset. For repeated / cross-query reads; `search_shard`
already reads each union cell once per batch so it does not need it.

## Benchmark — 4 variants on one real shortlist (measured)
```bash
uv run … python -m pyramid_newformat.bench \
  --source-prefix s3://…/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://…/kup_newformat/pyramid_dinov2vitg14_p448/kup \
  --work-dir /home/ubuntu/work/_nf_work --n-cells 10 --device cuda --out /home/ubuntu/work/nf_bench.json
```
Reports, **kept separate** (never conflated): GET/HEAD/retry + received bytes (real), useful vs
overread bytes (from the plan), cache hit/miss, reader time (measured around the read, not a
Future-wait), H2D transfer, wall, peak RSS/VRAM — for `old-per-cell`, `adjacent`, `merge`, `span`.
A second pass with a fresh `h5py` handle is **not** labelled "warm" (it does not guarantee a cold OS
page cache).

## Tests (synthetic, no S3/torch)
```bash
cd ~/work/geoLocalisation
uv run --python 3.11 --with h5py --with numpy --with pytest pytest pyramid_newformat/tests -q
```
`test_newformat.py` = convert/reader/loader end-to-end; `test_refactor.py` = the read-reduction work
(scheduler planning + budgets, `CellCache`, reader dedup/arbitrary-order/`out` buffers, true coords,
resume-after-failed-upload, bench counters/variants).

## GPU search over the new format (external top-K indices + directory)
Chunked index interface + drop-in store so the faithful-MAGSAC++ reranker runs on this format:
```python
from pyramid_newformat.loader import ChunkedCellLoader, NewFormatCellStore
# fetch cells by index (cell_id OR global_cell_idx), in GPU-ready chunks:
ld = ChunkedCellLoader("/home/ubuntu/work/mpp_local/kup", device="cuda", chunk_cells=8)
for cell_ids, batch in ld.iter_chunks([0, 5, 12, "kup:73_lvl0"]):   # batch: (C,25,8,32,32,1024) fp16 on GPU
    ...
# or let the existing reranker use it as the store:
store = NewFormatCellStore("/home/ubuntu/work/mpp_local/kup")   # .has/.cell/.config
```
End-to-end rerank of 1..N frames with externally-supplied top-K cell indices over a local directory:
```bash
cd ~/work/geoLocalisation
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm python -m pyramid_newformat.search_newformat \
  --queries kup=/home/ubuntu/work/out/gallery_h5/geo_iso_noverlap/queries/query_kup_d1024.h5 \
  --indices /home/ubuntu/work/out/gallery_h5/kup/shortlist_kup_prod_k100.json \
  --dataset-dir /home/ubuntu/work/mpp_local/kup \
  --topk 5 --level-agg sum --cell-agg mean --device cuda \
  --out /home/ubuntu/work/kup_newfmt_search.json
```
`--indices` = `{query_id:[idx...]}` or a shortlist JSON; idx is a `global_cell_idx` or `cell_id`.

### Minimum-reads shard search (`search_shard`)
Reads **shard-by-shard**, each shard's needed cells as coalesced sequential runs into a **reusable
buffer** (`read_cells_into`, no `np.stack`), scores per level on GPU, with a real 1-shard-ahead
prefetch (double-buffered). Each union cell is read once across all queries. Per-shard progress logs
`read=.. gpu=..`; the final `speed` block separates `wall_s / read_s / gpu_s`.
```bash
uv run … python -m pyramid_newformat.search_shard \
  --queries kup=/…/query_kup_d1024.h5 --indices /…/shortlist_kup_prod_k100.json \
  --dataset-dir /home/ubuntu/work/mpp_local/kup --gpu-cells 4 --topk 5 --device cuda \
  --out /home/ubuntu/work/kup_shard_search.json
```
- `--gpu-cells N` bounds the per-level GPU batch (`crops = N·25`) independently of read size — raise it
  only if VRAM allows; it does **not** change read count.
- `--no-prefetch` disables the read-ahead for low-RAM hosts; with it off **no** background read is
  started (so the first shard is not read twice), at the cost of no read/compute overlap.
RAM ≈ one shard's needed cells (×2 with prefetch); each cell is 400 MiB, so `cells-per-shard` at
convert time bounds this.

## Inspect results on S3
```bash
aws s3 ls s3://geo-reference/embeddings/kup_newformat/  --recursive --human-readable
aws s3 ls s3://geo-reference/embeddings/kram_newformat/ --recursive --human-readable
```
