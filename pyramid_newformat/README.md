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
batch = r.read_cells(["kup:0_lvl0","kup:1_lvl0"])  # request order; consecutive cells coalesced into one read
```
Local cache mode: point `NewFormatReader` at a local dir (an `aws s3 sync` of the dest). GPU transfer is
a separate step (`torch.from_numpy(blk).to("cuda")`).

## Benchmark old vs new (measured)
```bash
uv run … python -m pyramid_newformat.bench \
  --source-prefix s3://…/kup/pyramid_dinov2vitg14_p448/kup \
  --destination-prefix s3://…/kup_newformat/pyramid_dinov2vitg14_p448/kup \
  --work-dir /home/ubuntu/work/_nf_work --n-cells 10 --device cuda
```

## Tests (synthetic, no S3/torch)
```bash
cd ~/work/geoLocalisation
uv run --python 3.11 --with h5py --with numpy --with pytest pytest pyramid_newformat/tests -q
```

## Inspect results on S3
```bash
aws s3 ls s3://geo-reference/embeddings/kup_newformat/  --recursive --human-readable
aws s3 ls s3://geo-reference/embeddings/kram_newformat/ --recursive --human-readable
```
