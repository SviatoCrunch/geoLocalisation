"""Walk a directory of HDF5 files and write a Markdown report of each: datasets (shape/dtype), attrs
(levels_m / step_m / output_px / backbone / ...), and - for group-structured stores like the rerank
store (``p{i}/l{L}``) - the pyramid LEVELS present, per-level grid shape, and the position count.

Read-only. Self-contained (h5py + numpy). Run::

    uv run --python 3.11 --with h5py --with numpy python -m patch_rerank.h5_report \
      --root /home/ubuntu/work/out/gallery_h5 --out /home/ubuntu/work/out/gallery_h5/H5_REPORT.md
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _fmt(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def _attrval(v):
    if isinstance(v, (bytes,)):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(str(x) for x in v.tolist()[:12]) + ("..." if v.size > 12 else "") + "]"
    return str(v)


def summarize(path):
    import h5py
    out = [f"### `{path.name}`  ({_fmt(os.path.getsize(path))})", ""]
    try:
        f = h5py.File(path, "r")
    except Exception as e:                                    # noqa: BLE001
        return "\n".join(out + [f"- **could not open**: {e}", ""])
    with f:
        if f.attrs:
            out.append("**attrs:** " + ", ".join(f"`{k}`={_attrval(v)}" for k, v in f.attrs.items()))
            out.append("")
        top_ds, top_groups = [], []
        for k in f.keys():
            (top_ds if isinstance(f[k], h5py.Dataset) else top_groups).append(k)

        if top_ds:
            out.append("| dataset | shape | dtype |")
            out.append("|---|---|---|")
            for k in top_ds:
                d = f[k]
                out.append(f"| `{k}` | {tuple(d.shape)} | {d.dtype} |")
            out.append("")

        if top_groups:
            # group-structured (store: p{i}/l{L}, or query: img/ groups). Sample the first group.
            n_groups = len(top_groups)
            sample = top_groups[0]
            g = f[sample]
            sub = list(g.keys()) if hasattr(g, "keys") else []
            out.append(f"**{n_groups} groups** (e.g. `{sample}`) with sub-datasets:")
            out.append("")
            out.append("| sub-dataset | shape | dtype |")
            out.append("|---|---|---|")
            for s in sub:
                d = g[s]
                shp = tuple(d.shape) if hasattr(d, "shape") else "?"
                dt = str(d.dtype) if hasattr(d, "dtype") else "group"
                out.append(f"| `{s}` | {shp} | {dt} |")
            out.append("")
            # pyramid levels if sub-datasets look like l<NNN>
            levels = sorted(int(s[1:]) for s in sub if s.startswith("l") and s[1:].isdigit())
            if levels:
                shapes = {int(s[1:]): tuple(g[s].shape) for s in sub if s.startswith("l") and s[1:].isdigit()}
                out.append(f"**pyramid levels** ({len(levels)}): " +
                           ", ".join(f"{L}m {shapes[L]}" for L in levels))
                out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory to scan recursively for *.h5")
    ap.add_argument("--out", default=None, help="write Markdown here (else stdout)")
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    files = sorted(p for p in root.rglob("*.h5")
                   if len(p.relative_to(root).parts) <= args.max_depth)
    hdr = [f"# HDF5 report - `{root}`", "", f"{len(files)} files.", ""]
    body = []
    for p in files:
        print(f"[scan] {p}", flush=True)
        body.append(summarize(p))
    doc = "\n".join(hdr + body)
    if args.out:
        Path(args.out).expanduser().write_text(doc, encoding="utf-8")
        print(f"[ok] -> {args.out}", flush=True)
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
