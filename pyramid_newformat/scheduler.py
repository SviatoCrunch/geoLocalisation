"""Shared read planner: turn a batch's needed cells into a minimal set of physical reads, decoupled
from GPU order. Pure (no I/O) → unit-testable.

Pipeline (spec §2): dedup → drop cached → group by file → sort by local_cell_idx → contiguous runs →
optionally merge across small gaps (bounded by ``max_gap_cells`` and ``max_overread_ratio``) → cap each
read by ``max_read_bytes`` → keep each read's cell list (mapping to queries done by the caller).

A gap costs a whole cell of overread (400 MiB) — merging is only worth it when it removes a GET/seek
and the overread stays within budget (``max_overread_ratio``). ``max_read_bytes`` bounds the peak
buffer; no single read exceeds it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Read:
    file: str
    lo: int                     # inclusive local_cell_idx span actually read (may include gap cells)
    hi: int
    cells: list                 # [(local_idx, cell_id)] NEEDED within [lo, hi] (read order)

    @property
    def span(self) -> int:
        return self.hi - self.lo + 1


def plan_reads(needed, bytes_per_cell, *, max_read_bytes=0, max_gap_cells=0,
               max_overread_ratio=0.0, cached=None):
    """``needed``: iterable of ``(file, local_idx, cell_id)``. Returns ``(reads, stats)``.

    ``mode`` is implied by the knobs: ``max_gap_cells=0`` → contiguous-only; >0 with
    ``max_overread_ratio`` → merge near ranges; a huge ``max_gap_cells`` → shard-span read.
    """
    cached = cached or set()
    items_by_file = {}
    seen = set()
    for f, li, c in needed:
        if c in cached or c in seen:          # drop cached + duplicate cell_ids (read once)
            continue
        seen.add(c)
        items_by_file.setdefault(f, []).append((int(li), c))

    reads = []
    for f in sorted(items_by_file):           # files in sorted (sequential-read) order, deterministic
        items = items_by_file[f]
        items.sort()
        cur_lo, cur_hi, cur_cells = None, None, None
        runs = []
        for li, c in items:
            if cur_lo is None:
                cur_lo, cur_hi, cur_cells = li, li, [(li, c)]
                continue
            gap = li - cur_hi - 1
            span = li - cur_lo + 1
            over = span - (len(cur_cells) + 1)
            over_ratio = over / span if span else 0.0
            size_ok = (max_read_bytes <= 0) or (span * bytes_per_cell <= max_read_bytes)
            if size_ok and (gap == 0 or (gap <= max_gap_cells and over_ratio <= max_overread_ratio)):
                cur_hi = li; cur_cells.append((li, c))
            else:
                runs.append((cur_lo, cur_hi, cur_cells)); cur_lo, cur_hi, cur_cells = li, li, [(li, c)]
        if cur_lo is not None:
            runs.append((cur_lo, cur_hi, cur_cells))
        for lo, hi, cells in runs:                               # split any run over the byte budget
            if max_read_bytes > 0 and (hi - lo + 1) * bytes_per_cell > max_read_bytes:
                maxspan = max(1, max_read_bytes // bytes_per_cell)
                s = lo
                while s <= hi:
                    e = min(hi, s + maxspan - 1)
                    sc = [(li, c) for (li, c) in cells if s <= li <= e]
                    if sc:
                        reads.append(Read(f, min(li for li, _ in sc), max(li for li, _ in sc), sc))
                    s = e + 1
            else:
                reads.append(Read(f, lo, hi, cells))

    n_cells = sum(len(r.cells) for r in reads)
    read_cells = sum(r.span for r in reads)
    useful = n_cells * bytes_per_cell
    total = read_cells * bytes_per_cell
    stats = {"n_reads": len(reads), "n_cells": n_cells, "useful_bytes": useful, "read_bytes": total,
             "overread_bytes": total - useful, "overread_ratio": (total - useful) / total if total else 0.0,
             "peak_read_bytes": max((r.span * bytes_per_cell for r in reads), default=0)}
    return reads, stats
