"""Logical-batch planner + microbatch ranges (DSS).

Ported verbatim from ``siam_model_stage4_full_gallery.batching.dss``. One logical batch
= 1 seed + (B_log/2−1) similarity neighbours + (B_log/2) random fillers, with NO
duplicate query id or canonical tile id, and a pair-frequency counter so no pair is a
filler too often. If dedup starves the batch it stays SHORT (``B_log_actual`` records it).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchPlan:
    pair_indices: list
    seed: int
    neighbours_used: list
    random_fillers: list
    B_log_target: int
    B_log_actual: int
    freq_snapshot: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"seed": self.seed, "pair_indices": self.pair_indices,
                "neighbours_used": self.neighbours_used, "random_fillers": self.random_fillers,
                "B_log_target": self.B_log_target, "B_log_actual": self.B_log_actual,
                "no_duplicate_query_ids": True, "no_duplicate_canonical_tiles": True}


def plan_logical_batch(pairs, neighbour_cache, *, B_log: int, seed_pair: int, rng,
                       freq_counter: dict | None = None) -> BatchPlan:
    """Assemble ONE logical batch (see module docstring). ``rng`` is a numpy RandomState."""
    freq = freq_counter if freq_counter is not None else {}
    by_index = {p.pair_index: p for p in pairs}
    chosen, used_q, used_t = [], set(), set()

    def _try_add(pi: int) -> bool:
        p = by_index.get(int(pi))
        if p is None or p.pair_index in chosen:
            return False
        if p.query_id in used_q or p.canonical_tile_row in used_t:
            return False
        chosen.append(p.pair_index)
        used_q.add(p.query_id)
        used_t.add(p.canonical_tile_row)
        freq[p.pair_index] = freq.get(p.pair_index, 0) + 1
        return True

    _try_add(seed_pair)
    n_neighbours = max(0, B_log // 2 - 1)
    n_random = B_log - 1 - n_neighbours
    neighbours_used = []
    for nb in neighbour_cache.of(seed_pair):
        if len(neighbours_used) >= n_neighbours:
            break
        if _try_add(nb):
            neighbours_used.append(nb)
    random_fillers = []
    all_idx = [p.pair_index for p in pairs]
    rng.shuffle(all_idx)
    all_idx.sort(key=lambda pi: (freq.get(pi, 0)))
    want = n_random + (n_neighbours - len(neighbours_used))
    for pi in all_idx:
        if len(chosen) >= B_log:
            break
        if len(random_fillers) >= want:
            break
        if _try_add(pi):
            random_fillers.append(pi)
    return BatchPlan(pair_indices=list(chosen), seed=int(seed_pair),
                     neighbours_used=neighbours_used, random_fillers=random_fillers,
                     B_log_target=int(B_log), B_log_actual=len(chosen), freq_snapshot=dict(freq))


def microbatch_ranges(B: int, B_phys: int) -> list:
    """Physical microbatch ranges for a two-pass GradCache split of a logical batch."""
    return [range(s, min(s + B_phys, B)) for s in range(0, B, B_phys)]
