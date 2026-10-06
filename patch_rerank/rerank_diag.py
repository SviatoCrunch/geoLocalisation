"""Full diagnostics archive for the cv2 (``cpu_magsac`` = ``cv2.USAC_MAGSAC``) pyramid rerank.

This records **every** intermediate result of the existing search (``search_pyramid_s3`` with
``--verify-backend cpu_magsac``) without changing a single score, rank, best position/level or
coordinate. For each ``query × candidate cell × position × level`` it stores the inputs/outputs of the
one cv2 call that defined that level's rerank score — correspondences → H + mask → inliers → level
score — plus the position/cell aggregation and the full coarse-100 / reranked-100 / final top-K.

Layout under ``--diag-dir`` (all appended incrementally, so a killed run keeps its finished part):

  manifest.json              run_id, git commit, config fingerprint, store/query/cv2/method metadata
  records.jsonl              one row per query×cell×position×level  (incl. cv2_called=false skips)
  aggregation.jsonl          one row per query×cell  (all level scores → position → cell score)
  summary.jsonl              one row per query  (coarse top-N, reranked top-N, final top-K, metrics)
  arrays/<safe_query>.h5     per record group "<safe_cell>/p{i}/l{L}": qm, rm, q_idx, r_idx, H, mask,
                             residuals  (everything needed to re-analyse the geometry offline)
  completed.json             {fingerprint, queries:[...]}  — resume marker (refuses to mix configs)

Schema relationships: a records.jsonl row's (query_id, cell_id, position_id, level_id) is the key; its
``h5``/``h5_group`` point at the arrays group; aggregation.jsonl rows key on (query_id, cell_id) and
reference the per-level rows; summary.jsonl rows key on query_id. See RERANK_DIAG.md for the reader.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def _safe(s: str) -> str:
    return s.replace(":", "_").replace("/", "_")


def config_fingerprint(config: dict) -> str:
    """Stable hash of the run config — resume refuses to append to an archive built with a different
    one (so an archive never mixes two configurations)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def git_commit(repo: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo),
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class DiagWriter:
    """Incremental writer for one diagnostics run. One instance per process; call :meth:`begin_query`
    / :meth:`record_level` / :meth:`finalize_cell` / :meth:`finalize_query` around the existing search,
    and :meth:`close` at the end. Opening with ``resume=True`` on an existing dir skips already-finished
    queries (same fingerprint required)."""

    def __init__(self, diag_dir: str | Path, manifest: dict, fingerprint: str, *, resume: bool = True,
                 arrays_s3_bucket: str | None = None, arrays_s3_prefix: str | None = None,
                 s3_client=None):
        """``arrays_s3_bucket``/``arrays_s3_prefix`` (optional): after each query is finalized, upload
        its ``arrays/<query>.h5`` to ``s3://<bucket>/<prefix>/<query>.h5`` and delete the local copy —
        so only the compact jsonl + completed.json stay on disk (the full geometry lives on S3). Without
        them, arrays stay local."""
        self.dir = Path(diag_dir).expanduser()
        (self.dir / "arrays").mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self._completed_path = self.dir / "completed.json"
        self._completed: set[str] = set()
        self._manifest_path = self.dir / "manifest.json"
        self._s3_bucket = arrays_s3_bucket
        self._s3_prefix = (arrays_s3_prefix or "").rstrip("/")
        self._s3 = s3_client

        if self._completed_path.exists():
            prev = json.loads(self._completed_path.read_text())
            if prev.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"diag-dir {self.dir} holds a run with a DIFFERENT config fingerprint "
                    f"({prev.get('fingerprint')} != {fingerprint}); refusing to mix configs — use a "
                    f"fresh --diag-dir or delete the old archive")
            if resume:
                self._completed = set(prev.get("queries", []))
        # (re)write the manifest (config is identical by fingerprint; harmless to refresh)
        self._manifest_path.write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, **manifest}, indent=2))

        # append-mode handles; a resumed run keeps the finished rows and appends new ones
        self._rec_f = open(self.dir / "records.jsonl", "a", encoding="utf-8")
        self._agg_f = open(self.dir / "aggregation.jsonl", "a", encoding="utf-8")
        self._sum_f = open(self.dir / "summary.jsonl", "a", encoding="utf-8")
        self._h5 = None
        self._cur_q = None
        self._cell_levels: dict = {}           # (cell_id) -> {(i, L): level_score} for current query

    # -- resume ------------------------------------------------------------
    def is_done(self, query_id: str) -> bool:
        return query_id in self._completed

    # -- per-query lifecycle ----------------------------------------------
    def begin_query(self, query_id: str):
        import h5py
        if self._h5 is not None:
            self._h5.close()
        self._cur_q = query_id
        self._cell_levels = {}
        self._h5 = h5py.File(self.dir / "arrays" / (_safe(query_id) + ".h5"), "a")

    def _write(self, fh, obj):
        fh.write(json.dumps(obj, default=_json_default) + "\n")
        fh.flush()

    def record_level(self, *, query_id, cell_id, coarse_rank, position_id, level_id, level_m,
                     meta: dict, n_query_patches, n_map_patches, n_mutual, level_score,
                     cv2_called: bool, status: str, fail_reason, method_int, reproj_thresh,
                     n_inliers, verify_s, arrays: dict | None):
        """Append one query×cell×position×level record (+ its arrays group when cv2 was called). The
        ``level_score`` passed in is the value the service actually used — stored verbatim."""
        self._cell_levels.setdefault(cell_id, {})[(position_id, level_id)] = float(level_score)
        h5_group = None
        if arrays is not None and self._h5 is not None:
            h5_group = f"{_safe(cell_id)}/p{position_id}/l{int(level_id)}"
            g = self._h5.require_group(h5_group)
            for k, v in arrays.items():
                if k in g:
                    del g[k]
                if v is None:
                    continue
                g.create_dataset(k, data=np.asarray(v))
            g.attrs.update({"n_inliers": int(n_inliers), "level_score": float(level_score),
                            "status": status, "method_int": int(method_int),
                            "reproj_thresh": float(reproj_thresh),
                            "H_direction": "rm(map_crop)->qm(query_frame)"})
        row = {"query_id": query_id, "cell_id": cell_id, "coarse_rank": int(coarse_rank),
               "position_id": int(position_id), "level_id": int(level_id), "level_m": float(level_m),
               "n_query_patches": int(n_query_patches), "n_map_patches": int(n_map_patches),
               "n_mutual": int(n_mutual), "n_inliers": int(n_inliers),
               "level_score": float(level_score), "cv2_called": bool(cv2_called),
               "status": status, "fail_reason": fail_reason, "method_int": int(method_int),
               "reproj_thresh": float(reproj_thresh), "verify_s": float(verify_s),
               "h5": (self._arrays_location(query_id) if h5_group else None), "h5_group": h5_group,
               **meta}
        self._write(self._rec_f, row)

    def finalize_cell(self, *, query_id, cell_id, coarse_rank, pyr, blv, cd, level_agg):
        """Record the position/cell aggregation for one candidate: all level scores per position, the
        position score (sum/max over levels, verbatim ``pyr``), the cell score inputs, best pos/level."""
        levels = [int(x) for x in cd.levels]
        per_pos = []
        for i in range(cd.n_pos):
            lvl_scores = {str(int(L)): self._cell_levels.get(cell_id, {}).get((i, int(L)))
                          for L in levels}
            per_pos.append({"position_id": i, "pos_lat": float(cd.lat[i]), "pos_lon": float(cd.lon[i]),
                            "level_scores": lvl_scores, "position_score": float(pyr[i]),
                            "best_level_m": int(blv[i])})
        arr = np.asarray(pyr, float)
        bi = int(arr.argmax())
        self._write(self._agg_f, {
            "query_id": query_id, "cell_id": cell_id, "coarse_rank": int(coarse_rank),
            "level_agg": level_agg, "positions": per_pos,
            "position_scores": [float(x) for x in pyr], "n_positions": int(cd.n_pos),
            "cell_score_mean": float(arr.mean()), "cell_score_min": float(arr.min()),
            "cell_score_max": float(arr.max()),
            "best_position_id": bi, "best_level_m": int(blv[bi]),
            "best_h5_group": f"{_safe(cell_id)}/p{bi}/l{int(blv[bi])}"})

    def _arrays_location(self, query_id: str) -> str:
        """Final location of a query's arrays H5 — the s3:// uri when offloading, else the local
        relative path. Used BOTH for the per-record ``h5`` pointer and the summary, so a records.jsonl
        row points at where its arrays actually end up (not a deleted local path)."""
        if self._s3_bucket:
            key = f"{self._s3_prefix}/{_safe(query_id)}.h5" if self._s3_prefix else f"{_safe(query_id)}.h5"
            return f"s3://{self._s3_bucket}/{key}"
        return f"arrays/{_safe(query_id)}.h5"

    def _offload_arrays(self, query_id: str) -> str:
        """Upload this query's arrays H5 to S3 and delete the local copy; return its location. Upload
        errors propagate (the run should fail loudly rather than silently drop geometry)."""
        if not self._s3_bucket:
            return self._arrays_location(query_id)
        local = self.dir / "arrays" / (_safe(query_id) + ".h5")
        _, _, key = self._arrays_location(query_id)[5:].partition("/")
        self._s3.upload_file(str(local), self._s3_bucket, key)
        local.unlink(missing_ok=True)
        return self._arrays_location(query_id)

    def finalize_query(self, *, query_id, gt, coarse_cells, reranked, final_topk, metrics):
        """Close the query's arrays file, offload it (S3 + local delete when configured), record the
        per-query summary (full coarse list, full reranked list, final top-K, metrics, arrays location)
        and mark it complete (so resume skips it)."""
        if self._h5 is not None:
            self._h5.close(); self._h5 = None
        arrays_location = self._offload_arrays(query_id)
        self._write(self._sum_f, {
            "query_id": query_id, "gt": gt, "arrays": arrays_location,
            "coarse_top": [{"cell_id": c, "coarse_rank": r} for r, c in enumerate(coarse_cells)],
            "reranked_top": reranked, "final_topk": final_topk, "metrics": metrics})
        self._completed.add(query_id)
        self._completed_path.write_text(json.dumps(
            {"fingerprint": self.fingerprint, "queries": sorted(self._completed)}))

    def close(self):
        if self._h5 is not None:
            self._h5.close(); self._h5 = None
        for fh in (self._rec_f, self._agg_f, self._sum_f):
            fh.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
