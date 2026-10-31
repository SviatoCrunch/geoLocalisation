"""Torch-free tests for the CSV motion-budget sampler (parse, selection, manifest schema)."""
from __future__ import annotations

from pathlib import Path

from clip_search.sample_motion import (
    clip_entry,
    frame_stem,
    make_qid,
    parse_csv,
    select_positions,
)


# ------------------------------- CSV row fixture -------------------------------

def _row(video_id, zone, video, pred_lat, pred_lon, gt_lat, gt_lon, matched):
    """Build a 31-column CSV row with the fields at their real indices."""
    r = [""] * 31
    r[0] = video_id
    r[4] = zone
    r[5] = video
    r[8] = str(pred_lat); r[9] = str(pred_lon)
    r[27] = str(gt_lat); r[28] = str(gt_lon)
    r[29] = matched
    return ",".join(r)


def _write_csv(tmp: Path) -> Path:
    rows = [
        _row("v1", "0-17-34-138", "s3://b/live/34/f/a.mp4", 49.70, 37.60, 49.70, 37.60, "true"),
        _row("v2", "0-17-34-138", "s3://b/live/138/f/b.mp4", 49.71, 37.61, 49.6999, 37.6101, "false"),
        _row("v3", "40-104-105", "s3://b/live/104/f/c.mp4", 48.60, 37.55, 48.60, 37.55, "true"),  # other zone
        _row("v4", "0-17-34-138", "local/not-s3.mp4", 49.70, 37.60, 49.70, 37.60, "true"),         # non-s3
        _row("v5", "0-17-34-138", "s3://b/live/34/f/a.mp4", 49.70, 37.60, 49.72, 37.62, "true"),   # dup url
    ]
    p = tmp / "det.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_parse_csv_zone_filter_and_gt(tmp_path):
    recs = parse_csv(_write_csv(tmp_path), zone="0-17-34-138")
    # v3 (other zone), v4 (non-s3), v5 (dup url) dropped → v1, v2
    assert [r["video_id"] for r in recs] == ["v1", "v2"]
    v2 = recs[1]
    assert v2["gt_lat"] == 49.6999 and v2["gt_lon"] == 37.6101     # verified pair (cols 27/28)
    assert v2["pred_lat"] == 49.71 and v2["pred_lon"] == 37.61     # predicted (cols 8/9)
    assert v2["matched"] is False
    assert recs[0]["matched"] is True


def test_parse_csv_no_zone(tmp_path):
    recs = parse_csv(_write_csv(tmp_path), zone=None)
    # all s3 rows, dedup by url → v1, v2, v3 (v4 non-s3 dropped, v5 dup dropped)
    assert {r["video_id"] for r in recs} == {"v1", "v2", "v3"}


# ------------------------------- stem / qid round-trip -------------------------------

def _parse_stem(stem):
    """Mirror of map_extract.helpers._parse_image_stem (imported there pulls torchvision)."""
    parts = stem.split("_")
    return parts[0], float(parts[1]), float(parts[2])


def test_frame_stem_parses_back():
    stem = frame_stem(42, 49.6999001, 37.6101002)
    idx, lat, lon = _parse_stem(stem)
    assert idx == "00000042"
    assert abs(lat - 49.6999001) < 1e-6 and abs(lon - 37.6101002) < 1e-6
    assert make_qid("kup", stem) == f"kup:{stem}"


# ------------------------------- select_positions -------------------------------

def test_select_positions_endpoints_and_budget():
    # one segment, uniform motion: positions 0..10, cum 0..10 (step 1 → mu=1)
    seg = [(list(range(11)), [float(i) for i in range(11)])]
    sel = select_positions(seg, frames_per_key=2.0)        # budget = 2
    assert sel[0] == 0 and sel[-1] == 10                   # endpoints always kept
    assert sel == sorted(set(sel))                         # unique, sorted
    assert 2 <= len(sel) <= 11


def test_select_positions_target_k():
    seg = [(list(range(21)), [float(i) for i in range(21)])]
    sel = select_positions(seg, target_k=5)
    assert 0 in sel and 20 in sel
    assert len(sel) <= 5                                   # fpk_for_k caps to ~target_k


def test_select_positions_two_segments_maps_original_indices():
    # second segment's local positions must map back to original frame indices (pos[lp])
    seg = [([0, 1, 2], [0.0, 1.0, 2.0]), ([50, 51, 52, 53], [0.0, 1.0, 2.0, 3.0])]
    sel = select_positions(seg, frames_per_key=1.0)
    assert 0 in sel and 2 in sel and 50 in sel and 53 in sel
    assert max(sel) == 53                                  # original index, not local


# ------------------------------- clip_entry schema -------------------------------

def test_clip_entry_schema():
    rec = {"video_id": "v1", "video_url": "s3://b/a.mp4", "gt_lat": 49.7, "gt_lon": 37.6,
           "pred_lat": 49.7, "pred_lon": 37.6, "matched": True}
    stem = frame_stem(0, 49.7, 37.6)
    fm = [{"global_idx": 0, "frame_index": 100, "stem": stem, "qid": make_qid("kup", stem),
           "file": f"{stem}.jpg", "quality": "good"}]
    e = clip_entry(rec, "kup", fm)
    assert e["clip_id"] == "kup_v1" and e["city"] == "kup"
    assert e["gt_lat"] == 49.7 and e["n_selected"] == 1
    assert e["frames"][0]["qid"] == f"kup:{stem}"


# ------------------------------- report_levels.build_report -------------------------------

def test_build_report_levels_detail():
    from clip_search.report_levels import build_report, _csv_rows
    locate = {"meta": {"levels_m": [1000, 500]},
              "per_query": {"kup:0001": {"topk": [
                  {"cell_id": "c1", "cell_score": 0.8, "lat": 49.7, "lon": 37.6, "level_m": 1000,
                   "level_scores": [0.8, 0.2], "winning_position_idx": 3, "dist_m": 5.0},
                  {"cell_id": "c2", "cell_score": 0.1, "lat": 49.9, "lon": 37.9, "level_m": 500,
                   "level_scores": [0.0, 0.1]}]}}}
    results = {"per_clip": {"kup_v1": {"n_frames_scored": 1, "gt": [49.7, 37.6], "by_n": {"10": {
                   "sum": [{"cell_id": "c1", "lat": 49.7, "lon": 37.6, "dist_m": 5.0, "level_m": 1000}]}}}},
               "summary": {"sum_top10": {"n_eval": 1}}}
    manifest = {"clips": [{"clip_id": "kup_v1", "video_id": "v1", "frames": [
        {"qid": "kup:0001", "frame_index": 100, "file": "x.jpg", "quality": "good"}]}]}

    rep = build_report(locate, results, manifest, agg="sum", n=10, topk_detail=100)
    v = rep["videos"][0]
    assert v["clip_id"] == "kup_v1" and v["video_id"] == "v1"
    assert v["pred"]["dist_m"] == 5.0 and v["pred"]["cell_id"] == "c1"
    fr = v["frames"][0]
    assert fr["frame_index"] == 100 and fr["quality"] == "good" and fr["n_cells"] == 2
    # every top-K cell carries per-level scores labelled by level_m
    assert fr["topk"][0]["levels"] == {"1000": 0.8, "500": 0.2}
    assert fr["topk"][0]["winning_position_idx"] == 3
    assert v["level_hist"] == {"1000": 1}                         # frame's top-1 level
    row = next(iter(_csv_rows(rep)))
    assert row["hit250"] == 1 and row["best_level_m"] == "1000" and row["dist_m"] == 5.0
