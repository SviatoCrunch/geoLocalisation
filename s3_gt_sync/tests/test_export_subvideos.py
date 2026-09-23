"""export_subvideos: clip naming + frame-routing (pure logic, no cv2/S3)."""
from s3_gt_sync.export_subvideos import clip_name, build_plan, owner, _safe


def test_clip_name_matches_parent_stem():
    sv = {"sub_index": 1, "start_frame": 130, "end_frame": 260, "boundary": "overlap_cut"}
    name = clip_name("chunk_104_c01", sv)
    assert name == "chunk_104_c01__sub01_f130-260_overlap_cut.mp4"
    assert name.startswith("chunk_104_c01__")            # shares the full-video stem prefix


def test_clip_name_missing_boundary_and_pads_index():
    assert clip_name("v", {"sub_index": 0, "start_frame": 0, "end_frame": 5}) == \
        "v__sub00_f0-5_na.mp4"


def test_clip_name_flags_too_short():
    sv = {"sub_index": 2, "start_frame": 40, "end_frame": 42, "boundary": "overlap_cut",
          "too_short": True}
    assert clip_name("v", sv) == "v__sub02_f40-42_overlap_cut_short.mp4"


def test_build_plan_sorted_by_start():
    rec = {"sub_videos": [
        {"sub_index": 1, "start_frame": 130, "end_frame": 260, "boundary": "overlap_cut"},
        {"sub_index": 0, "start_frame": 0, "end_frame": 118, "boundary": "first"},
    ]}
    plan = build_plan(rec, "chunk")
    assert [p["start"] for p in plan] == [0, 130]         # sorted, not JSON order
    assert plan[0]["sub_index"] == 0 and plan[1]["sub_index"] == 1


def test_owner_advances_over_disjoint_ranges():
    plan = [{"start": 0, "end": 10}, {"start": 20, "end": 30}]
    j = 0
    j = owner(5, plan, j);   assert j == 0                # inside first
    j = owner(15, plan, j);  assert j == 1                # gap -> pointer moves to next
    j = owner(25, plan, j);  assert j == 1                # inside second
    j = owner(40, plan, j);  assert j == 2                # past all -> len(plan)


def test_safe_sanitizes():
    assert _safe("chunk_104_2026-07-29_12-42-06_6s_c02") == "chunk_104_2026-07-29_12-42-06_6s_c02"
    assert _safe("a/b c") == "a_b_c"
