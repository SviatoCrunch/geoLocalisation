from datetime import date

from video_label_pipeline.source import LocalSource, date_span, fmt_s3_date, parse_date


def test_parse_date_formats():
    assert parse_date("10.10.2026") == date(2026, 10, 10)
    assert parse_date("2026-10-10") == date(2026, 10, 10)
    assert fmt_s3_date(date(2026, 10, 10)) == "10.10.2026"


def test_date_span_inclusive():
    span = date_span(date(2026, 10, 10), date(2026, 10, 12))
    assert span == [date(2026, 10, 10), date(2026, 10, 11), date(2026, 10, 12)]


def test_local_source_parses_and_filters_by_date(tmp_path):
    names = [
        "chunk_104_2026-10-10_08-07-07_20s_c02.mp4",
        "chunk_138_2026-10-12_09-00-00_5s_c01.mp4",
        "chunk_40_2026-10-20_09-00-00_5s_c01.mp4",   # out of range
        "not_a_chunk.mp4",                            # ignored
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"")

    refs = list(LocalSource(tmp_path, date_from="10.10.2026", date_to="15.10.2026"))
    got = {(r.station, r.date) for r in refs}
    assert got == {("104", "10.10.2026"), ("138", "12.10.2026")}
    assert all(r.s3_uri is None and r._local is not None for r in refs)
