"""Video sources: enumerate videos in a date range and fetch their bytes.

Two sources, both yielding :class:`VideoRef` objects (which carry ALL the S3
prefixes so the manifest can record them):

* :class:`S3Source`  -- ``s3://<bucket>/live/<station>/filtered/<DD.MM.YYYY>/*.mp4``
  (the mediamtx-recordings layout; uses the ``aws`` CLI, like
  quality-cls ``station_frames``).
* :class:`LocalSource` -- ``*.mp4`` under a directory, station/date parsed from
  the ``chunk_<station>_<YYYY-MM-DD>_...`` filename.

Date range is inclusive. ``VideoRef.open()`` is a context manager yielding a
local path (downloading to a temp dir for S3, cleaned up afterwards).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

DEFAULT_BUCKET = "mediamtx-recordings-crunch"
PREFIX_TEMPLATE = "live/{station}/filtered/{date}/"
STATIONS_PREFIX = "live/"

_DATE_INPUT_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")
_S3_DATE_FORMAT = "%d.%m.%Y"

# chunk_<station>_<YYYY-MM-DD>_<HH-MM-SS>_<...>.mp4
_LOCAL_NAME_RE = re.compile(r"^chunk_(?P<station>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_")


def parse_date(s: str) -> date:
    for fmt in _DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date {s!r}. Use DD.MM.YYYY / YYYY-MM-DD / DD-MM-YYYY / DD/MM/YYYY.")


def fmt_s3_date(d: date) -> str:
    return d.strftime(_S3_DATE_FORMAT)


def date_span(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("date_to is before date_from")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


@dataclass
class VideoRef:
    station: str
    date: str                      # normalised DD.MM.YYYY
    chunk: str                     # filename stem
    s3_uri: str | None = None
    bucket: str | None = None
    key: str | None = None
    _local: Path | None = field(default=None, repr=False)

    @contextmanager
    def open(self) -> Iterator[Path]:
        if self._local is not None:
            yield self._local
            return
        if not self.s3_uri:
            raise RuntimeError(f"VideoRef {self.chunk} has neither a local path nor an s3_uri")
        tmp = Path(tempfile.mkdtemp(prefix="vlp_"))
        dst = tmp / Path(self.key or self.chunk).name
        cp = subprocess.run(["aws", "s3", "cp", self.s3_uri, str(dst), "--quiet"],
                            capture_output=True, text=True)
        if cp.returncode != 0 or not dst.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            raise IOError(f"aws s3 cp failed for {self.s3_uri}: {cp.stderr.strip()}")
        try:
            yield dst
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class S3Source:
    def __init__(self, stations, date_from: str, date_to: str, bucket: str = DEFAULT_BUCKET):
        self.bucket = bucket
        self.dates = date_span(parse_date(date_from), parse_date(date_to))
        self.stations = self._resolve_stations(stations)

    def _aws(self, *args) -> str:
        out = subprocess.run(["aws", *args], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"aws {' '.join(args[:2])} failed:\n{out.stderr.strip()}")
        return out.stdout

    def _resolve_stations(self, spec) -> list[str]:
        if isinstance(spec, (list, tuple)):
            return [str(s).strip() for s in spec if str(s).strip()]
        spec = str(spec).strip()
        if spec.lower() != "all":
            return [s.strip() for s in spec.split(",") if s.strip()]
        txt = self._aws("s3api", "list-objects-v2", "--bucket", self.bucket,
                        "--prefix", STATIONS_PREFIX, "--delimiter", "/",
                        "--query", "CommonPrefixes[].Prefix", "--output", "text")
        stations = []
        for prefix in txt.split():
            parts = [p for p in prefix.split("/") if p]
            if len(parts) == 2:
                stations.append(parts[1])
        return sorted(stations, key=lambda s: (len(s), s))

    def _list_chunks(self, station: str, date_str: str) -> list[str]:
        prefix = PREFIX_TEMPLATE.format(station=station, date=date_str)
        txt = self._aws("s3api", "list-objects-v2", "--bucket", self.bucket, "--prefix", prefix,
                        "--query", "Contents[?ends_with(Key, `.mp4`)].Key", "--output", "text")
        return [k for k in txt.split() if k and k != "None"]

    def __iter__(self) -> Iterator[VideoRef]:
        for station in self.stations:
            for d in self.dates:
                ds = fmt_s3_date(d)
                for key in sorted(self._list_chunks(station, ds)):
                    yield VideoRef(station=station, date=ds, chunk=Path(key).stem,
                                   s3_uri=f"s3://{self.bucket}/{key}", bucket=self.bucket, key=key)


class LocalSource:
    def __init__(self, directory, date_from: str | None = None, date_to: str | None = None,
                 glob: str = "**/*.mp4"):
        self.dir = Path(directory)
        self.glob = glob
        self.lo = parse_date(date_from) if date_from else None
        self.hi = parse_date(date_to) if date_to else None

    def __iter__(self) -> Iterator[VideoRef]:
        for p in sorted(self.dir.glob(self.glob)):
            m = _LOCAL_NAME_RE.match(p.name)
            if not m:
                continue
            d = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
            if (self.lo and d < self.lo) or (self.hi and d > self.hi):
                continue
            yield VideoRef(station=m.group("station"), date=fmt_s3_date(d),
                           chunk=p.stem, _local=p)
