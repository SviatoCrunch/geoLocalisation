"""Download the four pure SegProp reference modules for parity testing.

These are the authors' own files (github.com/vlicaret/segprop, no license), fetched
into this git-ignored folder so ``tests/test_segprop_parity.py`` can check our port
byte-for-byte against the original. Not redistributed; fetched on demand.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/vlicaret/segprop/master/{}"
FILES = ("util.py", "flow.py", "map2d.py", "classmap.py")


def fetch(dest: Path | None = None) -> bool:
    dest = dest or Path(__file__).parent
    ok = True
    for f in FILES:
        try:
            urllib.request.urlretrieve(RAW.format(f), dest / f)
            print(f"fetched {f}")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {f}: {e}")
            ok = False
    return ok


def available(dest: Path | None = None) -> bool:
    dest = dest or Path(__file__).parent
    return all((dest / f).exists() for f in FILES)


if __name__ == "__main__":
    raise SystemExit(0 if fetch() else 1)
