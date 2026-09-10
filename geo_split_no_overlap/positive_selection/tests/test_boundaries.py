"""Architecture test: enforce the one-way dependency boundary.

  * positive_selection (non-test code) must NOT import the split layer
    (components / optimizer / audit / spatial_conflicts / io / cli / schemas).
  * split-layer modules must NOT import positive_selection INTERNALS
    (strategies/* or adapters/*) — only the public API.
"""
from __future__ import annotations

import re
from pathlib import Path

_PS = Path(__file__).resolve().parents[1]          # .../positive_selection
_SPLIT = _PS.parent                                # .../geo_split_no_overlap

_FORBIDDEN_IN_PS = ("components", "optimizer", "audit", "spatial_conflicts",
                    "geo_split_no_overlap.io", "geo_split_no_overlap.cli", "schemas")


def _py_files(root: Path):
    return [p for p in root.rglob("*.py") if "tests" not in p.parts]


def test_positive_selection_does_not_import_split_layer():
    offenders = []
    for f in _py_files(_PS):
        text = f.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if not (s.startswith("import ") or s.startswith("from ")):
                continue
            for bad in _FORBIDDEN_IN_PS:
                if bad in s:
                    offenders.append(f"{f.name}: {s}")
    assert not offenders, f"positive_selection must not import split layer: {offenders}"


def test_split_layer_uses_only_public_positive_api():
    split_only = [p for p in _SPLIT.glob("*.py")]   # top-level split modules
    offenders = []
    pat = re.compile(r"positive_selection\.(strategies|adapters)")
    for f in split_only:
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if (s.startswith("import ") or s.startswith("from ")) and pat.search(s):
                offenders.append(f"{f.name}: {s}")
    assert not offenders, f"split layer must use only the public API: {offenders}"
