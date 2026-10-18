"""Minimal test runner for environments without pytest.

Imports the test modules, runs every ``test_*`` function, injects a fresh temp
dir for any that take a ``tmp_path`` parameter, and reports pass/fail.
"""
from __future__ import annotations

import importlib
import inspect
import tempfile
import traceback
from pathlib import Path

MODULES = ["test_segment", "test_diversity", "test_daynight", "test_pipeline",
           "test_source", "test_frames_tree"]


def main() -> int:
    passed = failed = 0
    for modname in MODULES:
        mod = importlib.import_module(f"video_label_pipeline.tests.{modname}")
        for name, fn in sorted(vars(mod).items()):
            if not (name.startswith("test_") and inspect.isfunction(fn)):
                continue
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(tempfile.mkdtemp(prefix="vlp_test_"))
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS {modname}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {modname}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
