"""
Run the whole QDebrisFlow test harness without requiring pytest.

    python tests/run_all.py

Each test module also runs standalone (python tests/test_xxx.py) and is
pytest-compatible (functions named test_*).
"""

from __future__ import annotations

import importlib
import sys
import traceback

import _pathsetup  # noqa: F401

_MODULES = [
    "test_backends_equiv",
    "test_mass_conservation",
    "test_motion",
    "test_normal_depth",
    "test_slope_correction",
    "test_entrainment",
    "test_eight_conn",
    "test_two_phase",
    "test_ab_harness",
]


def main() -> int:
    print(f"Backend in-process: {'numba' if _pathsetup.numba_active() else 'numpy'}")
    failures = 0
    for modname in _MODULES:
        mod = importlib.import_module(modname)
        tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
        for fn in tests:
            label = f"{modname}.{fn.__name__}"
            try:
                print(f"\n>>> {label}")
                fn()
                print(f"    PASS")
            except Exception:
                failures += 1
                print(f"    FAIL")
                traceback.print_exc()
    print("\n" + "=" * 50)
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
