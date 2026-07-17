"""
Shared path / backend setup for the QDebrisFlow test harness.

- Adds the QDebrisFlowEN plugin directory to sys.path so `core.*` imports work
  outside QGIS (the core solver depends only on numpy + optional numba).
- If the env var QDF_NO_NUMBA=1 is set, blocks `import numba` so the solver
  falls back to its pure-NumPy kernels. This lets the equivalence test run the
  two backends in separate subprocesses.

Import this module FIRST, before importing anything from `core`.
"""

from __future__ import annotations

import os
import sys

# Windows consoles default to cp1251 here; force UTF-8 so diagnostic prints with
# non-ASCII glyphs don't raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

# Block numba (must happen before core.solver is imported anywhere).
if os.environ.get("QDF_NO_NUMBA") == "1":
    # Setting a module to None makes `import numba` raise ImportError, which
    # drives solver.py into its `except ImportError` → NumPy-fallback branch.
    sys.modules["numba"] = None  # type: ignore[assignment]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

# Default to the English copy; both copies are functionally identical.
_PLUGIN_DIR = os.environ.get(
    "QDF_PLUGIN_DIR",
    os.path.join(_REPO_ROOT, "QDebrisFlowEN"),
)

if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)


def numba_active() -> bool:
    """True if the solver imported with the Numba JIT backend."""
    import core.solver as _s
    return bool(getattr(_s, "_NUMBA_AVAILABLE", False))


def plugin_dir() -> str:
    return _PLUGIN_DIR
