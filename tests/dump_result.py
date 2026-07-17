"""
Run a named scenario and dump its result arrays to an .npz file.

Used by test_backends_equiv.py, which invokes this script in separate
subprocesses (with and without QDF_NO_NUMBA=1) and compares the outputs.

Usage:  python dump_result.py <out.npz> [scenario]
        scenario in {"canon", "canon_slope"}  (default "canon")
"""

from __future__ import annotations

import sys

import numpy as np

import _pathsetup
import synthetic

SCENARIOS = {
    "canon":         synthetic.canonical_scenario,
    "canon_slope":   synthetic.canonical_scenario_slope,
    "canon_entrain": synthetic.canonical_scenario_entrain,
    "canon_8conn":   synthetic.canonical_scenario_8conn,
    "canon_2phase":  synthetic.canonical_scenario_2phase,
}


def main(out_path: str, scenario: str = "canon") -> None:
    res = SCENARIOS[scenario]()
    np.savez(
        out_path,
        h_max=res.h_max,
        V_max=res.V_max,
        h_final=res.h_final,
        t_arrival=np.where(np.isinf(res.t_arrival), -1.0, res.t_arrival),
        n_steps=np.array([res.n_steps]),
        numba=np.array([_pathsetup.numba_active()]),
    )
    print(f"[{'numba' if _pathsetup.numba_active() else 'numpy'}] {scenario}: "
          f"steps={res.n_steps} h_max={res.h_max.max():.4f} -> {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "result.npz"
    scen = sys.argv[2] if len(sys.argv) > 2 else "canon"
    main(out, scen)
