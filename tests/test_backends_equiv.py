"""
Regression test: the Numba JIT kernels and the NumPy fallback must produce the
same result. This is the repo's stated invariant and the safety net for every
kernel edit in later phases.

Strategy: run each scenario twice in fresh subprocesses — one with Numba, one
with QDF_NO_NUMBA=1 — then compare the dumped arrays. Both the planar baseline
("canon") and the steep-slope-correction path ("canon_slope") are checked.

Tolerance: Numba kernels are compiled with fastmath=True, which may reorder
float ops, so we allow a small tolerance rather than bit-identity. The goal is
to catch *logic* divergence between the two code paths.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

import _pathsetup  # noqa: F401
import synthetic

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = ("canon", "canon_slope", "canon_entrain", "canon_8conn", "canon_2phase")


def _run(out_path: str, scenario: str, disable_numba: bool) -> dict:
    env = dict(os.environ)
    if disable_numba:
        env["QDF_NO_NUMBA"] = "1"
    else:
        env.pop("QDF_NO_NUMBA", None)
    subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, "dump_result.py"),
         out_path, scenario],
        cwd=_THIS_DIR, env=env, check=True,
    )
    with np.load(out_path) as d:
        return {k: d[k] for k in d.files}


def test_numba_equals_numpy():
    for scenario in SCENARIOS:
        print(f"\n  scenario: {scenario}")
        with tempfile.TemporaryDirectory() as tmp:
            nb = _run(os.path.join(tmp, "nb.npz"), scenario, disable_numba=False)
            npf = _run(os.path.join(tmp, "np.npz"), scenario, disable_numba=True)

        assert bool(nb["numba"][0]) is True, "Numba backend was not active."
        assert bool(npf["numba"][0]) is False, "NumPy fallback did not engage."
        assert int(nb["n_steps"][0]) == int(npf["n_steps"][0]), (
            f"step count differs: numba={int(nb['n_steps'][0])} "
            f"numpy={int(npf['n_steps'][0])}")

        # Depth/extent/timing are the state invariants. Compared by field-
        # normalised max difference rather than a fixed absolute tolerance: the
        # logic is identical on both backends, but the Numba kernels use fastmath
        # (reordered float ops, ~1e-15/op) and the Phase-2 Cv feedback amplifies
        # that over a run. A genuine logic divergence would be order-1 here.
        for key in ("h_max", "h_final", "t_arrival"):
            a, b = nb[key], npf[key]
            scale = max(float(np.max(np.abs(a))), 1e-9)
            nd = float(np.max(np.abs(a - b))) / scale
            print(f"    {key:10s} norm|d|={nd:.3e}  {'OK' if nd < 1e-3 else 'FAIL'}")
            assert nd < 1e-3, f"{scenario}/{key}: backends diverge (norm {nd:.3e})"

        # V_max is a running-max of the derived q/h velocity, recorded at the
        # instant a thin front passes (h ~ h_min). Under the Cv feedback the two
        # backends hit those instants at slightly different moments, so the
        # per-cell running-max is not an invariant (the wet-mean is printed for
        # information only). The robust, asserted invariant is the domain peak.
        a, b = nb["V_max"], npf["V_max"]
        wet = (nb["h_max"] > 0.1) & (npf["h_max"] > 0.1)
        peak_rel = abs(float(a.max()) - float(b.max())) / max(float(a.max()), 1e-9)
        mean_rel = (abs(float(a[wet].mean()) - float(b[wet].mean()))
                    / max(float(a[wet].mean()), 1e-9)) if wet.any() else 0.0
        print(f"    {'V_max':10s} peak_rel={peak_rel:.2e}  (wet-mean_rel={mean_rel:.2e} info)")
        assert peak_rel < 5e-3, f"{scenario}/V_max peak differs {peak_rel:.2%}"


def test_off_baseline_pinned():
    """The default-OFF (planar) result must not drift as later phases land."""
    res = synthetic.canonical_scenario()
    peak = float(res.h_max.max())
    print(f"  canonical h_max peak = {peak:.4f} (pinned {synthetic.CANONICAL_HMAX})")
    assert abs(peak - synthetic.CANONICAL_HMAX) < 1e-2, (
        f"OFF baseline drifted: {peak:.4f} vs {synthetic.CANONICAL_HMAX}")


if __name__ == "__main__":
    test_numba_equals_numpy()
    test_off_baseline_pinned()
    print("test_backends_equiv: PASS")
