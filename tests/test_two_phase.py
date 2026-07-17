"""
Phase 4 (D) — two-phase / basal pore-pressure ratio λ (Iverson).

λ scales the yield stress by (1−λ): high pore pressure liquefies the basal layer,
cuts the yield, and lets the flow travel farther; λ decays (consolidation) so the
flow eventually stops. This is the main physical driver of debris-flow mobility.

Checks:
1. Runout grows monotonically with the initial pore-pressure ratio λ0.
2. two_phase ON with λ0 = 0 is identical to two_phase OFF (and λ field stays 0).
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401
import synthetic


def _runout(lam0, two_phase=True, Tc=1500.0):
    ny, nx = 80, 21
    dx = dy = 10.0
    dem, _, _ = synthetic.tilted_plane(ny=ny, nx=nx, slope=0.06)
    src = synthetic.point_inflow(
        rows=[2, 2, 2], cols=[9, 10, 11],
        times=[0.0, 60.0, 300.0, 4000.0], q_total=[0.0, 30.0, 30.0, 0.0], name="s")
    res = synthetic.run_scenario(
        dem, dx, dy, sources=[src],
        params_kw=dict(Cv=0.45, tau_yield_direct=2000.0, eta_direct=20.0,
                       manning_n=0.06, pore_pressure_lambda0=lam0,
                       pore_consolidation_time=Tc),
        config_kw=dict(t_end=900.0, dt_max=8.0, cfl_number=0.5, two_phase=two_phase),
    )
    wet = np.where(res.h_max.max(axis=1) > 0.05)[0]
    return int(wet.max()), res


def test_two_phase_runout_monotonic():
    r0, _ = _runout(0.0)
    r4, _ = _runout(0.4)
    r8, _ = _runout(0.8)
    print(f"  runout rows  λ0=0:{r0}  λ0=0.4:{r4}  λ0=0.8:{r8}")
    assert r0 < r4 < r8, "runout must increase with the pore-pressure ratio λ0"


def test_two_phase_off_is_inert():
    r_off, res_off = _runout(0.0, two_phase=False)
    r_on, res_on = _runout(0.0, two_phase=True)
    print(f"  runout OFF={r_off}  ON(λ0=0)={r_on}  max λ_final(on)={res_on.lambda_final.max():.3e}")
    assert r_off == r_on, "two_phase ON with λ0=0 must match two_phase OFF"
    assert float(res_on.lambda_final.max()) == 0.0, "λ should be 0 when λ0=0"


if __name__ == "__main__":
    test_two_phase_runout_monotonic()
    test_two_phase_off_is_inert()
    print("test_two_phase: PASS")
