"""
Phase 1 (C) — steep-slope correction.

The planar shallow-water formulation drives the flow with the surface gradient
measured over the *horizontal* spacing dx. The correction multiplies that term
by cos(theta), i.e. measures the gradient along the bed surface (true distance
dx/cos). For a uniform plane this turns the effective driving slope from
tan(theta) (planar) into tan(theta)*cos(theta) = sin(theta).

So at steady uniform flow:
    OFF  -> normal depth solves Sf(h) = tan(theta)
    ON   -> normal depth solves Sf(h) = sin(theta)   (smaller slope -> deeper)

We measure the established reach below the inlet on a steep plane with the
correction OFF then ON, and check the ON/OFF depth ratio matches the analytic
normal-depth ratio. Comparing the ratio cancels measurement/transient bias.
"""

from __future__ import annotations

import math

import numpy as np

import _pathsetup  # noqa: F401
import synthetic
from test_normal_depth import normal_depth


def _upper_reach_depth(slope, slope_correction, params_kw, Q_total=30.0):
    ny, nx = 140, 10
    dx = dy = 5.0
    dem, _, _ = synthetic.tilted_plane(ny=ny, nx=nx, slope=slope, dx=dx, dy=dy)
    src = synthetic.point_inflow(
        rows=[1] * nx, cols=list(range(nx)),
        times=[0.0, 60.0, 4000.0], q_total=[0.0, Q_total, Q_total], name="top")
    res = synthetic.run_scenario(
        dem, dx, dy, sources=[src], params_kw=params_kw,
        config_kw=dict(t_end=1500.0, dt_max=8.0, cfl_number=0.5,
                       output_interval=300.0, slope_correction=slope_correction))
    band = res.h_final[int(0.08 * ny):int(0.22 * ny), :]
    h_min = 0.005
    return (float(np.median(band[band > h_min])) if np.any(band > h_min) else 0.0,
            Q_total / (nx * dx))


def test_slope_correction_shifts_tan_to_sin():
    theta = math.radians(25.0)
    S = math.tan(theta)                 # planar bed slope used to build the DEM
    sin_t = math.sin(theta)
    params_kw = dict(Cv=0.45, tau_yield_direct=350.0, eta_direct=30.0,
                     manning_n=0.07, K_visc=24.0)

    from core.rheology import OBrienParameters
    p = OBrienParameters(**params_kw)

    h_off, q = _upper_reach_depth(S, False, params_kw)
    h_on, _ = _upper_reach_depth(S, True, params_kw)

    nd_off = normal_depth(q, S, p.gamma_mixture, p.tau_yield, p.eta,
                          p.manning_n, p.K_visc)
    nd_on = normal_depth(q, sin_t, p.gamma_mixture, p.tau_yield, p.eta,
                         p.manning_n, p.K_visc)

    print(f"  theta=25 deg  q={q:.3f}  tan={S:.3f} sin={sin_t:.3f}")
    print(f"  OFF: h_sim={h_off:.3f}  h_n(tan)={nd_off:.3f}  ratio={h_off/nd_off:.2f}")
    print(f"  ON : h_sim={h_on:.3f}  h_n(sin)={nd_on:.3f}  ratio={h_on/nd_on:.2f}")
    print(f"  depth ratio  sim ON/OFF={h_on/h_off:.3f}  analytic={nd_on/nd_off:.3f}")

    # Each backend run tracks its own analytic normal depth.
    assert 0.8 < h_off / nd_off < 1.25, "OFF depth far from tan(theta) normal depth"
    assert 0.8 < h_on / nd_on < 1.25, "ON depth far from sin(theta) normal depth"
    # The correction must deepen the flow, by the analytic amount (ratio cancels bias).
    assert h_on > h_off, "correction did not deepen the flow"
    sim_r, ana_r = h_on / h_off, nd_on / nd_off
    assert abs(sim_r / ana_r - 1.0) < 0.10, (
        f"ON/OFF depth ratio {sim_r:.3f} != analytic {ana_r:.3f}")


if __name__ == "__main__":
    test_slope_correction_shifts_tan_to_sin()
    print("test_slope_correction: PASS")
