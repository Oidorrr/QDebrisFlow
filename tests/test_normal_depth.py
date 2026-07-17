"""
Analytic check (loose): steady uniform sheet flow on a tilted plane should
approach the O'Brien normal depth h_n, i.e. the depth at which the friction
slope Sf equals the bed slope S for the imposed unit discharge q.

This is a transient/2-D simulation, so we use a generous tolerance and treat
the printed ratio as the primary signal — the goal is to catch gross errors,
not to certify exact convergence.
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401
import synthetic

G = 9.81


def normal_depth(q, S, gamma, tau_y, eta, n, K=24.0):
    """Solve Sf(h)=S for h by bisection (Sf is monotonically decreasing in h)."""
    def Sf(h):
        return (tau_y / (gamma * h)
                + K * eta * q / (8.0 * gamma * h ** 3)
                + n * n * q * q / h ** (10.0 / 3.0))
    lo, hi = 1e-4, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Sf(mid) > S:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_normal_depth_within_factor():
    ny, nx = 140, 10
    dx = dy = 5.0
    slope = 0.10
    dem, _, _ = synthetic.tilted_plane(ny=ny, nx=nx, slope=slope, dx=dx, dy=dy)

    params_kw = dict(Cv=0.45, tau_yield_direct=300.0, eta_direct=20.0,
                     manning_n=0.05, K_visc=24.0)
    from core.rheology import OBrienParameters
    p = OBrienParameters(**params_kw)

    # Feed the entire top row so the flow is ~1-D downslope.
    width_m = nx * dx
    Q_total = 30.0                      # m^3/s across the top
    q_unit = Q_total / width_m          # m^2/s per unit width
    src = synthetic.point_inflow(
        rows=[1] * nx, cols=list(range(nx)),
        times=[0.0, 60.0, 4000.0], q_total=[0.0, Q_total, Q_total],
        name="toprow",
    )
    res = synthetic.run_scenario(
        dem, dx, dy, sources=[src], params_kw=params_kw,
        config_kw=dict(t_end=1500.0, dt_max=8.0, cfl_number=0.5,
                       output_interval=200.0),
    )

    # Measure the established uniform reach just below the inlet. Further down
    # the slope the flow accelerates into the artificial 30 m/s velocity cap and
    # then pools against the closed bottom wall (backwater) — neither is the
    # normal-depth regime, so we sample rows ~8-22 % down the domain.
    band = res.h_final[int(0.08 * ny):int(0.22 * ny), :]
    h_sim = float(np.median(band[band > p.h_min])) if np.any(band > p.h_min) else 0.0
    h_n = normal_depth(q_unit, slope, p.gamma_mixture, p.tau_yield, p.eta,
                       p.manning_n, p.K_visc)
    ratio = h_sim / h_n if h_n > 0 else float("nan")
    print(f"  q={q_unit:.3f} m^2/s  h_sim={h_sim:.3f} m  h_n={h_n:.3f} m  "
          f"ratio={ratio:.2f}")
    assert 0.75 < ratio < 1.35, (
        f"simulated depth {h_sim:.3f} m far from normal depth {h_n:.3f} m "
        f"(ratio {ratio:.2f})")


if __name__ == "__main__":
    test_normal_depth_within_factor()
    print("test_normal_depth: PASS")
