"""
Physical sanity of the yield-stress stopping criterion and of gravity-driven
motion.

1. Quiescence: a uniform sheet thinner than the stop depth h_stop = τy/(γm·S)
   on a tilted plane must NOT move (the yield term holds it in place).
2. Motion: a deep blob (h > h_stop) on the same slope must travel downhill.
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401
import synthetic


def test_thin_sheet_does_not_move():
    slope = 0.05
    dem, dx, dy = synthetic.tilted_plane(ny=40, nx=20, slope=slope)
    params_kw = dict(Cv=0.45, tau_yield_direct=2000.0, eta_direct=20.0)

    from core.rheology import OBrienParameters
    p = OBrienParameters(**params_kw)
    h_stop = p.h_stop(slope)
    h_uniform = 0.4 * h_stop          # comfortably below threshold
    print(f"  h_stop={h_stop:.3f} m, uniform h={h_uniform:.3f} m")

    h0 = np.full(dem.shape, h_uniform, dtype=np.float64)
    res = synthetic.run_scenario(
        dem, dx, dy, h0=h0, params_kw=params_kw,
        config_kw=dict(t_end=120.0, dt_max=10.0, cfl_number=0.5),
    )
    moved = float(np.max(np.abs(res.h_final - h0)))
    print(f"  max|dh| = {moved:.3e} m, V_max = {res.V_max.max():.3e} m/s")
    assert moved < 1e-9, f"sheet below yield threshold moved by {moved:.3e} m"
    assert res.V_max.max() < 1e-9


def test_deep_blob_flows_downhill():
    ny, nx = 60, 21
    slope = 0.12
    dem, dx, dy = synthetic.tilted_plane(ny=ny, nx=nx, slope=slope)
    radius = 3
    ci = ny // 4                       # start in the upper quarter
    h0 = np.zeros(dem.shape)
    h0[ci - radius:ci + radius + 1, nx // 2 - radius:nx // 2 + radius + 1] = 4.0
    blob_bottom = ci + radius

    res = synthetic.run_scenario(
        dem, dx, dy, h0=h0,
        params_kw=dict(Cv=0.45, tau_yield_direct=150.0, eta_direct=10.0),
        config_kw=dict(t_end=300.0, dt_max=10.0, cfl_number=0.5),
    )
    wet_rows = np.where(res.h_max.max(axis=1) > 0.01)[0]
    reach = int(wet_rows.max())
    print(f"  blob bottom row={blob_bottom}, flow reached row={reach}")
    assert reach > blob_bottom + 5, "deep blob failed to travel downhill"


if __name__ == "__main__":
    test_thin_sheet_does_not_move()
    test_deep_blob_flows_downhill()
    print("test_motion: PASS")
