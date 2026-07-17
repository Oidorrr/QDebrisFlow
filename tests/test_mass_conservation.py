"""
Mass conservation: the grid has no flux faces at its outer boundary (Qx has
shape (ny, nx-1), Qy has (ny-1, nx)), so an all-valid DEM is effectively a
closed (no-flux) domain. With no inflow, total volume must be conserved to
machine precision regardless of how the blob spreads or pools.
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401
import synthetic


def _loss_pct(dem, dx, dy, h0, params_kw, t_end=400.0):
    res = synthetic.run_scenario(
        dem, dx, dy, h0=h0, params_kw=params_kw,
        config_kw=dict(t_end=t_end, dt_max=10.0, cfl_number=0.5,
                       output_interval=50.0),
    )
    vb = res.volume_balance()
    return abs(vb["V_loss_pct"]), res


def test_flat_closed_domain_conserves():
    dem, dx, dy = synthetic.flat(ny=40, nx=40)
    h0 = synthetic.central_blob(40, 40, depth=2.0, radius=4)
    loss, _ = _loss_pct(dem, dx, dy, h0,
                        dict(Cv=0.45, tau_yield_direct=200.0, eta_direct=20.0))
    print(f"  flat:    |V_loss| = {loss:.2e} %")
    assert loss < 1e-6, f"flat domain leaked volume: {loss:.3e} %"


def test_tilted_closed_domain_conserves():
    dem, dx, dy = synthetic.tilted_plane(ny=60, nx=30, slope=0.10)
    h0 = synthetic.central_blob(60, 30, depth=3.0, radius=4)
    loss, _ = _loss_pct(dem, dx, dy, h0,
                        dict(Cv=0.45, tau_yield_direct=200.0, eta_direct=20.0))
    print(f"  tilted:  |V_loss| = {loss:.2e} %")
    assert loss < 1e-6, f"tilted domain leaked volume: {loss:.3e} %"


if __name__ == "__main__":
    test_flat_closed_domain_conserves()
    test_tilted_closed_domain_conserves()
    print("test_mass_conservation: PASS")
