"""
Phase 3 (B) — 8-connectivity (diagonal fluxes), mobility-preserving (normalized).

Key facts established for this LIA solver:
- The cardinal-only scheme is already isotropic to first order, so 8-connectivity
  yields a negligible accuracy change (this is NOT a cellular-automaton / D8 model).
- The normalization (cardinal:diagonal weights 0.8:0.2, summing to 1) keeps the
  total transport equal to the 4-connectivity baseline, so enabling diagonals does
  NOT inflate mobility.

So the asserted invariants are: mass conservation with diagonals on, and that the
normalized 8-connectivity reproduces the 4-connectivity spreading (mobility), which
is the whole point of the normalization. (Numba≡NumPy on the diagonal path is
covered by test_backends_equiv via the canon_8conn scenario.)
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401
import synthetic


def _dam_break(eight):
    ny = nx = 61
    dx = dy = 10.0
    dem, _, _ = synthetic.flat(ny, nx, z=100.0)
    h0 = np.zeros((ny, nx))
    h0[28:33, 28:33] = 5.0
    return synthetic.run_scenario(
        dem, dx, dy, h0=h0,
        params_kw=dict(Cv=0.30, tau_yield_direct=1.0, eta_direct=1.0, manning_n=0.03),
        config_kw=dict(t_end=120.0, dt_max=5.0, cfl_number=0.4,
                       eight_connectivity=eight),
    )


def _axis_extent(res):
    ny, nx = res.h_max.shape
    ci, cj = ny // 2, nx // 2
    ii, jj = np.where(res.h_max > 0.05)
    di, dj = ii - ci, jj - cj
    dist = np.sqrt((di * 10.0) ** 2 + (dj * 10.0) ** 2)
    axis = (di == 0) | (dj == 0)
    return float(dist[axis].max())


def test_eight_conn_conserves_mass():
    res = _dam_break(eight=True)
    loss = abs(res.volume_balance()["V_loss_pct"])
    print(f"  8-conn |V_loss| = {loss:.2e} %")
    assert loss < 1e-9, f"8-connectivity leaked volume: {loss:.3e} %"


def test_eight_conn_preserves_mobility():
    """Normalized diagonals must not change the spreading rate vs 4-connectivity."""
    a4 = _axis_extent(_dam_break(eight=False))
    a8 = _axis_extent(_dam_break(eight=True))
    ratio = a8 / a4
    print(f"  axis extent 4-conn={a4:.1f}  8-conn={a8:.1f}  ratio={ratio:.4f}")
    assert 0.95 < ratio < 1.05, (
        f"normalized 8-connectivity changed mobility by {100*(ratio-1):+.1f}% "
        f"(should be ~0 — diagonals are mobility-preserving)")


if __name__ == "__main__":
    test_eight_conn_conserves_mass()
    test_eight_conn_preserves_mobility()
    print("test_eight_conn: PASS")
