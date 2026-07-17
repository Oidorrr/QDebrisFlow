"""
Synthetic terrains and a thin scenario runner for the QDebrisFlow core solver.

Everything here imports `core.*` lazily (inside functions) so that the
QDF_NO_NUMBA switch in _pathsetup can take effect before the solver module is
first imported.
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401  (side effect: sys.path + optional numba block)


# ── Terrains ──────────────────────────────────────────────────────────────────

def flat(ny: int = 40, nx: int = 40, z: float = 100.0):
    """Perfectly flat terrain. Returns (dem, dx, dy)."""
    return np.full((ny, nx), float(z)), 10.0, 10.0


def tilted_plane(ny: int = 60, nx: int = 30, slope: float = 0.10,
                 dx: float = 10.0, dy: float = 10.0, base: float = 200.0):
    """
    Plane tilted along +i (row index increases downhill).
    `slope` = tan(theta) = drop per metre. Returns (dem, dx, dy).
    """
    drop_per_row = slope * dy
    rows = np.arange(ny, dtype=np.float64)[:, None]
    dem = base - drop_per_row * rows * np.ones((1, nx))
    return dem, dx, dy


def central_blob(ny: int, nx: int, depth: float = 2.0, radius: int = 3):
    """Initial-depth array h0 with a square blob of `depth` near the centre."""
    h0 = np.zeros((ny, nx), dtype=np.float64)
    ci, cj = ny // 2, nx // 2
    h0[ci - radius:ci + radius + 1, cj - radius:cj + radius + 1] = depth
    return h0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_scenario(dem, dx, dy, *, h0=None, sources=None,
                 params_kw=None, config_kw=None, nodata=-9999.0):
    """
    Build TerrainGrid/OBrienParameters/SourceCondition/SimulationConfig and run.
    Returns the SimulationResult.
    """
    from core.rheology import OBrienParameters
    from core.solver import (TerrainGrid, SimulationConfig,
                             SourceCondition, DebrisFlowSolver2D)

    terrain = TerrainGrid(dem=np.ascontiguousarray(dem, dtype=np.float64),
                          dx=float(dx), dy=float(dy), nodata=float(nodata))
    params = OBrienParameters(**(params_kw or {}))
    source = SourceCondition(h0_array=h0, inflow_sources=list(sources or []))
    config = SimulationConfig(**(config_kw or {}))
    solver = DebrisFlowSolver2D(terrain, params, source, config)
    return solver.run()


def point_inflow(rows, cols, times, q_total, name="test"):
    """Convenience InflowSource builder."""
    from core.solver import InflowSource
    return InflowSource(
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        times=np.asarray(times, dtype=np.float64),
        q_total=np.asarray(q_total, dtype=np.float64),
        name=name,
    )


def canonical_scenario():
    """
    A single non-trivial scenario reused by several tests: a tilted plane fed by
    a three-cell hydrograph near the top. Returns the SimulationResult.
    Deterministic — used to compare the Numba and NumPy backends.
    """
    dem, dx, dy = tilted_plane(ny=50, nx=30, slope=0.08)
    src = point_inflow(
        rows=[2, 2, 2], cols=[13, 14, 15],
        times=[0.0, 60.0, 300.0, 600.0],
        q_total=[0.0, 40.0, 40.0, 0.0],
        name="canon",
    )
    return run_scenario(
        dem, dx, dy,
        sources=[src],
        params_kw=dict(Cv=0.45, tau_yield_direct=800.0, eta_direct=30.0,
                       manning_n=0.06),
        config_kw=dict(t_end=600.0, dt_max=10.0, cfl_number=0.5,
                       output_interval=120.0),
    )


# h_max peak of canonical_scenario with the OFF (planar) baseline. Pinned so that
# later phases cannot silently change the validated default-OFF behaviour.
CANONICAL_HMAX = 2.7729


def canonical_scenario_slope():
    """
    Same hydrograph as canonical_scenario but on a steeper plane with the
    Phase-1 steep-slope correction ON — exercises the cos(theta) code path so
    the backend-equivalence test covers it too.
    """
    dem, dx, dy = tilted_plane(ny=50, nx=30, slope=0.20)
    src = point_inflow(
        rows=[2, 2, 2], cols=[13, 14, 15],
        times=[0.0, 60.0, 300.0, 600.0],
        q_total=[0.0, 40.0, 40.0, 0.0],
        name="canon_slope",
    )
    return run_scenario(
        dem, dx, dy,
        sources=[src],
        params_kw=dict(Cv=0.45, tau_yield_direct=800.0, eta_direct=30.0,
                       manning_n=0.06),
        config_kw=dict(t_end=600.0, dt_max=10.0, cfl_number=0.5,
                       output_interval=120.0, slope_correction=True),
    )


def canonical_scenario_entrain():
    """
    Tilted plane fed by a hydrograph with bed entrainment ON and exponential-mode
    rheology (so the evolving Cv drives τy, η). Exercises the Phase-2 per-cell
    field + erosion substep for the backend-equivalence check.
    """
    dem, dx, dy = tilted_plane(ny=50, nx=30, slope=0.12)
    src = point_inflow(
        rows=[2, 2, 2], cols=[13, 14, 15],
        times=[0.0, 60.0, 300.0, 600.0],
        q_total=[0.0, 40.0, 40.0, 0.0],
        name="canon_entrain",
    )
    return run_scenario(
        dem, dx, dy,
        sources=[src],
        params_kw=dict(Cv=0.40, manning_n=0.06, entrain_coef=0.2,
                       bed_friction_angle_deg=37.0),
        config_kw=dict(t_end=600.0, dt_max=10.0, cfl_number=0.5,
                       output_interval=120.0, entrainment=True),
    )


def canonical_scenario_8conn():
    """Tilted plane fed by a hydrograph with 8-connectivity ON — exercises the
    Phase-3 diagonal flux path for backend equivalence."""
    dem, dx, dy = tilted_plane(ny=50, nx=30, slope=0.12)
    src = point_inflow(
        rows=[2, 2, 2], cols=[13, 14, 15],
        times=[0.0, 60.0, 300.0, 600.0],
        q_total=[0.0, 40.0, 40.0, 0.0],
        name="canon_8conn",
    )
    return run_scenario(
        dem, dx, dy,
        sources=[src],
        params_kw=dict(Cv=0.45, tau_yield_direct=800.0, eta_direct=30.0,
                       manning_n=0.06),
        config_kw=dict(t_end=600.0, dt_max=10.0, cfl_number=0.5,
                       output_interval=120.0, eight_connectivity=True),
    )


def canonical_scenario_2phase():
    """Tilted plane fed by a hydrograph with the two-phase pore-pressure model ON
    (λ0 > 0) — exercises the Phase-4 λ-modulated yield + advection for backend
    equivalence."""
    dem, dx, dy = tilted_plane(ny=50, nx=30, slope=0.12)
    src = point_inflow(
        rows=[2, 2, 2], cols=[13, 14, 15],
        times=[0.0, 60.0, 300.0, 600.0],
        q_total=[0.0, 40.0, 40.0, 0.0],
        name="canon_2phase",
    )
    return run_scenario(
        dem, dx, dy,
        sources=[src],
        params_kw=dict(Cv=0.45, tau_yield_direct=800.0, eta_direct=30.0,
                       manning_n=0.06, pore_pressure_lambda0=0.6,
                       pore_consolidation_time=1200.0),
        config_kw=dict(t_end=600.0, dt_max=10.0, cfl_number=0.5,
                       output_interval=120.0, two_phase=True),
    )
