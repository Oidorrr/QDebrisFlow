"""
Validate the headless A/B orchestration (montecito_ab.run_ab) WITHOUT GDAL, on a
synthetic tilted-plane DEM with a synthetic source, observed-inundation mask and
depth points. This exercises the LHS sampling, per-config flag wiring and the
core.sweep_runner integration end-to-end; only the GDAL data loaders (read_dem /
rasterize_*) are excluded (they run under the OSGeo4W Python with the real DEM).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import _pathsetup  # noqa: F401  (adds QDebrisFlowEN to path; optional numba block)
import numpy as np

import synthetic
import montecito_ab as ab


def _synthetic_case():
    dem, dx, dy = synthetic.tilted_plane(ny=30, nx=20, slope=0.08)
    geo = dict(nx=20, ny=30, dx=dx, dy=dy, xmin=0.0, ymax=300.0,
               nodata=-9999.0, geotransform=(0.0, dx, 0.0, 300.0, 0.0, -dy))
    src_cells = [dict(rows=np.array([2, 2, 2], np.int64),
                      cols=np.array([9, 10, 11], np.int64),
                      times=np.array([0., 60., 300., 600.]),
                      base_q_vals=np.array([0., 40., 40., 0.]), name="s")]
    bv = ab.base_volume(src_cells)
    obs_mask = np.zeros((30, 20), bool)
    obs_mask[5:20, 7:13] = True
    obs_rows = np.array([10, 12, 14], np.int64)
    obs_cols = np.array([10, 10, 10], np.int64)
    obs_depths = np.array([1.0, 0.8, 0.5])
    return dem, geo, src_cells, bv, obs_mask, obs_rows, obs_cols, obs_depths


def test_run_ab_smoke():
    (dem, geo, src_cells, bv, obs_mask,
     obs_rows, obs_cols, obs_depths) = _synthetic_case()
    pr = {"log_vol":   (np.log10(bv * 0.5), np.log10(bv * 2.0)),
          "tau_yield": (200., 800.), "eta": (5., 30.),
          "manning_n": (0.04, 0.10), "K_visc": (24., 24.)}
    new_params = dict(entrain_coef=0.3, bed_friction_angle_deg=37.0, Cv_bed=0.0,
                      pore_pressure_lambda0=0.7, pore_consolidation_time=900.0)
    configs = ["baseline", "entrain", "two_phase"]
    res = ab.run_ab(
        dem, geo, src_cells, bv, obs_mask, obs_rows, obs_cols, obs_depths,
        pr, n_runs=4, seed=1, t_end=600.0, dt_max=10.0,
        new_params=new_params, configs=configs, workers=1,
        plugin_dir=ab.EN, progress=print)

    for name in configs:
        assert name in res and res[name] is not None, f"{name}: no successful run"
        cm = res[name]["Cm"]
        assert cm is not None and 0.0 <= cm <= 1.0, f"{name}: bad Cm {cm}"
    print("  Cm by config:", {k: round(v["Cm"], 4) for k, v in res.items()})


if __name__ == "__main__":
    test_run_ab_smoke()
    print("test_ab_harness: PASS")
