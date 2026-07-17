#!/usr/bin/env python
"""
Headless A/B benchmark on the 2018 Montecito debris flow.

Runs the LHS calibration sweep (varying vol / tau_yield / eta / manning_n / K_visc,
exactly like the plugin's "LHS analysis" tab) once per physics-flag configuration,
and reports the best combined metric Cm (0 = best, Barnhart 2021) for each. This
shows which of the Phase 1-4 physics additions actually lowers Cm versus the
4-connectivity, single-phase baseline.

The data loaders need GDAL (osgeo), which ships with QGIS but not with a bare
Python. Run this with the OSGeo4W / QGIS Python, e.g. on Windows:

    & "C:\\OSGeo4W\\bin\\python-qgis.bat" montecito_ab.py ^
        --dem "C:\\path\\to\\5m_Montecito_32611.tif"

The DEM is NOT in the repo (it is the QGIS layer "5m_Montecito_32611",
EPSG:32611). Pass its path with --dem. Source lines, observed polygon/points and
hydrographs are read from ./data and ./data/scenarios.

The A/B orchestration (run_ab) is pure NumPy + core.sweep_runner and is unit
-tested without GDAL in tests/test_ab_harness.py on a synthetic DEM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
EN   = os.path.join(REPO, "QDebrisFlowEN")
DATA = os.path.join(REPO, "data")
if EN not in sys.path:
    sys.path.insert(0, EN)

from core import sweep_runner as sr   # headless-safe (no PyQt5/QGIS)


# ── physics-flag configurations to compare ───────────────────────────────────
FLAG_CONFIGS = {
    "baseline":  {},
    "slope":     {"slope_correction": True},
    "entrain":   {"entrainment": True},
    "8conn":     {"eight_connectivity": True},
    "two_phase": {"two_phase": True},
    "all":       {"slope_correction": True, "entrainment": True,
                  "eight_connectivity": True, "two_phase": True},
}


# ── GDAL/OGR data loaders (imported lazily so run_ab works without GDAL) ──────

def read_dem(path: str):
    from osgeo import gdal
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"GDAL cannot open DEM: {path}")
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem = band.ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    geo = dict(nx=ds.RasterXSize, ny=ds.RasterYSize,
               dx=abs(gt[1]), dy=abs(gt[5]), xmin=gt[0], ymax=gt[3],
               nodata=float(nodata) if nodata is not None else -9999.0,
               geotransform=tuple(float(x) for x in gt))
    ds = None
    return dem, geo


def _rasterize(layer, geo, burn=1, all_touched=True):
    from osgeo import gdal
    ras = gdal.GetDriverByName("MEM").Create(
        "", geo["nx"], geo["ny"], 1, gdal.GDT_Byte)
    ras.SetGeoTransform(geo["geotransform"])
    ras.GetRasterBand(1).Fill(0)
    opts = ["ALL_TOUCHED=TRUE"] if all_touched else []
    gdal.RasterizeLayer(ras, [1], layer, burn_values=[burn], options=opts)
    mask = ras.GetRasterBand(1).ReadAsArray().astype(bool)
    ras = None
    return mask


def rasterize_lines(shp_path, geo):
    """One (fid, rows, cols) per line feature."""
    from osgeo import ogr
    vec = ogr.Open(shp_path)
    lyr = vec.GetLayer()
    out = []
    for feat in lyr:
        fid = int(feat.GetFID())
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        mem = ogr.GetDriverByName("Memory").CreateDataSource("t")
        ml = mem.CreateLayer("s", srs=lyr.GetSpatialRef(),
                             geom_type=ogr.wkbLineString)
        f2 = ogr.Feature(ml.GetLayerDefn())
        f2.SetGeometry(geom.Clone())
        ml.CreateFeature(f2)
        mask = _rasterize(ml, geo)
        rr, cc = np.where(mask)
        out.append((fid, rr.astype(np.int64), cc.astype(np.int64)))
        mem = None
    vec = None
    return out


def rasterize_polygon(shp_path, geo):
    from osgeo import ogr
    vec = ogr.Open(shp_path)
    mask = _rasterize(vec.GetLayer(), geo, all_touched=True)
    vec = None
    return mask


def extract_points(shp_path, field, geo):
    from osgeo import ogr
    gt = geo["geotransform"]
    xmin, ymax, dx, dy = gt[0], gt[3], geo["dx"], geo["dy"]
    nx, ny = geo["nx"], geo["ny"]
    vec = ogr.Open(shp_path)
    lyr = vec.GetLayer()
    rows, cols, depths = [], [], []
    for feat in lyr:
        g = feat.GetGeometryRef()
        if g is None:
            continue
        col = int((g.GetX() - xmin) / dx)
        row = int((ymax - g.GetY()) / dy)
        if 0 <= row < ny and 0 <= col < nx:
            try:
                d = float(feat.GetField(field))
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row); cols.append(col); depths.append(d)
    vec = None
    return (np.array(rows, np.int64), np.array(cols, np.int64),
            np.array(depths, np.float64))


# ── source cells + base volume + LHS (mirror dialog.py exactly) ──────────────

def build_src_cells(line_feats, params_json):
    hydro = params_json.get("hydrographs", {})
    names = {int(r["fid"]): r["name"] for r in params_json.get("source_table", [])}
    src = []
    for fid, rows, cols in line_feats:
        h = hydro.get(str(fid))
        if h is None or len(rows) == 0:
            continue
        src.append(dict(rows=rows, cols=cols,
                        times=np.asarray(h["times"], np.float64),
                        base_q_vals=np.asarray(h["q_vals"], np.float64),
                        name=names.get(fid, str(fid))))
    return src


def base_volume(src_cells):
    trapz = getattr(np, "trapezoid", None) or np.trapz
    bv = 0.0
    for sc in src_cells:
        t, q = sc["times"], sc["base_q_vals"]
        finite = t < 1e8
        if int(finite.sum()) >= 2:
            bv += float(trapz(q[finite], t[finite]))
    return bv


def build_param_ranges(s: dict) -> dict:
    def rng(vary, mn, mx, fx):
        return (s[mn], s[mx]) if s.get(vary) else (s[fx], s[fx])
    if s.get("vol_vary"):
        lo = math.log10(max(s["vol_min"], 1.0))
        hi = math.log10(max(s["vol_max"], 2.0))
    else:
        fv = max(s.get("vol_fixed", 1e5), 1.0)
        lo = hi = math.log10(fv)
    return {
        "log_vol":   (lo, hi),
        "tau_yield": rng("tau_vary", "tau_min", "tau_max", "tau_fixed"),
        "eta":       rng("eta_vary", "eta_min", "eta_max", "eta_fixed"),
        "manning_n": rng("n_vary",   "n_min",   "n_max",   "n_fixed"),
        "K_visc":    rng("K_vary",   "K_min",   "K_max",   "K_fixed"),
    }


def lhs_samples(param_ranges, n, seed=42):
    rng = np.random.default_rng(seed)
    keys = list(param_ranges.keys())
    unit = np.zeros((n, len(keys)))
    for j in range(len(keys)):
        unit[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    out = []
    for i in range(n):
        pt = {}
        for j, key in enumerate(keys):
            lo, hi = param_ranges[key]
            v = lo + unit[i, j] * (hi - lo)
            if key.startswith("log_"):
                pt[key[4:]] = 10.0 ** v
            else:
                pt[key] = v
        out.append(pt)
    return out


# ── A/B core (pure NumPy + sweep_runner; unit-tested without GDAL) ────────────

def run_jobs(jobs, workers=1):
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            return pool.map(sr.run_one_sample, jobs)
    return [sr.run_one_sample(j) for j in jobs]


def run_ab(dem, geo, src_cells, base_vol, obs_mask, obs_rows, obs_cols,
           obs_depths, param_ranges, *, n_runs=500, seed=42, t_end=7200.0,
           dt_max=10.0, cfl_number=0.5, depth_threshold=0.5, Cv=0.45,
           Cv_max=0.615, rho_s=2650.0, rho_w=1000.0, new_params=None,
           configs=None, workers=1, plugin_dir=EN, progress=None):
    """Run the LHS sweep for each flag config; return {name: best_row}."""
    new_params = new_params or {}
    samples = lhs_samples(param_ranges, n_runs, seed)
    geo_ser = dict(nx=int(geo["nx"]), ny=int(geo["ny"]),
                   dx=float(geo["dx"]), dy=float(geo["dy"]),
                   nodata=float(geo.get("nodata", -9999.0)),
                   geotransform=tuple(float(x) for x in geo["geotransform"]))
    results = {}
    for name in (configs or list(FLAG_CONFIGS.keys())):
        job_base = dict(
            plugin_dir=plugin_dir, dem=dem, geo_info=geo_ser,
            src_cells=src_cells, base_vol=base_vol, obs_mask=obs_mask,
            obs_rows=obs_rows, obs_cols=obs_cols, obs_depths=obs_depths,
            Cv=Cv, Cv_max=Cv_max, rho_s=rho_s, rho_w=rho_w,
            param_ranges=param_ranges, t_end=t_end, dt_max=dt_max,
            cfl_number=cfl_number, depth_threshold=depth_threshold,
            **new_params, **FLAG_CONFIGS[name])
        jobs = [dict(job_base, run_id=i, sample=s) for i, s in enumerate(samples)]
        rows = run_jobs(jobs, workers)
        ok = [r for r in rows if r.get("Cm") is not None]
        best = min(ok, key=lambda r: r["Cm"], default=None)
        results[name] = best
        if progress:
            cm = best["Cm"] if best else float("nan")
            progress(f"  {name:10s}  best Cm = {cm:.4f}  ({len(ok)}/{len(rows)} ok)")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def _catchment_paths(catchment):
    c = catchment.lower()
    ds = os.path.join(DATA, "Dataset_9January2018_Debris_Flow_Santa_Barbara")
    tag = {"montecito": "Montecito", "romero": "Romero",
           "san_ysidro": "San_Ysidro"}[c]
    line = {"montecito": "Line_montecito", "romero": "Line_romero_creek",
            "san_ysidro": "Line_san_ysidro"}[c]
    return dict(
        params=os.path.join(DATA, "scenarios", tag, f"{tag}_params.json"),
        lhs=os.path.join(DATA, "scenarios", "LHS", "lhs_settings.json"),
        source=os.path.join(DATA, "start_lines", f"{line}.shp"),
        obs_poly=os.path.join(ds, f"Inindation_poly_{tag}.shp"),
        obs_pts=os.path.join(ds, f"Inindation_points_{tag}.shp"),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Montecito A/B physics benchmark")
    ap.add_argument("--dem", required=True, help="path to the DEM GeoTIFF")
    ap.add_argument("--catchment", default="montecito",
                    choices=["montecito", "romero", "san_ysidro"])
    ap.add_argument("--n-runs", type=int, default=None, help="override LHS n_runs")
    ap.add_argument("--t-end", type=float, default=7200.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of: " + ", ".join(FLAG_CONFIGS))
    ap.add_argument("--out", default="montecito_ab_results.csv")
    # fixed physics params used when a flag is ON
    ap.add_argument("--entrain-coef", type=float, default=0.3)
    ap.add_argument("--phi", type=float, default=37.0)
    ap.add_argument("--cv-bed", type=float, default=0.0)
    ap.add_argument("--lambda0", type=float, default=0.7)
    ap.add_argument("--tc", type=float, default=900.0)
    a = ap.parse_args(argv)

    pth = _catchment_paths(a.catchment)
    with open(pth["params"], encoding="utf-8") as f:
        params_json = json.load(f)
    with open(pth["lhs"], encoding="utf-8") as f:
        lhs = json.load(f)
    n_runs = a.n_runs or int(lhs.get("n_runs", 500))

    print(f"Reading DEM {a.dem} ...")
    dem, geo = read_dem(a.dem)
    print(f"  DEM {geo['ny']}x{geo['nx']} @ {geo['dx']}x{geo['dy']} m")
    src_cells = build_src_cells(rasterize_lines(pth["source"], geo), params_json)
    bv = base_volume(src_cells)
    print(f"  {len(src_cells)} sources, base volume {bv:.0f} m^3")
    obs_mask = rasterize_polygon(pth["obs_poly"], geo)
    obs_rows, obs_cols, obs_depths = extract_points(
        pth["obs_pts"], lhs.get("depth_field", "Max_Flow_D"), geo)
    print(f"  obs polygon {int(obs_mask.sum())} cells, {len(obs_depths)} depth points")

    new_params = dict(entrain_coef=a.entrain_coef, bed_friction_angle_deg=a.phi,
                      Cv_bed=a.cv_bed, pore_pressure_lambda0=a.lambda0,
                      pore_consolidation_time=a.tc)
    print(f"Running A/B: {n_runs} LHS runs x {len(a.configs or FLAG_CONFIGS)} "
          f"configs (workers={a.workers})")
    results = run_ab(
        dem, geo, src_cells, bv, obs_mask, obs_rows, obs_cols, obs_depths,
        build_param_ranges(lhs), n_runs=n_runs, seed=int(lhs.get("seed", 42)),
        t_end=a.t_end, depth_threshold=lhs.get("depth_threshold", 0.5),
        new_params=new_params, configs=a.configs, workers=a.workers,
        progress=print)

    base_cm = results.get("baseline", {}).get("Cm") if results.get("baseline") else None
    print("\n=== A/B summary (Cm: 0 = best) ===")
    fields = ["config", "Cm", "dCm_vs_baseline", "omega_Tm", "vol_m3",
              "tau_yield_Pa", "eta_Pas", "manning_n", "K_visc"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, best in results.items():
            if not best:
                print(f"  {name:10s}  (no successful run)"); continue
            d = best["Cm"] - base_cm if base_cm is not None else float("nan")
            print(f"  {name:10s}  Cm={best['Cm']:.4f}  ΔCm={d:+.4f}  "
                  f"(vol={best['vol_m3']:.0f}, τy={best['tau_yield_Pa']}, "
                  f"η={best['eta_Pas']}, n={best['manning_n']}, K={best['K_visc']})")
            w.writerow(dict(config=name, Cm=best["Cm"], dCm_vs_baseline=d,
                            omega_Tm=best["omega_Tm"], vol_m3=best["vol_m3"],
                            tau_yield_Pa=best["tau_yield_Pa"], eta_Pas=best["eta_Pas"],
                            manning_n=best["manning_n"], K_visc=best["K_visc"]))
    print(f"\nWritten: {a.out}")


if __name__ == "__main__":
    main()
