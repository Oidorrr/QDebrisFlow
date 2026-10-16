# QDebrisFlow

![QGIS](https://img.shields.io/badge/QGIS-3.16%2B-589632)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-1.0-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)

*Read this in other languages: [Русский](README.ru.md).*

QGIS plugin for modelling **post-fire debris flows** using the **O'Brien & Julien
quadratic rheology** combined with the **local inertial approximation (LIA)** of
the shallow water equations.

> The plugin is marked **experimental** in its QGIS metadata.

---

## Overview

QDebrisFlow models post-fire debris flows based on the quadratic rheological
model of O'Brien and Julien (1985, 1988) and the local inertial approximation of
the shallow water equations (LIA). It is based on the FLO-2D formulation.

It ships in two parallel copies that are functionally identical and differ only
in the language of their user interface:

- **`QDebrisFlowEN/`** — English UI
- **`QDebrisFlowRU/`** — Russian UI

---

## Method

The friction slope follows the O'Brien & Julien quadratic form (FLO-2D):

```
Sf = τy/(γm·h)  +  K·η·V/(8·γm·h²)  +  n²·V²/h^(4/3)
```

- **Yield + viscous + turbulent terms** combined in a single friction slope.
- **Numerical scheme**: Local Inertial Approximation (Bates et al., 2010) on a
  staggered grid. Manning and viscous terms are semi-implicit and use the full
  discharge vector (coupled friction, Cozzolino et al. 2021, eqs. 19–21).
- **Yield stress**: the impulse g·Δt·τy/γm acts against the trial discharge vector
  before the implicit division, so motion starts exactly at h·|∇η| > τy/γm (= h_stop)
  for any flow direction, time step, viscosity and K.
- **Options** (Solver Settings): θ-weighting of the discharge (de Almeida et al.
  2012; default θ = 1 = original scheme, 0.8–0.9 damps oscillations in steep
  low-friction channels) and a face-depth limiter hf ≤ ½(hL+hR) (default ON as in
  earlier versions). The limiter is the plugin's own modification: no literature
  source was found (the LIA papers use hf = max(η) − max(z) without it). It slows
  wetting fronts: on the test fan the arrival time is 8–13 % later at K = 2285 and
  15–18 % later at K = 24.
- **Inflow**: the exact volume of the piecewise-linear hydrograph is injected over
  every time step; the time step also respects CFL for the depth the inflow adds.
- **Compute backend**: optional **Numba JIT**; the solver falls back to pure NumPy
  when Numba is unavailable (results agree to round-off in stable flows; in
  oscillating steep channels the difference grows). The plugin GUI requires
  Numba to launch a run.

---

## Features

The dialog is organised into five tabs:

1. **Layers and sources** — select a DEM raster and a line source layer; assign an
   independent hydrograph `Q(t)` to each line feature (constant discharge, a
   time/Q table, or a CSV file); load all parameters from a JSON file.
2. **O'Brien Parameters** — sediment concentration `Cv`, component densities
   (`ρs`, `ρw`), yield stress `τy` and mixture viscosity `η` (direct input *or*
   exponential coefficients), Manning `n`, and the viscous shape factor `K`.
3. **Solver Settings** — simulation duration, maximum time step `dt_max`, the CFL
   number (0.10–0.70), θ-weighting and the face-depth limiter.
4. **Output** — file prefix, output folder, output-layer selection, depth-threshold
   filtering of the written rasters, the depth that defines the arrival time, and
   saving all parameters as JSON. Every run also writes `<prefix>_run.json` (all
   parameters, library versions, volume balance and the counters `edge_h_max_m` and
   `vcap_cells`; the warning texts themselves are only in the log).
5. **LHS analysis** — calibrate/validate against an observed inundation polygon and
   field depth-measurement points using Latin Hypercube Sampling, quality metrics,
   and a quintile sensitivity analysis; supports parallel worker threads.

The **Stop** button next to *Run simulation* at the bottom of the window interrupts
a running simulation.

---

## Outputs

| Layer        | Description                          |
|--------------|--------------------------------------|
| `h_max`      | Maximum depth envelope               |
| `V_max`      | Maximum velocity                     |
| `h_final`    | Final deposit                        |
| `t_arrival`  | Flow arrival time map (first time h ≥ the chosen depth, default 5 mm) |
| `HV2_max`    | Maximum over time of the instantaneous h·v² [m³/s²] (impact assessment; `h_max·V_max²` overestimates it) |
| `Fr_hv2`     | Froude number at the moment of peak h·v² (where the LIA is used outside its range) |

Rasters are written as GeoTIFF and loaded into QGIS with a pseudocolour style.

---

## Repository structure

```
QDebrisFlow/
├── LICENSE
├── README.md
├── README.ru.md
├── CITATION.cff
├── requirements.txt
├── setup.py
├── .gitignore
├── QDebrisFlowEN/              # English UI version
│   ├── __init__.py
│   ├── metadata.txt
│   ├── plugin.py
│   ├── dialog.py
│   ├── qgis_runner.py         # raster read/write + rasterisation (GDAL/OGR)
│   ├── dem_preprocess.py      # standalone utilities (sink filling, flow accumulation,
│   │                          # DEM checks); not used by the GUI
│   ├── icon.png
│   └── core/
│       ├── __init__.py
│       ├── rheology.py        # O'Brien & Julien quadratic rheology
│       ├── solver.py          # 2-D LIA solver (Numba JIT / NumPy)
│       └── sweep_runner.py    # single LHS-run executor + metrics
└── QDebrisFlowRU/             # Russian UI version (same structure)
```

---

## Requirements

- **QGIS ≥ 3.16** (provides PyQt5, the QGIS Python API, and GDAL/OGR)
- **NumPy** (bundled with QGIS)
- **Numba** — required to run simulations from the plugin GUI; the solver enables
  it only with NumPy ≥ 1.22 (for Numba ≥ 0.56), otherwise it falls back to NumPy and
  the GUI refuses to run with a message naming both versions
- Tested with QGIS 3.44.8 LTR (Python 3.12, NumPy 2.4, Numba 0.65.1, GDAL 3.12)

See [`requirements.txt`](requirements.txt).

---

## Installation

1. Copy one of the plugin folders (`QDebrisFlowEN` or `QDebrisFlowRU`) into your
   QGIS plugins directory:
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Install **Numba** into the QGIS Python environment. On Windows, in the
   *OSGeo4W Shell*:
   ```
   pip install numba
   ```
   Restart QGIS afterwards.
3. Enable the plugin in **Plugins → Manage and Install Plugins**.

---

## Usage

1. **Layers and sources** — choose the DEM and the line source layer, click
   *Load features*, and define a hydrograph for each source line.
2. **O'Brien Parameters** — set the rheology (`Cv`, densities, `τy`, `η`, `n`, `K`).
3. **Solver Settings** — set the duration, `dt_max`, and CFL number.
4. **Output** — choose the prefix, folder, and which output layers to produce.
5. Click **Run simulation**. Results are written as GeoTIFF and added to the map.

Parameter sets can be saved to / loaded from JSON on the *Output* and
*Layers and sources* tabs.

---

## LHS calibration & sensitivity analysis

The **LHS analysis** tab compares simulated against observed data:

- Inputs: an observed **inundation polygon**, **field depth points** and an optional
  **evaluation area** (cells outside it are ignored in TP/FP/FN). The evaluation
  area is the polygon of the mapped territory, not the inundation polygon: choosing
  the inundation polygon itself there makes FP ≡ 0. Vector layers in another CRS
  are reprojected to the DEM CRS.
- Quality metrics: `TP`, `FP`, `FN`, `ΩTm` (≡ 1 − CSI), depth over-/under-estimation
  (`Δo`, `Δu`), and the combined metric `Cm` (0 = best).
- Sampling: volume, `η` and `K` are sampled log-uniformly, `τy` and `n` uniformly.
- Sensitivity: quintiles of `Cm` and Spearman ρ with a 95 % bootstrap interval for V,
  `τy`, `η`, `n`, `K`, and a second table with volume control (Cm residuals after a
  regression on log₁₀V — the plugin's own procedure).
- Results are appended to the CSV after every run (a `_context.json` file keeps the
  settings) and the best run's `h_max` raster is added to QGIS.

---

## Limitations

- **Closed boundaries.** The edge of the DEM and nodata cells are walls: the flow
  cannot leave the domain and accumulates at the edge. A warning is shown when the
  flow reaches the boundary; extend the DEM beyond the expected runout.
- **Local inertial approximation.** Convective acceleration is neglected; the LIA
  agrees well with the full shallow-water equations only for Fr < 0.5 (de Almeida &
  Bates 2013). Steep channels are often supercritical — check the `Fr_hv2` layer.
- **Oscillations in steep channels.** With low friction (small K) the scheme can
  pulsate; a symptom is velocity stuck at the 30 m/s cap (reported after the run).
  θ-weighting 0.8–0.9 damps it.
- **Rheological parameters.** K = 24 is the smooth wide-channel value; FLO-2D uses
  up to ~50 000 on rough channels and 2285 for most urban studies. In the
  exponential mode the plugin uses τy = α₁·e^(β₁Cv), η = α₂·e^(β₂Cv) — the opposite
  of the FLO-2D indices; the default α/β have no established source.
- **First-order scheme on a regular grid**: results depend on the cell size;
  no entrainment/erosion, a single-phase mixture.

---

## References

The following works are referenced in the source code:

- O'Brien & Julien (1985, 1988) — quadratic rheological model
- Bates et al. (2010) — local inertial approximation (LIA)
- de Almeida et al. (2012) — θ-weighting of the LIA discharge
- de Almeida & Bates (2013) — applicability of the LIA (Fr < 0.5)
- Cozzolino et al. (2021) — coupled friction on staggered grids
- Jakob, Stein & Ulmi (2012) — debris-flow intensity index d·v²
- Barnhart et al. (2021) — calibration metrics, LHS, sensitivity analysis (FLO-2D)
- Heiser, Scheidl & Kaitna (2017) — `ΩT` fit metric
- Planchon & Darboux (2001) — depression filling
- Morris (1991) — parameter screening

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
Copyright © 2026 Aidar Khaybulin.

## Citation

If you use this software, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## Acknowledgments

The code generation, testing, and documentation were supported by the language models Claude (Anthropic)
