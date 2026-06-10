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
- **Numerical scheme**: Local Inertial Approximation (Bates et al., 2010) with the
  flux limiter of Almeida et al. (2012).
- **Compute backend**: optional **Numba JIT**; the solver falls back to pure NumPy
  when Numba is unavailable (the physics and results are identical). The plugin
  GUI requires Numba to launch a run.

---

## Features

The dialog is organised into five tabs:

1. **Layers and sources** — select a DEM raster and a line source layer; assign an
   independent hydrograph `Q(t)` to each line feature (constant discharge, a
   time/Q table, or a CSV file).
2. **O'Brien Parameters** — sediment concentration `Cv`, component densities
   (`ρs`, `ρw`), yield stress `τy` and mixture viscosity `η` (direct input *or*
   exponential coefficients), Manning `n`, and the viscous shape factor `K`.
3. **Solver Settings** — simulation duration, maximum time step `dt_max`, and the
   CFL number.
4. **Output** — file prefix, output folder, output-layer selection, depth-threshold
   filtering of the written rasters, and save/load of all parameters as JSON.
5. **LHS analysis** — calibrate/validate against an observed inundation polygon and
   field depth-measurement points using Latin Hypercube Sampling, quality metrics,
   and a quintile sensitivity analysis; supports parallel worker threads.

---

## Outputs

| Layer        | Description                          |
|--------------|--------------------------------------|
| `h_max`      | Maximum depth envelope               |
| `V_max`      | Maximum velocity                     |
| `h_final`    | Final deposit                        |
| `t_arrival`  | Flow arrival time map                |

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
│   ├── dem_preprocess.py      # sink filling, flow accumulation, DEM checks
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
- **Numba** — required to run simulations from the plugin GUI

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

- Inputs: an observed **inundation polygon** and **field depth points**.
- Quality metrics: `TP`, `FP`, `FN`, `ΩTm`, depth over-/under-estimation
  (`Δo`, `Δu`), and the combined metric `Cm` (0 = best).
- A **quintile sensitivity analysis** indicates which parameters (`τy`, `η`, `n`)
  influence the result.
- Results are written to CSV and the best run's `h_max` raster is added to QGIS.

---

## References

The following works are referenced in the source code:

- O'Brien & Julien (1985, 1988) — quadratic rheological model
- Bates et al. (2010) — local inertial approximation (LIA)
- Almeida et al. (2012) — flux limiter
- Barnhart et al. (2021) — calibration metrics, LHS, sensitivity analysis (FLO-2D)
- Heiser (2017) — `ΩT` fit metric
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
