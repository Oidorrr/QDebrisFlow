# -*- coding: utf-8 -*-
"""
Debris Flow O'Brien — simulation dialog v1.0

New in v1.0:
  - Tab 1 "Layers and sources":
      Table of features of the line layer. For each line (by ID)
      its own hydrograph Q(t) is specified independently.
  - Tab 4 "Output":
      prefix and folder moved here; checkboxes for layer selection.
  - HydrographEditorDialog:
      Pop-up hydrograph editor for a single feature.
"""

import json
import os
import re
import time
import traceback

import numpy as np

from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
    QWidget,
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (QgsMapLayerProxyModel, QgsWkbTypes, QgsGeometry, QgsPointXY,
                       QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsProject)
from qgis.gui import QgsMapLayerComboBox

_H_MIN = 0.005   # min. depth of a wet cell [m] — hidden from the user


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper functions (sweep)
# ─────────────────────────────────────────────────────────────────────────────

def _lhs_samples(param_ranges: dict, n: int, seed: int = 42) -> list:
    """
    Latin Hypercube Sampling — continuous coverage of the space without a fixed step.
    Keys "log_X" are automatically expanded: val → 10^val, key → "X".
    """
    rng = np.random.default_rng(seed)
    keys = list(param_ranges.keys())
    k = len(keys)
    unit = np.zeros((n, k))
    for j in range(k):
        perm = rng.permutation(n)
        unit[:, j] = (perm + rng.uniform(size=n)) / n
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


def _spearman(x, y) -> float:
    """Spearman rank correlation (no tie correction; LHS values are continuous)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / d) if d > 0 else float("nan")


def _spearman_ci(x, y, n_boot: int = 1000, seed: int = 0) -> tuple:
    """95 % bootstrap interval of the Spearman correlation."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.array([_spearman(x[i], y[i]) for i in
                     (rng.integers(0, n, n) for _ in range(n_boot))])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _volume_residuals(log_v, cm) -> np.ndarray:
    """Cm with the linear effect of log10(V) removed (mean Cm kept) — volume control."""
    log_v = np.asarray(log_v, float); cm = np.asarray(cm, float)
    A = np.vstack([log_v, np.ones_like(log_v)]).T
    coef = np.linalg.lstsq(A, cm, rcond=None)[0]
    return cm - A @ coef + cm.mean()



def _plugin_version() -> str:
    """Plugin version from metadata.txt (for run records)."""
    import configparser
    cp = configparser.ConfigParser()
    try:
        cp.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.txt"),
                encoding="utf-8")
        return cp.get("general", "version", fallback="?")
    except Exception:
        return "?"


def _runtime_info() -> dict:
    """Versions of the libraries that determine the numerics (for run records)."""
    info = {"plugin_version": _plugin_version(), "numpy": np.__version__}
    try:
        import numba
        info["numba"] = numba.__version__
    except Exception:
        info["numba"] = None
    try:
        from qgis.core import Qgis
        info["qgis"] = Qgis.version()
    except Exception:
        pass
    return info


def _jsonable(obj):
    """Parameters → JSON-safe (layers → name/source/CRS, numpy → lists, callables and
    keys starting with "_" (runtime-only data) dropped)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()
                if not callable(v) and not str(k).startswith("_")}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "source") and hasattr(obj, "name") and hasattr(obj, "crs"):
        return {"name": obj.name(), "source": obj.source(), "crs": obj.crs().authid()}
    if hasattr(obj, "authid") and hasattr(obj, "toWkt"):      # QgsCoordinateReferenceSystem
        return obj.authid() or obj.toWkt()
    return str(obj)


def _read_hydrograph_csv(path: str):
    """Reads (time [s], Q [m³/s]) from a CSV/text file: delimiter ',', ';', tab or spaces;
    a decimal comma is accepted with ';', tab or space delimiters; a header line is skipped."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    sample = "\n".join(lines[:20])

    def _comma_splits_numbers():
        # "," is the delimiter only if it splits a line into two numbers: in "300 12,5"
        # it is a decimal comma with a space delimiter
        for ln in lines[:20]:
            parts = ln.split(",")
            try:
                float(parts[0]); float(parts[1])
                return True
            except (ValueError, IndexError):
                continue
        return False

    delim = ";" if ";" in sample else ("\t" if "\t" in sample else (
        "," if "," in sample and _comma_splits_numbers() else None))
    times, qs = [], []
    for k, line in enumerate(lines):
        parts = line.split(delim) if delim else line.split()
        if delim != ",":
            parts = [s.strip().replace(",", ".") for s in parts]
        try:
            t, q = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            if not times:              # header line(s)
                continue
            raise ValueError(f"line {k + 1}: expected two numbers, got {line!r}")
        times.append(t); qs.append(q)
    if len(times) < 2:
        raise ValueError("fewer than 2 numeric rows")
    return times, qs



def _to_dem_crs(geom, src_crs, dem_crs):
    """Copy of a geometry transformed from src_crs to the DEM CRS (no-op if they are equal)."""
    g = QgsGeometry(geom)
    if src_crs.isValid() and dem_crs.isValid() and src_crs != dem_crs:
        g.transform(QgsCoordinateTransform(src_crs, dem_crs, QgsProject.instance()))
    return g


def _layer_wkts(layer, dem_crs) -> list:
    """WKT of every feature of a layer, in the DEM CRS (read in the main thread)."""
    return [_to_dem_crs(f.geometry(), layer.crs(), dem_crs).asWkt()
            for f in layer.getFeatures() if f.hasGeometry()]


def _layer_points(layer, field: str, dem_crs) -> list:
    """(x, y, depth) of every point of a point layer in the DEM CRS; MultiPoint is split."""
    out = []
    for f in layer.getFeatures():
        if not f.hasGeometry():
            continue
        try:
            d = float(f[field])
        except (KeyError, TypeError, ValueError):
            continue
        g = _to_dem_crs(f.geometry(), layer.crs(), dem_crs)
        pts = g.asMultiPoint() if g.isMultipart() else [g.asPoint()]
        out.extend((p.x(), p.y(), d) for p in pts)
    return out


def _feature_wkt(layer, fid, dem_crs) -> str:
    """WKT of one feature in the DEM CRS, or '' if it is missing / has no geometry."""
    f = layer.getFeature(fid)
    if not f.isValid() or not f.hasGeometry():
        return ""
    return _to_dem_crs(f.geometry(), layer.crs(), dem_crs).asWkt()


def _check_dem_crs(dem_layer) -> str:
    """'' if the DEM CRS is usable, otherwise the message for the user."""
    crs = dem_layer.crs()
    if crs.isValid() and crs.isGeographic():
        return (f"The DEM is in a geographic CRS ({crs.authid()}): its cell size is in degrees,\n"
                "not metres, and the simulation would be meaningless.\n"
                "Reproject the DEM to a projected CRS in metres (e.g. UTM) and run again.")
    if not crs.isValid():
        return ("The DEM has no coordinate reference system: its cell size and the\n"
                "reprojection of the vector layers cannot be checked.\n"
                "Assign the correct projected CRS in metres (Layer Properties → Source) and run again.")
    from qgis.core import QgsUnitTypes
    if crs.mapUnits() != QgsUnitTypes.DistanceMeters:
        # the solver takes the cell size and depths as metres (feet would give a 10.76× area)
        return (f"The DEM CRS ({crs.authid()}) is not in metres "
                f"({QgsUnitTypes.toString(crs.mapUnits())}): the solver treats the cell size as metres.\n"
                "Reproject the DEM to a projected CRS in metres (e.g. UTM) and run again.")
    return ""


def _detach_layers(params: dict, dem_layer) -> None:
    """Main thread: replace the QGIS layer objects in the worker parameters by their
    name/source/CRS (the same record _jsonable writes), so that the worker thread never
    calls methods of a layer that may be removed from the project during the run.
    A DEM that GDAL cannot open is read here with the QGIS provider."""
    from osgeo import gdal
    from .qgis_runner import read_dem_array
    params["dem_crs"] = QgsCoordinateReferenceSystem(dem_layer.crs())
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        readable = gdal.Open(dem_layer.source()) is not None
    finally:
        gdal.PopErrorHandler()
    if not readable:
        params["_dem_preloaded"] = read_dem_array(dem_layer)
    for k, v in list(params.items()):
        if hasattr(v, "source") and hasattr(v, "name") and hasattr(v, "crs"):
            params[k] = _jsonable(v)


def _read_job_dem(p: dict):
    """Worker thread: the DEM by the path and CRS taken in the main thread (or preloaded there)."""
    from .qgis_runner import read_dem_array
    if p.get("_dem_preloaded") is not None:
        return p["_dem_preloaded"]
    return read_dem_array(None, source=p["dem_layer"]["source"], crs=p["dem_crs"])


def _ogr_mem_datasource():
    """In-memory OGR datasource: driver "MEM" (vector-capable since GDAL 3.11), otherwise
    the older "Memory" (deprecated since GDAL 3.11)."""
    from osgeo import ogr
    for name in ("MEM", "Memory"):
        drv = ogr.GetDriverByName(name)
        try:
            ds = drv.CreateDataSource("tmp") if drv is not None else None
        except Exception:
            ds = None
        if ds is not None:
            return ds
    raise RuntimeError("no in-memory OGR vector driver (MEM / Memory)")



# ─────────────────────────────────────────────────────────────────────────────
# Helper dialog: hydrograph editor for a single line
# ─────────────────────────────────────────────────────────────────────────────

class HydrographEditorDialog(QDialog):
    """
    Pop-up dialog for specifying the hydrograph Q(t) of a single source.

    Returns via .get_hydrograph() → (times: list[float], q: list[float])
    or None if the user pressed Cancel.
    """

    def __init__(self, parent, feature_id, feature_name: str,
                 init_data=None):
        super().__init__(parent)
        self.setWindowTitle(f"Hydrograph — {feature_name} (ID={feature_id})")
        self.setMinimumWidth(420)
        self._result = None
        self._build_ui(init_data)

    def _build_ui(self, init=None):
        layout = QVBoxLayout(self)

        # Hydrograph type
        self.combo_type = QComboBox()
        self.combo_type.addItems([
            "Constant discharge (Constant Q)",
            "Table (Time, Q)",
            "CSV file (time, Q)",
        ])
        layout.addWidget(QLabel("Hydrograph type:"))
        layout.addWidget(self.combo_type)

        # Section 1: constant Q
        self._sec_const = QWidget()
        lay_c = QFormLayout(self._sec_const)
        lay_c.setContentsMargins(0, 4, 0, 4)
        self.spin_q = QDoubleSpinBox()
        self.spin_q.setRange(0, 1_000_000); self.spin_q.setValue(100)
        self.spin_q.setSuffix(" m³/s"); self.spin_q.setDecimals(1)
        lay_c.addRow("Q:", self.spin_q)
        layout.addWidget(self._sec_const)

        # Section 2: table
        self._sec_table = QWidget()
        lay_t = QVBoxLayout(self._sec_table)
        lay_t.setContentsMargins(0, 4, 0, 4)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Q (m³/s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(4)
        defaults = [(0, 0), (300, 100), (600, 50), (900, 0)]
        for i, (t, q) in enumerate(defaults):
            self.table.setItem(i, 0, QTableWidgetItem(str(t)))
            self.table.setItem(i, 1, QTableWidgetItem(str(q)))
        self.table.setMaximumHeight(160)
        lay_t.addWidget(self.table)
        row_btns = QHBoxLayout()
        btn_add = QPushButton("+ row")
        btn_add.clicked.connect(
            lambda: self.table.setRowCount(self.table.rowCount() + 1))
        btn_del = QPushButton("− row")
        btn_del.clicked.connect(
            lambda: self.table.setRowCount(max(2, self.table.rowCount() - 1)))
        row_btns.addWidget(btn_add); row_btns.addWidget(btn_del)
        row_btns.addStretch()
        lay_t.addLayout(row_btns)
        layout.addWidget(self._sec_table)

        # Section 3: CSV
        self._sec_csv = QWidget()
        lay_csv = QHBoxLayout(self._sec_csv)
        lay_csv.setContentsMargins(0, 4, 0, 4)
        self.btn_csv = QPushButton("📂 Choose CSV...")
        self.btn_csv.clicked.connect(self._load_csv)
        self.lbl_csv = QLabel("not selected")
        self.lbl_csv.setStyleSheet("color:gray")
        lay_csv.addWidget(self.btn_csv); lay_csv.addWidget(self.lbl_csv)
        lay_csv.addStretch()
        layout.addWidget(self._sec_csv)

        # OK / Cancel buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Switching signal
        self.combo_type.currentIndexChanged.connect(self._update_sections)

        # Restore saved data
        if init:
            self._restore(init)
        else:
            self._update_sections(0)

    def _update_sections(self, idx: int = -1):
        if idx < 0:
            idx = self.combo_type.currentIndex()
        self._sec_const.setVisible(idx == 0)
        self._sec_table.setVisible(idx == 1)
        self._sec_csv.setVisible(idx == 2)

    def _load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "CSV hydrograph", "", "CSV (*.csv);;All (*)")
        if not fname:
            return
        try:
            times, qs = _read_hydrograph_csv(fname)
            self.table.setRowCount(len(times))
            for i, (t, q) in enumerate(zip(times, qs)):
                self.table.setItem(i, 0, QTableWidgetItem(f"{t:.10g}"))
                self.table.setItem(i, 1, QTableWidgetItem(f"{q:.10g}"))
            self.lbl_csv.setText(os.path.basename(fname))
            self.lbl_csv.setStyleSheet("color:green")
            self.combo_type.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.warning(self, "CSV error", str(e))

    def _restore(self, data: dict):
        """Restore previously saved data."""
        typ = data.get("type", "const")
        if typ == "const":
            self.combo_type.setCurrentIndex(0)
            self.spin_q.setValue(data.get("q_const", 100.0))
        else:
            self.combo_type.setCurrentIndex(1)
            times = data.get("times", [])
            q_vals = data.get("q_vals", [])
            self.table.setRowCount(max(len(times), 2))
            for i, (t, q) in enumerate(zip(times, q_vals)):
                self.table.setItem(i, 0, QTableWidgetItem(f"{t:.10g}"))
                self.table.setItem(i, 1, QTableWidgetItem(f"{q:.10g}"))
        self._update_sections()

    def _on_ok(self):
        try:
            self._result = self._parse()
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _parse(self) -> dict:
        """Parse the entered data → dict for storage."""
        idx = self.combo_type.currentIndex()
        if idx == 0:
            q = self.spin_q.value()
            return {"type": "const", "q_const": q,
                    "times": [0.0, 1e9], "q_vals": [q, q]}
        # Table / CSV
        import math
        rows, bad = [], []
        for i in range(self.table.rowCount()):
            ti = self.table.item(i, 0)
            qi = self.table.item(i, 1)
            t_txt = ti.text().strip() if ti is not None else ""
            q_txt = qi.text().strip() if qi is not None else ""
            if not t_txt and not q_txt:
                continue                      # empty row
            try:
                t, q = float(t_txt.replace(",", ".")), float(q_txt.replace(",", "."))
            except ValueError:                # also a row with only one of the two cells filled
                bad.append(i + 1)
                continue
            if not (math.isfinite(t) and math.isfinite(q)):
                bad.append(i + 1)             # nan / inf
                continue
            rows.append((t, q))
        if bad:
            raise ValueError(f"Rows {bad}: not a finite number or an incomplete row "
                             "(use digits in both cells, e.g. 12.5 or 12,5).")
        if any(q < 0 for _, q in rows):
            raise ValueError("The discharge Q cannot be negative.")
        if any(t < 0 for t, _ in rows):
            raise ValueError("Time cannot be negative.")
        rows.sort(key=lambda r: r[0])          # np.interp needs increasing times
        times = [r[0] for r in rows]
        q_vals = [r[1] for r in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Two rows have the same time — times must differ.")
        if len(times) < 2:
            raise ValueError(
                "At least 2 rows with numbers are required.\n"
                "Example: (0, 0) and (300, 100).")
        if not any(q > 0 for q in q_vals):
            raise ValueError("The discharge Q is zero in every row: the hydrograph has no volume.")
        return {"type": "table", "times": times, "q_vals": q_vals,
                "q_const": 0.0}

    def get_result(self):
        return self._result


# ─────────────────────────────────────────────────────────────────────────────
# Background simulation thread
# ─────────────────────────────────────────────────────────────────────────────

class SimulationWorker(QThread):
    progress    = pyqtSignal(float, float)
    log_message = pyqtSignal(str)
    finished    = pyqtSignal(object)
    error       = pyqtSignal(str)
    cancelled   = pyqtSignal()

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    # ── Main thread method ──────────────────────────────────────────────────

    def run(self):
        try:
            # relative imports: a top-level `core` of another plugin (or of the other
            # language copy) loaded earlier must not be picked up instead of ours
            from .core.rheology import OBrienParameters
            from .core.solver   import (TerrainGrid, SimulationConfig,
                                        SourceCondition, InflowSource,
                                        DebrisFlowSolver2D, hydrograph_volume)
            from .qgis_runner   import write_geotiff

            p = self.params

            # ── DEM ───────────────────────────────────────────────────────────
            # layers were replaced by name/source/CRS in the main thread (_detach_layers)
            self.log_message.emit("Reading DEM...")
            dem, geo_info = _read_job_dem(p)

            # ── CRS check (compared in the main thread) ───────────────────────
            if p.get("_crs_note"):
                self.log_message.emit(p["_crs_note"])

            # ── Rasterisation per feature → one InflowSource each ────────────
            # p["sources"] = list of {fid, name, times, q_vals}
            inflow_sources = []
            for src_info in p["sources"]:
                fid  = src_info["fid"]
                name = src_info["name"]
                self.log_message.emit(f"  Rasterising: {name} (ID={fid})...")

                wkt = src_info.get("wkt")
                if not wkt:
                    self.log_message.emit(f"    ⚠ Feature ID={fid} not found, skipping.")
                    continue

                rows, cols = self._rasterise_feature(wkt, geo_info)
                if len(rows) == 0:
                    self.log_message.emit(
                        f"    ⚠ Feature ID={fid} does not intersect the DEM, skipping.")
                    continue
                # same validity rule as TerrainGrid.mask_valid: the solver drops nodata cells
                _z = dem[rows, cols]
                n_valid = int(np.sum((_z != geo_info["nodata"]) & np.isfinite(_z)))
                if n_valid == 0:
                    self.log_message.emit(
                        f"    ⚠ Feature ID={fid} lies entirely on DEM nodata cells, skipping.")
                    continue
                if n_valid < len(rows):
                    self.log_message.emit(
                        f"    ⚠ {len(rows) - n_valid} of {len(rows)} cells are DEM nodata — "
                        "the inflow is spread over the remaining cells.")

                self.log_message.emit(
                    f"    ✓ {len(rows)} cells, "
                    f"Q_max={max(src_info['q_vals']):.1f} m³/s")

                if max(src_info["times"]) < 1e8:      # not a constant hydrograph
                    _beyond = hydrograph_volume(src_info["times"], src_info["q_vals"],
                                                p["t_end"], 1e12)
                    if _beyond > 0:
                        self.log_message.emit(f"    ⚠ {_beyond:.0f} m³ of this hydrograph come after t_end and will not be injected.")
                inflow_sources.append(InflowSource(
                    rows    = rows,
                    cols    = cols,
                    times   = np.array(src_info["times"],  dtype=np.float64),
                    q_total = np.array(src_info["q_vals"], dtype=np.float64),
                    name    = name,
                ))

            if not inflow_sources:
                raise RuntimeError(
                    "No source feature intersects valid (non-nodata) DEM cells.\n"
                    "Check the CRS and extent of the line layer.")

            source = SourceCondition.from_sources(inflow_sources)
            self.log_message.emit(
                f"Total sources: {len(inflow_sources)}")

            # ── O'Brien parameters ────────────────────────────────────────────
            self.log_message.emit("Initialising O'Brien rheology...")
            obrien = OBrienParameters(
                Cv               = p["Cv"],
                Cv_max           = p["Cv_max"],
                rho_sediment     = p.get("rho_s", 2650.0),
                rho_water        = p.get("rho_w", 1000.0),
                tau_yield_direct = p.get("tau_yield_direct", 0.0),
                direct_mode      = p.get("direct_mode"),
                eta_direct       = p.get("eta_direct",       0.0),
                alpha1           = p["alpha1"],
                beta1            = p["beta1"],
                alpha2           = p["alpha2"],
                beta2            = p["beta2"],
                manning_n        = p["manning_n"],
                K_visc           = p.get("K_visc", 24.0),
                h_min            = _H_MIN,
            )
            self.log_message.emit(obrien.summary())

            terrain = TerrainGrid(
                dem    = dem,
                dx     = geo_info["dx"],
                dy     = geo_info["dy"],
                nodata = geo_info["nodata"],
            )

            config = SimulationConfig(
                t_end             = p["t_end"],
                dt_max            = p["dt_max"],
                cfl_number        = p.get("cfl_number", 0.60),
                face_depth_limiter = p.get("face_depth_limiter", True),
                theta             = p.get("theta", 1.0),
                arrival_depth     = p.get("arrival_depth"),
                cancel_check      = lambda: self._cancelled,
                progress_callback = lambda tc, te: (
                    self.progress.emit(tc, te),
                    self.log_message.emit(f"  t = {tc:.0f} s / {te:.0f} s")
                ),
            )

            self.log_message.emit("Starting simulation...")
            result = DebrisFlowSolver2D(terrain, obrien, source, config).run()

            # ── Writing results ────────────────────────────────────────────────
            out_dir = p["output_dir"]
            os.makedirs(out_dir, exist_ok=True)
            prefix  = p["output_prefix"]
            wanted  = p["output_layers"]   # set of keys

            # Apply the depth threshold if specified
            h_thr = p.get("h_threshold", 0.0)

            def _mask_by_depth(arr):
                if h_thr <= 0.0:
                    return arr
                out = arr.copy()
                out[result.h_max < h_thr] = -9999.0
                return out

            def _thr(arr):
                if h_thr <= 0.0:
                    return arr
                out = arr.copy()
                out[out < h_thr] = -9999.0
                return out

            # Full map: key → (array, zero_as_nodata)
            all_outputs = {
                "h_max":     (_thr(result.h_max),   True),
                "V_max":     (_mask_by_depth(result.V_max),   True),
                "HV2_max":   (_mask_by_depth(result.HV2_max), True),
                "Fr_hv2":    (_mask_by_depth(result.Fr_hv2),  True),
                "h_final":   (_thr(result.h_final),  True),
                "t_arrival": (np.where(np.isinf(result.t_arrival),
                                       -9999.0, result.t_arrival), False),
            }

            paths = {}
            for key, (arr, znod) in all_outputs.items():
                if key not in wanted:
                    continue
                name = f"{prefix}_{key}"
                path = os.path.join(out_dir, f"{name}.tif")
                write_geotiff(arr, geo_info, path, zero_as_nodata=znod)
                paths[name] = path
                self.log_message.emit(f"  Saved: {path}")

            _w01 = result.h_max >= 0.1
            _nw = max(int(_w01.sum()), 1)
            fr05 = 100.0 * float(np.sum(result.Fr_hv2[_w01] > 0.5)) / _nw
            fr1 = 100.0 * float(np.sum(result.Fr_hv2[_w01] > 1.0)) / _nw
            vb = result.volume_balance()
            self.log_message.emit(
                f"\n── Summary ──\n"
                f"  h_max peak:  {result.h_max.max():.3f} m\n"
                f"  V_max peak:  {result.V_max.max():.3f} m/s\n"
                f"  h·v² peak:   {result.HV2_max.max():.3f} m³/s² "
                f"(h_max·V_max² would give {float(np.max(result.h_max * result.V_max**2)):.3f})\n"
                f"  Fr at peak h·v²: > 0.5 on {fr05:.1f} %, > 1 on {fr1:.1f} % of the area with h_max ≥ 0.1 m\n"
                f"               (LIA is outside its validity range there, de Almeida & Bates 2013)\n"
                f"  Steps:       {result.n_steps}\n"
                f"  Sim. time:   {result.t_simulated:.1f} s\n"
                f"  Inflow vol.: {result.total_inflow_vol:.1f} m³\n"
                f"  Δ mass:      {vb.get('V_loss_pct', float('nan')):.3f}%"
            )
            if result.edge_h_max >= _H_MIN:
                self.log_message.emit(
                    f"\n⚠ The flow reached the domain boundary (h_max at the edge = "
                    f"{result.edge_h_max:.3f} m).\n"
                    "  The boundary is a closed wall: the fluid accumulates there instead of\n"
                    "  leaving, so h_max/h_final near the edge are overestimated.\n"
                    "  Nodata cells inside the DEM are walls too and also trigger this warning.\n"
                    "  Enlarge the DEM beyond the expected runout (or fill the nodata holes).")
            if result.n_vcap_cells:
                self.log_message.emit(
                    f"\n⚠ The velocity reached the 30 m/s cap in {result.n_vcap_cells} cells.\n"
                    "  This is a numerical artefact (LIA oscillations in steep channels or\n"
                    "  cells emptied in one step); V_max and h·v² there are not reliable.\n"
                    "  Try θ-weighting 0.8–0.9 on the Solver Settings tab.")

            rec_path = os.path.join(out_dir, f"{prefix}_run.json")
            from .core import solver as _solver_mod
            run_rec = dict(
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
                runtime=dict(_runtime_info(), backend=(
                    "Numba JIT" if _solver_mod._NUMBA_AVAILABLE else "NumPy")),
                parameters=_jsonable(p),
                rheology=dict(mode=obrien.mode_name, tau_yield_Pa=float(obrien.tau_yield),
                              eta_Pas=float(obrien.eta), rho_mixture=float(obrien.rho_mixture),
                              h_min=float(obrien.h_min)),
                dem=dict(nx=int(geo_info["nx"]), ny=int(geo_info["ny"]), dx=float(geo_info["dx"]),
                         dy=float(geo_info["dy"]), crs=p["dem_layer"]["crs"],
                         source=p["dem_layer"]["source"]),
                results=dict(n_steps=int(result.n_steps), t_simulated=float(result.t_simulated),
                             inflow_vol_m3=float(result.total_inflow_vol),
                             volume_balance=_jsonable(vb),
                             h_max_peak_m=float(result.h_max.max()),
                             V_max_peak_ms=float(result.V_max.max()),
                             HV2_max_peak=float(result.HV2_max.max()),
                             Fr_hv2_gt05_pct=fr05, Fr_hv2_gt1_pct=fr1,
                             edge_h_max_m=float(result.edge_h_max),
                             vcap_cells=int(result.n_vcap_cells)),
                outputs=paths,
            )
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump(run_rec, f, ensure_ascii=False, indent=2)
            self.log_message.emit(f"  Run record (all parameters, versions, balance): {rec_path}")

            self.finished.emit({
                "result":   result,
                "paths":    paths,
                "geo_info": geo_info,
                "prefix":   prefix,
                "wanted":   wanted,
            })

        except Exception as e:
            if type(e).__name__ == "SimulationCancelled":
                self.log_message.emit("\n⚠ Simulation cancelled by the user.")
                self.cancelled.emit()
                return
            self.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    # ── Rasterisation of a single feature ─────────────────────────────────────

    def _rasterise_feature(self, wkt: str, geo_info: dict):
        """
        Rasterises a single feature (line) → (rows_array, cols_array).
        Uses GDAL via OGR with an in-memory layer, without writing to disk.
        Fallback: iterate over cells using QGIS geometry.distance().
        """
        try:
            from osgeo import gdal, ogr, osr

            nx, ny = geo_info["nx"], geo_info["ny"]
            gt     = geo_info["geotransform"]
            crs    = geo_info.get("crs")

            # In-memory vector dataset with a single feature
            vec_ds      = _ogr_mem_datasource()

            srs = None
            if crs:
                srs = osr.SpatialReference()
                try:
                    srs.ImportFromWkt(crs.toWkt())
                except Exception:
                    srs = None

            lyr = vec_ds.CreateLayer("src", srs=srs,
                                     geom_type=ogr.wkbLineString)
            fld = ogr.FieldDefn("fid", ogr.OFTInteger)
            lyr.CreateField(fld)

            ogr_feat = ogr.Feature(lyr.GetLayerDefn())
            ogr_feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
            ogr_feat.SetField("fid", 1)
            lyr.CreateFeature(ogr_feat)

            # In-memory raster
            mem_ras_drv = gdal.GetDriverByName("MEM")
            ras_ds = mem_ras_drv.Create("", nx, ny, 1, gdal.GDT_Byte)
            ras_ds.SetGeoTransform(gt)
            band = ras_ds.GetRasterBand(1)
            band.Fill(0)
            gdal.RasterizeLayer(ras_ds, [1], lyr, burn_values=[1])

            mask = band.ReadAsArray().astype(bool)
            ras_ds = vec_ds = None
            return np.where(mask)

        except Exception as e:
            self.log_message.emit(
                f"    ⚠ GDAL line rasterisation failed ({e}) — QGIS fallback (slower, wider footprint).")
            # Fallback: QGIS distance-based
            dx, dy   = geo_info["dx"], geo_info["dy"]
            xmin     = geo_info["xmin"]
            ymax     = geo_info["ymax"]
            nx, ny   = geo_info["nx"], geo_info["ny"]
            geom     = QgsGeometry.fromWkt(wkt)
            thresh   = (dx**2 + dy**2)**0.5 * 0.5
            mask     = np.zeros((ny, nx), dtype=bool)
            for row in range(ny):
                cy = ymax - (row + 0.5) * dy
                for col in range(nx):
                    cx = xmin + (col + 0.5) * dx
                    pt = QgsGeometry.fromPointXY(QgsPointXY(cx, cy))
                    if geom.distance(pt) <= thresh:
                        mask[row, col] = True
            return np.where(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Background LHS-sweep thread
# ─────────────────────────────────────────────────────────────────────────────

class SweepWorker(QThread):
    """
    Sequential LHS-sweep in a background QThread.
    A parallel mp.Pool is incompatible with the QGIS Qt event loop,
    so runs proceed one after another, but the UI is not blocked.
    Signals:
        run_done(dict)    — result of a single run
        sweep_log(str)    — text message
        sweep_finished()  — all runs completed
        sweep_error(str)  — critical error
    """
    run_done          = pyqtSignal(dict)
    sweep_log         = pyqtSignal(str)
    sweep_finished    = pyqtSignal()
    sweep_error       = pyqtSignal(str)
    sweep_best_result = pyqtSignal(object)  # dict: h_max, geo_info, params, metrics

    def __init__(self, params: dict):
        super().__init__()
        self.params     = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    # ── main method ─────────────────────────────────────────────────────────

    def run(self):
        import sys, time
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

        # relative imports (see SimulationWorker.run)
        from .core import sweep_runner as _sr
        from .core.solver import hydrograph_volume as _hydro_volume

        _csv_f = None
        try:
            p = self.params

            # ── DEM and source rasterisation ──────────────────────────────────
            # layers were replaced by name/source/CRS in the main thread (_detach_layers)
            self.sweep_log.emit("Reading DEM...")
            dem, geo_info = _read_job_dem(p)

            self.sweep_log.emit("Rasterising sources...")
            src_cells = []
            for src_info in p["sources"]:
                wkt = src_info.get("wkt")
                if not wkt:
                    self.sweep_log.emit(f"⚠ Source {src_info['name']} not found, skipping.")
                    continue
                rows, cols = self._rasterise_feature(wkt, geo_info)
                if len(rows) == 0:
                    self.sweep_log.emit(
                        f"⚠ Source {src_info['name']} does not intersect the DEM, skipping.")
                    continue
                # a source entirely on nodata is dropped by the solver: keep it out of
                # the base volume as well (same validity rule as TerrainGrid.mask_valid)
                _z = dem[rows, cols]
                n_valid = int(np.sum((_z != geo_info["nodata"]) & np.isfinite(_z)))
                if n_valid == 0:
                    self.sweep_log.emit(
                        f"⚠ Source {src_info['name']} lies entirely on DEM nodata cells, skipping.")
                    continue
                if n_valid < len(rows):
                    self.sweep_log.emit(
                        f"⚠ Source {src_info['name']}: {len(rows) - n_valid} of {len(rows)} cells "
                        "are DEM nodata — the inflow is spread over the remaining cells.")
                src_cells.append({
                    "rows":        rows,
                    "cols":        cols,
                    "times":       np.array(src_info["times"],  dtype=np.float64),
                    "base_q_vals": np.array(src_info["q_vals"], dtype=np.float64),
                    "name":        src_info["name"],
                })

            if not src_cells:
                self.sweep_error.emit("No source intersects valid (non-nodata) DEM cells.")
                return

            # Base volume
            base_vol = 0.0
            for sc in src_cells:
                if float(np.max(sc["times"])) < 1e8:
                    _beyond = _hydro_volume(sc["times"], sc["base_q_vals"],
                                            float(p["t_end"]), 1e12)
                    if _beyond > 0:
                        self.sweep_log.emit(f"⚠ Source {sc['name']}: {_beyond:.0f} m³ come after t_end and are not injected.")
                # volume actually injected within [0, t_end] (also for constant Q)
                base_vol += _hydro_volume(sc["times"], sc["base_q_vals"],
                                          0.0, float(p["t_end"]))
            if not (base_vol > 0):          # NaN is rejected too
                self.sweep_error.emit(
                    "Base volume = 0. Check the source hydrographs.")
                return
            self.sweep_log.emit(
                f"Base volume within [0, t_end]: {base_vol:.0f} m³ "
                f"({len(src_cells)} sources)")

            # ── Observed inundation ───────────────────────────────────────────
            self.sweep_log.emit("Rasterising the inundation polygon...")
            obs_mask = self._rasterize_polygon(p["obs_poly_wkts"], geo_info)
            self.sweep_log.emit(f"Observed inundation: {obs_mask.sum()} cells")
            eval_mask = None
            if p.get("eval_poly_wkts"):
                eval_mask = self._rasterize_polygon(p["eval_poly_wkts"], geo_info)
                obs_out = int(np.sum(obs_mask & ~eval_mask))
                self.sweep_log.emit(
                    f"Evaluation area: {int(eval_mask.sum())} cells"
                    + (f" (⚠ {obs_out} observed cells lie outside it)" if obs_out else ""))

            # ── Depth points ──────────────────────────────────────────────────
            self.sweep_log.emit("Reading depth points...")
            n_nonfinite = sum(1 for _x, _y, _d in p["obs_pts"] if not np.isfinite(_d))
            if n_nonfinite:
                self.sweep_log.emit(
                    f"⚠ {n_nonfinite} depth point(s) with a non-finite depth (NaN/inf) skipped.")
            obs_rows, obs_cols, obs_depths = self._extract_depth_points(
                p["obs_pts"], geo_info)
            self.sweep_log.emit(f"Measurement points in domain: {len(obs_depths)}")
            n_outside = len(p["obs_pts"]) - n_nonfinite - len(obs_depths)
            if n_outside:
                self.sweep_log.emit(f"⚠ {n_outside} depth point(s) outside the DEM skipped.")
            if len(obs_depths) == 0:
                self.sweep_log.emit(
                    "⚠ No usable depth points: Cm = ΩTm, the depths are not taken into account\n"
                    "  (check the depth field: empty, non-numeric or \"0,7\" values are skipped).")

            # ── LHS sampling ──────────────────────────────────────────────────
            n_runs    = p["n_runs"]
            seed      = p.get("seed", 42)
            n_workers = max(1, int(p.get("n_workers", 1)))
            samples   = _lhs_samples(p["param_ranges"], n_runs, seed=seed)
            self.sweep_log.emit(
                f"\nStarting {n_runs} runs (LHS seed={seed}, "
                f"workers={n_workers})...")

            # ── Shared job context (serialisable) ─────────────────────────────
            # geo_info from read_dem_array contains crs (QgsCoordinateReferenceSystem)
            # and possibly GDAL objects — all unsuitable for pickle.
            # The child process (sweep_runner) needs only dx, dy, nodata.
            _gt = geo_info.get("geotransform", ())
            _geo_serial = dict(
                nx           = int(geo_info["nx"]),
                ny           = int(geo_info["ny"]),
                dx           = float(geo_info["dx"]),
                dy           = float(geo_info["dy"]),
                xmin         = float(geo_info.get("xmin", 0.0)),
                ymax         = float(geo_info.get("ymax", 0.0)),
                # read_dem_array already substitutes -9999 for a missing nodata;
                # no `or` here: nodata = 0 must stay 0, as in the single run
                nodata       = float(geo_info["nodata"]),
                # Explicit conversion of each element → plain Python float
                geotransform = tuple(float(x) for x in _gt),
                # crs is NOT included — QgsCoordinateReferenceSystem is not picklable
            )

            job_base = dict(
                plugin_dir      = plugin_dir,
                cancel_check    = (lambda: self._cancelled),
                dem             = dem,
                geo_info        = _geo_serial,   # only primitives → picklable
                src_cells       = src_cells,
                base_vol        = base_vol,
                obs_mask        = obs_mask,
                eval_mask       = eval_mask,
                obs_rows        = obs_rows,
                obs_cols        = obs_cols,
                obs_depths      = obs_depths,
                Cv              = p["Cv"],
                Cv_max          = p["Cv_max"],
                rho_s           = p.get("rho_s", 2650.0),
                rho_w           = p.get("rho_w", 1000.0),
                param_ranges    = p["param_ranges"],
                t_end           = p["t_end"],
                dt_max          = p["dt_max"],
                cfl_number      = p.get("cfl_number", 0.60),
                face_depth_limiter = p.get("face_depth_limiter", True),
                theta           = p.get("theta", 1.0),
                depth_threshold = p["depth_threshold"],
            )

            # List of jobs
            job_args = [
                dict(job_base, run_id=i, sample=s)
                for i, s in enumerate(samples)
            ]

            # No check needed: threads do not require pickle
            import csv as _csv
            os.makedirs(p["out_dir"], exist_ok=True)
            csv_path = os.path.join(p["out_dir"], p["csv_name"])
            _csv_f = open(csv_path, "w", newline="", encoding="utf-8")
            _csv_w = _csv.DictWriter(_csv_f, fieldnames=["run_id", "status", "vol_m3", "vol_in_m3", "tau_yield_Pa", "eta_Pas", "manning_n",
                       "K_visc", "h_max_m", "edge_h_max_m", "vcap_cells", "wall_s", "TP", "FP",
                       "FN", "omega_Tm", "delta_o_c", "delta_u", "Cm", "depth_bias_m", "eta_remaining_min"],
                                     extrasaction="ignore", restval="")
            _csv_w.writeheader(); _csv_f.flush()
            from .core import solver as _solver_mod
            ctx = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       runtime=dict(_runtime_info(), backend=(
                           "Numba JIT" if _solver_mod._NUMBA_AVAILABLE else "NumPy")),
                       csv=csv_path, note="keys with prefix log_ are sampled uniformly in log10 and hold log10 values",
                       base_volume_m3=base_vol, parameters=_jsonable(p))
            with open(os.path.splitext(csv_path)[0] + "_context.json", "w",
                      encoding="utf-8") as f:
                json.dump(ctx, f, ensure_ascii=False, indent=2)
            self.sweep_log.emit(f"Results are appended to {csv_path} after every run")
            t_start   = time.time()
            _best_Cm  = float("inf")
            _best_pkg = None
            completed = 0

            # ── Helper function to process a single result ────────────────────
            def _process_result(row_data: dict):
                nonlocal completed, _best_Cm, _best_pkg

                h_max_arr = row_data.pop("h_max", None)  # remove from the CSV row
                eta_min   = round(
                    (time.time() - t_start) / max(completed, 1)
                    * (n_runs - completed) / 60, 1)
                row_data["eta_remaining_min"] = eta_min   # remaining time [min], not η
                _csv_w.writerow(row_data); _csv_f.flush()

                self.run_done.emit(row_data)

                rid   = row_data.get("run_id", "?")
                cm_v  = row_data.get("Cm")
                status= row_data.get("status", "")

                if status == "ok":
                    self.sweep_log.emit(
                        f"  ✓ [{completed}/{n_runs}]  "
                        f"V={row_data.get('vol_m3', '?'):.0f}m³  "
                        f"τy={row_data.get('tau_yield_Pa', '?'):.1f}Pa  "
                        f"η={row_data.get('eta_Pas', '?'):.1f}Pa·s  "
                        f"n={row_data.get('manning_n', '?'):.3f}  "
                        f"K={row_data.get('K_visc', '?'):.1f}  "
                        f"Cm={cm_v}  ETA {eta_min:.1f}min")
                else:
                    self.sweep_log.emit(
                        f"  ✗ [{rid+1}/{n_runs}]  {status}")

                if cm_v is not None and h_max_arr is not None and cm_v < _best_Cm:
                    _best_Cm  = cm_v
                    _best_pkg = dict(
                        h_max    = h_max_arr.copy(),
                        geo_info = geo_info,
                        params   = dict(
                            vol_m3       = row_data.get("vol_m3"),
                            tau_yield_Pa = row_data.get("tau_yield_Pa"),
                            eta_Pas      = row_data.get("eta_Pas"),
                            manning_n    = row_data.get("manning_n"),
                            K_visc       = row_data.get("K_visc"),
                        ),
                        metrics  = {k: row_data[k] for k in
                                    ("TP","FP","FN","omega_Tm",
                                     "delta_o_c","delta_u","Cm","depth_bias_m")
                                    if k in row_data},
                    )

            # ── Execution: sequential (n=1) or via threads (n>1) ──────────────
            if n_workers == 1:
                # Sequential mode
                for job in job_args:
                    if self._cancelled:
                        self.sweep_log.emit("⚠ Sweep cancelled by the user.")
                        break
                    row_data  = _sr.run_one_sample(job)
                    completed += 1
                    _process_result(row_data)

            else:
                # Parallel mode via ThreadPoolExecutor.
                # Threads run inside the QGIS process → no spawn needed,
                # no pickle needed. Numba njit releases the GIL during execution
                # → real parallelism at the C/JIT-code level.
                from concurrent.futures import ThreadPoolExecutor, as_completed

                self.sweep_log.emit(
                    f"⚙ Starting {n_workers} parallel threads...")

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_map = {
                        executor.submit(_sr.run_one_sample, job): job["run_id"]
                        for job in job_args
                    }
                    for future in as_completed(future_map):
                        if self._cancelled:
                            self.sweep_log.emit("⚠ Sweep cancelled by the user.")
                            # Cancel futures that have not started yet
                            for f in future_map:
                                f.cancel()
                            break
                        try:
                            row_data = future.result()
                        except Exception as exc:
                            rid = future_map[future]
                            row_data = dict(
                                run_id=rid, status=f"error: {exc}",
                                h_max=None,
                                **{k: None for k in [
                                    "vol_m3","vol_in_m3","tau_yield_Pa","eta_Pas","manning_n",
                                    "K_visc","h_max_m","edge_h_max_m","vcap_cells","wall_s","eta_remaining_min",
                                    "TP","FP","FN","omega_Tm",
                                    "delta_o_c","delta_u","Cm","depth_bias_m"]},
                            )
                        completed += 1
                        _process_result(row_data)

            if _best_pkg is not None:
                self.sweep_best_result.emit(_best_pkg)
            self.sweep_finished.emit()

        except Exception as e:
            self.sweep_error.emit(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        finally:
            if _csv_f is not None:
                _csv_f.close()

    # ── Helper methods ────────────────────────────────────────────────────────

    def _rasterise_feature(self, wkt, geo_info):
        """Same rasterisation as in SimulationWorker."""
        try:
            from osgeo import gdal, ogr, osr
            nx, ny = geo_info["nx"], geo_info["ny"]
            gt     = geo_info["geotransform"]
            crs    = geo_info.get("crs")
            mem_vec = _ogr_mem_datasource()
            srs = None
            if crs:
                srs = osr.SpatialReference()
                try:
                    srs.ImportFromWkt(crs.toWkt())
                except Exception:
                    srs = None
            lyr = mem_vec.CreateLayer("src", srs=srs,
                                      geom_type=ogr.wkbLineString)
            lyr.CreateField(ogr.FieldDefn("fid", ogr.OFTInteger))
            ogr_feat = ogr.Feature(lyr.GetLayerDefn())
            ogr_feat.SetGeometry(
                ogr.CreateGeometryFromWkt(wkt))
            ogr_feat.SetField("fid", 1)
            lyr.CreateFeature(ogr_feat)
            ras_ds = gdal.GetDriverByName("MEM").Create(
                "", nx, ny, 1, gdal.GDT_Byte)
            ras_ds.SetGeoTransform(gt)
            ras_ds.GetRasterBand(1).Fill(0)
            gdal.RasterizeLayer(ras_ds, [1], lyr, burn_values=[1])
            mask = ras_ds.GetRasterBand(1).ReadAsArray().astype(bool)
            ras_ds = mem_vec = None
            return np.where(mask)
        except Exception as e:
            self.sweep_log.emit(
                f"  ⚠ GDAL line rasterisation failed ({e}) — QGIS fallback (slower, wider footprint).")
            dx, dy = geo_info["dx"], geo_info["dy"]
            xmin, ymax = geo_info["xmin"], geo_info["ymax"]
            nx, ny = geo_info["nx"], geo_info["ny"]
            geom   = QgsGeometry.fromWkt(wkt)
            thresh = (dx**2 + dy**2)**0.5 * 0.5
            mask   = np.zeros((ny, nx), dtype=bool)
            for row in range(ny):
                cy = ymax - (row + 0.5) * dy
                for col in range(nx):
                    cx = xmin + (col + 0.5) * dx
                    pt = QgsGeometry.fromPointXY(QgsPointXY(cx, cy))
                    if geom.distance(pt) <= thresh:
                        mask[row, col] = True
            return np.where(mask)

    def _rasterize_polygon(self, wkts, geo_info) -> np.ndarray:
        """
        Rasterises the inundation polygon layer → bool mask (ny, nx).
        Value 1 = inundation was observed.
        """
        try:
            from osgeo import gdal, ogr, osr
            nx, ny = geo_info["nx"], geo_info["ny"]
            gt     = geo_info["geotransform"]
            crs    = geo_info.get("crs")

            mem_vec = _ogr_mem_datasource()
            srs = None
            if crs:
                srs = osr.SpatialReference()
                try:
                    srs.ImportFromWkt(crs.toWkt())
                except Exception:
                    srs = None

            lyr = mem_vec.CreateLayer("obs", srs=srs,
                                      geom_type=ogr.wkbPolygon)
            lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))

            for wkt in wkts:
                ogr_feat = ogr.Feature(lyr.GetLayerDefn())
                ogr_feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
                ogr_feat.SetField("val", 1)
                lyr.CreateFeature(ogr_feat)

            ras_ds = gdal.GetDriverByName("MEM").Create(
                "", nx, ny, 1, gdal.GDT_Byte)
            ras_ds.SetGeoTransform(gt)
            ras_ds.GetRasterBand(1).Fill(0)
            gdal.RasterizeLayer(ras_ds, [1], lyr,
                                 options=["ATTRIBUTE=val"])
            mask = ras_ds.GetRasterBand(1).ReadAsArray().astype(bool)
            ras_ds = mem_vec = None
            return mask

        except Exception as e:
            # Fallback via QGIS geometry (slow)
            self.sweep_log.emit(
                f"  ⚠ GDAL polygon rasterize failed: {e} — fallback QGIS")
            nx, ny = geo_info["nx"], geo_info["ny"]
            dx, dy = geo_info["dx"], geo_info["dy"]
            xmin, ymax = geo_info["xmin"], geo_info["ymax"]
            mask = np.zeros((ny, nx), dtype=bool)
            polys = [QgsGeometry.fromWkt(w) for w in wkts]
            for row in range(ny):
                cy = ymax - (row + 0.5) * dy
                for col in range(nx):
                    cx = xmin + (col + 0.5) * dx
                    pt = QgsGeometry.fromPointXY(QgsPointXY(cx, cy))
                    if any(g.contains(pt) for g in polys):
                        mask[row, col] = True
            return mask

    def _extract_depth_points(self, pts, geo_info: dict):
        """
        Snaps observed depth points [(x, y, depth) in the DEM CRS] to grid cells.
        Returns (row_arr, col_arr, depth_arr) — only points within the domain.
        """
        gt   = geo_info["geotransform"]
        dx   = geo_info["dx"]
        dy   = geo_info["dy"]
        xmin = gt[0]
        ymax = gt[3]
        ny   = geo_info["ny"]
        nx   = geo_info["nx"]

        rows, cols, depths = [], [], []
        for x, y, d in pts:
            # a NaN/inf depth would make Σ|d_obs| = NaN and zero Δo,c and Δu in every run
            if not np.isfinite(d):
                continue
            # floor (not int): points left of / above the grid are dropped
            col = int(np.floor((x - xmin) / dx))
            row = int(np.floor((ymax - y) / dy))
            if 0 <= row < ny and 0 <= col < nx:
                rows.append(row)
                cols.append(col)
                depths.append(d)

        return (np.array(rows, dtype=np.int64),
                np.array(cols, dtype=np.int64),
                np.array(depths, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────────────

class DebrisFlowDialog(QDialog):
    """
    Simulation dialog with four tabs:
      1. Layers and sources — DEM, line layer, source table
      2. O'Brien Parameters — rheology
      3. Solver Settings    — time parameters
      4. Output             — folder, prefix, layer selection
    """

    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface  = iface
        self.worker = None
        # Hydrograph storage: {feature_id (int): dict}
        self._hydro_data     = {}
        # id of the line layer the hydrographs belong to (fids are per layer)
        self._hydro_layer_id = None
        self.sweep_worker    = None
        self._sweep_results  = []
        self._sweep_best_pkg = None
        self.setWindowTitle(
            "Debris Flow Modelling — O'Brien Quadratic  (v1.0)")
        self.setMinimumWidth(660)
        self.resize(740, 580)
        self._build_ui()

    # ── Main layout ───────────────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_sources(),   "1. Layers and sources")
        self.tabs.addTab(self._tab_rheology(),  "2. O'Brien Parameters")
        self.tabs.addTab(self._tab_solver(),    "3. Solver Settings")
        self.tabs.addTab(self._tab_output(),    "4. Output")
        self.tabs.addTab(self._tab_sweep(),     "5. LHS analysis")
        main.addWidget(self.tabs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier", 9))
        self.log_box.setMaximumHeight(150)
        main.addWidget(self.log_box)

        btns = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run simulation")
        self.btn_run.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_run.clicked.connect(self._on_run)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_run)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setToolTip("Stop the running simulation at the next time step.")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btns.addWidget(self.btn_stop)
        btns.addWidget(btn_close)
        main.addLayout(btns)

    # ── Tab 1: Layers and sources ─────────────────────────────────────────────

    def _tab_sources(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # DEM
        grp_dem = QGroupBox("Digital Elevation Model (DEM)")
        fl_dem = QFormLayout(grp_dem)
        self.dem_combo = QgsMapLayerComboBox()
        self.dem_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        fl_dem.addRow("DEM layer:", self.dem_combo)
        layout.addWidget(grp_dem)

        # Source line layer
        grp_src = QGroupBox("Source line layer")
        lay_src = QVBoxLayout(grp_src)

        src_row = QHBoxLayout()
        self.src_line_combo = QgsMapLayerComboBox()
        self.src_line_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        src_row.addWidget(self.src_line_combo)
        btn_load = QPushButton("⟳ Load features")
        btn_load.setToolTip(
            "Read the features (lines) from the selected layer\n"
            "and fill the source table.")
        btn_load.clicked.connect(self._load_features)
        src_row.addWidget(btn_load)
        lay_src.addLayout(src_row)

        # Feature table
        info_lbl = QLabel(
            "Specify the hydrograph Q(t) for each line.\n"
            "Click \"Edit\" in the row of the desired feature.")
        info_lbl.setStyleSheet("color:#555; font-size:10px;")
        lay_src.addWidget(info_lbl)

        self.tbl_sources = QTableWidget()
        self.tbl_sources.setColumnCount(5)
        self.tbl_sources.setHorizontalHeaderLabels(
            ["ID", "Description", "Q_max (m³/s)", "V to t_end (m³)", "Hydrograph"])
        hdr = self.tbl_sources.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tbl_sources.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_sources.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_sources.setMinimumHeight(160)
        lay_src.addWidget(self.tbl_sources)

        layout.addWidget(grp_src)

        # ── Parameter loading ─────────────────────────────────────────────────
        grp_load = QGroupBox("Load parameters from file")
        grp_load.setStyleSheet(
            "QGroupBox{font-size:11px;color:#555;"
            "border:1px solid #bbb;border-radius:4px;margin-top:4px}"
            "QGroupBox::title{padding:0 4px}")
        load_row = QHBoxLayout(grp_load)
        load_row.setContentsMargins(6, 10, 6, 6)
        load_row.addWidget(QLabel("File:"))
        self.edit_load_path = QLineEdit()
        self.edit_load_path.setPlaceholderText("Path to JSON parameter file …")
        self.edit_load_path.setToolTip(
            "JSON file previously saved with the \"Save parameters\" button\n"
            "on the \"Output\" tab.\n"
            "Layers are matched by name in the current QGIS project.")
        load_row.addWidget(self.edit_load_path)
        btn_lb = QPushButton("…")
        btn_lb.setFixedWidth(28)
        btn_lb.clicked.connect(self._browse_load_file)
        load_row.addWidget(btn_lb)
        btn_lp = QPushButton("📂 Load")
        btn_lp.setToolTip(
            "Restore all parameters from a JSON file.\n"
            "Layers, hydrographs, rheology, solver settings\n"
            "and output parameters will be filled from the file.")
        btn_lp.clicked.connect(self._load_params)
        load_row.addWidget(btn_lp)
        layout.addWidget(grp_load)

        layout.addStretch()
        return w

    # ── Tab 2: O'Brien Parameters ──────────────────────────────────────────────

    def _tab_rheology(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Concentration
        grp_conc = QGroupBox("Sediment concentration")
        fl = QFormLayout(grp_conc)
        self.spin_Cv = QDoubleSpinBox()
        self.spin_Cv.setRange(0.01, 0.74); self.spin_Cv.setValue(0.40)
        self.spin_Cv.setDecimals(3)
        self.spin_Cv.setToolTip(
            "Volumetric sediment concentration [-].\n"
            "Post-fire range: 0.30–0.55")
        fl.addRow("Cv:", self.spin_Cv)
        self.spin_Cv_max = QDoubleSpinBox()
        self.spin_Cv_max.setRange(0.60, 0.90); self.spin_Cv_max.setValue(0.615)
        self.spin_Cv_max.setDecimals(3)
        self.spin_Cv_max.setToolTip(
            "Upper bound of the allowed Cv (input check only: Cv must be < Cv_max).\n"
            "It does not enter any formula and does not affect the result.\n"
            "Increase it only if Cv ≥ 0.615 is needed.")
        fl.addRow("Cv_max:", self.spin_Cv_max)
        layout.addWidget(grp_conc)

        # ── Component densities ───────────────────────────────────────────────
        grp_rho = QGroupBox("Component densities")
        fl_rho  = QFormLayout(grp_rho)

        self.spin_rho_s = QDoubleSpinBox()
        self.spin_rho_s.setRange(1000.0, 3500.0)
        self.spin_rho_s.setValue(2650.0)
        self.spin_rho_s.setDecimals(0)
        self.spin_rho_s.setSingleStep(10.0)
        self.spin_rho_s.setSuffix(" kg/m³")
        self.spin_rho_s.setToolTip(
            "Density of the solid phase ρs [kg/m³].\n\n"
            "Typical values:\n"
            "  2650 — quartz/granite (FLO-2D standard, Barnhart 2021)\n"
            "  2700–2800 — andesite, basalt\n"
            "  2600–2700 — limestone\n"
            "  1200–2200 — volcanic ash\n\n"
            "Barnhart et al. (2021): ρs is fixed at 2650 kg/m³,\n"
            "because the Morris (1991) screening showed insensitivity\n"
            "of the results to its variation.\n\n"
            "Affects: γm = ρm·g → τy/(γm·h), K·η/(8·γm·h²).")
        fl_rho.addRow("ρs — sediment:", self.spin_rho_s)

        self.spin_rho_w = QDoubleSpinBox()
        self.spin_rho_w.setRange(900.0, 1100.0)
        self.spin_rho_w.setValue(1000.0)
        self.spin_rho_w.setDecimals(0)
        self.spin_rho_w.setSingleStep(1.0)
        self.spin_rho_w.setSuffix(" kg/m³")
        self.spin_rho_w.setToolTip(
            "Density of the liquid phase ρw [kg/m³].\n\n"
            "  1000 — fresh water (standard)\n"
            "  1025 — sea water\n\n"
            "For all post-fire events in the mainland zone\n"
            "leave it at 1000 kg/m³.")
        fl_rho.addRow("ρw — liquid:", self.spin_rho_w)

        lbl_rho_note = QLabel(
            "ρm = ρs·Cv + ρw·(1–Cv)  →  γm = ρm·g  "
            "[used in the O'Brien formula]")
        lbl_rho_note.setWordWrap(True)
        lbl_rho_note.setStyleSheet("color:#666; font-size:10px; margin-top:2px")
        fl_rho.addRow(lbl_rho_note)

        # ρm indicator (recomputed when Cv, ρs, ρw change)
        self._lbl_rho_m = QLabel()
        self._lbl_rho_m.setStyleSheet("color:#1565C0; font-size:10px; font-weight:bold")
        fl_rho.addRow("ρm (current):", self._lbl_rho_m)
        self.spin_rho_s.valueChanged.connect(self._update_rho_m)
        self.spin_rho_w.valueChanged.connect(self._update_rho_m)

        layout.addWidget(grp_rho)

        # Mode switch
        grp_mode = QGroupBox("Mode for specifying τy and η")
        lay_mode = QVBoxLayout(grp_mode)
        self._radio_direct = QRadioButton("Direct input")
        self._radio_exp = QRadioButton(
            "Exponential coefficients")
        self._radio_direct.setChecked(True)
        self._mode_grp = QButtonGroup()
        self._mode_grp.addButton(self._radio_direct, 0)
        self._mode_grp.addButton(self._radio_exp, 1)
        lay_mode.addWidget(self._radio_direct)
        lay_mode.addWidget(self._radio_exp)
        layout.addWidget(grp_mode)

        # Mode 1: direct input
        self._sec_direct = QGroupBox("Direct parameters (Mode 1)")
        self._sec_direct.setStyleSheet(
            "QGroupBox{border:2px solid #2196F3;border-radius:4px;"
            "margin-top:6px;font-weight:bold}"
            "QGroupBox::title{color:#1565C0}")
        fl_d = QFormLayout(self._sec_direct)
        self.spin_tau = QDoubleSpinBox()
        self.spin_tau.setRange(0.0, 50000); self.spin_tau.setValue(600)
        self.spin_tau.setDecimals(2); self.spin_tau.setSuffix(" Pa")
        self.spin_tau.setSingleStep(50)
        self.spin_tau.setToolTip(
            "Yield stress [Pa] "
            "Post-fire debris flow: 200–800 Pa. 0 = no yield stress (purely viscous).")
        fl_d.addRow("τy — Yield stress:", self.spin_tau)
        self._lbl_hstop = QLabel()
        self._lbl_hstop.setStyleSheet("color:#555;font-size:10px;padding-left:4px")
        fl_d.addRow("", self._lbl_hstop)
        self.spin_tau.valueChanged.connect(self._update_hstop)
        self.spin_Cv.valueChanged.connect(self._update_hstop)
        self.spin_rho_s.valueChanged.connect(self._update_hstop)
        self.spin_rho_w.valueChanged.connect(self._update_hstop)
        self.spin_eta = QDoubleSpinBox()
        self.spin_eta.setRange(0.001, 10000); self.spin_eta.setValue(10.0)
        self.spin_eta.setDecimals(3); self.spin_eta.setSuffix(" Pa·s")
        self.spin_eta.setSingleStep(5)
        self.spin_eta.setToolTip(
            "Mixture viscosity [Pa·s] "
            "Post-fire debris flow: 5–30 Pa·s.")
        fl_d.addRow("η — Mixture viscosity:", self.spin_eta)
        layout.addWidget(self._sec_direct)

        # Mode 2: exponential coefficients
        self._sec_exp = QGroupBox("O'Brien exponential coefficients (Mode 2)")
        fl_e = QFormLayout(self._sec_exp)
        warn = QLabel(
            "⚠ The coefficients are calibrated on laboratory samples.\n"
            "For field conditions use Mode 1.")
        warn.setStyleSheet("color:#B26A00;font-size:10px")
        fl_e.addRow(warn)
        info = QLabel(
            "τy = α₁·exp(β₁·Cv),  η = α₂·exp(β₂·Cv)  (this plugin).\n"
            "In FLO-2D and O'Brien & Julien (1988) the indices are the other way\n"
            "round: η = α₁·exp(β₁·Cv), τy = α₂·exp(β₂·Cv) — do not copy α₁/β₁ blindly.\n"
            "The FLO-2D coefficient table has only the footnote \"1 Pa = 10 dynes/cm²;\n"
            "1 Pa·s = 10 poises\": if your α are in dynes/cm² and poises, multiply by 0.1.\n"
            "Default values: source not established; at Cv = 0.40 they give\n"
            "τy ≈ 0.8 Pa and η ≈ 26 Pa·s.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#555;font-size:10px")
        fl_e.addRow(info)
        self.spin_a1 = QDoubleSpinBox()
        self.spin_a1.setRange(0.0001,10); self.spin_a1.setValue(0.00423)
        self.spin_a1.setDecimals(5); self.spin_a1.setSingleStep(0.0001)
        self.spin_a1.setToolTip("Yield stress coefficient: τy = α₁·exp(β₁·Cv) [Pa]")
        fl_e.addRow("α₁ (τy) [Pa]:", self.spin_a1)
        self.spin_b1 = QDoubleSpinBox()
        self.spin_b1.setRange(1,30); self.spin_b1.setValue(13.11)
        self.spin_b1.setDecimals(2)
        self.spin_b1.setToolTip("Yield stress exponent β₁ (dimensionless)")
        fl_e.addRow("β₁ (τy) [-]:", self.spin_b1)
        self.spin_a2 = QDoubleSpinBox()
        self.spin_a2.setRange(0.001,10); self.spin_a2.setValue(0.0311)
        self.spin_a2.setDecimals(4); self.spin_a2.setSingleStep(0.001)
        self.spin_a2.setToolTip("Viscosity coefficient: η = α₂·exp(β₂·Cv) [Pa·s]")
        fl_e.addRow("α₂ (η) [Pa·s]:", self.spin_a2)
        self.spin_b2 = QDoubleSpinBox()
        self.spin_b2.setRange(1,40); self.spin_b2.setValue(16.81)
        self.spin_b2.setDecimals(2)
        self.spin_b2.setToolTip("Viscosity exponent β₂ (dimensionless)")
        fl_e.addRow("β₂ (η) [-]:", self.spin_b2)
        layout.addWidget(self._sec_exp)

        # Manning + K
        grp_n = QGroupBox("Turbulent and viscous terms")
        fl_n = QFormLayout(grp_n)
        self.spin_n = QDoubleSpinBox()
        self.spin_n.setRange(0.01, 0.30); self.spin_n.setValue(0.06)
        self.spin_n.setDecimals(3)
        self.spin_n.setToolTip(
            "Manning coefficient.\n"
            "Mountain channels after fire: 0.04–0.12")
        fl_n.addRow("Manning n:", self.spin_n)

        self.spin_K = QDoubleSpinBox()
        self.spin_K.setRange(0.001, 10000.0); self.spin_K.setValue(24.0)
        self.spin_K.setDecimals(3); self.spin_K.setSingleStep(1.0)
        self.spin_K.setToolTip(
            "K — dimensionless shape coefficient of the cross-section\n"
            "in the viscous term Sf_visc = K·η·V / (8·γm·h²).\n\n"
            "Theoretical value (O'Brien & Julien 1985):\n"
            "  K = 24 — wide rectangular channel (Bingham flow)\n\n"
            "FLO-2D Mudflow Guidelines: K grows with roughness and irregular\n"
            "cross-sections up to ~50 000; K = 2285 (calibrated on Rudd Creek)\n"
            "is used for most urban studies.\n"
            "Barnhart et al. (2021, Table 1): log K from 1.38 to 3.70.\n\n"
            "Default 24 = the smooth wide-channel value: the viscous resistance\n"
            "is then ~100× weaker than with the FLO-2D practice value 2285 —\n"
            "calibrate K or justify the choice.\n"
            "NB: in FLO-2D K = 0 means \"compute K from Manning n\"; here K = 0\n"
            "would switch the viscous term off (the field starts at 0.001).")
        fl_n.addRow("K — shape factor [-]:", self.spin_K)
        layout.addWidget(grp_n)

        self._radio_direct.toggled.connect(self._on_rheo_mode)
        self._on_rheo_mode()
        self._update_hstop()
        self.spin_Cv.valueChanged.connect(self._update_rho_m)
        self._update_rho_m()   # first calculation on open
        layout.addStretch()
        return w

    # ── Tab 3: Solver Settings ─────────────────────────────────────────────────

    def _tab_solver(self) -> QWidget:
        w = QWidget()
        layout = QFormLayout(w)

        self.spin_tend = QSpinBox()
        self.spin_tend.setRange(60, 86400); self.spin_tend.setValue(3600)
        self.spin_tend.valueChanged.connect(lambda _v: self._refresh_volumes())
        self.spin_tend.setToolTip(
            "Simulated time [s], counted from the start of the hydrographs.\n"
            "It must cover the hydrograph (the \"V to t_end\" column on the Layers\n"
            "tab shows how much is injected) and the time the flow needs to stop.\n"
            "The run also ends earlier if the flow has come to complete rest.")
        self.spin_tend.setSuffix(" s")
        layout.addRow("Simulation duration:", self.spin_tend)

        self.spin_dtmax = QDoubleSpinBox()
        self.spin_dtmax.setRange(0.1, 300); self.spin_dtmax.setValue(5.0)
        self.spin_dtmax.setSuffix(" s")
        self.spin_dtmax.setToolTip(
            "Maximum internal time step [s].\n"
            "The actual step is ≤ dt_max and additionally limited by CFL.\n"
            "Do NOT confuse with the progress-bar update interval (60 s).\n\n"
            "It matters while the domain is dry or very shallow: once the flow is\n"
            "established the CFL step is usually far smaller (for h = 1 m and a\n"
            "5 m grid about 1 s). On 30–90 m grids, however, the CFL step can exceed\n"
            "5 s, and then dt_max changes the result (test fan, 90 m grid: h_max\n"
            "differs by up to 1.7 m between dt_max = 60 s and 5 s). The first step\n"
            "is also CFL-checked with the depth the inflow adds.\n"
            "Check: halve dt_max and compare h_max.")
        layout.addRow("Max. step dt_max:", self.spin_dtmax)

        self.spin_cfl = QDoubleSpinBox()
        self.spin_cfl.setRange(0.10, 0.70); self.spin_cfl.setValue(0.60)
        self.spin_cfl.setDecimals(2); self.spin_cfl.setSingleStep(0.05)
        self.spin_cfl.setToolTip(
            "Courant–Friedrichs–Lewy number (CFL).\n"
            "Controls the numerical stability of the explicit LIA scheme.\n\n"
            "Recommended values:\n"
            "  0.40–0.50 — conservative (plugin recommendation)\n"
            "  0.60      — standard (default)\n"
            "Allowed range 0.10–0.70: Bates et al. (2010, eq. 14) give a\n"
            "coefficient in the range 0.2–0.7.")
        layout.addRow("CFL number:", self.spin_cfl)

        self.chk_face_lim = QCheckBox("Face-depth limiter  hf ≤ ½(hL+hR)")
        self.chk_face_lim.setChecked(True)
        self.chk_face_lim.setToolTip(
            "Limits the flow depth on a cell face to the mean depth of the two cells.\n"
            "This is the plugin's own modification: no literature source was found\n"
            "(the LIA papers use hf = max(η) − max(z) without it).\n"
            "Slows the front on dry slopes (test fan: arrival time with the limiter\n"
            "8–13 % later at K = 2285 and 15–18 % later at K = 24)\n"
            "and amplifies pulsations in steep channels.\n"
            "ON by default — as in the published runs.")
        layout.addRow(self.chk_face_lim)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setRange(0.50, 1.00); self.spin_theta.setValue(1.00)
        self.spin_theta.setDecimals(2); self.spin_theta.setSingleStep(0.05)
        self.spin_theta.setToolTip(
            "θ-weighting of the discharge (de Almeida et al. 2012):\n"
            "q* = θ·q + (1−θ)/2·(q_prev + q_next).\n"
            "Damps numerical oscillations of the LIA scheme in steep channels\n"
            "with low friction (velocities stuck at the 30 m/s cap are a symptom).\n\n"
            "1.00 — original Bates et al. (2010) scheme, as in the published runs.\n"
            "0.80 — Landlab default; 0.7–0.9 is the usual range.")
        layout.addRow("θ-weighting:", self.spin_theta)

        lbl = QLabel(
            "<i>h_min = 5 mm is fixed automatically for LIA stability.</i>")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#666;font-size:10px")
        layout.addRow(lbl)
        return w

    # ── Tab 4: Output ──────────────────────────────────────────────────────────

    def _tab_output(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # File location
        grp_loc = QGroupBox("Output location")
        fl = QFormLayout(grp_loc)

        self.edit_prefix = QLineEdit("Debris")
        self.edit_prefix.setToolTip(
            "Base name of the output files.\n"
            "Result: <prefix>_h_max.tif, <prefix>_V_max.tif, ...")
        fl.addRow("File prefix:", self.edit_prefix)

        out_row = QHBoxLayout()
        self.edit_outdir = QLineEdit(os.path.expanduser("~/debris_output"))
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.edit_outdir)
        out_row.addWidget(btn_browse)
        fl.addRow("Output folder:", out_row)
        layout.addWidget(grp_loc)

        # Output layer selection
        grp_layers = QGroupBox("Output layers (select the ones you need)")
        fl_l = QVBoxLayout(grp_layers)

        self.chk_h_max = QCheckBox("Peak flow thickness  (h_max)")
        self.chk_h_max.setChecked(True)
        self.chk_h_max.setToolTip(
            "Maximum depth over the whole simulation.\n"
            "The main product for hazard assessment.")
        fl_l.addWidget(self.chk_h_max)

        self.chk_V_max = QCheckBox(
            "Peak flow velocity  (V_max)")
        self.chk_V_max.setChecked(True)
        self.chk_V_max.setToolTip(
            "Maximum flow velocity [m/s] over the whole time.\n"
            "Used for assessing the dynamic load.")
        fl_l.addWidget(self.chk_V_max)

        self.chk_HV2_max = QCheckBox(
            "Peak momentum flux  h·v²  (HV2_max)")
        self.chk_HV2_max.setChecked(False)
        self.chk_HV2_max.setToolTip(
            "Maximum over time of the instantaneous product h(t)·v(t)² [m³/s²].\n"
            "Use it for impact assessment (force per unit width ≈ ρm·h·v²).\n"
            "Not the same as h_max·V_max²: that product combines two separate\n"
            "envelopes (front velocity × body depth) and overestimates.")
        fl_l.addWidget(self.chk_HV2_max)

        self.chk_Fr_hv2 = QCheckBox(
            "Froude number at peak h·v²  (Fr_hv2)")
        self.chk_Fr_hv2.setChecked(False)
        self.chk_Fr_hv2.setToolTip(
            "Fr = V/√(g·h) at the moment of the peak momentum flux h·v² in each cell.\n"
            "Shows where the local inertial approximation (LIA) is used outside\n"
            "its validity range: good agreement with the full equations only for\n"
            "Fr < 0.5, growing errors above (de Almeida & Bates 2013).\n"
            "Diagnostic only. A plain maximum of Fr is not used: it is set by the\n"
            "thin film at the front and exceeds 1 almost everywhere.")
        fl_l.addWidget(self.chk_Fr_hv2)

        self.chk_h_final = QCheckBox(
            "Final flow thickness  (h_final)")
        self.chk_h_final.setChecked(False)
        self.chk_h_final.setToolTip(
            "Deposit thickness at the moment the simulation ends.\n"
            "Shows where the flow stopped.")
        fl_l.addWidget(self.chk_h_final)

        self.chk_t_arrival = QCheckBox(
            "Flow arrival time  (t_arrival)")
        self.chk_t_arrival.setChecked(False)
        self.chk_t_arrival.setToolTip(
            "The time [s] when the flow first reached a cell.\n"
            "Used for evacuation planning.")
        fl_l.addWidget(self.chk_t_arrival)
        arr_row = QHBoxLayout()
        arr_row.addWidget(QLabel("   arrival when depth ≥"))
        self.spin_arrival_thr = QDoubleSpinBox()
        self.spin_arrival_thr.setRange(0.005, 5.0); self.spin_arrival_thr.setValue(0.005)
        self.spin_arrival_thr.setDecimals(3); self.spin_arrival_thr.setSingleStep(0.05)
        self.spin_arrival_thr.setSuffix(" m")
        self.spin_arrival_thr.setToolTip(
            "Depth at which a cell counts as reached for t_arrival.\n"
            "0.005 m = the 5 mm wet film (previous behaviour). For hazard maps\n"
            "a meaningful depth (e.g. 0.1–0.3 m) is often more useful.\n"
            "Set before the run: t_arrival is recorded during the simulation.")
        arr_row.addWidget(self.spin_arrival_thr); arr_row.addStretch()
        fl_l.addLayout(arr_row)

        layout.addWidget(grp_layers)

        # Depth threshold (filtering of depth layers on write)
        grp_thresh = QGroupBox("Depth threshold — filtering when writing the raster")
        fl_th = QFormLayout(grp_thresh)

        self.chk_threshold = QCheckBox(
            "Show only cells with depth ≥ threshold")
        self.chk_threshold.setChecked(False)
        self.chk_threshold.setToolTip(
            "If enabled — cells with depth below the threshold\n"
            "are written as NoData (transparent in QGIS).\n"
            "Applies to the h_max and h_final layers; in V_max, h·v² and Fr\n"
            "the cells where h_max is below the threshold are masked too.")
        self.chk_threshold.toggled.connect(
            lambda on: self.spin_threshold.setEnabled(on))
        fl_th.addRow(self.chk_threshold)

        thresh_row = QHBoxLayout()
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.00, 10.0)
        self.spin_threshold.setValue(0.10)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setSuffix(" m")
        self.spin_threshold.setEnabled(False)
        self.spin_threshold.setToolTip(
            "Minimum depth [m].\n"
            "Cells with h < threshold → NoData (transparent in QGIS).\n\n"
            "Examples:\n"
            "  0.01 m — remove the thin film at the front\n"
            "  0.10 m — only significant flooding zones\n"
            "  0.30 m — hazard threshold for pedestrians\n"
            "  1.00 m — hazard threshold for buildings")
        thresh_row.addWidget(self.spin_threshold)

        for label, val in [("1 cm", 0.01), ("10 cm", 0.10),
                            ("30 cm", 0.30), ("1 m", 1.00)]:
            btn = QPushButton(label)
            btn.setFixedWidth(46)
            btn.setToolTip(f"Set threshold to {val} m")
            btn.clicked.connect(
                lambda checked, v=val: (
                    self.spin_threshold.setValue(v),
                    self.chk_threshold.setChecked(True),
                )
            )
            thresh_row.addWidget(btn)

        fl_th.addRow("Threshold:", thresh_row)

        lbl_note = QLabel(
            "<i>Does not affect the simulation — only the raster output.<br>"
            "The full data inside the plugin is kept unchanged.</i>")
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color:#666; font-size:10px;")
        fl_th.addRow(lbl_note)

        layout.addWidget(grp_thresh)

        # ── Parameter saving ───────────────────────────────────────────────────
        grp_save = QGroupBox("Save parameters to file")
        grp_save.setStyleSheet(
            "QGroupBox{font-size:11px;color:#555;"
            "border:1px solid #bbb;border-radius:4px;margin-top:4px}"
            "QGroupBox::title{padding:0 4px}")
        save_row = QHBoxLayout(grp_save)
        save_row.setContentsMargins(6, 10, 6, 6)
        save_row.addWidget(QLabel("File:"))
        self.edit_save_path = QLineEdit(
            os.path.join(os.path.expanduser("~"), "debris_params.json"))
        self.edit_save_path.setToolTip(
            "Path to the JSON file for saving all parameters:\n"
            "layers, hydrographs, rheology, solver, output.")
        save_row.addWidget(self.edit_save_path)
        btn_sb = QPushButton("…")
        btn_sb.setFixedWidth(28)
        btn_sb.clicked.connect(self._browse_save_file)
        save_row.addWidget(btn_sb)
        btn_sp = QPushButton("💾 Save")
        btn_sp.setToolTip(
            "Save all current parameters to a JSON file.\n"
            "The file can be loaded on the \"Layers and sources\" tab.")
        btn_sp.clicked.connect(self._save_params)
        save_row.addWidget(btn_sp)
        layout.addWidget(grp_save)

        layout.addStretch()
        return w

    # ── Tab 5: LHS analysis ────────────────────────────────────────────────────

    def _tab_sweep(self) -> QWidget:
        w  = QWidget()
        lo = QVBoxLayout(w)

        # ── Saving / loading LHS settings ───────────────────────────────────
        grp_lhs_io = QGroupBox("Analysis settings")
        grp_lhs_io.setStyleSheet(
            "QGroupBox{"
            "  border:1px solid #90CAF9;"
            "  border-radius:6px;"
            "  background:#F3F8FF;"
            "  margin-top:8px;"
            "  font-weight:bold;"
            "}"
            "QGroupBox::title{"
            "  color:#1565C0;"
            "  padding:0 6px;"
            "}")
        io_lo = QVBoxLayout(grp_lhs_io)
        io_lo.setSpacing(4)

        io_path_row = QHBoxLayout()
        lbl_io = QLabel("File:")
        lbl_io.setFixedWidth(34)
        io_path_row.addWidget(lbl_io)
        self.edit_lhs_path = QLineEdit()
        self.edit_lhs_path.setPlaceholderText(
            "Path to the LHS analysis settings file (.json)  …")
        self.edit_lhs_path.setToolTip(
            "JSON file for saving/loading all parameters of the\n"
            "LHS analysis tab: observed data, ranges,\n"
            "number of runs, seed and output folder.")
        io_path_row.addWidget(self.edit_lhs_path)
        btn_lhs_br = QPushButton("…")
        btn_lhs_br.setFixedWidth(28)
        btn_lhs_br.setToolTip("Choose file")
        btn_lhs_br.clicked.connect(self._browse_lhs_file)
        io_path_row.addWidget(btn_lhs_br)
        io_lo.addLayout(io_path_row)

        io_btn_row = QHBoxLayout()
        btn_lhs_save = QPushButton("💾  Save LHS settings")
        btn_lhs_save.setToolTip(
            "Save all parameters of the LHS tab to a JSON file.\n"
            "Convenient for storing different analysis scenarios.")
        btn_lhs_save.clicked.connect(self._save_lhs_params)
        btn_lhs_load = QPushButton("📂  Load LHS settings")
        btn_lhs_load.setToolTip(
            "Restore LHS analysis parameters from a JSON file.\n"
            "Layers are matched by name in the current QGIS project.")
        btn_lhs_load.clicked.connect(self._load_lhs_params)
        io_btn_row.addWidget(btn_lhs_save)
        io_btn_row.addWidget(btn_lhs_load)
        io_lo.addLayout(io_btn_row)
        lo.addWidget(grp_lhs_io)

        # ── Observed data ─────────────────────────────────────────────────────
        grp_obs = QGroupBox("Observed data (real event)")
        fl_obs  = QFormLayout(grp_obs)

        self.combo_obs_poly = QgsMapLayerComboBox()
        self.combo_obs_poly.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.combo_obs_poly.setToolTip(
            "SHP layer of the real inundation zone (polygon).\n"
            "Used for computing TP, FP, FN, ΩTm.")
        fl_obs.addRow("Inundation zone (SHP):", self.combo_obs_poly)

        self.combo_eval_poly = QgsMapLayerComboBox()
        self.combo_eval_poly.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.combo_eval_poly.setAllowEmptyLayer(True)
        self.combo_eval_poly.setLayer(None)
        self.combo_eval_poly.setToolTip(
            "Optional polygon of the area where the inundation was mapped.\n"
            "Cells outside it are ignored in TP/FP/FN, so that flow in channels\n"
            "upstream of the mapped zone is not counted as over-prediction.\n"
            "Empty = the whole DEM (previous behaviour).\n"
            "⚠ This is the polygon of the MAPPED territory, not the inundation polygon:\n"
            "choosing the inundation polygon itself here makes FP ≡ 0.")
        fl_obs.addRow("Evaluation area (optional):", self.combo_eval_poly)

        self.combo_obs_pts = QgsMapLayerComboBox()
        self.combo_obs_pts.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.combo_obs_pts.setToolTip(
            "SHP layer of points with field measurements of peak flow depths.")
        self.combo_obs_pts.layerChanged.connect(self._on_obs_pts_changed)
        fl_obs.addRow("Depth points (SHP):", self.combo_obs_pts)

        self.combo_depth_field = QComboBox()
        self.combo_depth_field.setToolTip(
            "Numeric attribute of the point layer containing\n"
            "the peak flow depth value [m].")
        fl_obs.addRow("Depth field [m]:", self.combo_depth_field)
        # the point combo may already hold a layer (picked before layerChanged was connected)
        self._on_obs_pts_changed(self.combo_obs_pts.currentLayer())

        self.spin_depth_thr = QDoubleSpinBox()
        self.spin_depth_thr.setRange(0.01, 5.0)
        self.spin_depth_thr.setValue(0.5)
        self.spin_depth_thr.setSuffix(" m")
        self.spin_depth_thr.setDecimals(2)
        self.spin_depth_thr.setToolTip(
            "Threshold h [m] for classifying a cell as \"inundated\".\n"
            "Barnhart et al. (2021) justify 0.5 m (Section 7.2).")
        fl_obs.addRow("Depth threshold:", self.spin_depth_thr)
        lo.addWidget(grp_obs)

        # ── Parameters: range or fixed value ─────────────────────────────────
        grp_rng = QGroupBox(
            "Variation parameters — range or fixed value")
        rng_lo = QVBoxLayout(grp_rng)
        rng_lo.setSpacing(4)

        def _param_row(label, chk_attr, fix_attr, mn_attr, mx_attr,
                       fix_def, lo_def, hi_def, suffix, decs, step, rng):
            """Parameter row with a range/fixed toggle."""
            cw = QWidget()
            ch = QHBoxLayout(cw); ch.setContentsMargins(0, 0, 0, 0)

            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.setMinimumWidth(175)
            setattr(self, chk_attr, chk)
            ch.addWidget(chk)

            # Range widget
            rw = QWidget()
            rh = QHBoxLayout(rw); rh.setContentsMargins(0, 0, 0, 0)
            s_min = QDoubleSpinBox()
            s_min.setRange(*rng); s_min.setValue(lo_def)
            s_min.setDecimals(decs); s_min.setSuffix(suffix)
            s_min.setSingleStep(step)
            setattr(self, mn_attr, s_min)
            s_max = QDoubleSpinBox()
            s_max.setRange(*rng); s_max.setValue(hi_def)
            s_max.setDecimals(decs); s_max.setSuffix(suffix)
            s_max.setSingleStep(step)
            setattr(self, mx_attr, s_max)
            rh.addWidget(QLabel("from")); rh.addWidget(s_min)
            rh.addWidget(QLabel("to")); rh.addWidget(s_max)
            ch.addWidget(rw)

            # Fixed-value widget
            fw = QWidget()
            fh = QHBoxLayout(fw); fh.setContentsMargins(0, 0, 0, 0)
            fh.addWidget(QLabel("Value:"))
            s_fix = QDoubleSpinBox()
            s_fix.setRange(*rng); s_fix.setValue(fix_def)
            s_fix.setDecimals(decs); s_fix.setSuffix(suffix)
            s_fix.setSingleStep(step)
            setattr(self, fix_attr, s_fix)
            fh.addWidget(s_fix)
            fw.setVisible(False)
            ch.addWidget(fw)

            def _tog(on, _rw=rw, _fw=fw):
                _rw.setVisible(on); _fw.setVisible(not on)
            chk.toggled.connect(_tog)
            rng_lo.addWidget(cw)

        _param_row("Vary volume V",
                   "chk_vol_vary", "spin_vol_fixed",
                   "spin_vol_min",  "spin_vol_max",
                   fix_def=100_000, lo_def=50_000, hi_def=2_000_000,
                   suffix=" m³", decs=0, step=10_000,
                   rng=(1000, 1e8))

        _param_row("Vary τy",
                   "chk_tau_vary", "spin_tau_fixed",
                   "spin_tau_min",  "spin_tau_max",
                   fix_def=600.0, lo_def=100.0, hi_def=1200.0,
                   suffix=" Pa", decs=2, step=50,
                   rng=(0.01, 50000))

        _param_row("Vary η",
                   "chk_eta_vary", "spin_eta_fixed",
                   "spin_eta_min",  "spin_eta_max",
                   fix_def=10.0, lo_def=2.0, hi_def=50.0,
                   suffix=" Pa·s", decs=2, step=1.0,
                   rng=(0.001, 10000))

        _param_row("Vary Manning n",
                   "chk_n_vary",  "spin_n_fixed",
                   "spin_n_min",  "spin_n_max",
                   fix_def=0.06, lo_def=0.04, hi_def=0.15,
                   suffix="", decs=4, step=0.01,
                   rng=(0.001, 0.30))

        _param_row("Vary K — shape factor",
                   "chk_K_vary",  "spin_K_fixed",
                   "spin_K_min",  "spin_K_max",
                   fix_def=24.0, lo_def=24.0, hi_def=2400.0,
                   suffix="", decs=3, step=1.0,
                   rng=(0.001, 10000.0))

        # Fixed values are set directly on this tab.
        # (Cv is taken from the "O'Brien Parameters" tab.)
        note_rng = QLabel(
            "Checked — the parameter varies within a range (LHS sampling).  "
            "Unchecked — the parameter is fixed, one value for all runs.\n"
            "When a parameter is unchecked, the value is taken from its \"Value:\" "
            "field on this tab; the \"O'Brien Parameters\" tab is not used for it.")
        note_rng.setWordWrap(True)
        note_rng.setStyleSheet("color:#777;font-size:10px;margin-top:2px")
        rng_lo.addWidget(note_rng)
        lo.addWidget(grp_rng)

        # ── Sweep settings ─────────────────────────────────────────────────────
        grp_cfg = QGroupBox("Analysis settings")
        fl_cfg  = QFormLayout(grp_cfg)

        self.spin_n_runs = QSpinBox()
        self.spin_n_runs.setRange(10, 1000); self.spin_n_runs.setValue(100)
        self.spin_n_runs.setToolTip(
            "Number of LHS runs.\n"
            "Barnhart et al.: 100·Np ≈ 300–500.\n"
            "For an initial analysis: 50–100.")
        fl_cfg.addRow("Number of runs:", self.spin_n_runs)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999); self.spin_seed.setValue(42)
        self.spin_seed.setToolTip(
            "Generator seed — the same seed → reproducible result.")
        fl_cfg.addRow("Random seed:", self.spin_seed)

        # Parallel workers
        self.spin_n_workers = QSpinBox()
        self.spin_n_workers.setRange(1, 16)
        self.spin_n_workers.setValue(1)
        self.spin_n_workers.setToolTip(
            "Number of parallel threads for the LHS analysis.\n\n"
            "1 — sequential mode (recommended on the first run).\n"
            "2–8 — parallel threads (ThreadPoolExecutor).\n\n"
            "Threads run inside the QGIS process — no launching of\n"
            "child processes is needed. With Numba JIT the threads really speed\n"
            "up the computation, because njit releases the GIL during execution.\n\n"
            "Do not exceed the number of physical cores of your CPU.\n"
            "Maximum: 16.")
        fl_cfg.addRow("Parallel workers:", self.spin_n_workers)

        out_row2 = QHBoxLayout()
        self.edit_sweep_outdir = QLineEdit(os.path.expanduser("~/debris_sweep"))
        btn_sweep_out = QPushButton("…")
        btn_sweep_out.setFixedWidth(30)
        btn_sweep_out.clicked.connect(self._browse_sweep_outdir)
        out_row2.addWidget(self.edit_sweep_outdir)
        out_row2.addWidget(btn_sweep_out)
        fl_cfg.addRow("Output folder:", out_row2)

        self.edit_sweep_csv = QLineEdit("sweep_results.csv")
        self.edit_sweep_csv.setToolTip(
            "Name of the CSV file (saved in the output folder).\n"
            "Contains all 5 parameters (V, τy, η, n, K) + metrics for each scenario.")
        fl_cfg.addRow("CSV file:", self.edit_sweep_csv)
        lo.addWidget(grp_cfg)

        # ── Sweep log ───────────────────────────────────────────────────────────
        self.sweep_log_box = QTextEdit()
        self.sweep_log_box.setReadOnly(True)
        self.sweep_log_box.setFont(QFont("Courier", 9))
        self.sweep_log_box.setMaximumHeight(130)
        lo.addWidget(self.sweep_log_box)

        # ── Buttons ──────────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_sweep_run = QPushButton("▶  Run LHS analysis")
        self.btn_sweep_run.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_sweep_run.clicked.connect(self._on_sweep_run)
        self.btn_sweep_cancel = QPushButton("⏹  Stop")
        self.btn_sweep_cancel.setEnabled(False)
        self.btn_sweep_cancel.clicked.connect(self._on_sweep_cancel)
        btn_row.addWidget(self.btn_sweep_run)
        btn_row.addWidget(self.btn_sweep_cancel)
        lo.addLayout(btn_row)

        # Wrap in a QScrollArea to work on laptops
        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    # ── Tab 5 slots ────────────────────────────────────────────────────────────

    def _on_obs_pts_changed(self, layer):
        """Fill combo_depth_field with the numeric fields of the new point layer."""
        self.combo_depth_field.clear()
        if layer is None:
            return
        # isNumeric() also covers Integer64 (GPKG/SHP); the feature id "fid" is never a depth
        for field in layer.fields():
            if field.isNumeric() and field.name().lower() != "fid":
                self.combo_depth_field.addItem(field.name())
        # Fallback: add all fields if there are no numeric ones
        if self.combo_depth_field.count() == 0:
            for field in layer.fields():
                self.combo_depth_field.addItem(field.name())

    def _browse_lhs_file(self):
        current = self.edit_lhs_path.text().strip()
        start = os.path.dirname(current) if current else os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self, "LHS analysis settings file",
            os.path.join(start, "lhs_settings.json"),
            "JSON settings (*.json);;All Files (*)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.edit_lhs_path.setText(path)

    def _save_lhs_params(self):
        path = self.edit_lhs_path.text().strip()
        if not path:
            QMessageBox.warning(
                self, "No path",
                "Specify a file path before saving.")
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        def _layer_name(combo):
            layer = combo.currentLayer()
            return layer.name() if layer else ""

        payload = {
            "version":         "lhs_1.0",
            "obs_poly_name":   _layer_name(self.combo_obs_poly),
            "eval_poly_name":  _layer_name(self.combo_eval_poly),
            "obs_pts_name":    _layer_name(self.combo_obs_pts),
            "depth_field":     self.combo_depth_field.currentText(),
            "depth_threshold": self.spin_depth_thr.value(),
            # Variation parameters
            "vol_vary":  self.chk_vol_vary.isChecked(),
            "vol_min":   self.spin_vol_min.value(),
            "vol_max":   self.spin_vol_max.value(),
            "vol_fixed": self.spin_vol_fixed.value(),
            "tau_vary":  self.chk_tau_vary.isChecked(),
            "tau_min":   self.spin_tau_min.value(),
            "tau_max":   self.spin_tau_max.value(),
            "tau_fixed": self.spin_tau_fixed.value(),
            "eta_vary":  self.chk_eta_vary.isChecked(),
            "eta_min":   self.spin_eta_min.value(),
            "eta_max":   self.spin_eta_max.value(),
            "eta_fixed": self.spin_eta_fixed.value(),
            "n_vary":    self.chk_n_vary.isChecked(),
            "n_min":     self.spin_n_min.value(),
            "n_max":     self.spin_n_max.value(),
            "n_fixed":   self.spin_n_fixed.value(),
            "K_vary":    self.chk_K_vary.isChecked(),
            "K_min":     self.spin_K_min.value(),
            "K_max":     self.spin_K_max.value(),
            "K_fixed":   self.spin_K_fixed.value(),
            # Analysis settings
            "n_runs":    self.spin_n_runs.value(),
            "n_workers": self.spin_n_workers.value(),
            "seed":      self.spin_seed.value(),
            "out_dir":   self.edit_sweep_outdir.text(),
            "csv_name":  self.edit_sweep_csv.text(),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            QMessageBox.information(
                self, "LHS settings saved",
                f"LHS analysis settings saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Save error", f"Failed to write the file:\n{e}")

    def _load_lhs_params(self):
        path = self.edit_lhs_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load LHS settings",
                os.path.expanduser("~"),
                "JSON settings (*.json);;All Files (*)")
            if not path:
                return
            self.edit_lhs_path.setText(path)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "File not found",
                                f"The file does not exist:\n{path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Read error",
                                 f"Failed to read the JSON:\n{e}")
            return

        warnings = []
        from qgis.core import QgsProject

        def _set_layer(combo, name):
            if not name:
                return
            m = QgsProject.instance().mapLayersByName(name)
            if m:
                combo.setLayer(m[0])
            else:
                warnings.append(f"Layer \"{name}\" not found in the project.")

        _set_layer(self.combo_obs_poly, d.get("obs_poly_name", ""))
        if d.get("eval_poly_name"):
            _set_layer(self.combo_eval_poly, d["eval_poly_name"])
        else:
            # optional layer: an empty name means "no evaluation area", not "keep the old one"
            self.combo_eval_poly.setLayer(None)
        _set_layer(self.combo_obs_pts,  d.get("obs_pts_name",  ""))
        if d.get("depth_field"):
            idx = self.combo_depth_field.findText(d["depth_field"])
            if idx >= 0:
                self.combo_depth_field.setCurrentIndex(idx)
        if "depth_threshold" in d:
            self.spin_depth_thr.setValue(d["depth_threshold"])

        def _sv_chk(chk, spin_min, spin_max, spin_fix,
                    key_vary, key_min, key_max, key_fix):
            if key_vary in d:  chk.setChecked(d[key_vary])
            if key_min  in d:  spin_min.setValue(d[key_min])
            if key_max  in d:  spin_max.setValue(d[key_max])
            if key_fix  in d:  spin_fix.setValue(d[key_fix])

        _sv_chk(self.chk_vol_vary, self.spin_vol_min, self.spin_vol_max,
                self.spin_vol_fixed, "vol_vary","vol_min","vol_max","vol_fixed")
        _sv_chk(self.chk_tau_vary, self.spin_tau_min, self.spin_tau_max,
                self.spin_tau_fixed, "tau_vary","tau_min","tau_max","tau_fixed")
        _sv_chk(self.chk_eta_vary, self.spin_eta_min, self.spin_eta_max,
                self.spin_eta_fixed, "eta_vary","eta_min","eta_max","eta_fixed")
        _sv_chk(self.chk_n_vary,   self.spin_n_min,   self.spin_n_max,
                self.spin_n_fixed,   "n_vary","n_min","n_max","n_fixed")
        _sv_chk(self.chk_K_vary,   self.spin_K_min,   self.spin_K_max,
                self.spin_K_fixed,   "K_vary","K_min","K_max","K_fixed")

        if "n_runs"  in d: self.spin_n_runs.setValue(int(d["n_runs"]))
        if "n_workers" in d: self.spin_n_workers.setValue(int(d["n_workers"]))
        if "seed"    in d: self.spin_seed.setValue(int(d["seed"]))
        if "out_dir" in d: self.edit_sweep_outdir.setText(d["out_dir"])
        if "csv_name" in d: self.edit_sweep_csv.setText(d["csv_name"])

        msg = "LHS settings loaded."
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(f"• {w}" for w in warnings)
            QMessageBox.warning(self, "LHS loaded (with warnings)", msg)
        else:
            QMessageBox.information(self, "LHS settings loaded", msg)

    def _browse_sweep_outdir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Folder for LHS analysis results",
            self.edit_sweep_outdir.text())
        if path:
            self.edit_sweep_outdir.setText(path)

    def _on_sweep_cancel(self):
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.cancel()
            self.sweep_log_box.append(
                "⚠ Cancelling — the running simulation(s) stop at the next time step...")
            self.btn_sweep_cancel.setEnabled(False)

    def _on_sweep_run(self):
        """Validation, parameter assembly and launch of SweepWorker."""
        # ── Numba JIT check ────────────────────────────────────────────────────
        _nb_ok, _nb_reason = self._check_numba()
        if not _nb_ok:
            QMessageBox.critical(
                self, "Numba JIT not found",
                "Numba JIT is required to run the LHS analysis.\n\n"
                + _nb_reason)
            return

        dem_layer = self.dem_combo.currentLayer()
        src_layer = self.src_line_combo.currentLayer()
        obs_poly  = self.combo_obs_poly.currentLayer()
        obs_pts   = self.combo_obs_pts.currentLayer()
        depth_fld = self.combo_depth_field.currentText()

        if dem_layer is None:
            QMessageBox.warning(
                self, "Error", "Select a DEM (Tab 1)."); return
        if src_layer is None:
            QMessageBox.warning(
                self, "Error",
                "Select a source line layer (Tab 1)."); return
        if self.tbl_sources.rowCount() == 0:
            QMessageBox.warning(
                self, "Error",
                "Load the sources — click \"Load features\" (Tab 1)."); return
        if src_layer.id() != self._hydro_layer_id:
            QMessageBox.warning(
                self, "Error",
                "The source table belongs to another line layer.\n"
                "Click \"Load features\" for the selected layer (Tab 1)."); return
        if obs_poly is None:
            QMessageBox.warning(
                self, "Error",
                "Select the polygon layer of the observed inundation."); return
        if obs_pts is None:
            QMessageBox.warning(
                self, "Error",
                "Select the point layer of field depth measurements."); return
        if not depth_fld:
            QMessageBox.warning(
                self, "Error",
                "Select the depth field [m] in the point layer."); return

        # Range validation (only for the varied parameters)
        _crs_msg = _check_dem_crs(dem_layer)
        if _crs_msg:
            QMessageBox.warning(self, "CRS", _crs_msg); return
        Cv = self.spin_Cv.value(); Cv_max = self.spin_Cv_max.value()
        if Cv >= Cv_max:
            # otherwise every run fails one by one with a ValueError
            QMessageBox.warning(self, "Parameter",
                                f"Cv ({Cv}) must be < Cv_max ({Cv_max}).\n"
                                "(\"O'Brien Parameters\" tab)"); return

        err = []
        if self.chk_vol_vary.isChecked() and \
                self.spin_vol_min.value() >= self.spin_vol_max.value():
            err.append("V_min must be less than V_max")
        if self.chk_tau_vary.isChecked() and \
                self.spin_tau_min.value() >= self.spin_tau_max.value():
            err.append("τy_min must be less than τy_max")
        if self.chk_eta_vary.isChecked() and \
                self.spin_eta_min.value() >= self.spin_eta_max.value():
            err.append("η_min must be less than η_max")
        if self.chk_n_vary.isChecked() and \
                self.spin_n_min.value() >= self.spin_n_max.value():
            err.append("n_min must be less than n_max")
        if self.chk_K_vary.isChecked() and \
                self.spin_K_min.value() >= self.spin_K_max.value():
            err.append("K_min must be less than K_max")
        if err:
            QMessageBox.warning(self, "Range error",
                                 "\n".join(err)); return

        out_dir = self.edit_sweep_outdir.text().strip()
        if not out_dir:
            QMessageBox.warning(
                self, "Error", "Specify an output folder."); return
        csv_name = self.edit_sweep_csv.text().strip() or "sweep_results.csv"
        if os.path.exists(os.path.join(out_dir, csv_name)):
            # the CSV, <csv>_context.json and best_h_max_<csv>.tif of the previous analysis
            # would be overwritten without a trace
            if QMessageBox.question(
                    self, "File exists",
                    f"{os.path.join(out_dir, csv_name)} already exists.\n"
                    "Overwrite the results of the previous LHS analysis\n"
                    "(CSV, _context.json and the best h_max raster)?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return

        # Assemble sources from the Tab 1 table
        sources = []
        for row in range(self.tbl_sources.rowCount()):
            fid_item = self.tbl_sources.item(row, 0)
            if fid_item is None:
                continue
            try:
                fid = int(fid_item.text())
            except ValueError:
                continue
            name_item = self.tbl_sources.item(row, 1)
            fname = name_item.text() if name_item else f"source_{fid}"
            data  = self._hydro_data.get(fid)
            if data is None:
                continue
            sources.append({
                "fid":    fid, "name": fname,
                "wkt":    _feature_wkt(src_layer, fid, dem_layer.crs()),
                "times":  data["times"],
                "q_vals": data["q_vals"],
            })

        if not sources:
            QMessageBox.warning(
                self, "Error",
                "No sources with hydrographs.\n"
                "Specify hydrographs on Tab 1."); return

        # LHS ranges — lo==hi for a fixed parameter
        import math

        def _rng(chk, s_min, s_max, s_fix):
            if chk.isChecked():
                return (s_min.value(), s_max.value())
            v = s_fix.value()
            return (v, v)

        # η and K span orders of magnitude (K: 24…2400) → log-uniform sampling,
        # otherwise ~90 % of the K samples would lie above 260
        def _log_rng(chk, s_min, s_max, s_fix):
            lo, hi = _rng(chk, s_min, s_max, s_fix)
            return (math.log10(lo), math.log10(hi))

        if self.chk_vol_vary.isChecked():
            lo_vol = math.log10(max(self.spin_vol_min.value(), 1.0))
            hi_vol = math.log10(max(self.spin_vol_max.value(), 2.0))
        else:
            fv = max(self.spin_vol_fixed.value(), 1.0)
            lo_vol = hi_vol = math.log10(fv)

        param_ranges = {
            "log_vol":   (lo_vol, hi_vol),
            "tau_yield": _rng(self.chk_tau_vary, self.spin_tau_min,
                               self.spin_tau_max, self.spin_tau_fixed),
            "log_eta":   _log_rng(self.chk_eta_vary, self.spin_eta_min,
                                   self.spin_eta_max, self.spin_eta_fixed),
            "manning_n": _rng(self.chk_n_vary,  self.spin_n_min,
                               self.spin_n_max,  self.spin_n_fixed),
            "log_K_visc": _log_rng(self.chk_K_vary, self.spin_K_min,
                                   self.spin_K_max, self.spin_K_fixed),
        }

        params = dict(
            dem_layer       = dem_layer,
            source_layer    = src_layer,
            sources         = sources,
            obs_poly_layer  = obs_poly,
            obs_poly_wkts   = _layer_wkts(obs_poly, dem_layer.crs()),
            obs_pts         = _layer_points(obs_pts, depth_fld, dem_layer.crs()),
            eval_poly_wkts  = (_layer_wkts(self.combo_eval_poly.currentLayer(), dem_layer.crs())
                               if self.combo_eval_poly.currentLayer() is not None else None),
            eval_poly_layer = self.combo_eval_poly.currentLayer(),
            obs_pts_layer   = obs_pts,
            depth_field     = depth_fld,
            depth_threshold = self.spin_depth_thr.value(),
            Cv              = self.spin_Cv.value(),
            Cv_max          = self.spin_Cv_max.value(),
            param_ranges    = param_ranges,
            n_runs          = self.spin_n_runs.value(),
            seed            = self.spin_seed.value(),
            t_end           = float(self.spin_tend.value()),
            dt_max          = self.spin_dtmax.value(),
            out_dir         = out_dir,
            csv_name        = csv_name,
            cfl_number      = self.spin_cfl.value(),
            face_depth_limiter = self.chk_face_lim.isChecked(),
            theta           = self.spin_theta.value(),
            n_workers       = self.spin_n_workers.value(),
            rho_s           = self.spin_rho_s.value(),
            rho_w           = self.spin_rho_w.value(),
        )
        try:
            _detach_layers(params, dem_layer)
        except Exception as e:
            QMessageBox.critical(self, "DEM", f"Cannot read the DEM:\n{type(e).__name__}: {e}")
            return

        self._sweep_results  = []
        self._sweep_best_pkg = None
        self.sweep_log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_sweep_run.setEnabled(False)
        self.btn_sweep_cancel.setEnabled(True)
        def _rtxt(chk, s_min, s_max, s_fix, fmt):
            """Range of a varied parameter, or its fixed value (what is actually used)."""
            if chk.isChecked():
                return f"{s_min.value():{fmt}} – {s_max.value():{fmt}}"
            return f"{s_fix.value():{fmt}} (fixed)"

        self.sweep_log_box.append(
            f"Starting LHS analysis: {params['n_runs']} runs…\n"
            f"Volume: {_rtxt(self.chk_vol_vary, self.spin_vol_min, self.spin_vol_max, self.spin_vol_fixed, '.0f')} m³\n"
            f"τy: {_rtxt(self.chk_tau_vary, self.spin_tau_min, self.spin_tau_max, self.spin_tau_fixed, '.0f')} Pa\n"
            f"η: {_rtxt(self.chk_eta_vary, self.spin_eta_min, self.spin_eta_max, self.spin_eta_fixed, '.1f')} Pa·s\n"
            f"n: {_rtxt(self.chk_n_vary, self.spin_n_min, self.spin_n_max, self.spin_n_fixed, '.3f')}\n"
            f"K: {_rtxt(self.chk_K_vary, self.spin_K_min, self.spin_K_max, self.spin_K_fixed, '.1f')}")

        self._t_sweep_start = time.time()   # LHS timer

        self.sweep_worker = SweepWorker(params)
        self.sweep_worker.run_done.connect(self._on_sweep_run_done)
        self.sweep_worker.sweep_log.connect(self._on_sweep_log_msg)
        self.sweep_worker.sweep_finished.connect(self._on_sweep_finished)
        self.sweep_worker.sweep_error.connect(self._on_sweep_error)
        self.sweep_worker.sweep_best_result.connect(self._on_sweep_best_result)
        self.sweep_worker.start()

    def _on_sweep_run_done(self, row_data: dict):
        self._sweep_results.append(row_data)
        n       = len(self._sweep_results)
        n_total = self.spin_n_runs.value()
        self.progress_bar.setValue(int(100 * n / max(n_total, 1)))

    def _on_sweep_log_msg(self, msg: str):
        self.sweep_log_box.append(msg)
        sb = self.sweep_log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_sweep_best_result(self, pkg: dict):
        """Store the best-run package for creating a vector layer."""
        self._sweep_best_pkg = pkg

    def _on_sweep_error(self, msg: str):
        self.btn_sweep_run.setEnabled(True)
        self.btn_sweep_cancel.setEnabled(False)
        self.sweep_log_box.append(f"\n⛔ ERROR:\n{msg}")
        QMessageBox.critical(self, "LHS analysis error", msg[:800])

    def _on_sweep_finished(self):
        self.btn_sweep_run.setEnabled(True)
        self.btn_sweep_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        elapsed_s = time.time() - getattr(self, "_t_sweep_start", time.time())
        n_ok = sum(1 for r in self._sweep_results if r.get("status") == "ok")
        avg_s = elapsed_s / max(n_ok, 1)
        self.sweep_log_box.append(
            f"\n✓ LHS completed: {n_ok} runs in "
            f"{elapsed_s:.0f} s ({elapsed_s/60:.1f} min), "
            f"avg. {avg_s:.1f} s/run")

        if not self._sweep_results:
            QMessageBox.warning(
                self, "LHS analysis", "No successful runs."); return

        out_dir  = self.edit_sweep_outdir.text().strip()
        csv_name = self.edit_sweep_csv.text().strip() or "sweep_results.csv"
        csv_path = os.path.join(out_dir, csv_name)
        os.makedirs(out_dir, exist_ok=True)

        # CSV rows were written by the worker right after each run (nothing is lost on a crash)
        if not os.path.isfile(csv_path):
            csv_path = None
        elif any(r.get("status") == "ok" for r in self._sweep_results):
            self.sweep_log_box.append(f"\n📋 CSV saved: {csv_path}")
        else:
            self.sweep_log_box.append("\n⚠ No successful runs for CSV.")

        # ── h_max raster layer of the best run ───────────────────────────────
        raster_path = None
        if self._sweep_best_pkg is not None:
            base = csv_name.replace(".csv", "")
            raster_path = os.path.join(out_dir, f"best_h_max_{base}.tif")
            try:
                from .qgis_runner import write_geotiff
                # Apply the threshold: pixels < threshold → 0 → nodata (zero_as_nodata=True)
                thr   = self.spin_depth_thr.value()
                h_out = self._sweep_best_pkg["h_max"].copy()
                h_out[h_out < thr] = 0.0
                write_geotiff(
                    h_out,
                    self._sweep_best_pkg["geo_info"],
                    raster_path,
                    zero_as_nodata=True,
                )
                self.sweep_log_box.append(
                    f"🗺 h_max raster (threshold ≥ {thr:.2f} m): {raster_path}")
                from qgis.core import QgsRasterLayer, QgsProject
                rlayer = QgsRasterLayer(raster_path, "LHS_best_h_max")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    # Apply pseudocolour (depth style)
                    try:
                        from .plugin import DebrisFlowPlugin
                        DebrisFlowPlugin._apply_pseudocolour(rlayer, "depth")
                    except Exception:
                        pass
                else:
                    self.sweep_log_box.append(
                        "⚠ The raster was created but cannot be loaded into QGIS.")
            except Exception as e:
                self.sweep_log_box.append(
                    f"⚠ Error creating the raster: {e}")
                raster_path = None

        # ── Results dialog ─────────────────────────────────────────────────────
        dlg = SweepResultsDialog(
            self, self._sweep_results, raster_path, csv_path)
        dlg.exec_()

    def _browse_save_file(self):
        """File-selection dialog for saving parameters."""
        current = self.edit_save_path.text().strip()
        start   = os.path.dirname(current) if current else os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save simulation parameters",
            os.path.join(start, "debris_params.json"),
            "JSON parameters (*.json);;All Files (*)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.edit_save_path.setText(path)

    def _browse_load_file(self):
        """File-selection dialog for loading parameters."""
        current = self.edit_load_path.text().strip()
        start   = os.path.dirname(current) if current else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load simulation parameters",
            start, "JSON parameters (*.json);;All Files (*)")
        if path:
            self.edit_load_path.setText(path)

    def _save_params(self):
        """
        Collect all parameters from the UI and write them to a JSON file.
        File structure:
          {
            "version": "1.0",
            "layers": { "dem_name": ..., "source_name": ... },
            "hydrographs": { "<fid>": {type, q_const/times/q_vals}, ... },
            "source_table": [ {fid, name, q_max}, ... ],
            "obrien": { Cv, Cv_max, direct_mode, tau_yield, eta,
                        alpha1, beta1, alpha2, beta2, manning_n },
            "solver": { t_end, dt_max, cfl_number, face_depth_limiter, theta },
            "output": { prefix, outdir,
                        h_max, V_max, HV2_max, Fr_hv2, h_final, t_arrival,
                        threshold_enabled, threshold_value }
          }
        """
        path = self.edit_save_path.text().strip()
        if not path:
            QMessageBox.warning(
                self, "No path",
                "Specify a file path on the \"Output\" tab\n"
                "before saving.")
            return

        if not path.lower().endswith(".json"):
            path += ".json"

        # --- layers ---
        dem_layer = self.dem_combo.currentLayer()
        src_layer = self.src_line_combo.currentLayer()
        layers_sec = {
            "dem_name":    dem_layer.name()    if dem_layer else "",
            "source_name": src_layer.name()    if src_layer else "",
        }

        # --- hydrographs: int keys → str for JSON ---
        hydro_sec = {
            str(fid): data
            for fid, data in self._hydro_data.items()
        }

        # --- source table rows (for restoring descriptions) ---
        table_rows = []
        for row in range(self.tbl_sources.rowCount()):
            fid_item  = self.tbl_sources.item(row, 0)
            name_item = self.tbl_sources.item(row, 1)
            q_item    = self.tbl_sources.item(row, 2)
            if fid_item is None:
                continue
            table_rows.append({
                "fid":  fid_item.text(),
                "name": name_item.text() if name_item else "",
                "q_max_display": q_item.text() if q_item else "—",
            })

        # --- O'Brien ---
        obrien_sec = {
            "Cv":          self.spin_Cv.value(),
            "Cv_max":      self.spin_Cv_max.value(),
            "direct_mode": self._radio_direct.isChecked(),
            "tau_yield":   self.spin_tau.value(),
            "eta":         self.spin_eta.value(),
            "alpha1":      self.spin_a1.value(),
            "beta1":       self.spin_b1.value(),
            "alpha2":      self.spin_a2.value(),
            "beta2":       self.spin_b2.value(),
            "manning_n":   self.spin_n.value(),
            "K_visc":      self.spin_K.value(),
            "rho_s":       self.spin_rho_s.value(),
            "rho_w":       self.spin_rho_w.value(),
        }

        # --- Solver ---
        solver_sec = {
            "t_end":  self.spin_tend.value(),
            "dt_max": self.spin_dtmax.value(),
            "cfl_number": self.spin_cfl.value(),
            "face_depth_limiter": self.chk_face_lim.isChecked(),
            "theta":  self.spin_theta.value(),
        }

        # --- Output ---
        output_sec = {
            "prefix":            self.edit_prefix.text(),
            "outdir":            self.edit_outdir.text(),
            "h_max":             self.chk_h_max.isChecked(),
            "V_max":             self.chk_V_max.isChecked(),
            "HV2_max":           self.chk_HV2_max.isChecked(),
            "Fr_hv2":            self.chk_Fr_hv2.isChecked(),
            "h_final":           self.chk_h_final.isChecked(),
            "t_arrival":         self.chk_t_arrival.isChecked(),
            "arrival_depth":     self.spin_arrival_thr.value(),
            "threshold_enabled": self.chk_threshold.isChecked(),
            "threshold_value":   self.spin_threshold.value(),
        }

        payload = {
            "version":      "1.0",
            "layers":       layers_sec,
            "source_table": table_rows,
            "hydrographs":  hydro_sec,
            "obrien":       obrien_sec,
            "solver":       solver_sec,
            "output":       output_sec,
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            QMessageBox.information(
                self, "Parameters saved",
                f"Simulation parameters saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Save error",
                f"Failed to write the file:\n{e}")

    def _load_params(self):
        """
        Load parameters from a JSON file and restore the UI.
        Layers are looked up by name in the current QGIS project.
        """
        path = self.edit_load_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load simulation parameters",
                os.path.expanduser("~"),
                "JSON parameters (*.json);;All Files (*)")
            if not path:
                return
            self.edit_load_path.setText(path)

        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "File not found",
                f"The file does not exist:\n{path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Read error",
                f"Failed to read the JSON:\n{e}")
            return

        warnings = []

        # --- Layers: look up by name in the QGIS project ---
        from qgis.core import QgsProject
        layers_sec = data.get("layers", {})

        dem_name = layers_sec.get("dem_name", "")
        if dem_name:
            matches = QgsProject.instance().mapLayersByName(dem_name)
            if matches:
                self.dem_combo.setLayer(matches[0])
            else:
                warnings.append(
                    f"DEM layer \"{dem_name}\" not found in the project.")

        src_name = layers_sec.get("source_name", "")
        if src_name:
            matches = QgsProject.instance().mapLayersByName(src_name)
            if matches:
                self.src_line_combo.setLayer(matches[0])
            else:
                warnings.append(
                    f"Source layer \"{src_name}\" not found in the project.")

        # --- Hydrographs ---
        hydro_sec = data.get("hydrographs", {})
        self._hydro_data = {}
        _src = self.src_line_combo.currentLayer()
        self._hydro_layer_id = _src.id() if (src_name and _src is not None
                                             and _src.name() == src_name) else None
        for fid_str, hdata in hydro_sec.items():
            try:
                self._hydro_data[int(fid_str)] = hdata
            except ValueError:
                pass  # invalid key — skip

        # --- Source table: restore rows from source_table ---
        table_rows = data.get("source_table", [])
        if table_rows and self._hydro_data:
            self.tbl_sources.setRowCount(0)
            for row_data in table_rows:
                try:
                    fid  = int(row_data.get("fid", -1))
                except (ValueError, TypeError):
                    continue
                fname  = row_data.get("name", f"Line {fid}")
                q_disp = row_data.get("q_max_display", "—")

                row = self.tbl_sources.rowCount()
                self.tbl_sources.insertRow(row)

                id_item = QTableWidgetItem(str(fid))
                id_item.setTextAlignment(Qt.AlignCenter)
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_sources.setItem(row, 0, id_item)

                name_item = QTableWidgetItem(fname)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_sources.setItem(row, 1, name_item)

                q_item = QTableWidgetItem(q_disp)
                q_item.setTextAlignment(Qt.AlignCenter)
                q_item.setFlags(q_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_sources.setItem(row, 2, q_item)

                btn = QPushButton("✏ Edit...")
                btn.setProperty("fid",   fid)
                btn.setProperty("fname", fname)
                btn.setProperty("row",   row)
                btn.clicked.connect(self._edit_hydro)
                self.tbl_sources.setCellWidget(row, 4, btn)

            self._refresh_volumes()
            self.tbl_sources.resizeRowsToContents()

        # --- O'Brien ---
        ob = data.get("obrien", {})
        if ob:
            _sv = lambda spin, key, default=None: (
                spin.setValue(ob[key]) if key in ob else None)
            _sv(self.spin_Cv,     "Cv")
            _sv(self.spin_Cv_max, "Cv_max")
            _sv(self.spin_tau,    "tau_yield")
            _sv(self.spin_eta,    "eta")
            _sv(self.spin_a1,     "alpha1")
            _sv(self.spin_b1,     "beta1")
            _sv(self.spin_a2,     "alpha2")
            _sv(self.spin_b2,     "beta2")
            _sv(self.spin_n,      "manning_n")
            if "K_visc" in ob: self.spin_K.setValue(ob["K_visc"])
            if "rho_s"  in ob: self.spin_rho_s.setValue(ob["rho_s"])
            if "rho_w"  in ob: self.spin_rho_w.setValue(ob["rho_w"])
            if "direct_mode" in ob:          # absent (older file): keep the current mode
                direct = bool(ob["direct_mode"])
                self._radio_direct.setChecked(direct)
                self._radio_exp.setChecked(not direct)
            self._on_rheo_mode()
            self._update_hstop()

        # --- Solver ---
        sv = data.get("solver", {})
        if sv:
            if "t_end"  in sv: self.spin_tend.setValue(int(sv["t_end"]))
            if "dt_max" in sv: self.spin_dtmax.setValue(sv["dt_max"])
            if "cfl_number" in sv: self.spin_cfl.setValue(float(sv["cfl_number"]))
            if "face_depth_limiter" in sv:
                self.chk_face_lim.setChecked(bool(sv["face_depth_limiter"]))
            if "theta" in sv: self.spin_theta.setValue(float(sv["theta"]))

        # --- Output ---
        out = data.get("output", {})
        if out:
            if "prefix"  in out: self.edit_prefix.setText(out["prefix"])
            if "outdir"  in out: self.edit_outdir.setText(out["outdir"])
            if "h_max"   in out: self.chk_h_max.setChecked(out["h_max"])
            if "V_max"   in out: self.chk_V_max.setChecked(out["V_max"])
            if "HV2_max" in out: self.chk_HV2_max.setChecked(out["HV2_max"])
            if "Fr_hv2" in out: self.chk_Fr_hv2.setChecked(out["Fr_hv2"])
            if "h_final" in out: self.chk_h_final.setChecked(out["h_final"])
            if "t_arrival" in out: self.chk_t_arrival.setChecked(out["t_arrival"])
            if "arrival_depth" in out: self.spin_arrival_thr.setValue(float(out["arrival_depth"]))
            thr_on  = out.get("threshold_enabled", False)
            thr_val = out.get("threshold_value", 0.10)
            self.chk_threshold.setChecked(thr_on)
            self.spin_threshold.setValue(thr_val)
            self.spin_threshold.setEnabled(thr_on)

        # keys that change the result but are missing in older files: the current
        # widget values stay in effect, so the user must see which ones they are
        _missing = ([f"obrien.{k}" for k in ("K_visc", "rho_s", "rho_w", "direct_mode") if k not in ob]
                    + [f"solver.{k}" for k in ("cfl_number", "face_depth_limiter", "theta") if k not in sv])
        if _missing:
            warnings.append("Not in the file (older version), the current values are kept: "
                            + ", ".join(_missing))

        # --- Summary ---
        if warnings:
            QMessageBox.warning(
                self,
                "Parameters loaded (with warnings)",
                "Parameters restored.\n\n"
                "Warnings:\n" + "\n".join(f"• {w}" for w in warnings))
        else:
            QMessageBox.information(
                self, "Parameters loaded",
                f"All parameters restored from:\n{path}")

    # ── Slots ───────────────────────────────────────────────────────────────────

    def _load_features(self):
        """Load the features of the line layer into the source table."""
        layer = self.src_line_combo.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "No layer",
                                "Select a source line layer.")
            return
        if layer.geometryType() != QgsWkbTypes.LineGeometry:
            QMessageBox.warning(self, "Invalid type",
                                "The layer must be a line layer (LineString).")
            return

        # hydrographs are keyed by fid, which is only unique within one layer:
        # a different layer must not inherit them
        if layer.id() != self._hydro_layer_id:
            # hydrographs from a JSON whose line layer was not found by name: offer to keep
            # them for the selected layer instead of silently replacing them by 100 m³/s
            keep = (self._hydro_layer_id is None and bool(self._hydro_data)
                    and all(layer.getFeature(fid).isValid() for fid in self._hydro_data)
                    and QMessageBox.question(
                        self, "Hydrographs from the file",
                        f"Hydrographs loaded from the parameter file ({len(self._hydro_data)}) "
                        f"are not bound to a layer.\nApply them to the features of \"{layer.name()}\" "
                        "with the same IDs?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes)
            if not keep:
                self._hydro_data = {}
            self._hydro_layer_id = layer.id()
        feats = list(layer.getFeatures())
        if not feats:
            QMessageBox.warning(self, "Empty",
                                "The layer contains no features.")
            return

        self.tbl_sources.setRowCount(0)

        for feat in feats:
            fid   = feat.id()
            # Try to get a readable name from the 'name' or 'id' field
            fname = None
            fields = [f.name().lower() for f in feat.fields()]
            for candidate in ("name", "label", "id", "fid"):
                if candidate in fields:
                    val = feat[candidate]
                    if val is not None:
                        fname = str(val)
                        break
            if fname is None:
                fname = f"Line {fid}"

            row = self.tbl_sources.rowCount()
            self.tbl_sources.insertRow(row)

            # ID
            id_item = QTableWidgetItem(str(fid))
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_sources.setItem(row, 0, id_item)

            # Description
            name_item = QTableWidgetItem(fname)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_sources.setItem(row, 1, name_item)

            # Q_max (updated when editing the hydrograph)
            data = self._hydro_data.get(fid)
            q_max = max(data["q_vals"]) if data else 0.0
            q_item = QTableWidgetItem(f"{q_max:.1f}" if q_max > 0 else "—")
            q_item.setTextAlignment(Qt.AlignCenter)
            q_item.setFlags(q_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_sources.setItem(row, 2, q_item)

            # Edit button
            btn = QPushButton("✏ Edit...")
            btn.setProperty("fid", fid)
            btn.setProperty("fname", fname)
            btn.setProperty("row", row)
            btn.clicked.connect(self._edit_hydro)
            self.tbl_sources.setCellWidget(row, 4, btn)

            # If there is no data for this fid — set a default
            if fid not in self._hydro_data:
                self._hydro_data[fid] = {
                    "type": "const", "q_const": 100.0,
                    "times": [0.0, 1e9], "q_vals": [100.0, 100.0],
                }

        self._refresh_volumes()
        self.tbl_sources.resizeRowsToContents()

    def _refresh_volumes(self):
        """Volume of every hydrograph injected within [0, t_end] (column 3)."""
        from .core.solver import hydrograph_volume
        t_end = float(self.spin_tend.value()) if hasattr(self, "spin_tend") else 0.0
        for row in range(self.tbl_sources.rowCount()):
            it = self.tbl_sources.item(row, 0)
            try:
                fid = int(it.text()) if it is not None else None
            except ValueError:
                fid = None
            data = self._hydro_data.get(fid) if fid is not None else None
            txt, after = "—", 0.0
            if data:
                v = hydrograph_volume(data["times"], data["q_vals"], 0.0, t_end)
                if data.get("type") != "const":
                    after = hydrograph_volume(data["times"], data["q_vals"], t_end, 1e12)
                txt = f"{v:.0f}" + (" ⚠" if after > 0 else "")
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setToolTip("Volume injected within [0, t_end]; ⚠ = part of the hydrograph lies after\n"
                        "t_end and will not be injected (increase the simulation duration).")
            self.tbl_sources.setItem(row, 3, item)

    def _edit_hydro(self):
        """Open the hydrograph editor for the selected feature."""
        btn   = self.sender()
        fid   = btn.property("fid")
        fname = btn.property("fname")
        row   = btn.property("row")

        init  = self._hydro_data.get(fid)
        dlg   = HydrographEditorDialog(self, fid, fname, init)

        if dlg.exec_() == QDialog.Accepted:
            result = dlg.get_result()
            if result:
                self._hydro_data[fid] = result
                q_max = max(result["q_vals"])
                item = self.tbl_sources.item(row, 2)
                if item:
                    item.setText(f"{q_max:.1f}")
                self._refresh_volumes()

    def _on_rheo_mode(self):
        direct = self._radio_direct.isChecked()
        self._sec_direct.setVisible(direct)
        self._sec_exp.setVisible(not direct)

    def _update_rho_m(self):
        """Updates the indicator of the current mixture density ρm."""
        try:
            Cv  = self.spin_Cv.value()
            rho_s = self.spin_rho_s.value()
            rho_w = self.spin_rho_w.value()
            rho_m = rho_s * Cv + rho_w * (1.0 - Cv)
            self._lbl_rho_m.setText(f"{rho_m:.1f} kg/m³")
        except Exception:
            pass

    def _update_hstop(self):
        try:
            tau = self.spin_tau.value()
            Cv  = self.spin_Cv.value()
            rho_m   = self.spin_rho_s.value() * Cv + self.spin_rho_w.value() * (1. - Cv)
            gamma_m = rho_m * 9.81
            self._lbl_hstop.setText(
                f"h_stop: {tau/(gamma_m*0.10):.3f} m (10%),  "
                f"{tau/(gamma_m*0.05):.3f} m (5%)")
        except Exception:
            pass

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Output folder", self.edit_outdir.text())
        if path:
            self.edit_outdir.setText(path)

    # ── Run ─────────────────────────────────────────────────────────────────────

    # ── Numba JIT check ──────────────────────────────────────────────────────────

    @staticmethod
    def _check_numba() -> tuple:
        """
        Checks the availability of Numba JIT independently of the plugin import paths.
        Returns (ok: bool, reason: str).
        """
        import sys

        # Step 1: import Numba directly (independent of the plugin package)
        try:
            import numba
            from numba import njit as _njit_test  # noqa: F401
            numba_version = numba.__version__
        except ImportError as e:
            return False, (
                "Numba was not found in the QGIS Python environment.\n"
                f"Details: {e}\n\n"
                "Installation in the OSGeo4W Shell console:\n"
                "  pip install numba\n"
                "After installation, restart QGIS.")
        except Exception as e:
            return False, (
                f"Numba is installed but does not initialise.\n"
                f"Error ({type(e).__name__}): {e}\n\n"
                "Possible causes:\n"
                "  • Incompatibility between Numba and NumPy\n"
                "  • Missing llvmlite or LLVM DLL (Windows)\n"
                "Try: pip install --upgrade numba")

        # Step 2: check the flag in our solver (only if the module is already imported);
        # a top-level `core.solver` of another plugin is not ours and is ignored
        solver_mod = sys.modules.get(f"{__package__}.core.solver")
        if solver_mod is not None and not getattr(solver_mod, '_NUMBA_AVAILABLE', True):
            return False, (
                f"Numba {numba_version} was found but is disabled in solver.\n"
                "The plugin may have been loaded before Numba was installed.\n"
                "Restart QGIS and load the plugin again.")

        return True, ""

    def _on_run(self):
        # ── Numba JIT check ────────────────────────────────────────────────────
        self._nb_ok, _nb_reason = self._check_numba()
        if not self._nb_ok:
            QMessageBox.critical(
                self, "Numba JIT not found",
                "Numba JIT is required to run the simulation.\n\n"
                + _nb_reason)
            return

        self._t_run_start = time.time()   # run timer

        # ── Validation ─────────────────────────────────────────────────────────
        dem_layer = self.dem_combo.currentLayer()
        src_layer = self.src_line_combo.currentLayer()

        if dem_layer is None:
            QMessageBox.warning(self, "No data", "Select a DEM layer."); return
        if src_layer is None:
            QMessageBox.warning(self, "No data",
                                "Select a source line layer."); return

        if self.tbl_sources.rowCount() == 0:
            QMessageBox.warning(
                self, "Sources not loaded",
                "Click \"Load features\" to load the source lines."); return
        if src_layer.id() != self._hydro_layer_id:
            # the table and hydrographs belong to another layer: the fids would pick
            # geometries of the new layer with the old names and hydrographs
            QMessageBox.warning(
                self, "Sources not loaded",
                "The source table belongs to another line layer.\n"
                "Click \"Load features\" for the selected layer."); return

        out_dir = self.edit_outdir.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No data",
                                "Specify an output folder.\n"
                                "(\"Output\" tab)"); return
        # check the folder before the run: an error after it would lose all results
        try:
            os.makedirs(out_dir, exist_ok=True)
            _test = os.path.join(out_dir, f".qdebrisflow_write_test_{os.getpid()}")
            with open(_test, "w"):
                pass
            os.remove(_test)
        except OSError as e:
            QMessageBox.warning(self, "Output folder",
                                f"Cannot create or write to the output folder:\n{out_dir}\n\n{e}")
            return

        prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_",
                        self.edit_prefix.text().strip())
        if not prefix:
            QMessageBox.warning(self, "No data",
                                "Specify a file prefix.\n"
                                "(\"Output\" tab)"); return

        wanted = set()
        if self.chk_h_max.isChecked():    wanted.add("h_max")
        if self.chk_V_max.isChecked():    wanted.add("V_max")
        if self.chk_HV2_max.isChecked():  wanted.add("HV2_max")
        if self.chk_Fr_hv2.isChecked():   wanted.add("Fr_hv2")
        if self.chk_h_final.isChecked():  wanted.add("h_final")
        if self.chk_t_arrival.isChecked(): wanted.add("t_arrival")

        if not wanted:
            QMessageBox.warning(
                self, "No output layers",
                "Select at least one output layer\n"
                "on the \"Output\" tab."); return

        _crs_msg = _check_dem_crs(dem_layer)
        if _crs_msg:
            QMessageBox.warning(self, "CRS", _crs_msg); return

        Cv = self.spin_Cv.value(); Cv_max = self.spin_Cv_max.value()
        if Cv >= Cv_max:
            QMessageBox.warning(self, "Parameter",
                                f"Cv ({Cv}) must be < Cv_max ({Cv_max})."); return

        # Assemble the list of sources from the table
        sources = []
        for row in range(self.tbl_sources.rowCount()):
            fid_item = self.tbl_sources.item(row, 0)
            if fid_item is None:
                continue
            try:
                fid = int(fid_item.text())
            except ValueError:
                continue
            name_item = self.tbl_sources.item(row, 1)
            fname = name_item.text() if name_item else f"source_{fid}"
            data  = self._hydro_data.get(fid)
            if data is None:
                continue
            sources.append({
                "fid":    fid,
                "name":   fname,
                "wkt":    _feature_wkt(src_layer, fid, dem_layer.crs()),
                "times":  data["times"],
                "q_vals": data["q_vals"],
            })

        if not sources:
            QMessageBox.warning(self, "No sources",
                                "Add at least one source."); return

        direct = self._radio_direct.isChecked()
        params = {
            "dem_layer":        dem_layer,
            "source_layer":     src_layer,
            "sources":          sources,
            "output_dir":       out_dir,
            "output_prefix":    prefix,
            "output_layers":    wanted,
            "Cv":               Cv,
            "Cv_max":           Cv_max,
            "direct_mode":      direct,
            "tau_yield_direct": self.spin_tau.value() if direct else 0.0,
            "eta_direct":       self.spin_eta.value() if direct else 0.0,
            "alpha1":           self.spin_a1.value(),
            "beta1":            self.spin_b1.value(),
            "alpha2":           self.spin_a2.value(),
            "beta2":            self.spin_b2.value(),
            "manning_n":        self.spin_n.value(),
            "K_visc":           self.spin_K.value(),
            "cfl_number":       self.spin_cfl.value(),
            "face_depth_limiter": self.chk_face_lim.isChecked(),
            "theta":            self.spin_theta.value(),
            "t_end":            float(self.spin_tend.value()),
            "dt_max":           self.spin_dtmax.value(),
            "arrival_depth":    self.spin_arrival_thr.value(),
            "h_threshold":      (self.spin_threshold.value()
                                 if self.chk_threshold.isChecked() else 0.0),
        }
        if dem_layer.crs() != src_layer.crs():
            params["_crs_note"] = (f"CRS: DEM={dem_layer.crs().authid()}, "
                                   f"lines={src_layer.crs().authid()} — lines reprojected to the DEM CRS.")
        try:
            _detach_layers(params, dem_layer)
        except Exception as e:
            QMessageBox.critical(self, "DEM", f"Cannot read the DEM:\n{type(e).__name__}: {e}")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳  Running...")
        self.log_box.clear()
        self.progress_bar.setValue(0)

        self.worker = SimulationWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()
        self.btn_stop.setEnabled(True)

    def _on_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.btn_stop.setEnabled(False)
            self.log_box.append("Stopping…")

    def _on_cancelled(self):
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run simulation")
        self.progress_bar.setValue(0)

    def _on_progress(self, t_cur: float, t_end: float):
        self.progress_bar.setValue(
            int(100 * t_cur / t_end) if t_end > 0 else 0)

    def _on_log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def _on_finished(self, payload: dict):
        elapsed_s = time.time() - getattr(self, "_t_run_start", time.time())
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run simulation")
        self.progress_bar.setValue(100)
        self.log_box.append(
            f"\n✓ Simulation completed in {elapsed_s:.1f} s "
            f"({elapsed_s/60:.1f} min)")

        paths   = payload["paths"]
        prefix  = payload["prefix"]
        wanted  = payload["wanted"]

        # Nice names for the message
        layer_names = {
            "h_max":     "Peak thickness",
            "V_max":     "Peak velocity",
            "HV2_max":   "Peak momentum flux h·v²",
            "Fr_hv2":    "Froude number at peak h·v²",
            "h_final":   "Final deposits",
            "t_arrival": "Arrival time",
        }

        from .plugin import DebrisFlowPlugin
        lines = []
        for key in ("h_max", "V_max", "HV2_max", "Fr_hv2", "h_final", "t_arrival"):
            if key not in wanted:
                continue
            name = f"{prefix}_{key}"
            path = paths.get(name)
            if not path:
                continue
            style = "velocity" if key == "V_max" else (
                    "intensity" if key == "HV2_max" else (
                    "froude"   if key == "Fr_hv2" else (
                    "time"     if key == "t_arrival" else "depth")))
            DebrisFlowPlugin.load_result_layer(path, name, style)
            lines.append(f"  • {name}  — {layer_names.get(key,'')}")

        folder = list(paths.values())[0] if paths else out_dir
        QMessageBox.information(
            self, "Simulation completed",
            "Layers added to QGIS:\n" + "\n".join(lines) +
            f"\n\nFolder:\n{os.path.dirname(folder)}")

    def _on_error(self, msg: str):
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run simulation")
        self.log_box.append(f"\n⛔ ERROR:\n{msg}")
        QMessageBox.critical(self, "Simulation error", msg[:700])


# ─────────────────────────────────────────────────────────────────────────────
# LHS analysis results dialog
# ─────────────────────────────────────────────────────────────────────────────

class SweepResultsDialog(QDialog):
    """
    Displays the results of the LHS sweep:
      - best parameters (minimum Cm)
      - table of the top 10 runs
      - sensitivity analysis: quintiles of Cm, Spearman ρ, volume control
      - paths to the raster and CSV
    """

    def __init__(self, parent, results: list, raster_path, csv_path):
        super().__init__(parent)
        self.setWindowTitle("LHS analysis — Results")
        self.setMinimumSize(760, 640)
        self._build_ui(results, raster_path, csv_path)

    def _build_ui(self, results: list, raster_path, csv_path):
        lo = QVBoxLayout(self)

        ok_rows = sorted(
            [r for r in results
             if r.get("status") == "ok" and r.get("Cm") is not None],
            key=lambda r: float(r["Cm"]),
        )
        n_ok    = len(ok_rows)
        n_total = len(results)

        if not ok_rows:
            lo.addWidget(QLabel("No successful runs."))
            bb = QDialogButtonBox(QDialogButtonBox.Close)
            bb.rejected.connect(self.reject)
            lo.addWidget(bb)
            return

        best = ok_rows[0]

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_lo = QVBoxLayout(inner)

        # ── Best parameters ──────────────────────────────────────────────────
        grp_best = QGroupBox("🏆  Best run (minimum Cm)")
        grp_best.setStyleSheet(
            "QGroupBox{border:2px solid #2196F3;border-radius:6px;"
            "margin-top:8px;font-weight:bold}"
            "QGroupBox::title{color:#1565C0;padding:0 4px}")
        fl_b = QFormLayout(grp_best)

        def _lbl(key, fmt=".3f", unit=""):
            v = best.get(key)
            if v is None:
                return QLabel("—")
            try:
                return QLabel(f"{float(v):{fmt}} {unit}".strip())
            except (ValueError, TypeError):
                return QLabel(str(v))

        fl_b.addRow("Total volume V:",          _lbl("vol_m3",       ".0f", "m³"))
        fl_b.addRow("τy — Yield stress:",       _lbl("tau_yield_Pa", ".1f", "Pa"))
        fl_b.addRow("η — Mixture viscosity:",   _lbl("eta_Pas",      ".2f", "Pa·s"))
        fl_b.addRow("Manning n:",               _lbl("manning_n",    ".4f"))
        fl_b.addRow("K — shape factor:",        _lbl("K_visc",       ".1f"))

        sep = QLabel("─" * 50)
        sep.setStyleSheet("color:#ccc; font-size:8px")
        fl_b.addRow(sep)
        tp = best.get("TP", "—"); fp = best.get("FP", "—")
        fn = best.get("FN", "—")
        fl_b.addRow("TP / FP / FN:",
                    QLabel(f"{tp} / {fp} / {fn}"))
        fl_b.addRow("ΩTm (0=best):",            _lbl("omega_Tm"))
        fl_b.addRow("Δo,c (depth overest.):",   _lbl("delta_o_c"))
        fl_b.addRow("Δu  (depth underest.):",   _lbl("delta_u"))
        fl_b.addRow("Cm (overall, 0=best):",    _lbl("Cm"))

        note_dep = QLabel(
            "<i>Δo and Δu are normalised to Σ|d_obs| → independent metrics "
            "(Δo + Δu ≠ 1).<br>"
            "A value > 0.5: the mean error exceeds half the observed depth.</i>")
        note_dep.setWordWrap(True)
        note_dep.setStyleSheet("color:#666; font-size:10px")
        fl_b.addRow(note_dep)
        inner_lo.addWidget(grp_best)

        # ── Top-10 table ──────────────────────────────────────────────────────
        header_lbl = QLabel(
            f"<b>Top {min(10, n_ok)} runs by Cm "
            f"({n_ok} successful out of {n_total}):</b>")
        inner_lo.addWidget(header_lbl)

        cols = ["run_id", "vol_m3", "tau_yield_Pa", "eta_Pas",
                "manning_n", "K_visc", "TP", "FP", "FN",
                "omega_Tm", "delta_o_c", "delta_u", "Cm"]
        col_lbl = ["ID", "V [m³]", "τy [Pa]", "η [Pa·s]",
                   "n", "K", "TP", "FP", "FN",
                   "ΩTm", "Δo,c", "Δu", "Cm"]

        tbl = QTableWidget(min(10, n_ok), len(cols))
        tbl.setHorizontalHeaderLabels(col_lbl)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.setMaximumHeight(220)

        highlight = QColor(195, 225, 255)

        for r_idx, row in enumerate(ok_rows[:10]):
            for c_idx, key in enumerate(cols):
                v = row.get(key)
                if v is None:
                    text = "—"
                elif isinstance(v, float):
                    if abs(v) >= 1e4:
                        text = f"{v:.0f}"
                    elif abs(v) >= 1:
                        text = f"{v:.2f}"
                    else:
                        text = f"{v:.4f}"
                else:
                    text = str(v)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if r_idx == 0:
                    item.setBackground(highlight)
                tbl.setItem(r_idx, c_idx, item)

        inner_lo.addWidget(tbl)

        # ── Sensitivity analysis: quintiles of Cm, rank correlation ─────────
        grp_q = QGroupBox("Parameter sensitivity (quintiles of Cm, rank correlation)")
        grp_q.setStyleSheet(
            "QGroupBox{border:1px solid #aaa;border-radius:4px;margin-top:6px}"
            "QGroupBox::title{padding:0 4px}")
        grp_q_lo = QVBoxLayout(grp_q)

        note_q = QLabel(
            "<b>How to read it:</b> for each parameter the runs are split into 5 quintiles "
            "(Q1 = smallest values … Q5 = largest) and the mean Cm of every quintile is "
            "shown. ρ is the Spearman rank correlation between the parameter and Cm with "
            "a 95 % bootstrap interval; an interval that excludes 0 indicates an influence.<br>"
            "<b>Volume control</b> (second table): Cm residuals after a linear regression "
            "of Cm on log₁₀V, so that the dominant effect of the volume does not mask the "
            "other parameters. This is the plugin's own procedure and it differs from "
            "Barnhart et al. (2021, Section 7.5), who split each input into five bins and "
            "show the smoothed mean Cm against the effective volume, without correlations "
            "or regression.")
        note_q.setWordWrap(True)
        note_q.setStyleSheet("font-size:11px; color:#333; margin-bottom:4px")
        grp_q_lo.addWidget(note_q)

        cm_raw = {id(r): float(r["Cm"]) for r in ok_rows}
        cm_vc = dict(cm_raw)
        _vr = [r for r in ok_rows if (r.get("vol_m3") or 0) > 0]
        if len(_vr) >= 10:
            _res = _volume_residuals(np.log10([float(r["vol_m3"]) for r in _vr]),
                                     [cm_raw[id(r)] for r in _vr])
            for r, v in zip(_vr, _res):
                cm_vc[id(r)] = float(v)

        def _sens_table(params_sens, cm_of):
            tbl = QTableWidget()
            tbl.setColumnCount(8)
            tbl.setHorizontalHeaderLabels(["Parameter", "Q1 (min.)", "Q2", "Q3", "Q4", "Q5 (max.)", "Cm spread", "ρ [95 % CI]"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.verticalHeader().setVisible(False)
            for param_key, param_label in params_sens:
                vals = [(float(r.get(param_key)), cm_of[id(r)], r.get("run_id") or 0)
                        for r in ok_rows if r.get(param_key) is not None]
                if len(vals) < 10:
                    continue
                if np.ptp([v[0] for v in vals]) == 0:
                    # the parameter was fixed: quintiles would only re-sort by Cm itself
                    row_i = tbl.rowCount()
                    tbl.insertRow(row_i)
                    for c_i, txt in enumerate([param_label] + ["—"] * 5 + ["fixed", "—"]):
                        cell = QTableWidgetItem(txt)
                        cell.setTextAlignment(Qt.AlignCenter)
                        tbl.setItem(row_i, c_i, cell)
                    continue
                # ties in x are ordered by run, not by Cm
                vals.sort(key=lambda x: (x[0], x[2]))
                n_q = len(vals)
                q_size = n_q // 5
                q_means = []
                for qi in range(5):
                    start = qi * q_size
                    end   = (qi + 1) * q_size if qi < 4 else n_q
                    q_cm  = [v[1] for v in vals[start:end]]
                    q_means.append(float(np.mean(q_cm)) if q_cm else float("nan"))
                spread = max(q_means) - min(q_means)
                xs = np.array([v[0] for v in vals]); ys = np.array([v[1] for v in vals])
                if np.ptp(xs) > 0:
                    rho = _spearman(xs, ys)
                    lo, hi = _spearman_ci(xs, ys)
                else:
                    rho = lo = hi = float("nan")
                sig = (lo == lo) and not (lo <= 0.0 <= hi)
                is_important = spread > 0.05 or sig

                row_i = tbl.rowCount()
                tbl.insertRow(row_i)
                lbl_item = QTableWidgetItem(param_label)
                lbl_item.setTextAlignment(Qt.AlignCenter)
                if is_important:
                    lbl_item.setForeground(QColor("#1565C0"))
                    lbl_item.setFont(QFont("", -1, QFont.Bold))
                tbl.setItem(row_i, 0, lbl_item)
                for qi, cm_mean in enumerate(q_means):
                    cell = QTableWidgetItem(f"{cm_mean:.3f}" if cm_mean == cm_mean else "—")
                    cell.setTextAlignment(Qt.AlignCenter)
                    if cm_mean == cm_mean:
                        g = int(min(255, max(0, 255 * (1 - cm_mean))))
                        cell.setBackground(QColor(255 - g // 2, 255, 255 - g // 2))
                    tbl.setItem(row_i, qi + 1, cell)
                spread_item = QTableWidgetItem(f"{spread:.3f}")
                spread_item.setTextAlignment(Qt.AlignCenter)
                if is_important:
                    spread_item.setForeground(QColor("#C00000"))
                    spread_item.setFont(QFont("", -1, QFont.Bold))
                tbl.setItem(row_i, 6, spread_item)
                rho_item = QTableWidgetItem(
                    f"{rho:+.2f} [{lo:+.2f}; {hi:+.2f}]" if rho == rho else "—")
                rho_item.setTextAlignment(Qt.AlignCenter)
                if sig:
                    rho_item.setForeground(QColor("#C00000"))
                tbl.setItem(row_i, 7, rho_item)
            tbl.setMaximumHeight(28 * (tbl.rowCount() + 1) + 10)
            return tbl

        params_all = [("vol_m3", "V [m³]"), ("tau_yield_Pa", "τy [Pa]"), ("eta_Pas", "η [Pa·s]"),
                       ("manning_n", "n"), ("K_visc", "K")]
        sens_tbl = _sens_table(params_all, cm_raw)
        if sens_tbl.rowCount() > 0:
            grp_q_lo.addWidget(sens_tbl)
            grp_q_lo.addWidget(QLabel("With volume control (Cm residuals after regression on log₁₀V):"))
            grp_q_lo.addWidget(_sens_table(params_all[1:], cm_vc))
            note_interp = QLabel(
                "Cm spread > 0.05 (heuristic threshold) or a ρ interval that excludes 0 → "
                "the parameter AFFECTS the result (highlighted in blue/red)."
            )
            note_interp.setWordWrap(True)
            note_interp.setStyleSheet("font-size:10px; color:#555; margin-top:4px")
            grp_q_lo.addWidget(note_interp)
        else:
            grp_q_lo.addWidget(QLabel("Not enough data for analysis (< 10 runs)."))

        inner_lo.addWidget(grp_q)

        # ── File paths ───────────────────────────────────────────────────────
        if csv_path and os.path.exists(csv_path):
            lbl_csv = QLabel(f"📋 CSV: <code>{csv_path}</code>")
            lbl_csv.setWordWrap(True)
            inner_lo.addWidget(lbl_csv)
        if raster_path and os.path.exists(raster_path):
            lbl_r = QLabel(
                f"🗺 The h_max raster of the best run has been added to QGIS: "
                f"<code>{os.path.basename(raster_path)}</code>")
            lbl_r.setWordWrap(True)
            inner_lo.addWidget(lbl_r)

        scroll.setWidget(inner)
        lo.addWidget(scroll)

        # ── Close button ───────────────────────────────────────────────────────
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lo.addWidget(bb)
