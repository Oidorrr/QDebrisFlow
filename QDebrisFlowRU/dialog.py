# -*- coding: utf-8 -*-
"""
Debris Flow O'Brien — диалог симуляции v1.0

Новое в v1.0:
  - Вкладка 1 «Слои и источники»:
      Таблица объектов линейного слоя. Для каждой линии (по ID)
      независимо задаётся свой гидрограф Q(t).
  - Вкладка 4 «Выходные данные»:
      Перемещены prefix и папка; чекбоксы для выбора слоёв.
  - HydrographEditorDialog:
      Всплывающий редактор гидрографа для одного объекта.
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
from qgis.core import QgsMapLayerProxyModel, QgsWkbTypes, QgsGeometry, QgsPointXY
from qgis.gui import QgsMapLayerComboBox

_H_MIN = 0.005   # мин. глубина мокрой ячейки [м] — скрыто от пользователя
_CFL   = 0.40    # число Куранта — скрыто от пользователя


# ─────────────────────────────────────────────────────────────────────────────
# Модульные вспомогательные функции (sweep)
# ─────────────────────────────────────────────────────────────────────────────

def _lhs_samples(param_ranges: dict, n: int, seed: int = 42) -> list:
    """
    Latin Hypercube Sampling — непрерывный охват пространства без фикс. шага.
    Ключи "log_X" автоматически раскрываются: val → 10^val, ключ → "X".
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



# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный диалог: редактор гидрографа одной линии
# ─────────────────────────────────────────────────────────────────────────────

class HydrographEditorDialog(QDialog):
    """
    Всплывающий диалог для задания гидрографа Q(t) одного источника.

    Возвращает через .get_hydrograph() → (times: list[float], q: list[float])
    или None если пользователь нажал Cancel.
    """

    def __init__(self, parent, feature_id, feature_name: str,
                 init_data=None):
        super().__init__(parent)
        self.setWindowTitle(f"Гидрограф — {feature_name} (ID={feature_id})")
        self.setMinimumWidth(420)
        self._result = None
        self._build_ui(init_data)

    def _build_ui(self, init=None):
        layout = QVBoxLayout(self)

        # Тип гидрографа
        self.combo_type = QComboBox()
        self.combo_type.addItems([
            "Постоянный расход (Constant Q)",
            "Таблица (Время, Q)",
            "CSV файл (time, Q)",
        ])
        layout.addWidget(QLabel("Тип гидрографа:"))
        layout.addWidget(self.combo_type)

        # Секция 1: постоянный Q
        self._sec_const = QWidget()
        lay_c = QFormLayout(self._sec_const)
        lay_c.setContentsMargins(0, 4, 0, 4)
        self.spin_q = QDoubleSpinBox()
        self.spin_q.setRange(0, 1_000_000); self.spin_q.setValue(100)
        self.spin_q.setSuffix(" м³/с"); self.spin_q.setDecimals(1)
        lay_c.addRow("Q:", self.spin_q)
        layout.addWidget(self._sec_const)

        # Секция 2: таблица
        self._sec_table = QWidget()
        lay_t = QVBoxLayout(self._sec_table)
        lay_t.setContentsMargins(0, 4, 0, 4)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Время (с)", "Q (м³/с)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(4)
        defaults = [(0, 0), (300, 100), (600, 50), (900, 0)]
        for i, (t, q) in enumerate(defaults):
            self.table.setItem(i, 0, QTableWidgetItem(str(t)))
            self.table.setItem(i, 1, QTableWidgetItem(str(q)))
        self.table.setMaximumHeight(160)
        lay_t.addWidget(self.table)
        row_btns = QHBoxLayout()
        btn_add = QPushButton("+ строка")
        btn_add.clicked.connect(
            lambda: self.table.setRowCount(self.table.rowCount() + 1))
        btn_del = QPushButton("− строка")
        btn_del.clicked.connect(
            lambda: self.table.setRowCount(max(2, self.table.rowCount() - 1)))
        row_btns.addWidget(btn_add); row_btns.addWidget(btn_del)
        row_btns.addStretch()
        lay_t.addLayout(row_btns)
        layout.addWidget(self._sec_table)

        # Секция 3: CSV
        self._sec_csv = QWidget()
        lay_csv = QHBoxLayout(self._sec_csv)
        lay_csv.setContentsMargins(0, 4, 0, 4)
        self.btn_csv = QPushButton("📂 Выбрать CSV...")
        self.btn_csv.clicked.connect(self._load_csv)
        self.lbl_csv = QLabel("не выбран")
        self.lbl_csv.setStyleSheet("color:gray")
        lay_csv.addWidget(self.btn_csv); lay_csv.addWidget(self.lbl_csv)
        lay_csv.addStretch()
        layout.addWidget(self._sec_csv)

        # Кнопки OK / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Сигнал переключения
        self.combo_type.currentIndexChanged.connect(self._update_sections)

        # Восстановить сохранённые данные
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
            self, "CSV гидрограф", "", "CSV (*.csv);;All (*)")
        if not fname:
            return
        try:
            data = np.loadtxt(fname, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            self.table.setRowCount(len(data))
            for i, (t, q) in enumerate(data[:, :2]):
                self.table.setItem(i, 0, QTableWidgetItem(f"{t:.1f}"))
                self.table.setItem(i, 1, QTableWidgetItem(f"{q:.3f}"))
            self.lbl_csv.setText(os.path.basename(fname))
            self.lbl_csv.setStyleSheet("color:green")
            self.combo_type.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка CSV", str(e))

    def _restore(self, data: dict):
        """Восстановить ранее сохранённые данные."""
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
                self.table.setItem(i, 0, QTableWidgetItem(f"{t:.1f}"))
                self.table.setItem(i, 1, QTableWidgetItem(f"{q:.3f}"))
        self._update_sections()

    def _on_ok(self):
        try:
            self._result = self._parse()
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _parse(self) -> dict:
        """Разобрать введённые данные → dict для хранения."""
        idx = self.combo_type.currentIndex()
        if idx == 0:
            q = self.spin_q.value()
            return {"type": "const", "q_const": q,
                    "times": [0.0, 1e9], "q_vals": [q, q]}
        # Таблица / CSV
        times, q_vals = [], []
        for i in range(self.table.rowCount()):
            ti = self.table.item(i, 0)
            qi = self.table.item(i, 1)
            if ti is None or qi is None or not ti.text() or not qi.text():
                continue
            try:
                times.append(float(ti.text()))
                q_vals.append(float(qi.text()))
            except ValueError:
                pass
        if len(times) < 2:
            raise ValueError(
                "Нужно минимум 2 строки с числами.\n"
                "Пример: (0, 0) и (300, 100).")
        return {"type": "table", "times": times, "q_vals": q_vals,
                "q_const": 0.0}

    def get_result(self):
        return self._result


# ─────────────────────────────────────────────────────────────────────────────
# Фоновый поток симуляции
# ─────────────────────────────────────────────────────────────────────────────

class SimulationWorker(QThread):
    progress    = pyqtSignal(float, float)
    log_message = pyqtSignal(str)
    finished    = pyqtSignal(object)
    error       = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    # ── Главный метод потока ──────────────────────────────────────────────────

    def run(self):
        try:
            import sys
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            from core.rheology import OBrienParameters
            from core.solver   import (TerrainGrid, SimulationConfig,
                                        SourceCondition, InflowSource,
                                        DebrisFlowSolver2D)
            from qgis_runner   import read_dem_array, write_geotiff

            p = self.params

            # ── DEM ───────────────────────────────────────────────────────────
            self.log_message.emit("Чтение DEM...")
            dem, geo_info = read_dem_array(p["dem_layer"])

            # ── CRS check ─────────────────────────────────────────────────────
            src_layer = p["source_layer"]
            if p["dem_layer"].crs() != src_layer.crs():
                self.log_message.emit(
                    f"⚠ CRS не совпадают: DEM={p['dem_layer'].crs().authid()}, "
                    f"линии={src_layer.crs().authid()}. "
                    "Перепроецируйте линейный слой в CRS DEM.")

            # ── Растеризация по объектам → InflowSource на каждый ────────────
            # p["sources"] = list of {fid, name, times, q_vals}
            inflow_sources = []
            for src_info in p["sources"]:
                fid  = src_info["fid"]
                name = src_info["name"]
                self.log_message.emit(f"  Растеризация: {name} (ID={fid})...")

                feat = next(
                    (f for f in src_layer.getFeatures() if f.id() == fid),
                    None)
                if feat is None or not feat.geometry():
                    self.log_message.emit(f"    ⚠ Объект ID={fid} не найден, пропуск.")
                    continue

                rows, cols = self._rasterise_feature(feat, geo_info)
                if len(rows) == 0:
                    self.log_message.emit(
                        f"    ⚠ Объект ID={fid} не пересекает DEM, пропуск.")
                    continue

                self.log_message.emit(
                    f"    ✓ {len(rows)} ячеек, "
                    f"Q_max={max(src_info['q_vals']):.1f} м³/с")

                inflow_sources.append(InflowSource(
                    rows    = rows,
                    cols    = cols,
                    times   = np.array(src_info["times"],  dtype=np.float64),
                    q_total = np.array(src_info["q_vals"], dtype=np.float64),
                    name    = name,
                ))

            if not inflow_sources:
                raise RuntimeError(
                    "Ни один объект источника не пересекает DEM.\n"
                    "Проверьте CRS и экстент линейного слоя.")

            source = SourceCondition.from_sources(inflow_sources)
            self.log_message.emit(
                f"Итого источников: {len(inflow_sources)}")

            # ── O'Brien параметры ─────────────────────────────────────────────
            self.log_message.emit("Инициализация реологии O'Brien...")
            obrien = OBrienParameters(
                Cv               = p["Cv"],
                Cv_max           = p["Cv_max"],
                rho_sediment     = p.get("rho_s", 2650.0),
                rho_water        = p.get("rho_w", 1000.0),
                tau_yield_direct = p.get("tau_yield_direct", 0.0),
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
                progress_callback = lambda tc, te: (
                    self.progress.emit(tc, te),
                    self.log_message.emit(f"  t = {tc:.0f} с / {te:.0f} с")
                ),
            )

            self.log_message.emit("Запуск симуляции...")
            result = DebrisFlowSolver2D(terrain, obrien, source, config).run()

            # ── Запись результатов ────────────────────────────────────────────
            out_dir = p["output_dir"]
            os.makedirs(out_dir, exist_ok=True)
            prefix  = p["output_prefix"]
            wanted  = p["output_layers"]   # set of keys

            # Применяем порог глубины если задан
            h_thr = p.get("h_threshold", 0.0)

            def _thr(arr):
                if h_thr <= 0.0:
                    return arr
                out = arr.copy()
                out[out < h_thr] = -9999.0
                return out

            # Полная карта: ключ → (массив, zero_as_nodata)
            all_outputs = {
                "h_max":     (_thr(result.h_max),   True),
                "V_max":     (result.V_max,          True),
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
                self.log_message.emit(f"  Сохранён: {path}")

            vb = result.volume_balance()
            self.log_message.emit(
                f"\n── Итоги ──\n"
                f"  h_max пик:  {result.h_max.max():.3f} м\n"
                f"  V_max пик:  {result.V_max.max():.3f} м/с\n"
                f"  Шагов:      {result.n_steps}\n"
                f"  Время сим.: {result.t_simulated:.1f} с\n"
                f"  Δ масса:    {vb.get('V_loss_pct', float('nan')):.3f}%"
            )

            self.finished.emit({
                "result":   result,
                "paths":    paths,
                "geo_info": geo_info,
                "prefix":   prefix,
                "wanted":   wanted,
            })

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    # ── Растеризация одного объекта ───────────────────────────────────────────

    def _rasterise_feature(self, feature, geo_info: dict):
        """
        Растеризует один объект (линию) → (rows_array, cols_array).
        Использует GDAL через OGR с in-memory слоем, без записи на диск.
        Fallback: обход ячеек через QGIS geometry.distance().
        """
        try:
            from osgeo import gdal, ogr, osr

            nx, ny = geo_info["nx"], geo_info["ny"]
            gt     = geo_info["geotransform"]
            crs    = geo_info.get("crs")

            # In-memory vector dataset с одним объектом
            mem_vec_drv = ogr.GetDriverByName("Memory")
            vec_ds      = mem_vec_drv.CreateDataSource("tmp")

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
            wkt = feature.geometry().asWkt()
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

        except Exception:
            # Fallback: QGIS distance-based
            dx, dy   = geo_info["dx"], geo_info["dy"]
            xmin     = geo_info["xmin"]
            ymax     = geo_info["ymax"]
            nx, ny   = geo_info["nx"], geo_info["ny"]
            geom     = feature.geometry()
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
# Фоновый поток LHS-sweep
# ─────────────────────────────────────────────────────────────────────────────

class SweepWorker(QThread):
    """
    Последовательный LHS-sweep в фоновом QThread.
    Параллельный mp.Pool несовместим с Qt-event-loop QGIS,
    поэтому прогоны идут один за другим, но UI не блокируется.
    Сигналы:
        run_done(dict)    — результат одного прогона
        sweep_log(str)    — текстовое сообщение
        sweep_finished()  — все прогоны завершены
        sweep_error(str)  — критическая ошибка
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

    # ── главный метод ─────────────────────────────────────────────────────────

    def run(self):
        import sys, time
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        # Импортируем sweep_runner ПОСЛЕ добавления plugin_dir в sys.path
        from core import sweep_runner as _sr

        try:
            from qgis_runner import read_dem_array

            p = self.params

            # ── DEM и растеризация источников ─────────────────────────────────
            self.sweep_log.emit("Чтение DEM...")
            dem, geo_info = read_dem_array(p["dem_layer"])

            self.sweep_log.emit("Растеризация источников...")
            src_cells = []
            for src_info in p["sources"]:
                feat = next(
                    (f for f in p["source_layer"].getFeatures()
                     if f.id() == src_info["fid"]), None)
                if feat is None:
                    continue
                rows, cols = self._rasterise_feature(feat, geo_info)
                if len(rows) == 0:
                    continue
                src_cells.append({
                    "rows":        rows,
                    "cols":        cols,
                    "times":       np.array(src_info["times"],  dtype=np.float64),
                    "base_q_vals": np.array(src_info["q_vals"], dtype=np.float64),
                    "name":        src_info["name"],
                })

            if not src_cells:
                self.sweep_error.emit("Ни один источник не пересекает DEM.")
                return

            # Базовый объём
            base_vol = 0.0
            for sc in src_cells:
                t, q = sc["times"], sc["base_q_vals"]
                finite = t < 1e8
                if finite.sum() >= 2:
                    _trapz = getattr(np, "trapezoid", None) or np.trapz
                    base_vol += float(_trapz(q[finite], t[finite]))
            if base_vol <= 0:
                self.sweep_error.emit(
                    "Базовый объём = 0. Проверьте гидрографы источников.")
                return
            self.sweep_log.emit(
                f"Базовый объём: {base_vol:.0f} м³ "
                f"({len(src_cells)} источников)")

            # ── Наблюдаемая инундация ─────────────────────────────────────────
            self.sweep_log.emit("Растеризация полигона инундации...")
            obs_mask = self._rasterize_polygon(p["obs_poly_layer"], geo_info)
            self.sweep_log.emit(f"Наблюдаемая инундация: {obs_mask.sum()} ячеек")

            # ── Точки глубин ──────────────────────────────────────────────────
            self.sweep_log.emit("Чтение точек глубин...")
            obs_rows, obs_cols, obs_depths = self._extract_depth_points(
                p["obs_pts_layer"], p["depth_field"], geo_info)
            self.sweep_log.emit(f"Точек замеров в домене: {len(obs_depths)}")

            # ── LHS выборка ───────────────────────────────────────────────────
            n_runs    = p["n_runs"]
            seed      = p.get("seed", 42)
            n_workers = max(1, int(p.get("n_workers", 1)))
            samples   = _lhs_samples(p["param_ranges"], n_runs, seed=seed)
            self.sweep_log.emit(
                f"\nЗапуск {n_runs} прогонов (LHS seed={seed}, "
                f"ядра={n_workers})...")

            # ── Общий контекст задания (сериализуемый) ────────────────────────
            # geo_info из read_dem_array содержит crs (QgsCoordinateReferenceSystem)
            # и, возможно, GDAL-объекты — всё непригодно для pickle.
            # Дочернему процессу (sweep_runner) нужны только dx, dy, nodata.
            _gt = geo_info.get("geotransform", ())
            _geo_serial = dict(
                nx           = int(geo_info["nx"]),
                ny           = int(geo_info["ny"]),
                dx           = float(geo_info["dx"]),
                dy           = float(geo_info["dy"]),
                xmin         = float(geo_info.get("xmin", 0.0)),
                ymax         = float(geo_info.get("ymax", 0.0)),
                nodata       = float(geo_info.get("nodata") or -9999.0),
                # Явное приведение каждого элемента → чистые Python float
                geotransform = tuple(float(x) for x in _gt),
                # crs НЕ включается — QgsCoordinateReferenceSystem не picklable
            )

            job_base = dict(
                plugin_dir      = plugin_dir,
                dem             = dem,
                geo_info        = _geo_serial,   # только примитивы → picklable
                src_cells       = src_cells,
                base_vol        = base_vol,
                obs_mask        = obs_mask,
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
                depth_threshold = p["depth_threshold"],
            )

            # Список заданий
            job_args = [
                dict(job_base, run_id=i, sample=s)
                for i, s in enumerate(samples)
            ]

            # Проверка не нужна: потоки не требуют pickle
            t_start   = time.time()
            _best_Cm  = float("inf")
            _best_pkg = None
            completed = 0

            # ── Вспомогательная функция обработки одного результата ───────────
            def _process_result(row_data: dict):
                nonlocal completed, _best_Cm, _best_pkg

                h_max_arr = row_data.pop("h_max", None)  # убираем из CSV-строки
                eta_min   = round(
                    (time.time() - t_start) / max(completed, 1)
                    * (n_runs - completed) / 60, 1)
                row_data["eta_min"] = eta_min

                self.run_done.emit(row_data)

                rid   = row_data.get("run_id", "?")
                cm_v  = row_data.get("Cm")
                status= row_data.get("status", "")

                if status == "ok":
                    self.sweep_log.emit(
                        f"  ✓ [{completed}/{n_runs}]  "
                        f"V={row_data.get('vol_m3', '?'):.0f}м³  "
                        f"τy={row_data.get('tau_yield_Pa', '?'):.1f}Па  "
                        f"η={row_data.get('eta_Pas', '?'):.1f}Па·с  "
                        f"n={row_data.get('manning_n', '?'):.3f}  "
                        f"K={row_data.get('K_visc', '?'):.1f}  "
                        f"Cm={cm_v}  ETA {eta_min:.1f}мин")
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

            # ── Выполнение: последовательно (n=1) или через потоки (n>1) ──────
            if n_workers == 1:
                # Последовательный режим
                for job in job_args:
                    if self._cancelled:
                        self.sweep_log.emit("⚠ Sweep отменён пользователем.")
                        break
                    row_data  = _sr.run_one_sample(job)
                    completed += 1
                    _process_result(row_data)

            else:
                # Параллельный режим через ThreadPoolExecutor.
                # Потоки работают внутри QGIS-процесса → не нужен spawn,
                # не нужен pickle. Numba njit освобождает GIL при выполнении
                # → реальный параллелизм на уровне C/JIT-кода.
                from concurrent.futures import ThreadPoolExecutor, as_completed

                self.sweep_log.emit(
                    f"⚙ Запуск {n_workers} параллельных потоков...")

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_map = {
                        executor.submit(_sr.run_one_sample, job): job["run_id"]
                        for job in job_args
                    }
                    for future in as_completed(future_map):
                        if self._cancelled:
                            self.sweep_log.emit("⚠ Sweep отменён пользователем.")
                            # Отменяем ещё не запущенные futures
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
                                    "vol_m3","tau_yield_Pa","eta_Pas","manning_n",
                                    "K_visc","h_max_m","wall_s","eta_min",
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

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _rasterise_feature(self, feature, geo_info):
        """Та же растеризация что и в SimulationWorker."""
        try:
            from osgeo import gdal, ogr, osr
            nx, ny = geo_info["nx"], geo_info["ny"]
            gt     = geo_info["geotransform"]
            crs    = geo_info.get("crs")
            mem_vec = ogr.GetDriverByName("Memory").CreateDataSource("tmp")
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
                ogr.CreateGeometryFromWkt(feature.geometry().asWkt()))
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
        except Exception:
            dx, dy = geo_info["dx"], geo_info["dy"]
            xmin, ymax = geo_info["xmin"], geo_info["ymax"]
            nx, ny = geo_info["nx"], geo_info["ny"]
            geom   = feature.geometry()
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

    def _rasterize_polygon(self, poly_layer, geo_info) -> np.ndarray:
        """
        Растеризует полигональный слой инундации → bool-маска (ny, nx).
        Значение 1 = наблюдалась инундация.
        """
        try:
            from osgeo import gdal, ogr, osr
            nx, ny = geo_info["nx"], geo_info["ny"]
            gt     = geo_info["geotransform"]
            crs    = geo_info.get("crs")

            mem_vec = ogr.GetDriverByName("Memory").CreateDataSource("tmp")
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

            for feat in poly_layer.getFeatures():
                if feat.geometry() is None:
                    continue
                ogr_feat = ogr.Feature(lyr.GetLayerDefn())
                ogr_feat.SetGeometry(
                    ogr.CreateGeometryFromWkt(feat.geometry().asWkt()))
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
            # Fallback через QGIS geometry (медленно)
            self.sweep_log.emit(
                f"  ⚠ GDAL polygon rasterize failed: {e} — fallback QGIS")
            nx, ny = geo_info["nx"], geo_info["ny"]
            dx, dy = geo_info["dx"], geo_info["dy"]
            xmin, ymax = geo_info["xmin"], geo_info["ymax"]
            mask = np.zeros((ny, nx), dtype=bool)
            polys = [f.geometry() for f in poly_layer.getFeatures()
                     if f.geometry() is not None]
            for row in range(ny):
                cy = ymax - (row + 0.5) * dy
                for col in range(nx):
                    cx = xmin + (col + 0.5) * dx
                    pt = QgsGeometry.fromPointXY(QgsPointXY(cx, cy))
                    if any(g.contains(pt) for g in polys):
                        mask[row, col] = True
            return mask

    def _extract_depth_points(self, pts_layer, depth_field: str,
                               geo_info: dict):
        """
        Извлекает точки наблюдаемых глубин из point SHP.
        Возвращает (row_arr, col_arr, depth_arr) — только точки в домене.
        """
        gt   = geo_info["geotransform"]
        dx   = geo_info["dx"]
        dy   = geo_info["dy"]
        xmin = gt[0]
        ymax = gt[3]
        ny   = geo_info["ny"]
        nx   = geo_info["nx"]

        rows, cols, depths = [], [], []
        for feat in pts_layer.getFeatures():
            geom = feat.geometry()
            if geom is None:
                continue
            pt = geom.asPoint()
            col = int((pt.x() - xmin) / dx)
            row = int((ymax - pt.y()) / dy)
            if 0 <= row < ny and 0 <= col < nx:
                try:
                    d = float(feat[depth_field])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(row)
                cols.append(col)
                depths.append(d)

        return (np.array(rows, dtype=np.int64),
                np.array(cols, dtype=np.int64),
                np.array(depths, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Главный диалог
# ─────────────────────────────────────────────────────────────────────────────

class DebrisFlowDialog(QDialog):
    """
    Диалог симуляции с четырьмя вкладками:
      1. Слои и источники   — DEM, линейный слой, таблица источников
      2. O'Brien Parameters — реология
      3. Solver Settings    — временные параметры
      4. Выходные данные    — папка, prefix, выбор слоёв
    """

    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface  = iface
        self.worker = None
        # Хранилище гидрографов: {feature_id (int): dict}
        self._hydro_data     = {}
        self.sweep_worker    = None
        self._sweep_results  = []
        self._sweep_best_pkg = None
        self.setWindowTitle(
            "Моделирование селевых потоков — O'Brien Quadratic  (v1.0)")
        self.setMinimumWidth(660)
        self.resize(740, 580)
        self._build_ui()

    # ── Основная компоновка ───────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_sources(),   "1. Слои и источники")
        self.tabs.addTab(self._tab_rheology(),  "2. Параметры O'Brien")
        self.tabs.addTab(self._tab_solver(),    "3. Настройки решателя")
        self.tabs.addTab(self._tab_output(),    "4. Выходные данные")
        self.tabs.addTab(self._tab_sweep(),     "5. LHS-анализ")
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
        self.btn_run = QPushButton("▶  Запуск симуляции")
        self.btn_run.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_run.clicked.connect(self._on_run)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_run)
        btns.addWidget(btn_close)
        main.addLayout(btns)

    # ── Вкладка 1: Слои и источники ───────────────────────────────────────────

    def _tab_sources(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # DEM
        grp_dem = QGroupBox("ЦМР (DEM)")
        fl_dem = QFormLayout(grp_dem)
        self.dem_combo = QgsMapLayerComboBox()
        self.dem_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        fl_dem.addRow("DEM слой:", self.dem_combo)
        layout.addWidget(grp_dem)

        # Линейный слой источников
        grp_src = QGroupBox("Линейный слой источников")
        lay_src = QVBoxLayout(grp_src)

        src_row = QHBoxLayout()
        self.src_line_combo = QgsMapLayerComboBox()
        self.src_line_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        src_row.addWidget(self.src_line_combo)
        btn_load = QPushButton("⟳ Загрузить объекты")
        btn_load.setToolTip(
            "Прочитать объекты (линии) из выбранного слоя\n"
            "и заполнить таблицу источников.")
        btn_load.clicked.connect(self._load_features)
        src_row.addWidget(btn_load)
        lay_src.addLayout(src_row)

        # Таблица объектов
        info_lbl = QLabel(
            "Для каждой линии задайте гидрограф Q(t).\n"
            "Нажмите «Редактировать» в строке нужного объекта.")
        info_lbl.setStyleSheet("color:#555; font-size:10px;")
        lay_src.addWidget(info_lbl)

        self.tbl_sources = QTableWidget()
        self.tbl_sources.setColumnCount(4)
        self.tbl_sources.setHorizontalHeaderLabels(
            ["ID", "Описание", "Q_max (м³/с)", "Гидрограф"])
        hdr = self.tbl_sources.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tbl_sources.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_sources.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_sources.setMinimumHeight(160)
        lay_src.addWidget(self.tbl_sources)

        layout.addWidget(grp_src)

        # ── Загрузка параметров ───────────────────────────────────────────────
        grp_load = QGroupBox("Загрузка параметров из файла")
        grp_load.setStyleSheet(
            "QGroupBox{font-size:11px;color:#555;"
            "border:1px solid #bbb;border-radius:4px;margin-top:4px}"
            "QGroupBox::title{padding:0 4px}")
        load_row = QHBoxLayout(grp_load)
        load_row.setContentsMargins(6, 10, 6, 6)
        load_row.addWidget(QLabel("Файл:"))
        self.edit_load_path = QLineEdit()
        self.edit_load_path.setPlaceholderText("Путь к JSON-файлу параметров …")
        self.edit_load_path.setToolTip(
            "JSON-файл, сохранённый ранее кнопкой «Сохранить параметры»\n"
            "на вкладке «Выходные данные».\n"
            "Слои подбираются по имени в текущем проекте QGIS.")
        load_row.addWidget(self.edit_load_path)
        btn_lb = QPushButton("…")
        btn_lb.setFixedWidth(28)
        btn_lb.clicked.connect(self._browse_load_file)
        load_row.addWidget(btn_lb)
        btn_lp = QPushButton("📂 Загрузить")
        btn_lp.setToolTip(
            "Восстановить все параметры из JSON-файла.\n"
            "Слои, гидрографы, реология, настройки решателя\n"
            "и параметры вывода будут заполнены из файла.")
        btn_lp.clicked.connect(self._load_params)
        load_row.addWidget(btn_lp)
        layout.addWidget(grp_load)

        layout.addStretch()
        return w

    # ── Вкладка 2: Параметры O'Brien ──────────────────────────────────────────

    def _tab_rheology(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Концентрация
        grp_conc = QGroupBox("Седиментная концентрация")
        fl = QFormLayout(grp_conc)
        self.spin_Cv = QDoubleSpinBox()
        self.spin_Cv.setRange(0.01, 0.74); self.spin_Cv.setValue(0.40)
        self.spin_Cv.setDecimals(3)
        self.spin_Cv.setToolTip(
            "Объёмная концентрация осадка [-].\n"
            "Постпожарный диапазон: 0.30–0.55")
        fl.addRow("Cv:", self.spin_Cv)
        self.spin_Cv_max = QDoubleSpinBox()
        self.spin_Cv_max.setRange(0.60, 0.90); self.spin_Cv_max.setValue(0.615)
        self.spin_Cv_max.setDecimals(3)
        self.spin_Cv_max.setToolTip(
            "Предел упаковки Бэгнолда.\nУвеличить если Cv > 0.50.")
        fl.addRow("Cv_max:", self.spin_Cv_max)
        layout.addWidget(grp_conc)

        # ── Плотности компонентов ─────────────────────────────────────────────
        grp_rho = QGroupBox("Плотности компонентов")
        fl_rho  = QFormLayout(grp_rho)

        self.spin_rho_s = QDoubleSpinBox()
        self.spin_rho_s.setRange(1000.0, 3500.0)
        self.spin_rho_s.setValue(2650.0)
        self.spin_rho_s.setDecimals(0)
        self.spin_rho_s.setSingleStep(10.0)
        self.spin_rho_s.setSuffix(" кг/м³")
        self.spin_rho_s.setToolTip(
            "Плотность твёрдой фазы ρs [кг/м³].\n\n"
            "Типичные значения:\n"
            "  2650 — кварц/гранит (стандарт FLO-2D, Barnhart 2021)\n"
            "  2700–2800 — андезит, базальт\n"
            "  2600–2700 — известняк\n"
            "  1200–2200 — вулканический пепел\n\n"
            "Barnhart et al. (2021): ρs зафиксирована на 2650 кг/м³,\n"
            "т.к. скрининг Морриса (1991) показал нечувствительность\n"
            "результатов к её вариации.\n\n"
            "Влияет на: γm = ρm·g → τy/(γm·h), K·η/(8·γm·h²).")
        fl_rho.addRow("ρs — осадок:", self.spin_rho_s)

        self.spin_rho_w = QDoubleSpinBox()
        self.spin_rho_w.setRange(900.0, 1100.0)
        self.spin_rho_w.setValue(1000.0)
        self.spin_rho_w.setDecimals(0)
        self.spin_rho_w.setSingleStep(1.0)
        self.spin_rho_w.setSuffix(" кг/м³")
        self.spin_rho_w.setToolTip(
            "Плотность жидкой фазы ρw [кг/м³].\n\n"
            "  1000 — пресная вода (стандарт)\n"
            "  1025 — морская вода\n\n"
            "Для всех постпожарных событий материковой зоны\n"
            "оставьте 1000 кг/м³.")
        fl_rho.addRow("ρw — жидкость:", self.spin_rho_w)

        lbl_rho_note = QLabel(
            "ρm = ρs·Cv + ρw·(1–Cv)  →  γm = ρm·g  "
            "[используется в формуле O'Brien]")
        lbl_rho_note.setWordWrap(True)
        lbl_rho_note.setStyleSheet("color:#666; font-size:10px; margin-top:2px")
        fl_rho.addRow(lbl_rho_note)

        # Индикатор ρm (пересчитывается при изменении Cv, ρs, ρw)
        self._lbl_rho_m = QLabel()
        self._lbl_rho_m.setStyleSheet("color:#1565C0; font-size:10px; font-weight:bold")
        fl_rho.addRow("ρm (текущее):", self._lbl_rho_m)
        self.spin_rho_s.valueChanged.connect(self._update_rho_m)
        self.spin_rho_w.valueChanged.connect(self._update_rho_m)

        layout.addWidget(grp_rho)

        # Переключатель режима
        grp_mode = QGroupBox("Режим задания τy и η")
        lay_mode = QVBoxLayout(grp_mode)
        self._radio_direct = QRadioButton("Прямой ввод")
        self._radio_exp = QRadioButton(
            "Экспоненциальные коэффициенты")
        self._radio_direct.setChecked(True)
        self._mode_grp = QButtonGroup()
        self._mode_grp.addButton(self._radio_direct, 0)
        self._mode_grp.addButton(self._radio_exp, 1)
        lay_mode.addWidget(self._radio_direct)
        lay_mode.addWidget(self._radio_exp)
        layout.addWidget(grp_mode)

        # Режим 1: прямой ввод
        self._sec_direct = QGroupBox("Прямые параметры (Режим 1)")
        self._sec_direct.setStyleSheet(
            "QGroupBox{border:2px solid #2196F3;border-radius:4px;"
            "margin-top:6px;font-weight:bold}"
            "QGroupBox::title{color:#1565C0}")
        fl_d = QFormLayout(self._sec_direct)
        self.spin_tau = QDoubleSpinBox()
        self.spin_tau.setRange(0.01, 50000); self.spin_tau.setValue(600)
        self.spin_tau.setDecimals(2); self.spin_tau.setSuffix(" Па")
        self.spin_tau.setSingleStep(50)
        self.spin_tau.setToolTip(
            "Предел текучести [Па] "
            "Постпожарный сель: 200–800 Па.")
        fl_d.addRow("τy — Предел текучести:", self.spin_tau)
        self._lbl_hstop = QLabel()
        self._lbl_hstop.setStyleSheet("color:#555;font-size:10px;padding-left:4px")
        fl_d.addRow("", self._lbl_hstop)
        self.spin_tau.valueChanged.connect(self._update_hstop)
        self.spin_Cv.valueChanged.connect(self._update_hstop)
        self.spin_eta = QDoubleSpinBox()
        self.spin_eta.setRange(0.001, 10000); self.spin_eta.setValue(10.0)
        self.spin_eta.setDecimals(3); self.spin_eta.setSuffix(" Па·с")
        self.spin_eta.setSingleStep(5)
        self.spin_eta.setToolTip(
            "Вязкость смеси [Па·с] "
            "Постпожарный сель: 5–30 Па·с.")
        fl_d.addRow("η — Вязкость смеси:", self.spin_eta)
        layout.addWidget(self._sec_direct)

        # Режим 2: экспоненциальные коэффициенты
        self._sec_exp = QGroupBox("Экспоненциальные коэффициенты O'Brien (Режим 2)")
        fl_e = QFormLayout(self._sec_exp)
        warn = QLabel(
            "⚠ Коэффициенты откалиброваны на лабораторных образцах.\n"
            "Для полевых условий используйте Режим 1.")
        warn.setStyleSheet("color:#B26A00;font-size:10px")
        fl_e.addRow(warn)
        self.spin_a1 = QDoubleSpinBox()
        self.spin_a1.setRange(0.0001,10); self.spin_a1.setValue(0.00423)
        self.spin_a1.setDecimals(5); self.spin_a1.setSingleStep(0.0001)
        fl_e.addRow("α₁ [Па]:", self.spin_a1)
        self.spin_b1 = QDoubleSpinBox()
        self.spin_b1.setRange(1,30); self.spin_b1.setValue(13.11)
        self.spin_b1.setDecimals(2)
        fl_e.addRow("β₁ [-]:", self.spin_b1)
        self.spin_a2 = QDoubleSpinBox()
        self.spin_a2.setRange(0.001,10); self.spin_a2.setValue(0.0311)
        self.spin_a2.setDecimals(4); self.spin_a2.setSingleStep(0.001)
        fl_e.addRow("α₂ [Па·с]:", self.spin_a2)
        self.spin_b2 = QDoubleSpinBox()
        self.spin_b2.setRange(1,40); self.spin_b2.setValue(16.81)
        self.spin_b2.setDecimals(2)
        fl_e.addRow("β₂ [-]:", self.spin_b2)
        layout.addWidget(self._sec_exp)

        # Маннинг + K
        grp_n = QGroupBox("Турбулентный и вязкий члены")
        fl_n = QFormLayout(grp_n)
        self.spin_n = QDoubleSpinBox()
        self.spin_n.setRange(0.01, 0.30); self.spin_n.setValue(0.06)
        self.spin_n.setDecimals(3)
        self.spin_n.setToolTip(
            "Коэф. Маннинга.\n"
            "Горные каналы после пожара: 0.04–0.12")
        fl_n.addRow("Manning n:", self.spin_n)

        self.spin_K = QDoubleSpinBox()
        self.spin_K.setRange(0.001, 10000.0); self.spin_K.setValue(24.0)
        self.spin_K.setDecimals(3); self.spin_K.setSingleStep(1.0)
        self.spin_K.setToolTip(
            "K — безразмерный коэффициент формы поперечного сечения\n"
            "в вязком члене Sf_visc = K·η·V / (8·γm·h²).\n\n"
            "Теоретические значения (O'Brien & Julien 1985):\n"
            "  K = 24 — широкий прямоугольный канал (Бинго-поток)\n"
            "  K = 8  — трубное течение\n\n"
            "FLO-2D (Barnhart et al. 2021, Table 1):\n"
            "  log K от 1.38 до 3.70; экспертное мнение ≈ 2290.\n\n"
            "По умолчанию: 24 (аналитически строго обоснованное значение).")
        fl_n.addRow("K — коэф. формы [-]:", self.spin_K)
        layout.addWidget(grp_n)

        self._radio_direct.toggled.connect(self._on_rheo_mode)
        self._on_rheo_mode()
        self._update_hstop()
        self.spin_Cv.valueChanged.connect(self._update_rho_m)
        self._update_rho_m()   # первый расчёт при открытии
        layout.addStretch()
        return w

    # ── Вкладка 3: Solver Settings ────────────────────────────────────────────

    def _tab_solver(self) -> QWidget:
        w = QWidget()
        layout = QFormLayout(w)

        self.spin_tend = QSpinBox()
        self.spin_tend.setRange(60, 86400); self.spin_tend.setValue(3600)
        self.spin_tend.setSuffix(" с")
        layout.addRow("Длительность симуляции:", self.spin_tend)

        self.spin_dtmax = QDoubleSpinBox()
        self.spin_dtmax.setRange(0.1, 300); self.spin_dtmax.setValue(5.0)
        self.spin_dtmax.setSuffix(" с")
        self.spin_dtmax.setToolTip(
            "Максимальный внутренний шаг по времени [с].\n"
            "Реальный шаг ≤ dt_max и дополнительно ограничен CFL.\n"
            "НЕ путать с интервалом обновления прогресс-бара (60 с).\n\n"
            "Рекомендации:\n"
            "  DEM 90 м → 30–60 с\n"
            "  DEM 30 м → 10–30 с\n"
            "  DEM 10 м →  3–10 с\n"
            "  DEM  5 м →  1–5 с")
        layout.addRow("Макс. шаг dt_max:", self.spin_dtmax)

        self.spin_cfl = QDoubleSpinBox()
        self.spin_cfl.setRange(0.10, 0.90); self.spin_cfl.setValue(0.60)
        self.spin_cfl.setDecimals(2); self.spin_cfl.setSingleStep(0.05)
        self.spin_cfl.setToolTip(
            "Число Куранта–Фридрихса–Леви (CFL).\n"
            "Контролирует численную устойчивость явной схемы LIA.\n\n"
            "Рекомендуемые значения:\n"
            "  0.40–0.50 — консервативный (Bates et al. 2010)\n"
            "  0.60      — стандарт (по умолчанию)\n"
            "  > 0.70    — только при плоском рельефе, с осторожностью.\n\n"
            "При CFL > 0.9 возможна числовая неустойчивость.")
        layout.addRow("Число CFL:", self.spin_cfl)

        lbl = QLabel(
            "<i>h_min = 5 мм зафиксирован автоматически для устойчивости LIA.</i>")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#666;font-size:10px")
        layout.addRow(lbl)
        return w

    # ── Вкладка 4: Выходные данные ────────────────────────────────────────────

    def _tab_output(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Расположение файлов
        grp_loc = QGroupBox("Расположение результатов")
        fl = QFormLayout(grp_loc)

        self.edit_prefix = QLineEdit("Debris")
        self.edit_prefix.setToolTip(
            "Базовое имя выходных файлов.\n"
            "Результат: <prefix>_h_max.tif, <prefix>_V_max.tif, ...")
        fl.addRow("Префикс файлов:", self.edit_prefix)

        out_row = QHBoxLayout()
        self.edit_outdir = QLineEdit(os.path.expanduser("~/debris_output"))
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.edit_outdir)
        out_row.addWidget(btn_browse)
        fl.addRow("Папка для файлов:", out_row)
        layout.addWidget(grp_loc)

        # Выбор выходных слоёв
        grp_layers = QGroupBox("Выходные слои (выберите нужные)")
        fl_l = QVBoxLayout(grp_layers)

        self.chk_h_max = QCheckBox("Пиковая толщина потока  (h_max)")
        self.chk_h_max.setChecked(True)
        self.chk_h_max.setToolTip(
            "Максимальная глубина за всё время симуляции.\n"
            "Основной продукт для оценки опасности.")
        fl_l.addWidget(self.chk_h_max)

        self.chk_V_max = QCheckBox(
            "Пиковая скорость потока  (V_max)")
        self.chk_V_max.setChecked(True)
        self.chk_V_max.setToolTip(
            "Максимальная скорость потока [м/с] за всё время.\n"
            "Используется для оценки динамической нагрузки.")
        fl_l.addWidget(self.chk_V_max)

        self.chk_h_final = QCheckBox(
            "Финальная толщина потока  (h_final)")
        self.chk_h_final.setChecked(False)
        self.chk_h_final.setToolTip(
            "Толщина осадка в момент окончания симуляции.\n"
            "Показывает где поток остановился.")
        fl_l.addWidget(self.chk_h_final)

        self.chk_t_arrival = QCheckBox(
            "Время прихода потока  (t_arrival)")
        self.chk_t_arrival.setChecked(False)
        self.chk_t_arrival.setToolTip(
            "Момент времени [с] когда поток впервые достиг ячейки.\n"
            "Используется для планирования эвакуации.")
        fl_l.addWidget(self.chk_t_arrival)

        layout.addWidget(grp_layers)

        # Порог глубины (фильтрация слоёв глубины при записи)
        grp_thresh = QGroupBox("Порог глубины — фильтрация при записи растра")
        fl_th = QFormLayout(grp_thresh)

        self.chk_threshold = QCheckBox(
            "Показывать только ячейки с глубиной ≥ порога")
        self.chk_threshold.setChecked(False)
        self.chk_threshold.setToolTip(
            "Если включено — ячейки с глубиной меньше порога\n"
            "записываются как NoData (прозрачные в QGIS).\n"
            "Применяется к слоям h_max и h_final.")
        self.chk_threshold.toggled.connect(
            lambda on: self.spin_threshold.setEnabled(on))
        fl_th.addRow(self.chk_threshold)

        thresh_row = QHBoxLayout()
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.00, 10.0)
        self.spin_threshold.setValue(0.10)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setSuffix(" м")
        self.spin_threshold.setEnabled(False)
        self.spin_threshold.setToolTip(
            "Минимальная глубина [м].\n"
            "Ячейки с h < порога → NoData (прозрачные в QGIS).\n\n"
            "Примеры:\n"
            "  0.01 м — убрать тонкую плёнку на фронте\n"
            "  0.10 м — только значимые зоны затопления\n"
            "  0.30 м — порог опасности для пешеходов\n"
            "  1.00 м — порог опасности для зданий")
        thresh_row.addWidget(self.spin_threshold)

        for label, val in [("1 см", 0.01), ("10 см", 0.10),
                            ("30 см", 0.30), ("1 м", 1.00)]:
            btn = QPushButton(label)
            btn.setFixedWidth(46)
            btn.setToolTip(f"Установить порог {val} м")
            btn.clicked.connect(
                lambda checked, v=val: (
                    self.spin_threshold.setValue(v),
                    self.chk_threshold.setChecked(True),
                )
            )
            thresh_row.addWidget(btn)

        fl_th.addRow("Порог:", thresh_row)

        lbl_note = QLabel(
            "<i>Не влияет на симуляцию — только на запись растра.<br>"
            "Полные данные внутри плагина сохраняются без изменений.</i>")
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color:#666; font-size:10px;")
        fl_th.addRow(lbl_note)

        layout.addWidget(grp_thresh)

        # ── Сохранение параметров ─────────────────────────────────────────────
        grp_save = QGroupBox("Сохранение параметров в файл")
        grp_save.setStyleSheet(
            "QGroupBox{font-size:11px;color:#555;"
            "border:1px solid #bbb;border-radius:4px;margin-top:4px}"
            "QGroupBox::title{padding:0 4px}")
        save_row = QHBoxLayout(grp_save)
        save_row.setContentsMargins(6, 10, 6, 6)
        save_row.addWidget(QLabel("Файл:"))
        self.edit_save_path = QLineEdit(
            os.path.join(os.path.expanduser("~"), "debris_params.json"))
        self.edit_save_path.setToolTip(
            "Путь к JSON-файлу для сохранения всех параметров:\n"
            "слои, гидрографы, реология, решатель, выходные данные.")
        save_row.addWidget(self.edit_save_path)
        btn_sb = QPushButton("…")
        btn_sb.setFixedWidth(28)
        btn_sb.clicked.connect(self._browse_save_file)
        save_row.addWidget(btn_sb)
        btn_sp = QPushButton("💾 Сохранить")
        btn_sp.setToolTip(
            "Сохранить все текущие параметры в JSON-файл.\n"
            "Файл можно загрузить на вкладке «Слои и источники».")
        btn_sp.clicked.connect(self._save_params)
        save_row.addWidget(btn_sp)
        layout.addWidget(grp_save)

        layout.addStretch()
        return w

    # ── Вкладка 5: LHS-анализ ────────────────────────────────────────────────

    def _tab_sweep(self) -> QWidget:
        w  = QWidget()
        lo = QVBoxLayout(w)

        # ── Сохранение / загрузка настроек LHS ──────────────────────────────
        grp_lhs_io = QGroupBox("Настройки анализа")
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
        lbl_io = QLabel("Файл:")
        lbl_io.setFixedWidth(34)
        io_path_row.addWidget(lbl_io)
        self.edit_lhs_path = QLineEdit()
        self.edit_lhs_path.setPlaceholderText(
            "Путь к файлу настроек LHS-анализа (.json)  …")
        self.edit_lhs_path.setToolTip(
            "JSON-файл для сохранения/загрузки всех параметров\n"
            "вкладки LHS-анализ: наблюдаемые данные, диапазоны,\n"
            "число прогонов, seed и папка вывода.")
        io_path_row.addWidget(self.edit_lhs_path)
        btn_lhs_br = QPushButton("…")
        btn_lhs_br.setFixedWidth(28)
        btn_lhs_br.setToolTip("Выбрать файл")
        btn_lhs_br.clicked.connect(self._browse_lhs_file)
        io_path_row.addWidget(btn_lhs_br)
        io_lo.addLayout(io_path_row)

        io_btn_row = QHBoxLayout()
        btn_lhs_save = QPushButton("💾  Сохранить настройки LHS")
        btn_lhs_save.setToolTip(
            "Сохранить все параметры вкладки LHS в JSON-файл.\n"
            "Удобно для хранения разных сценариев анализа.")
        btn_lhs_save.clicked.connect(self._save_lhs_params)
        btn_lhs_load = QPushButton("📂  Загрузить настройки LHS")
        btn_lhs_load.setToolTip(
            "Восстановить параметры LHS-анализа из JSON-файла.\n"
            "Слои подбираются по имени в текущем проекте QGIS.")
        btn_lhs_load.clicked.connect(self._load_lhs_params)
        io_btn_row.addWidget(btn_lhs_save)
        io_btn_row.addWidget(btn_lhs_load)
        io_lo.addLayout(io_btn_row)
        lo.addWidget(grp_lhs_io)

        # ── Наблюдаемые данные ────────────────────────────────────────────────
        grp_obs = QGroupBox("Наблюдаемые данные (реальное событие)")
        fl_obs  = QFormLayout(grp_obs)

        self.combo_obs_poly = QgsMapLayerComboBox()
        self.combo_obs_poly.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.combo_obs_poly.setToolTip(
            "SHP-слой реальной зоны инундации (полигон).\n"
            "Используется для расчёта TP, FP, FN, ΩTm.")
        fl_obs.addRow("Зона инундации (SHP):", self.combo_obs_poly)

        self.combo_obs_pts = QgsMapLayerComboBox()
        self.combo_obs_pts.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.combo_obs_pts.setToolTip(
            "SHP-слой точек с полевыми замерами пиковых глубин потока.")
        self.combo_obs_pts.layerChanged.connect(self._on_obs_pts_changed)
        fl_obs.addRow("Точки глубин (SHP):", self.combo_obs_pts)

        self.combo_depth_field = QComboBox()
        self.combo_depth_field.setToolTip(
            "Числовой атрибут точечного слоя, содержащий\n"
            "значение пиковой глубины потока [м].")
        fl_obs.addRow("Поле глубины [м]:", self.combo_depth_field)

        self.spin_depth_thr = QDoubleSpinBox()
        self.spin_depth_thr.setRange(0.01, 5.0)
        self.spin_depth_thr.setValue(0.5)
        self.spin_depth_thr.setSuffix(" м")
        self.spin_depth_thr.setDecimals(2)
        self.spin_depth_thr.setToolTip(
            "Порог h [м] для классификации ячейки как «инундированной».\n"
            "Barnhart et al. (2021) обосновывают 0.5 м (Section 7.2).")
        fl_obs.addRow("Порог глубины:", self.spin_depth_thr)
        lo.addWidget(grp_obs)

        # ── Параметры: диапазон или фиксированное значение ───────────────────
        grp_rng = QGroupBox(
            "Параметры вариации — диапазон или фиксированное значение")
        rng_lo = QVBoxLayout(grp_rng)
        rng_lo.setSpacing(4)

        def _param_row(label, chk_attr, fix_attr, mn_attr, mx_attr,
                       fix_def, lo_def, hi_def, suffix, decs, step, rng):
            """Строка параметра с переключателем диапазон/фикс."""
            cw = QWidget()
            ch = QHBoxLayout(cw); ch.setContentsMargins(0, 0, 0, 0)

            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.setMinimumWidth(175)
            setattr(self, chk_attr, chk)
            ch.addWidget(chk)

            # Виджет диапазона
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
            rh.addWidget(QLabel("от")); rh.addWidget(s_min)
            rh.addWidget(QLabel("до")); rh.addWidget(s_max)
            ch.addWidget(rw)

            # Виджет фиксированного значения
            fw = QWidget()
            fh = QHBoxLayout(fw); fh.setContentsMargins(0, 0, 0, 0)
            fh.addWidget(QLabel("Значение:"))
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

        _param_row("Объём V варьировать",
                   "chk_vol_vary", "spin_vol_fixed",
                   "spin_vol_min",  "spin_vol_max",
                   fix_def=100_000, lo_def=50_000, hi_def=2_000_000,
                   suffix=" м³", decs=0, step=10_000,
                   rng=(1000, 1e8))

        _param_row("τy варьировать",
                   "chk_tau_vary", "spin_tau_fixed",
                   "spin_tau_min",  "spin_tau_max",
                   fix_def=600.0, lo_def=100.0, hi_def=1200.0,
                   suffix=" Па", decs=2, step=50,
                   rng=(0.01, 50000))

        _param_row("η варьировать",
                   "chk_eta_vary", "spin_eta_fixed",
                   "spin_eta_min",  "spin_eta_max",
                   fix_def=10.0, lo_def=2.0, hi_def=50.0,
                   suffix=" Па·с", decs=2, step=1.0,
                   rng=(0.001, 10000))

        _param_row("Manning n варьировать",
                   "chk_n_vary",  "spin_n_fixed",
                   "spin_n_min",  "spin_n_max",
                   fix_def=0.06, lo_def=0.04, hi_def=0.15,
                   suffix="", decs=4, step=0.01,
                   rng=(0.001, 0.30))

        _param_row("K — коэф. формы варьировать",
                   "chk_K_vary",  "spin_K_fixed",
                   "spin_K_min",  "spin_K_max",
                   fix_def=24.0, lo_def=24.0, hi_def=2400.0,
                   suffix="", decs=3, step=1.0,
                   rng=(0.001, 10000.0))

        # Фиксированные значения задаются прямо в этой вкладке.
        # (Cv берётся из вкладки «Параметры O'Brien».)
        note_rng = QLabel(
            "Включён — параметр варьируется в диапазоне (LHS-выборка).  "
            "Выключен — параметр фиксирован, одно значение для всех прогонов.\n"
            "При выключении τy, η, n значение берётся автоматически "
            "из вкладки «Параметры O'Brien».")
        note_rng.setWordWrap(True)
        note_rng.setStyleSheet("color:#777;font-size:10px;margin-top:2px")
        rng_lo.addWidget(note_rng)
        lo.addWidget(grp_rng)

        # ── Настройки sweep ───────────────────────────────────────────────────
        grp_cfg = QGroupBox("Настройки анализа")
        fl_cfg  = QFormLayout(grp_cfg)

        self.spin_n_runs = QSpinBox()
        self.spin_n_runs.setRange(10, 1000); self.spin_n_runs.setValue(100)
        self.spin_n_runs.setToolTip(
            "Число прогонов LHS.\n"
            "Barnhart et al.: 100·Np ≈ 300–500.\n"
            "Для первичного анализа: 50–100.")
        fl_cfg.addRow("Число прогонов:", self.spin_n_runs)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999); self.spin_seed.setValue(42)
        self.spin_seed.setToolTip(
            "Seed генератора — одинаковый seed → воспроизводимый результат.")
        fl_cfg.addRow("Random seed:", self.spin_seed)

        # Параллельные ядра
        self.spin_n_workers = QSpinBox()
        self.spin_n_workers.setRange(1, 16)
        self.spin_n_workers.setValue(1)
        self.spin_n_workers.setToolTip(
            "Число параллельных потоков для LHS-анализа.\n\n"
            "1 — последовательный режим (рекомендуется при первом запуске).\n"
            "2–8 — параллельные потоки (ThreadPoolExecutor).\n\n"
            "Потоки работают внутри QGIS-процесса — не нужен запуск\n"
            "дочерних процессов. При Numba JIT потоки реально ускоряют\n"
            "расчёт, так как njit освобождает GIL во время выполнения.\n\n"
            "Не превышайте число физических ядер вашего CPU.\n"
            "Максимум: 16.")
        fl_cfg.addRow("Параллельных ядер:", self.spin_n_workers)

        out_row2 = QHBoxLayout()
        self.edit_sweep_outdir = QLineEdit(os.path.expanduser("~/debris_sweep"))
        btn_sweep_out = QPushButton("…")
        btn_sweep_out.setFixedWidth(30)
        btn_sweep_out.clicked.connect(self._browse_sweep_outdir)
        out_row2.addWidget(self.edit_sweep_outdir)
        out_row2.addWidget(btn_sweep_out)
        fl_cfg.addRow("Папка результатов:", out_row2)

        self.edit_sweep_csv = QLineEdit("sweep_results.csv")
        self.edit_sweep_csv.setToolTip(
            "Имя CSV-файла (сохраняется в папке результатов).\n"
            "Содержит все 4 параметра + метрики для каждого сценария.")
        fl_cfg.addRow("Файл CSV:", self.edit_sweep_csv)
        lo.addWidget(grp_cfg)

        # ── Лог sweep ─────────────────────────────────────────────────────────
        self.sweep_log_box = QTextEdit()
        self.sweep_log_box.setReadOnly(True)
        self.sweep_log_box.setFont(QFont("Courier", 9))
        self.sweep_log_box.setMaximumHeight(130)
        lo.addWidget(self.sweep_log_box)

        # ── Кнопки ────────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_sweep_run = QPushButton("▶  Запустить LHS-анализ")
        self.btn_sweep_run.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_sweep_run.clicked.connect(self._on_sweep_run)
        self.btn_sweep_cancel = QPushButton("⏹  Остановить")
        self.btn_sweep_cancel.setEnabled(False)
        self.btn_sweep_cancel.clicked.connect(self._on_sweep_cancel)
        btn_row.addWidget(self.btn_sweep_run)
        btn_row.addWidget(self.btn_sweep_cancel)
        lo.addLayout(btn_row)

        # Оборачиваем в QScrollArea для работы на ноутбуках
        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    # ── Слоты вкладки 5 ──────────────────────────────────────────────────────

    def _on_obs_pts_changed(self, layer):
        """Заполнить combo_depth_field числовыми полями нового слоя точек."""
        self.combo_depth_field.clear()
        if layer is None:
            return
        numeric = {"double", "real", "float", "integer", "int64", "int",
                   "int2", "int4", "int8", "numeric", "decimal"}
        for field in layer.fields():
            if field.typeName().lower().split("(")[0] in numeric:
                self.combo_depth_field.addItem(field.name())
        # Fallback: добавить все поля если числовых нет
        if self.combo_depth_field.count() == 0:
            for field in layer.fields():
                self.combo_depth_field.addItem(field.name())

    def _browse_lhs_file(self):
        current = self.edit_lhs_path.text().strip()
        start = os.path.dirname(current) if current else os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self, "Файл настроек LHS-анализа",
            os.path.join(start, "lhs_settings.json"),
            "JSON настройки (*.json);;All Files (*)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.edit_lhs_path.setText(path)

    def _save_lhs_params(self):
        path = self.edit_lhs_path.text().strip()
        if not path:
            QMessageBox.warning(
                self, "Нет пути",
                "Укажите путь к файлу перед сохранением.")
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        def _layer_name(combo):
            layer = combo.currentLayer()
            return layer.name() if layer else ""

        payload = {
            "version":         "lhs_1.0",
            "obs_poly_name":   _layer_name(self.combo_obs_poly),
            "obs_pts_name":    _layer_name(self.combo_obs_pts),
            "depth_field":     self.combo_depth_field.currentText(),
            "depth_threshold": self.spin_depth_thr.value(),
            # Параметры варьирования
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
            # Настройки анализа
            "n_runs":    self.spin_n_runs.value(),
            "seed":      self.spin_seed.value(),
            "out_dir":   self.edit_sweep_outdir.text(),
            "csv_name":  self.edit_sweep_csv.text(),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            QMessageBox.information(
                self, "Настройки LHS сохранены",
                f"Настройки LHS-анализа сохранены в:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка сохранения", f"Не удалось записать файл:\n{e}")

    def _load_lhs_params(self):
        path = self.edit_lhs_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить настройки LHS",
                os.path.expanduser("~"),
                "JSON настройки (*.json);;All Files (*)")
            if not path:
                return
            self.edit_lhs_path.setText(path)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Файл не найден",
                                f"Файл не существует:\n{path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения",
                                 f"Не удалось прочитать JSON:\n{e}")
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
                warnings.append(f"Слой «{name}» не найден в проекте.")

        _set_layer(self.combo_obs_poly, d.get("obs_poly_name", ""))
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
        if "seed"    in d: self.spin_seed.setValue(int(d["seed"]))
        if "out_dir" in d: self.edit_sweep_outdir.setText(d["out_dir"])
        if "csv_name" in d: self.edit_sweep_csv.setText(d["csv_name"])

        msg = "Настройки LHS загружены."
        if warnings:
            msg += "\n\nПредупреждения:\n" + "\n".join(f"• {w}" for w in warnings)
            QMessageBox.warning(self, "LHS загружен (с предупреждениями)", msg)
        else:
            QMessageBox.information(self, "Настройки LHS загружены", msg)

    def _browse_sweep_outdir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка для результатов LHS-анализа",
            self.edit_sweep_outdir.text())
        if path:
            self.edit_sweep_outdir.setText(path)

    def _on_sweep_cancel(self):
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.cancel()
            self.sweep_log_box.append(
                "⚠ Отмена — ожидание завершения текущего прогона...")
            self.btn_sweep_cancel.setEnabled(False)

    def _on_sweep_run(self):
        """Валидация, сборка параметров и запуск SweepWorker."""
        # ── Проверка Numba JIT ────────────────────────────────────────────────
        _nb_ok, _nb_reason = self._check_numba()
        if not _nb_ok:
            QMessageBox.critical(
                self, "Numba JIT не найдена",
                "Для запуска LHS-анализа требуется Numba JIT.\n\n"
                + _nb_reason)
            return

        dem_layer = self.dem_combo.currentLayer()
        src_layer = self.src_line_combo.currentLayer()
        obs_poly  = self.combo_obs_poly.currentLayer()
        obs_pts   = self.combo_obs_pts.currentLayer()
        depth_fld = self.combo_depth_field.currentText()

        if dem_layer is None:
            QMessageBox.warning(
                self, "Ошибка", "Выберите DEM (Вкладка 1)."); return
        if src_layer is None:
            QMessageBox.warning(
                self, "Ошибка",
                "Выберите линейный слой источников (Вкладка 1)."); return
        if self.tbl_sources.rowCount() == 0:
            QMessageBox.warning(
                self, "Ошибка",
                "Загрузите источники — нажмите «Загрузить объекты» (Вкладка 1)."); return
        if obs_poly is None:
            QMessageBox.warning(
                self, "Ошибка",
                "Выберите полигональный слой наблюдаемой инундации."); return
        if obs_pts is None:
            QMessageBox.warning(
                self, "Ошибка",
                "Выберите точечный слой полевых замеров глубин."); return
        if not depth_fld:
            QMessageBox.warning(
                self, "Ошибка",
                "Выберите поле глубины [м] в точечном слое."); return

        # Валидация диапазонов (только для варьируемых параметров)
        err = []
        if self.chk_vol_vary.isChecked() and \
                self.spin_vol_min.value() >= self.spin_vol_max.value():
            err.append("V_min должен быть меньше V_max")
        if self.chk_tau_vary.isChecked() and \
                self.spin_tau_min.value() >= self.spin_tau_max.value():
            err.append("τy_min должен быть меньше τy_max")
        if self.chk_eta_vary.isChecked() and \
                self.spin_eta_min.value() >= self.spin_eta_max.value():
            err.append("η_min должен быть меньше η_max")
        if self.chk_n_vary.isChecked() and \
                self.spin_n_min.value() >= self.spin_n_max.value():
            err.append("n_min должен быть меньше n_max")
        if self.chk_K_vary.isChecked() and \
                self.spin_K_min.value() >= self.spin_K_max.value():
            err.append("K_min должен быть меньше K_max")
        if err:
            QMessageBox.warning(self, "Ошибка диапазонов",
                                 "\n".join(err)); return

        out_dir = self.edit_sweep_outdir.text().strip()
        if not out_dir:
            QMessageBox.warning(
                self, "Ошибка", "Укажите папку для результатов."); return

        # Сборка источников из таблицы вкладки 1
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
                "times":  data["times"],
                "q_vals": data["q_vals"],
            })

        if not sources:
            QMessageBox.warning(
                self, "Ошибка",
                "Нет источников с гидрографами.\n"
                "Задайте гидрографы на Вкладке 1."); return

        # LHS диапазоны — lo==hi для фиксированного параметра
        import math

        def _rng(chk, s_min, s_max, s_fix):
            if chk.isChecked():
                return (s_min.value(), s_max.value())
            v = s_fix.value()
            return (v, v)

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
            "eta":       _rng(self.chk_eta_vary, self.spin_eta_min,
                               self.spin_eta_max, self.spin_eta_fixed),
            "manning_n": _rng(self.chk_n_vary,  self.spin_n_min,
                               self.spin_n_max,  self.spin_n_fixed),
            "K_visc":    _rng(self.chk_K_vary,  self.spin_K_min,
                               self.spin_K_max,  self.spin_K_fixed),
        }

        params = dict(
            dem_layer       = dem_layer,
            source_layer    = src_layer,
            sources         = sources,
            obs_poly_layer  = obs_poly,
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
            csv_name        = (self.edit_sweep_csv.text().strip()
                               or "sweep_results.csv"),
            cfl_number      = self.spin_cfl.value(),
            n_workers       = self.spin_n_workers.value(),
            rho_s           = self.spin_rho_s.value(),
            rho_w           = self.spin_rho_w.value(),
        )

        self._sweep_results  = []
        self._sweep_best_pkg = None
        self.sweep_log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_sweep_run.setEnabled(False)
        self.btn_sweep_cancel.setEnabled(True)
        self.sweep_log_box.append(
            f"Запуск LHS-анализа: {params['n_runs']} прогонов…\n"
            f"Объём: {self.spin_vol_min.value():.0f} – "
            f"{self.spin_vol_max.value():.0f} м³\n"
            f"τy: {self.spin_tau_min.value():.0f} – "
            f"{self.spin_tau_max.value():.0f} Па\n"
            f"η: {self.spin_eta_min.value():.1f} – "
            f"{self.spin_eta_max.value():.1f} Па·с\n"
            f"n: {self.spin_n_min.value():.3f} – "
            f"{self.spin_n_max.value():.3f}")

        self._t_sweep_start = time.time()   # таймер LHS

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
        """Сохраняем пакет лучшего прогона для создания векторного слоя."""
        self._sweep_best_pkg = pkg

    def _on_sweep_error(self, msg: str):
        self.btn_sweep_run.setEnabled(True)
        self.btn_sweep_cancel.setEnabled(False)
        self.sweep_log_box.append(f"\n⛔ ОШИБКА:\n{msg}")
        QMessageBox.critical(self, "Ошибка LHS-анализа", msg[:800])

    def _on_sweep_finished(self):
        self.btn_sweep_run.setEnabled(True)
        self.btn_sweep_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        elapsed_s = time.time() - getattr(self, "_t_sweep_start", time.time())
        n_ok = sum(1 for r in self._sweep_results if r.get("status") == "ok")
        avg_s = elapsed_s / max(n_ok, 1)
        self.sweep_log_box.append(
            f"\n✓ LHS завершён: {n_ok} прогонов за "
            f"{elapsed_s:.0f} с ({elapsed_s/60:.1f} мин), "
            f"ср. {avg_s:.1f} с/прогон")

        if not self._sweep_results:
            QMessageBox.warning(
                self, "LHS-анализ", "Нет успешных прогонов."); return

        out_dir  = self.edit_sweep_outdir.text().strip()
        csv_name = self.edit_sweep_csv.text().strip() or "sweep_results.csv"
        csv_path = os.path.join(out_dir, csv_name)
        os.makedirs(out_dir, exist_ok=True)

        # ── Запись CSV ────────────────────────────────────────────────────────
        import csv as _csv
        ok_rows = [r for r in self._sweep_results if r.get("status") == "ok"]
        if ok_rows:
            fieldnames = list(ok_rows[0].keys())
            # Гарантируем, что все строки имеют те же ключи
            all_rows = []
            for r in self._sweep_results:
                row = {k: r.get(k, None) for k in fieldnames}
                all_rows.append(row)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            self.sweep_log_box.append(f"\n📋 CSV сохранён: {csv_path}")
        else:
            csv_path = None
            self.sweep_log_box.append("\n⚠ Нет успешных прогонов для CSV.")

        # ── Растровый слой h_max лучшего прогона ─────────────────────────────
        raster_path = None
        if self._sweep_best_pkg is not None:
            base = csv_name.replace(".csv", "")
            raster_path = os.path.join(out_dir, f"best_h_max_{base}.tif")
            try:
                from qgis_runner import write_geotiff
                # Применяем порог: пиксели < threshold → 0 → nodata (zero_as_nodata=True)
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
                    f"🗺 Растр h_max (порог ≥ {thr:.2f} м): {raster_path}")
                from qgis.core import QgsRasterLayer, QgsProject
                rlayer = QgsRasterLayer(raster_path, "LHS_best_h_max")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    # Применяем псевдоцвет (depth-стиль)
                    try:
                        from .plugin import DebrisFlowPlugin
                        DebrisFlowPlugin._apply_pseudocolour(rlayer, "depth")
                    except Exception:
                        pass
                else:
                    self.sweep_log_box.append(
                        "⚠ Растр создан, но не может быть загружен в QGIS.")
            except Exception as e:
                self.sweep_log_box.append(
                    f"⚠ Ошибка создания растра: {e}")
                raster_path = None

        # ── Диалог результатов ────────────────────────────────────────────────
        dlg = SweepResultsDialog(
            self, self._sweep_results, raster_path, csv_path)
        dlg.exec_()

    def _browse_save_file(self):
        """Диалог выбора файла для сохранения параметров."""
        current = self.edit_save_path.text().strip()
        start   = os.path.dirname(current) if current else os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить параметры симуляции",
            os.path.join(start, "debris_params.json"),
            "JSON параметры (*.json);;All Files (*)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.edit_save_path.setText(path)

    def _browse_load_file(self):
        """Диалог выбора файла для загрузки параметров."""
        current = self.edit_load_path.text().strip()
        start   = os.path.dirname(current) if current else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить параметры симуляции",
            start, "JSON параметры (*.json);;All Files (*)")
        if path:
            self.edit_load_path.setText(path)

    def _save_params(self):
        """
        Собрать все параметры из UI и записать в JSON-файл.
        Структура файла:
          {
            "version": "1.0",
            "layers": { "dem_name": ..., "source_name": ... },
            "hydrographs": { "<fid>": {type, q_const/times/q_vals}, ... },
            "source_table": [ {fid, name, q_max}, ... ],
            "obrien": { Cv, Cv_max, direct_mode, tau_yield, eta,
                        alpha1, beta1, alpha2, beta2, manning_n },
            "solver": { t_end, dt_max },
            "output": { prefix, outdir,
                        h_max, V_max, h_final, t_arrival,
                        threshold_enabled, threshold_value }
          }
        """
        path = self.edit_save_path.text().strip()
        if not path:
            QMessageBox.warning(
                self, "Нет пути",
                "Укажите путь к файлу на вкладке «Выходные данные»\n"
                "перед сохранением.")
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

        # --- hydrographs: ключи int → str для JSON ---
        hydro_sec = {
            str(fid): data
            for fid, data in self._hydro_data.items()
        }

        # --- source table rows (для восстановления описаний) ---
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
        }

        # --- Output ---
        output_sec = {
            "prefix":            self.edit_prefix.text(),
            "outdir":            self.edit_outdir.text(),
            "h_max":             self.chk_h_max.isChecked(),
            "V_max":             self.chk_V_max.isChecked(),
            "h_final":           self.chk_h_final.isChecked(),
            "t_arrival":         self.chk_t_arrival.isChecked(),
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
                self, "Параметры сохранены",
                f"Параметры симуляции сохранены в:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка сохранения",
                f"Не удалось записать файл:\n{e}")

    def _load_params(self):
        """
        Загрузить параметры из JSON-файла и восстановить UI.
        Слои ищутся по имени в текущем проекте QGIS.
        """
        path = self.edit_load_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить параметры симуляции",
                os.path.expanduser("~"),
                "JSON параметры (*.json);;All Files (*)")
            if not path:
                return
            self.edit_load_path.setText(path)

        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Файл не найден",
                f"Файл не существует:\n{path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка чтения",
                f"Не удалось прочитать JSON:\n{e}")
            return

        warnings = []

        # --- Слои: ищем по имени в проекте QGIS ---
        from qgis.core import QgsProject
        layers_sec = data.get("layers", {})

        dem_name = layers_sec.get("dem_name", "")
        if dem_name:
            matches = QgsProject.instance().mapLayersByName(dem_name)
            if matches:
                self.dem_combo.setLayer(matches[0])
            else:
                warnings.append(
                    f"DEM слой «{dem_name}» не найден в проекте.")

        src_name = layers_sec.get("source_name", "")
        if src_name:
            matches = QgsProject.instance().mapLayersByName(src_name)
            if matches:
                self.src_line_combo.setLayer(matches[0])
            else:
                warnings.append(
                    f"Слой источников «{src_name}» не найден в проекте.")

        # --- Гидрографы ---
        hydro_sec = data.get("hydrographs", {})
        self._hydro_data = {}
        for fid_str, hdata in hydro_sec.items():
            try:
                self._hydro_data[int(fid_str)] = hdata
            except ValueError:
                pass  # некорректный ключ — пропускаем

        # --- Таблица источников: восстановить строки из source_table ---
        table_rows = data.get("source_table", [])
        if table_rows and self._hydro_data:
            self.tbl_sources.setRowCount(0)
            for row_data in table_rows:
                try:
                    fid  = int(row_data.get("fid", -1))
                except (ValueError, TypeError):
                    continue
                fname  = row_data.get("name", f"Линия {fid}")
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

                btn = QPushButton("✏ Редактировать...")
                btn.setProperty("fid",   fid)
                btn.setProperty("fname", fname)
                btn.setProperty("row",   row)
                btn.clicked.connect(self._edit_hydro)
                self.tbl_sources.setCellWidget(row, 3, btn)

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
            direct = ob.get("direct_mode", True)
            self._radio_direct.setChecked(direct)
            self._radio_exp.setChecked(not direct)
            self._on_rheo_mode()
            self._update_hstop()

        # --- Solver ---
        sv = data.get("solver", {})
        if sv:
            if "t_end"  in sv: self.spin_tend.setValue(int(sv["t_end"]))
            if "dt_max" in sv: self.spin_dtmax.setValue(sv["dt_max"])

        # --- Output ---
        out = data.get("output", {})
        if out:
            if "prefix"  in out: self.edit_prefix.setText(out["prefix"])
            if "outdir"  in out: self.edit_outdir.setText(out["outdir"])
            if "h_max"   in out: self.chk_h_max.setChecked(out["h_max"])
            if "V_max"   in out: self.chk_V_max.setChecked(out["V_max"])
            if "h_final" in out: self.chk_h_final.setChecked(out["h_final"])
            if "t_arrival" in out: self.chk_t_arrival.setChecked(out["t_arrival"])
            thr_on  = out.get("threshold_enabled", False)
            thr_val = out.get("threshold_value", 0.10)
            self.chk_threshold.setChecked(thr_on)
            self.spin_threshold.setValue(thr_val)
            self.spin_threshold.setEnabled(thr_on)

        # --- Итог ---
        if warnings:
            QMessageBox.warning(
                self,
                "Параметры загружены (с предупреждениями)",
                "Параметры восстановлены.\n\n"
                "Предупреждения:\n" + "\n".join(f"• {w}" for w in warnings))
        else:
            QMessageBox.information(
                self, "Параметры загружены",
                f"Все параметры восстановлены из:\n{path}")

    # ── Слоты ─────────────────────────────────────────────────────────────────

    def _load_features(self):
        """Загрузить объекты линейного слоя в таблицу источников."""
        layer = self.src_line_combo.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "Нет слоя",
                                "Выберите линейный слой источников.")
            return
        if layer.geometryType() != QgsWkbTypes.LineGeometry:
            QMessageBox.warning(self, "Неверный тип",
                                "Слой должен быть линейным (LineString).")
            return

        feats = list(layer.getFeatures())
        if not feats:
            QMessageBox.warning(self, "Пусто",
                                "Слой не содержит объектов.")
            return

        self.tbl_sources.setRowCount(0)

        for feat in feats:
            fid   = feat.id()
            # Пробуем получить читаемое имя из поля 'name' или 'id'
            fname = None
            fields = [f.name().lower() for f in feat.fields()]
            for candidate in ("name", "label", "id", "fid"):
                if candidate in fields:
                    val = feat[candidate]
                    if val is not None:
                        fname = str(val)
                        break
            if fname is None:
                fname = f"Линия {fid}"

            row = self.tbl_sources.rowCount()
            self.tbl_sources.insertRow(row)

            # ID
            id_item = QTableWidgetItem(str(fid))
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_sources.setItem(row, 0, id_item)

            # Описание
            name_item = QTableWidgetItem(fname)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_sources.setItem(row, 1, name_item)

            # Q_max (обновляется при редактировании гидрографа)
            data = self._hydro_data.get(fid)
            q_max = max(data["q_vals"]) if data else 0.0
            q_item = QTableWidgetItem(f"{q_max:.1f}" if q_max > 0 else "—")
            q_item.setTextAlignment(Qt.AlignCenter)
            q_item.setFlags(q_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_sources.setItem(row, 2, q_item)

            # Кнопка редактирования
            btn = QPushButton("✏ Редактировать...")
            btn.setProperty("fid", fid)
            btn.setProperty("fname", fname)
            btn.setProperty("row", row)
            btn.clicked.connect(self._edit_hydro)
            self.tbl_sources.setCellWidget(row, 3, btn)

            # Если для этого fid нет данных — устанавливаем дефолт
            if fid not in self._hydro_data:
                self._hydro_data[fid] = {
                    "type": "const", "q_const": 100.0,
                    "times": [0.0, 1e9], "q_vals": [100.0, 100.0],
                }

        self.tbl_sources.resizeRowsToContents()

    def _edit_hydro(self):
        """Открыть редактор гидрографа для выбранного объекта."""
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

    def _on_rheo_mode(self):
        direct = self._radio_direct.isChecked()
        self._sec_direct.setVisible(direct)
        self._sec_exp.setVisible(not direct)

    def _update_rho_m(self):
        """Обновляет индикатор текущей плотности смеси ρm."""
        try:
            Cv  = self.spin_Cv.value()
            rho_s = self.spin_rho_s.value()
            rho_w = self.spin_rho_w.value()
            rho_m = rho_s * Cv + rho_w * (1.0 - Cv)
            self._lbl_rho_m.setText(f"{rho_m:.1f} кг/м³")
        except Exception:
            pass

    def _update_hstop(self):
        try:
            tau = self.spin_tau.value()
            Cv  = self.spin_Cv.value()
            rho_m   = 2650. * Cv + 1000. * (1. - Cv)
            gamma_m = rho_m * 9.81
            self._lbl_hstop.setText(
                f"h_stop: {tau/(gamma_m*0.10):.3f} м (10%),  "
                f"{tau/(gamma_m*0.05):.3f} м (5%)")
        except Exception:
            pass

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка для результатов", self.edit_outdir.text())
        if path:
            self.edit_outdir.setText(path)

    # ── Запуск ────────────────────────────────────────────────────────────────

    # ── Проверка Numba JIT ────────────────────────────────────────────────────

    @staticmethod
    def _check_numba() -> tuple:
        """
        Проверяет доступность Numba JIT независимо от путей импорта плагина.
        Возвращает (ok: bool, reason: str).
        """
        import sys

        # Шаг 1: Numba напрямую (не зависит от sys.path плагина)
        try:
            import numba
            from numba import njit as _njit_test  # noqa: F401
            numba_version = numba.__version__
        except ImportError as e:
            return False, (
                "Numba не найдена в среде Python QGIS.\n"
                f"Детали: {e}\n\n"
                "Установка в консоли OSGeo4W Shell:\n"
                "  pip install numba\n"
                "После установки перезапустите QGIS.")
        except Exception as e:
            return False, (
                f"Numba установлена, но не инициализируется.\n"
                f"Ошибка ({type(e).__name__}): {e}\n\n"
                "Возможные причины:\n"
                "  • Несовместимость Numba и NumPy\n"
                "  • Отсутствует llvmlite или LLVM DLL (Windows)\n"
                "Попробуйте: pip install --upgrade numba")

        # Шаг 2: проверяем флаг в solver (только если модуль уже импортирован)
        for key in ('core.solver', 'debris_flow_obrien.core.solver'):
            solver_mod = sys.modules.get(key)
            if solver_mod is not None:
                if not getattr(solver_mod, '_NUMBA_AVAILABLE', True):
                    return False, (
                        f"Numba {numba_version} найдена, но отключена в solver.\n"
                        "Возможно, плагин был загружен до установки Numba.\n"
                        "Перезапустите QGIS и загрузите плагин заново.")
                break

        return True, ""

    def _on_run(self):
        # ── Проверка Numba JIT ────────────────────────────────────────────────
        self._nb_ok, _nb_reason = self._check_numba()
        if not self._nb_ok:
            QMessageBox.critical(
                self, "Numba JIT не найдена",
                "Для запуска симуляции требуется Numba JIT.\n\n"
                + _nb_reason)
            return

        self._t_run_start = time.time()   # таймер запуска

        # ── Валидация ─────────────────────────────────────────────────────────
        dem_layer = self.dem_combo.currentLayer()
        src_layer = self.src_line_combo.currentLayer()

        if dem_layer is None:
            QMessageBox.warning(self, "Нет данных", "Выберите слой DEM."); return
        if src_layer is None:
            QMessageBox.warning(self, "Нет данных",
                                "Выберите линейный слой источников."); return

        if self.tbl_sources.rowCount() == 0:
            QMessageBox.warning(
                self, "Источники не загружены",
                "Нажмите «Загрузить объекты» чтобы загрузить линии источника."); return

        out_dir = self.edit_outdir.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Нет данных",
                                "Укажите папку для результатов.\n"
                                "(Вкладка «Выходные данные»)"); return

        prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_",
                        self.edit_prefix.text().strip())
        if not prefix:
            QMessageBox.warning(self, "Нет данных",
                                "Укажите префикс файлов.\n"
                                "(Вкладка «Выходные данные»)"); return

        wanted = set()
        if self.chk_h_max.isChecked():    wanted.add("h_max")
        if self.chk_V_max.isChecked():    wanted.add("V_max")
        if self.chk_h_final.isChecked():  wanted.add("h_final")
        if self.chk_t_arrival.isChecked(): wanted.add("t_arrival")

        if not wanted:
            QMessageBox.warning(
                self, "Нет выходных слоёв",
                "Отметьте хотя бы один выходной слой\n"
                "на вкладке «Выходные данные»."); return

        Cv = self.spin_Cv.value(); Cv_max = self.spin_Cv_max.value()
        if Cv >= Cv_max:
            QMessageBox.warning(self, "Параметр",
                                f"Cv ({Cv}) должен быть < Cv_max ({Cv_max})."); return

        # Собираем список источников из таблицы
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
                "times":  data["times"],
                "q_vals": data["q_vals"],
            })

        if not sources:
            QMessageBox.warning(self, "Нет источников",
                                "Добавьте хотя бы один источник."); return

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
            "t_end":            float(self.spin_tend.value()),
            "dt_max":           self.spin_dtmax.value(),
            "h_threshold":      (self.spin_threshold.value()
                                 if self.chk_threshold.isChecked() else 0.0),
        }

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳  Выполняется...")
        self.log_box.clear()
        self.progress_bar.setValue(0)

        self.worker = SimulationWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, t_cur: float, t_end: float):
        self.progress_bar.setValue(
            int(100 * t_cur / t_end) if t_end > 0 else 0)

    def _on_log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def _on_finished(self, payload: dict):
        elapsed_s = time.time() - getattr(self, "_t_run_start", time.time())
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Запуск симуляции")
        self.progress_bar.setValue(100)
        self.log_box.append(
            f"\n✓ Симуляция завершена за {elapsed_s:.1f} с "
            f"({elapsed_s/60:.1f} мин)")

        paths   = payload["paths"]
        prefix  = payload["prefix"]
        wanted  = payload["wanted"]

        # Красивые имена для сообщения
        layer_names = {
            "h_max":     "Пиковая толщина",
            "V_max":     "Пиковая скорость",
            "h_final":   "Финальные отложения",
            "t_arrival": "Время прихода",
        }

        from .plugin import DebrisFlowPlugin
        lines = []
        for key in ("h_max", "V_max", "h_final", "t_arrival"):
            if key not in wanted:
                continue
            name = f"{prefix}_{key}"
            path = paths.get(name)
            if not path:
                continue
            style = "velocity" if key == "V_max" else (
                    "time"     if key == "t_arrival" else "depth")
            DebrisFlowPlugin.load_result_layer(path, name, style)
            lines.append(f"  • {name}  — {layer_names.get(key,'')}")

        folder = list(paths.values())[0] if paths else out_dir
        QMessageBox.information(
            self, "Симуляция завершена",
            "Добавлены слои в QGIS:\n" + "\n".join(lines) +
            f"\n\nПапка:\n{os.path.dirname(folder)}")

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Запуск симуляции")
        self.log_box.append(f"\n⛔ ОШИБКА:\n{msg}")
        QMessageBox.critical(self, "Ошибка симуляции", msg[:700])


# ─────────────────────────────────────────────────────────────────────────────
# Диалог результатов LHS-анализа
# ─────────────────────────────────────────────────────────────────────────────

class SweepResultsDialog(QDialog):
    """
    Отображает итоги LHS-sweep:
      - лучшие параметры (минимальный Cm)
      - таблица топ-10 прогонов
      - квинтильный анализ чувствительности (Section 7.5)
      - пути к растру и CSV
    """

    def __init__(self, parent, results: list, raster_path, csv_path):
        super().__init__(parent)
        self.setWindowTitle("LHS-анализ — Результаты")
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
            lo.addWidget(QLabel("Нет успешных прогонов."))
            bb = QDialogButtonBox(QDialogButtonBox.Close)
            bb.rejected.connect(self.reject)
            lo.addWidget(bb)
            return

        best = ok_rows[0]

        # Скроллируемое содержимое
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_lo = QVBoxLayout(inner)

        # ── Лучшие параметры ─────────────────────────────────────────────────
        grp_best = QGroupBox("🏆  Лучший прогон (минимальный Cm)")
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

        fl_b.addRow("Общий объём V:",          _lbl("vol_m3",       ".0f", "м³"))
        fl_b.addRow("τy — Предел текучести:",  _lbl("tau_yield_Pa", ".1f", "Па"))
        fl_b.addRow("η — Вязкость смеси:",     _lbl("eta_Pas",      ".2f", "Па·с"))
        fl_b.addRow("Manning n:",              _lbl("manning_n",    ".4f"))
        fl_b.addRow("K — коэф. формы:",        _lbl("K_visc",       ".1f"))

        sep = QLabel("─" * 50)
        sep.setStyleSheet("color:#ccc; font-size:8px")
        fl_b.addRow(sep)
        tp = best.get("TP", "—"); fp = best.get("FP", "—")
        fn = best.get("FN", "—")
        fl_b.addRow("TP / FP / FN:",
                    QLabel(f"{tp} / {fp} / {fn}"))
        fl_b.addRow("ΩTm (0=лучший):",         _lbl("omega_Tm"))
        fl_b.addRow("Δo,c (переоценка глуб.):", _lbl("delta_o_c"))
        fl_b.addRow("Δu  (недооценка глуб.):",  _lbl("delta_u"))
        fl_b.addRow("Cm (итог, 0=лучший):",    _lbl("Cm"))

        note_dep = QLabel(
            "<i>Δo и Δu нормированы на Σ|d_наблюд| → независимые метрики "
            "(Δo + Δu ≠ 1).<br>"
            "Значение > 0.5: средняя ошибка превышает половину наблюд. глубины.</i>")
        note_dep.setWordWrap(True)
        note_dep.setStyleSheet("color:#666; font-size:10px")
        fl_b.addRow(note_dep)
        inner_lo.addWidget(grp_best)

        # ── Таблица топ-10 ────────────────────────────────────────────────────
        header_lbl = QLabel(
            f"<b>Топ-{min(10, n_ok)} прогонов по Cm "
            f"({n_ok} успешных из {n_total}):</b>")
        inner_lo.addWidget(header_lbl)

        cols = ["run_id", "vol_m3", "tau_yield_Pa", "eta_Pas",
                "manning_n", "K_visc", "TP", "FP", "FN",
                "omega_Tm", "delta_o_c", "delta_u", "Cm"]
        col_lbl = ["ID", "V [м³]", "τy [Па]", "η [Па·с]",
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

        # ── Квинтильный анализ чувствительности (Section 7.5) ────────────────
        grp_q = QGroupBox(
            "Анализ чувствительности к параметрам (Section 7.5 — контроль объёма)")
        grp_q.setStyleSheet(
            "QGroupBox{border:1px solid #aaa;border-radius:4px;margin-top:6px}"
            "QGroupBox::title{padding:0 4px}")
        grp_q_lo = QVBoxLayout(grp_q)

        note_q = QLabel(
            "<b>Что это значит (Barnhart et al. 2021, Section 7.5):</b><br>"
            "Для каждого параметра (τy, η, n) симуляции делятся на 5 квинтилей "
            "(Q1=малые значения…Q5=большие). Внутри каждого квинтиля вычисляется "
            "средний Cm. Если квинтили дают разные Cm → параметр важен "
            "(даже после контроля объёма). Если все квинтили близки → параметр неважен."
        )
        note_q.setWordWrap(True)
        note_q.setStyleSheet("font-size:11px; color:#333; margin-bottom:4px")
        grp_q_lo.addWidget(note_q)

        params_sens = [
            ("tau_yield_Pa", "τy [Па]"),
            ("eta_Pas",      "η [Па·с]"),
            ("manning_n",    "n"),
        ]

        sens_tbl = QTableWidget()
        sens_tbl.setColumnCount(7)  # параметр + Q1..Q5 + разброс
        sens_tbl.setHorizontalHeaderLabels(
            ["Параметр", "Q1 (мин.)", "Q2", "Q3", "Q4", "Q5 (макс.)",
             "Разброс Cm"])
        sens_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        sens_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        sens_tbl.verticalHeader().setVisible(False)

        for param_key, param_label in params_sens:
            vals = [(r.get(param_key), r.get("Cm"))
                    for r in ok_rows
                    if r.get(param_key) is not None and r.get("Cm") is not None]
            if len(vals) < 10:
                continue
            vals.sort(key=lambda x: x[0])
            n_q = len(vals)
            q_size = n_q // 5
            q_means = []
            for qi in range(5):
                start = qi * q_size
                end   = (qi + 1) * q_size if qi < 4 else n_q
                q_cm  = [v[1] for v in vals[start:end]]
                q_means.append(float(np.mean(q_cm)) if q_cm else float("nan"))

            spread = max(q_means) - min(q_means)
            is_important = spread > 0.05

            row_i = sens_tbl.rowCount()
            sens_tbl.insertRow(row_i)

            lbl_item = QTableWidgetItem(param_label)
            lbl_item.setTextAlignment(Qt.AlignCenter)
            if is_important:
                lbl_item.setForeground(QColor("#1565C0"))
                lbl_item.setFont(QFont("", -1, QFont.Bold))
            sens_tbl.setItem(row_i, 0, lbl_item)

            for qi, cm_mean in enumerate(q_means):
                cell = QTableWidgetItem(f"{cm_mean:.3f}" if cm_mean == cm_mean else "—")
                cell.setTextAlignment(Qt.AlignCenter)
                # Наглядная окраска: низкий Cm = зелёный (лучший)
                if cm_mean == cm_mean:
                    g = int(min(255, max(0, 255 * (1 - cm_mean))))
                    cell.setBackground(QColor(255 - g // 2, 255, 255 - g // 2))
                sens_tbl.setItem(row_i, qi + 1, cell)

            spread_item = QTableWidgetItem(f"{spread:.3f}")
            spread_item.setTextAlignment(Qt.AlignCenter)
            if is_important:
                spread_item.setForeground(QColor("#C00000"))
                spread_item.setFont(QFont("", -1, QFont.Bold))
            sens_tbl.setItem(row_i, 6, spread_item)

        if sens_tbl.rowCount() > 0:
            sens_tbl.setMaximumHeight(
                28 * (sens_tbl.rowCount() + 1) + 10)
            grp_q_lo.addWidget(sens_tbl)
            note_interp = QLabel(
                "Разброс Cm > 0.05 → параметр ВЛИЯЕТ на результат (выделен синим/красным).<br>"
                "Разброс Cm ≤ 0.05 → параметр малозначим при данном диапазоне."
            )
            note_interp.setWordWrap(True)
            note_interp.setStyleSheet("font-size:10px; color:#555; margin-top:4px")
            grp_q_lo.addWidget(note_interp)
        else:
            grp_q_lo.addWidget(QLabel("Недостаточно данных для анализа (< 10 прогонов)."))

        inner_lo.addWidget(grp_q)

        # ── Пути к файлам ────────────────────────────────────────────────────
        if csv_path and os.path.exists(csv_path):
            lbl_csv = QLabel(f"📋 CSV: <code>{csv_path}</code>")
            lbl_csv.setWordWrap(True)
            inner_lo.addWidget(lbl_csv)
        if raster_path and os.path.exists(raster_path):
            lbl_r = QLabel(
                f"🗺 Растр h_max лучшего прогона добавлен в QGIS: "
                f"<code>{os.path.basename(raster_path)}</code>")
            lbl_r.setWordWrap(True)
            inner_lo.addWidget(lbl_r)

        scroll.setWidget(inner)
        lo.addWidget(scroll)

        # ── Кнопка закрыть ────────────────────────────────────────────────────
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lo.addWidget(bb)
