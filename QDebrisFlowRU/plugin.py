# -*- coding: utf-8 -*-
"""
Main plugin class for the Debris Flow O'Brien simulator.
"""

import os
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProject, QgsRasterLayer


class DebrisFlowPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(
            icon,
            "Селевые потоки методом O'Brien",
            self.iface.mainWindow(),
        )
        self.action.setToolTip("QDebrisFlow 1.1")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("Селевые потоки", self.action)

    def unload(self):
        # перезагрузка плагина не должна оставлять старый экземпляр диалога; идущий
        # расчёт сначала отменяется, иначе его QThread был бы уничтожен во время работы
        # (простое закрытие окна по-прежнему не прерывает расчёт — так задумано)
        if self.dialog is not None:
            for w in (getattr(self.dialog, "worker", None),
                      getattr(self.dialog, "sweep_worker", None),
                      getattr(self.dialog, "sweep_best_worker", None)):
                if w is not None and w.isRunning():
                    w.cancel()
                    w.wait(30000)
            self.dialog.close()
            self.dialog.deleteLater()
            self.dialog = None
        self.iface.removePluginMenu("Селевые потоки", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.action:
            self.action.deleteLater()

    def run(self):
        from .dialog import DebrisFlowDialog
        if self.dialog is None:
            self.dialog = DebrisFlowDialog(self.iface)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    @staticmethod
    def load_result_layer(path: str, name: str, style: str = "depth") -> None:
        layer = QgsRasterLayer(path, name)
        if not layer.isValid():
            QMessageBox.critical(None, "Ошибка", f"Не удалось загрузить слой:\n{path}")
            return
        DebrisFlowPlugin._apply_pseudocolour(layer, style)
        # повторный запуск с тем же префиксом: убрать прежний слой этого файла, а не
        # добавлять дубликат, который держит старый файл открытым
        prj = QgsProject.instance()
        norm = os.path.normcase(os.path.abspath(path))
        old = [lid for lid, lyr in prj.mapLayers().items()
               if os.path.normcase(os.path.abspath(lyr.source().split("|")[0])) == norm]
        if old:
            prj.removeMapLayers(old)
        prj.addMapLayer(layer)

    @staticmethod
    def _apply_pseudocolour(layer, style: str) -> None:
        try:
            from qgis.core import (
                QgsRasterShader, QgsColorRampShader,
                QgsSingleBandPseudoColorRenderer,
            )
            from qgis.PyQt.QtGui import QColor

            stats = layer.dataProvider().bandStatistics(1)
            vmin = stats.minimumValue
            vmax = stats.maximumValue
            if vmax <= vmin:
                return

            shader = QgsRasterShader()
            color_ramp = QgsColorRampShader()
            color_ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)

            if style == "depth":
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 255), f"{vmin:.2f}"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.25, QColor(173, 216, 230), ""),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.5,  QColor(70,  130, 180), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(0,   0,   139), f"{vmax:.2f}"),
                ]
            elif style == "velocity":
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 200), f"{vmin:.2f} м/с"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.33, QColor(255, 200, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.66, QColor(255, 80, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(180, 0, 0), f"{vmax:.2f} м/с"),
                ]
            elif style == "intensity":
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 200), f"{vmin:.2f} м³/с²"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.33, QColor(255, 200, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.66, QColor(255, 80, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(120, 0, 60), f"{vmax:.2f} м³/с²"),
                ]
            elif style == "froude":
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(230, 240, 255), f"Fr {vmin:.2f}"),
                    QgsColorRampShader.ColorRampItem(min(max(0.5, vmin), vmax), QColor(120, 170, 230), "Fr 0.5"),
                    QgsColorRampShader.ColorRampItem(min(max(1.0, vmin), vmax), QColor(255, 160, 0), "Fr 1"),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(170, 0, 0), f"Fr {vmax:.2f}"),
                ]
            else:  # arrival time
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 0), f"{vmin:.0f} с"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.5, QColor(255, 140, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(139, 0, 0), f"{vmax:.0f} с"),
                ]

            color_ramp.setColorRampItemList(items)
            shader.setRasterShaderFunction(color_ramp)
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception as e:
            from qgis.core import QgsMessageLog, Qgis
            QgsMessageLog.logMessage(f"стиль псевдоцвета не применён к {layer.name()}: "
                                     f"{type(e).__name__}: {e}", "QDebrisFlow", Qgis.MessageLevel.Warning)