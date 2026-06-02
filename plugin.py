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
        self.action.setToolTip("Селевые потоки методом O'Brien")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("Селевые потоки", self.action)

    def unload(self):
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
        QgsProject.instance().addMapLayer(layer)

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
            color_ramp.setColorRampType(QgsColorRampShader.Interpolated)

            if style == "depth":
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 255), f"{vmin:.2f}"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.25, QColor(173, 216, 230), ""),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.5,  QColor(70,  130, 180), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(0,   0,   139), f"{vmax:.2f}"),
                ]
            elif style == "velocity":
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 200), f"{vmin:.2f} m/s"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.33, QColor(255, 200, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.66, QColor(255, 80, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(180, 0, 0), f"{vmax:.2f} m/s"),
                ]
            else:  # arrival time
                items = [
                    QgsColorRampShader.ColorRampItem(vmin, QColor(255, 255, 0), f"{vmin:.0f}s"),
                    QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * 0.5, QColor(255, 140, 0), ""),
                    QgsColorRampShader.ColorRampItem(vmax, QColor(139, 0, 0), f"{vmax:.0f}s"),
                ]

            color_ramp.setColorRampItemList(items)
            shader.setRasterShaderFunction(color_ramp)
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception:
            pass