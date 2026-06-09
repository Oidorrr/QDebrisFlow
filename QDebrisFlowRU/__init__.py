# -*- coding: utf-8 -*-
"""
QGIS Plugin entry point.
"""


def classFactory(iface):
    from .plugin import DebrisFlowPlugin
    return DebrisFlowPlugin(iface)