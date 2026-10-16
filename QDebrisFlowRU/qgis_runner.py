# -*- coding: utf-8 -*-
"""
QGIS-aware raster read/write helpers.
"""

from __future__ import annotations
import numpy as np


class RotatedRasterError(ValueError):
    """The DEM geotransform has rotation/shear terms or a south-up row order."""


def read_dem_array(layer, source=None, crs=None) -> tuple[np.ndarray, dict]:
    """Массив ЦМР и geo_info. Рабочий поток передаёт layer=None с путём источника
    и CRS, взятыми в главном потоке, чтобы не вызывать методы слоя; запасной путь
    через провайдер QGIS требует слой и тогда недоступен."""
    try:
        from osgeo import gdal
        if source is None:
            source = layer.source()
        ds = gdal.Open(source)
        if ds is None:
            raise RuntimeError("GDAL cannot open source")
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        dem = band.ReadAsArray().astype(np.float64)
        # масштаб/смещение полосы (упакованные целочисленные ЦМР): значение = raw·scale + offset;
        # nodata остаётся в исходных единицах
        scale, offset = band.GetScale(), band.GetOffset()
        if (scale not in (None, 1.0)) or (offset not in (None, 0.0)):
            valid = (dem != nodata) if nodata is not None else np.ones(dem.shape, dtype=bool)
            dem[valid] = dem[valid] * (1.0 if scale is None else scale) + (offset or 0.0)
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0 or gt[5] > 0 or gt[1] < 0:
            raise RotatedRasterError(f"ЦМР не ориентирована на север (геопреобразование {tuple(gt)}); перепроецируйте её в обычный растр.")
        nx, ny = ds.RasterXSize, ds.RasterYSize
        dx, dy = abs(gt[1]), abs(gt[5])
        xmin, ymax = gt[0], gt[3]
        ds = None

        geo_info = {
            "nx": nx, "ny": ny,
            "dx": dx, "dy": dy,
            "xmin": xmin, "ymax": ymax,
            "xmax": xmin + nx * dx,
            "ymin": ymax - ny * dy,
            "nodata": nodata if nodata is not None else -9999.0,
            "geotransform": gt,
            "crs": crs if crs is not None else layer.crs(),
        }
        return dem, geo_info
    except (RuntimeError, OSError, ImportError):
        # GDAL не открывает/не читает источник → провайдер QGIS (RotatedRasterError не перехватывается)
        if layer is None:
            raise
        return _read_dem_block(layer)


def _read_dem_block(layer) -> tuple[np.ndarray, dict]:
    extent = layer.extent()
    nx, ny = layer.width(), layer.height()
    dx = extent.width() / nx
    dy = extent.height() / ny
    provider = layer.dataProvider()
    nodata = provider.sourceNoDataValue(1)

    block = provider.block(1, extent, nx, ny)
    dem = np.zeros((ny, nx), dtype=np.float64)
    for r in range(ny):
        for c in range(nx):
            dem[r, c] = block.value(r, c) if not block.isNoData(r, c) else nodata

    geo_info = {
        "nx": nx, "ny": ny,
        "dx": dx, "dy": dy,
        "xmin": extent.xMinimum(),
        "ymin": extent.yMinimum(),
        "xmax": extent.xMaximum(),
        "ymax": extent.yMaximum(),
        "nodata": nodata if nodata is not None else -9999.0,
        "geotransform": (extent.xMinimum(), dx, 0, extent.yMaximum(), 0, -dy),
        "crs": layer.crs(),
    }
    return dem, geo_info


def rasterise_source(source_layer, geo_info: dict, default_depth: float) -> np.ndarray:
    try:
        return _rasterise_gdal(source_layer, geo_info, default_depth)
    except Exception:
        return _rasterise_qgis(source_layer, geo_info, default_depth)


def _rasterise_gdal(source_layer, geo_info: dict, depth: float) -> np.ndarray:
    from osgeo import gdal, ogr
    nx, ny = geo_info["nx"], geo_info["ny"]
    gt = geo_info["geotransform"]
    mem_driver = gdal.GetDriverByName("MEM")
    out_ds = mem_driver.Create("", nx, ny, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(gt)
    band = out_ds.GetRasterBand(1)
    band.Fill(0.0)
    band.SetNoDataValue(-9999.0)
    parts = source_layer.source().split("|")
    src_path = parts[0]
    vec_ds = ogr.Open(src_path)
    if vec_ds is None:
        raise RuntimeError(f"OGR cannot open {src_path}")
    # multi-layer sources (GeoPackage "|layername=…" / "|layerid=…"): take THAT layer,
    # not the first one in the file
    opts = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    if "layername" in opts:
        vec_lyr = vec_ds.GetLayerByName(opts["layername"])
    elif "layerid" in opts:
        vec_lyr = vec_ds.GetLayer(int(opts["layerid"]))
    else:
        vec_lyr = vec_ds.GetLayer()
    if vec_lyr is None:
        raise RuntimeError(f"OGR cannot find the layer in {source_layer.source()}")
    gdal.RasterizeLayer(out_ds, [1], vec_lyr, burn_values=[depth])
    h0 = band.ReadAsArray().astype(np.float64)
    out_ds = None
    vec_ds = None
    return np.maximum(h0, 0.0)


def _rasterise_qgis(source_layer, geo_info: dict, depth: float) -> np.ndarray:
    from qgis.core import QgsGeometry, QgsPointXY
    nx, ny = geo_info["nx"], geo_info["ny"]
    dx, dy = geo_info["dx"], geo_info["dy"]
    xmin, ymax = geo_info["xmin"], geo_info["ymax"]
    h0 = np.zeros((ny, nx), dtype=np.float64)
    geoms = [f.geometry() for f in source_layer.getFeatures()]
    for row in range(ny):
        cy = ymax - (row + 0.5) * dy
        for col in range(nx):
            cx = xmin + (col + 0.5) * dx
            pt = QgsGeometry.fromPointXY(QgsPointXY(cx, cy))
            if any(g.contains(pt) for g in geoms):
                h0[row, col] = depth
    return h0


def write_geotiff(
    array: np.ndarray,
    geo_info: dict,
    path: str,
    zero_as_nodata: bool = False,
    nodata_value: float = -9999.0,
) -> None:
    """
    Записывает массив в GeoTIFF.

    Args:
        array:          2-D массив значений.
        geo_info:       Геопространственные метаданные из read_dem_array().
        path:           Путь к выходному файлу.
        zero_as_nodata: Если True — ячейки со значением <= 0 записываются
                        как NoData (прозрачные в QGIS).
                        ИСПРАВЛЕНИЕ БАГ 5: убирает белую рамку вокруг потока.
        nodata_value:   Значение NoData (по умолчанию -9999).
    """
    from osgeo import gdal, osr

    ny, nx = array.shape
    gt  = geo_info["geotransform"]
    out = array.astype(np.float32).copy()

    # БАГ 5: маскируем ячейки без потока → прозрачные в QGIS
    if zero_as_nodata:
        out[out <= 0.0] = nodata_value

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER", "OVERWRITE=YES"]
    )
    ds.SetGeoTransform(gt)

    crs = geo_info.get("crs")
    if crs is not None:
        try:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.toWkt())
            ds.SetProjection(srs.ExportToWkt())
        except Exception:
            pass

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    # WriteRaster принимает сырые байты и не импортирует gdal_array,
    # поэтому работает при любой версии NumPy (1.20, 1.22, 1.26 и т.д.)
    band.WriteRaster(0, 0, nx, ny, out.tobytes(), nx, ny, gdal.GDT_Float32)
    band.FlushCache()
    ds = None