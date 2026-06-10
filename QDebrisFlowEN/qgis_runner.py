# -*- coding: utf-8 -*-
"""
QGIS-aware raster read/write helpers.
"""

from __future__ import annotations
import numpy as np


def read_dem_array(layer) -> tuple[np.ndarray, dict]:
    try:
        from osgeo import gdal
        source = layer.source()
        ds = gdal.Open(source)
        if ds is None:
            raise RuntimeError("GDAL cannot open source")
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        dem = band.ReadAsArray().astype(np.float64)
        gt = ds.GetGeoTransform()
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
            "crs": layer.crs(),
        }
        if nodata is not None:
            dem[dem == nodata] = nodata
        return dem, geo_info
    except Exception:
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
    src_path = source_layer.source().split("|")[0]
    vec_ds = ogr.Open(src_path)
    if vec_ds is None:
        raise RuntimeError(f"OGR cannot open {src_path}")
    vec_lyr = vec_ds.GetLayer()
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
    Writes the array to a GeoTIFF.

    Args:
        array:          2-D array of values.
        geo_info:       Geospatial metadata from read_dem_array().
        path:           Path to the output file.
        zero_as_nodata: If True — cells with a value <= 0 are written
                        as NoData (transparent in QGIS).
                        BUG 5 FIX: removes the white frame around the flow.
        nodata_value:   NoData value (default -9999).
    """
    from osgeo import gdal, osr

    ny, nx = array.shape
    gt  = geo_info["geotransform"]
    out = array.astype(np.float32).copy()

    # BUG 5: mask cells without flow → transparent in QGIS
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
    # WriteRaster accepts raw bytes and does not import gdal_array,
    # so it works with any NumPy version (1.20, 1.22, 1.26, etc.)
    band.WriteRaster(0, 0, nx, ny, out.tobytes(), nx, ny, gdal.GDT_Float32)
    band.FlushCache()
    ds = None