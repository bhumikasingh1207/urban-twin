"""Map-display helpers: reproject rasters to WGS84 and colourise them for Leaflet overlays."""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


def to_wgs84(arr: np.ndarray, meta: dict, categorical: bool):
    """Reproject a 2-D array to EPSG:4326. Returns (array, [[south, west], [north, east]])."""
    h, w = arr.shape
    left, bottom, right, top = array_bounds(h, w, meta["transform"])
    dst_tf, dw, dh = calculate_default_transform(meta["crs"], "EPSG:4326", w, h, left, bottom, right, top)
    if categorical:
        dst = np.zeros((dh, dw), dtype=arr.dtype)
        kw = dict(resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    else:
        dst = np.full((dh, dw), np.nan, dtype=np.float32)
        arr = arr.astype(np.float32)
        kw = dict(resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    reproject(arr, dst, src_transform=meta["transform"], src_crs=meta["crs"], dst_transform=dst_tf,
              dst_crs="EPSG:4326", **kw)
    west, south, east, north = array_bounds(dh, dw, dst_tf)
    return dst, [[south, west], [north, east]]


def colorize_categorical(arr: np.ndarray, colors: dict[int, str]) -> np.ndarray:
    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
    for k, hexcol in colors.items():
        m = arr == k
        rgba[m, :3] = (np.array(to_rgb(hexcol)) * 255).astype(np.uint8)
        rgba[m, 3] = 255
    return rgba


def colorize_continuous(arr: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[~np.isfinite(arr), 3] = 0
    return rgba


def true_colour(s2_arr: np.ndarray, meta: dict):
    """B04/B03/B02 -> RGBA with a 2-98 percentile stretch, reprojected to WGS84."""
    chans = []
    for i in (2, 1, 0):  # red, green, blue = B04, B03, B02
        band, bounds = to_wgs84(s2_arr[i], meta, categorical=False)
        chans.append(band)
    rgb = np.stack(chans, axis=-1)
    ok = np.isfinite(rgb).all(-1)
    lo, hi = (np.percentile(rgb[ok], [2, 98]) if ok.any() else (0, 1))
    out = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = (np.clip((np.nan_to_num(rgb) - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    out[..., 3] = np.where(ok, 255, 0)
    return out, bounds


def legend_html(title: str, items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:14px;height:14px;'
        f'background:{c};margin-right:6px;border:1px solid #555;vertical-align:middle"></span>{n}</div>'
        for n, c in items)
    return (f'<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 12px;'
            f'border:1px solid #888;border-radius:4px;font:12px sans-serif;color:#222">'
            f'<b>{title}</b>{rows}</div>')
