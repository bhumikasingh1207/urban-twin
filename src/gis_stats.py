"""Module 2 - GIS processing: area statistics, change detection, grid (zonal) statistics."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from .utils import CLASS_NAMES, pixel_area_km2

CHANGE_INFO = {
    1: ("No change (non built-up)", "#d9d9d9"),
    2: ("Built-up (stable)", "#7a7a7a"),
    3: ("New built-up", "#e31a1c"),
    4: ("Built-up loss", "#1f78b4"),
}


def class_area_table(lulc: np.ndarray, transform) -> pd.DataFrame:
    pa = pixel_area_km2(transform)
    valid_total = int((lulc > 0).sum())
    rows = []
    for c, name in CLASS_NAMES.items():
        n = int((lulc == c).sum())
        rows.append({"class_id": c, "class": name, "pixels": n, "area_km2": n * pa,
                     "percent": 100 * n / valid_total if valid_total else 0.0})
    return pd.DataFrame(rows)


def build_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0 = not comparable (no data in either year), 1..4 see CHANGE_INFO."""
    out = np.zeros(a.shape, dtype=np.uint8)
    ok = (a > 0) & (b > 0)
    ba, bb = a == 1, b == 1
    out[ok & ~ba & ~bb] = 1
    out[ok & ba & bb] = 2
    out[ok & ~ba & bb] = 3
    out[ok & ba & ~bb] = 4
    return out


def transition_matrix(a: np.ndarray, b: np.ndarray, transform) -> pd.DataFrame:
    """Area (km2) moving from class (rows, year A) to class (columns, year B)."""
    ok = (a > 0) & (b > 0)
    pa = pixel_area_km2(transform)
    names = list(CLASS_NAMES.values())
    m = np.zeros((5, 5))
    for i in range(1, 6):
        for j in range(1, 6):
            m[i - 1, j - 1] = ((a == i) & (b == j) & ok).sum() * pa
    return pd.DataFrame(m, index=names, columns=names)


def grid_stats(a: np.ndarray, b: np.ndarray, transform, crs, cell_m: float) -> gpd.GeoDataFrame:
    """Square grid cells with built-up share in year A/B and new built-up area. WGS84 output."""
    res = abs(transform.a)
    cell = max(int(round(cell_m / res)), 1)
    nh, nw = a.shape[0] // cell, a.shape[1] // cell
    if nh == 0 or nw == 0:
        raise ValueError("cell_m is larger than the study area.")

    def blocks(x):
        return x[: nh * cell, : nw * cell].reshape(nh, cell, nw, cell).swapaxes(1, 2).reshape(nh, nw, cell * cell)

    A, B = blocks(a), blocks(b)
    valid = (A > 0) & (B > 0)
    nvalid = valid.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa_pct = 100 * ((A == 1) & valid).sum(-1) / nvalid
        pb_pct = 100 * ((B == 1) & valid).sum(-1) / nvalid
    gain = ((A != 1) & (B == 1) & valid).sum(-1) * pixel_area_km2(transform)

    recs = []
    for i in range(nh):
        for j in range(nw):
            if nvalid[i, j] < 0.5 * cell * cell:
                continue
            x0, y1 = transform.c + j * cell * res, transform.f - i * cell * res
            recs.append({"row": i, "col": j, "built_pct_a": float(pa_pct[i, j]), "built_pct_b": float(pb_pct[i, j]),
                         "built_change_pp": float(pb_pct[i, j] - pa_pct[i, j]), "gain_km2": float(gain[i, j]),
                         "geometry": box(x0, y1 - cell * res, x0 + cell * res, y1)})
    return gpd.GeoDataFrame(recs, crs=crs).to_crs("EPSG:4326")
