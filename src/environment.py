"""Advanced module - environmental indicators: NDVI by class/year and land surface temperature (LST)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_feature_stack
from .utils import CLASS_NAMES, read_raster


def ndvi_by_class(paths, years) -> pd.DataFrame:
    rows = []
    for y in years:
        X, _, _ = build_feature_stack(paths.s2(y))
        ndvi = X[..., 6]
        lulc = read_raster(paths.lulc(y))[0][0]
        for c, name in CLASS_NAMES.items():
            m = (lulc == c) & np.isfinite(ndvi)
            rows.append({"year": y, "class": name, "mean_ndvi": float(ndvi[m].mean()) if m.any() else np.nan})
        m = (lulc > 0) & np.isfinite(ndvi)
        rows.append({"year": y, "class": "All classified", "mean_ndvi": float(ndvi[m].mean())})
    return pd.DataFrame(rows)


def lst_by_class(lst_path, lulc_path) -> tuple[pd.DataFrame, dict]:
    """Mean/median LST per class and a simple urban-heat contrast (built-up minus vegetation+cropland)."""
    lst = read_raster(lst_path)[0][0]
    lulc = read_raster(lulc_path)[0][0]
    if lst.shape != lulc.shape:
        raise ValueError("LST and LULC rasters are not on the same grid.")
    rows = []
    for c, name in CLASS_NAMES.items():
        m = (lulc == c) & np.isfinite(lst)
        rows.append({"class": name, "pixels": int(m.sum()),
                     "mean_lst_c": float(lst[m].mean()) if m.any() else np.nan,
                     "median_lst_c": float(np.median(lst[m])) if m.any() else np.nan})
    df = pd.DataFrame(rows)
    built = (lulc == 1) & np.isfinite(lst)
    green = np.isin(lulc, [2, 3]) & np.isfinite(lst)
    summary = {"uhi_contrast_c": float(lst[built].mean() - lst[green].mean()) if built.any() and green.any() else None,
               "valid_lst_pct": float(np.isfinite(lst).mean() * 100)}
    return df, summary
