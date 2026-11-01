"""Module 3b - training/validation labels.

Training labels are WEAK labels derived from ESA WorldCover (itself a classification product with
~75-80% published accuracy). We keep only pixels whose 3x3 neighbourhood has a single class
(homogeneous pixels) to reduce edge noise. These are NOT ground truth.

Independent test labels: a CSV of points (lon, lat, class_id) digitised by the team in QGIS from
high-resolution imagery. See README, section "Creating validation points".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.ndimage import maximum_filter, minimum_filter

# ESA WorldCover code -> project class. 70 (snow), 90 (wetland), 100 (moss) are ignored.
WC_TO_CLASS = {10: 2, 20: 2, 30: 2, 40: 3, 50: 1, 60: 5, 80: 4, 95: 2}


def map_worldcover(wc: np.ndarray) -> np.ndarray:
    out = np.zeros(wc.shape, dtype=np.uint8)
    for code, cls in WC_TO_CLASS.items():
        out[wc == code] = cls
    return out


def homogeneous_mask(cls: np.ndarray, size: int = 3) -> np.ndarray:
    return (maximum_filter(cls, size=size) == minimum_filter(cls, size=size)) & (cls > 0)


def sample_weak_labels(cls: np.ndarray, valid: np.ndarray, n_per_class: int, seed: int):
    """Stratified random sample of homogeneous, valid pixels. Returns rows, cols, labels."""
    rng = np.random.default_rng(seed)
    ok = homogeneous_mask(cls) & valid
    rows, cols, ys = [], [], []
    for c in np.unique(cls[ok]):
        idx = np.flatnonzero((ok & (cls == c)).ravel())
        take = rng.choice(idx, size=min(n_per_class, idx.size), replace=False)
        r, cc = np.unravel_index(take, cls.shape)
        rows.append(r), cols.append(cc), ys.append(np.full(take.size, c, dtype=np.uint8))
    if not rows:
        raise ValueError("No usable training pixels (check WorldCover/S2 alignment and cloud cover).")
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(ys)


def load_validation_points(csv_path, transform, crs, shape, valid):
    """Read independent validation points -> (rows, cols, labels) inside the grid and on valid pixels."""
    df = pd.read_csv(csv_path)
    missing = {"lon", "lat", "class_id"} - set(df.columns)
    if missing:
        raise ValueError(f"validation CSV is missing columns: {sorted(missing)}")
    df = df[df["class_id"].isin([1, 2, 3, 4, 5])]
    xs, ys = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(df["lon"].values, df["lat"].values)
    r, c = rowcol(transform, xs, ys)
    r, c = np.asarray(r), np.asarray(c)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    r, c, lab = r[inside], c[inside], df["class_id"].values[inside].astype(np.uint8)
    keep = valid[r, c]
    return r[keep], c[keep], lab[keep]
