"""Module 3a - spectral indices and the feature stack used by the classifier.

Limitations (state these in your report):
  NDVI  (NIR-Red)/(NIR+Red)       vegetation vigour; saturates over dense canopy; dry crops look like soil.
  NDBI  (SWIR1-NIR)/(SWIR1+NIR)   built-up proxy; bare soil and fallow fields also score high.
  NDWI  (Green-NIR)/(Green+NIR)   open water; shadows and some built-up surfaces can be confused with water.
  MNDWI (Green-SWIR1)/(Green+SWIR1) better water/built-up separation than NDWI in urban scenes.
"""
from __future__ import annotations

import numpy as np

from .utils import read_raster

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]
INDEX_NAMES = ["NDVI", "NDBI", "NDWI", "MNDWI"]
FEATURE_NAMES = BAND_NAMES + INDEX_NAMES


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
    r = r.astype(np.float32)
    r[~np.isfinite(r)] = np.nan
    return r


def compute_indices(green, red, nir, swir1) -> dict[str, np.ndarray]:
    return {
        "NDVI": normalized_difference(nir, red),
        "NDBI": normalized_difference(swir1, nir),
        "NDWI": normalized_difference(green, nir),
        "MNDWI": normalized_difference(green, swir1),
    }


def build_feature_stack(s2_path):
    """Return (X[h,w,F] float32, valid[h,w] bool, meta). Invalid pixels (cloud/no data) stay NaN."""
    arr, meta = read_raster(s2_path)
    refl = arr[: len(BAND_NAMES)].astype(np.float32)
    b02, b03, b04, b08, b11, b12 = refl
    idx = compute_indices(b03, b04, b08, b11)
    X = np.stack(list(refl) + [idx[n] for n in INDEX_NAMES], axis=-1)
    valid = np.isfinite(X).all(axis=-1)
    return X, valid, meta
