"""Module 1b - preprocessing checks: alignment, CRS, missing-value reporting."""
from __future__ import annotations

import numpy as np

from .utils import read_raster


def check_alignment(*paths) -> None:
    """All rasters must share CRS, transform and shape (guaranteed by acquisition; verified here)."""
    ref_meta, ref_shape = None, None
    for p in paths:
        arr, meta = read_raster(p)
        if ref_meta is None:
            ref_meta, ref_shape = meta, arr.shape[1:]
            continue
        if meta["crs"] != ref_meta["crs"] or meta["transform"] != ref_meta["transform"] or arr.shape[1:] != ref_shape:
            raise ValueError(f"{p} is not aligned with {paths[0]} (CRS/transform/shape differ)")


def missing_report(stack: np.ndarray, names: list[str]) -> dict:
    """Percentage of NaN pixels per band and of pixels invalid in any band."""
    nan = ~np.isfinite(stack)
    return {
        "per_band_missing_pct": {n: round(float(nan[i].mean() * 100), 3) for i, n in enumerate(names)},
        "any_band_missing_pct": round(float(nan.any(axis=0).mean() * 100), 3),
    }
