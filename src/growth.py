"""Advanced module - urban growth susceptibility model and what-if scenario allocation.

What it does
  1. Learns, from the change between two classified years (A -> B), which non-built pixels were
     converted to built-up, using spatial drivers (distance to built-up / water, neighbourhood
     built-up density, NDVI, NDBI).
  2. Applies the learned model to year B to produce a conversion-probability map.
  3. `allocate_growth` turns a planner's scenario (area to convert + protected zones) into a
     projected built-up map by converting the highest-probability pixels first.

Honest limitations: the labels (change) come from a classifier, so classification errors appear as
spurious change; the model is a SUSCEPTIBILITY ranking learned from one past period, not a
guaranteed forecast. Roads, zoning, terrain and population drivers are not included (future work).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from .features import build_feature_stack
from .model import spatial_block_split
from .utils import get_logger, pixel_area_km2, read_raster, write_raster

log = get_logger("growth")
DRIVER_NAMES = ["dist_to_builtup_m", "dist_to_water_m", "builtup_density_500m", "builtup_density_1500m", "NDVI", "NDBI"]


def driver_features(builtup: np.ndarray, water: np.ndarray, ndvi: np.ndarray, ndbi: np.ndarray, res: float):
    far = float(max(builtup.shape) * res)
    d_built = distance_transform_edt(~builtup) * res if builtup.any() else np.full(builtup.shape, far)
    d_water = distance_transform_edt(~water) * res if water.any() else np.full(builtup.shape, far)
    f = builtup.astype(np.float32)
    dens500 = uniform_filter(f, size=2 * max(int(500 / res), 1) + 1, mode="nearest")
    dens1500 = uniform_filter(f, size=2 * max(int(1500 / res), 1) + 1, mode="nearest")
    X = np.stack([d_built, d_water, dens500, dens1500, ndvi, ndbi], axis=-1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0)


def _drivers_for_year(paths, lulc, year, res):
    X, _, _ = build_feature_stack(paths.s2(year))
    ndvi, ndbi = X[..., 6], X[..., 7]  # FEATURE_NAMES order: 6 bands, NDVI, NDBI, ...
    return driver_features(lulc == 1, lulc == 4, ndvi, ndbi, res)


def train_growth_model(cfg: dict, paths, ya: int, yb: int) -> dict:
    la, meta = read_raster(paths.lulc(ya))
    lb, _ = read_raster(paths.lulc(yb))
    la, lb = la[0], lb[0]
    res = abs(meta["transform"].a)
    Xd = _drivers_for_year(paths, la, ya, res)

    cand = (la > 0) & (lb > 0) & (la != 1) & (la != 4)
    rows, cols = np.nonzero(cand)
    y = (lb[rows, cols] == 1).astype(int)
    rng = np.random.default_rng(cfg["model"]["seed"])
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if pos.size < 50:
        raise ValueError(f"Only {pos.size} converted pixels between {ya} and {yb}; "
                         "growth model needs a longer period or a larger study area.")
    half = cfg["growth"]["n_samples"] // 2
    sel = np.concatenate([rng.choice(pos, min(half, pos.size), replace=False),
                          rng.choice(neg, min(half, neg.size), replace=False)])
    rows, cols, y = rows[sel], cols[sel], y[sel]
    X = Xd[rows, cols]

    mc = cfg["model"]
    test = spatial_block_split(rows, cols, la.shape, mc["block_size_px"], mc["test_fraction"], mc["seed"])
    rf = RandomForestClassifier(n_estimators=mc["rf"]["n_estimators"], max_depth=mc["rf"]["max_depth"],
                                min_samples_leaf=mc["rf"]["min_samples_leaf"], class_weight="balanced",
                                n_jobs=-1, random_state=mc["seed"]).fit(X[~test], y[~test])
    p = rf.predict_proba(X[test])[:, 1]
    yt = y[test]
    n_years = max(yb - ya, 1)
    pa = pixel_area_km2(meta["transform"])
    metrics = {
        "period": [ya, yb], "n_train": int((~test).sum()), "n_test": int(test.sum()),
        "note": "Test set is class-balanced by sampling; AUC/AP are prevalence-independent, "
                "precision/recall are for the balanced sample only.",
        "roc_auc": float(roc_auc_score(yt, p)),
        "average_precision": float(average_precision_score(yt, p)),
        "precision@0.5": float(precision_score(yt, p >= 0.5, zero_division=0)),
        "recall@0.5": float(recall_score(yt, p >= 0.5, zero_division=0)),
        "baseline_auc_distance_only": float(roc_auc_score(yt, -X[test][:, 0])),
        "feature_importance": dict(sorted(zip(DRIVER_NAMES, map(float, rf.feature_importances_)), key=lambda t: -t[1])),
        "observed_gain_km2": float(((la != 1) & (lb == 1) & (la > 0)).sum() * pa),
        "observed_gain_km2_per_year": float(((la != 1) & (lb == 1) & (la > 0)).sum() * pa / n_years),
    }
    joblib.dump(rf, paths.models / "growth_rf.joblib", compress=3)

    # probability map for the next period, using year B as the starting state
    Xb = _drivers_for_year(paths, lb, yb, res)
    candb = (lb > 0) & (lb != 1) & (lb != 4)
    prob = np.full(lb.shape, np.nan, dtype=np.float32)
    idx = np.flatnonzero(candb.ravel())
    flat = Xb.reshape(-1, Xb.shape[-1])
    out = prob.ravel()
    for s in range(0, idx.size, 400_000):
        i = idx[s: s + 400_000]
        out[i] = rf.predict_proba(flat[i])[:, 1]
    write_raster(paths.out / "growth_prob.tif", out.reshape(lb.shape), meta["crs"], meta["transform"],
                 nodata=np.nan, descriptions=["conversion_probability"])
    (paths.out / "growth_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("growth model AUC=%.3f (distance-only baseline %.3f)", metrics["roc_auc"], metrics["baseline_auc_distance_only"])
    return metrics


def allocate_growth(prob: np.ndarray, candidate: np.ndarray, n_pixels: int, protected: np.ndarray | None = None) -> np.ndarray:
    """Convert the `n_pixels` most susceptible allowed pixels. Returns boolean mask of new built-up."""
    allowed = candidate & np.isfinite(prob)
    if protected is not None:
        allowed &= ~protected
    n_pixels = int(min(max(n_pixels, 0), allowed.sum()))
    new = np.zeros(prob.shape, dtype=bool)
    if n_pixels == 0:
        return new
    score = np.where(allowed, prob, -np.inf).ravel()
    top = np.argpartition(-score, n_pixels - 1)[:n_pixels]
    new.ravel()[top] = True
    return new
