"""Module 3c - Random Forest LULC classifier: training, honest evaluation, prediction.

Leakage control
---------------
Neighbouring pixels are highly correlated, so a random pixel split would put near-duplicates in
both train and test and inflate accuracy. We split by SPATIAL BLOCKS (default 1 km x 1 km): whole
blocks are either train or test. Independent validation points that fall on a training pixel are
removed before scoring.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix

from .features import FEATURE_NAMES, build_feature_stack
from .labels import load_validation_points, map_worldcover, sample_weak_labels
from .utils import CLASS_NAMES, get_logger, read_raster, write_raster

log = get_logger("model")


def spatial_block_split(rows, cols, shape, block: int, test_fraction: float, seed: int):
    """Boolean mask: True = sample belongs to a held-out spatial block."""
    n_cols_blocks = int(np.ceil(shape[1] / block))
    block_id = (rows // block) * n_cols_blocks + (cols // block)
    ids = np.unique(block_id)
    if ids.size < 2:
        raise ValueError("Study area has fewer than 2 spatial blocks; reduce block_size_px.")
    rng = np.random.default_rng(seed)
    n_test = int(np.clip(round(test_fraction * ids.size), 1, ids.size - 1))
    test_blocks = rng.choice(ids, size=n_test, replace=False)
    return np.isin(block_id, test_blocks)


def class_aware_block_split(rows, cols, y, shape, block, test_fraction, seed, min_per_class=10, attempts=200):
    """Spatial-block split that (when possible) keeps every class in BOTH train and test.

    Tries seeds seed, seed+1, ... and returns the first split where each class has at least
    `min_per_class` samples on each side. Falls back to the plain split and logs a warning.
    """
    best = None
    for k in range(attempts):
        test = spatial_block_split(rows, cols, shape, block, test_fraction, seed + k)
        if best is None:
            best = test
        if all((y[test] == c).sum() >= min_per_class and (y[~test] == c).sum() >= min_per_class
               for c in np.unique(y)):
            return test
    log.warning("Could not find a spatial split with every class in both train and test; "
                "classes with too few samples will be missing from the evaluation.")
    return best


def _scores(y_true, y_pred, labels) -> dict:
    names = [CLASS_NAMES[c] for c in labels]
    rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "per_class": {n: {k: float(rep[n][k]) for k in ("precision", "recall", "f1-score", "support")} for n in names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [CLASS_NAMES[c] for c in labels],
    }


def _plot_cm(cm, labels, title, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"), ax.set_ylabel("Reference"), ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(cfg: dict, paths) -> dict:
    mc, year = cfg["model"], cfg["train_year"]
    X_img, valid, meta = build_feature_stack(paths.s2(year))
    wc = read_raster(paths.worldcover())[0][0]
    if wc.shape != valid.shape:
        raise ValueError("WorldCover and Sentinel-2 grids differ - re-run acquisition.")
    cls = map_worldcover(wc)

    rows, cols, y = sample_weak_labels(cls, valid, mc["n_per_class"], mc["seed"])
    X = X_img[rows, cols]
    is_test = class_aware_block_split(rows, cols, y, valid.shape, mc["block_size_px"], mc["test_fraction"], mc["seed"])
    Xtr, ytr, Xte, yte = X[~is_test], y[~is_test], X[is_test], y[is_test]
    labels = sorted(set(np.unique(ytr)) & set(np.unique(yte)))
    log.info("train=%d test=%d classes=%s", len(ytr), len(yte), [CLASS_NAMES[c] for c in labels])

    rf = RandomForestClassifier(
        n_estimators=mc["rf"]["n_estimators"], max_depth=mc["rf"]["max_depth"],
        min_samples_leaf=mc["rf"]["min_samples_leaf"], class_weight="balanced",
        n_jobs=-1, random_state=mc["seed"],
    ).fit(Xtr, ytr)

    metrics = {
        "train_year": year,
        "label_source": "ESA WorldCover (weak labels, NOT ground truth)",
        "split": f"spatial blocks of {mc['block_size_px']} px, {int(mc['test_fraction']*100)}% of blocks held out",
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "classes_missing_from_train_or_test": [CLASS_NAMES[c] for c in np.unique(y) if c not in labels],
        "features": FEATURE_NAMES,
        "rf_params": {**mc["rf"], "class_weight": "balanced", "seed": mc["seed"]},
        "weak_label_test": _scores(yte, rf.predict(Xte), labels),
        "independent_validation": None,
        "feature_importance": dict(sorted(zip(FEATURE_NAMES, map(float, rf.feature_importances_)),
                                          key=lambda t: -t[1])),
    }
    _plot_cm(metrics["weak_label_test"]["confusion_matrix"], metrics["weak_label_test"]["labels"],
             "Held-out blocks vs WorldCover (weak labels)", paths.out / "confusion_matrix_weak.png")

    vpath = paths.validation_csv()
    if vpath.exists():
        vr, vc, vy = load_validation_points(vpath, meta["transform"], meta["crs"], valid.shape, valid)
        train_pix = set(zip(rows[~is_test].tolist(), cols[~is_test].tolist()))  # leakage guard
        keep = np.array([(r, c) not in train_pix for r, c in zip(vr.tolist(), vc.tolist())], dtype=bool)
        vr, vc, vy = vr[keep], vc[keep], vy[keep]
        if len(vy) >= 20:
            vlabels = sorted(set(np.unique(vy)))
            metrics["independent_validation"] = _scores(vy, rf.predict(X_img[vr, vc]), vlabels)
            metrics["independent_validation"]["source"] = "team-digitised validation points (QGIS)"
            _plot_cm(metrics["independent_validation"]["confusion_matrix"],
                     metrics["independent_validation"]["labels"],
                     "Independent validation points", paths.out / "confusion_matrix_independent.png")
        else:
            log.warning("Only %d usable validation points - need >=20; skipping independent evaluation.", len(vy))
    else:
        log.warning("No validation_points.csv found: only weak-label evaluation is available. "
                    "Create independent points in QGIS (see README).")

    joblib.dump(rf, paths.models / "rf_lulc.joblib", compress=3)
    (paths.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("weak-label test accuracy=%.4f kappa=%.4f", metrics["weak_label_test"]["overall_accuracy"],
             metrics["weak_label_test"]["kappa"])
    return metrics


def predict_array(model, X_img: np.ndarray, valid: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    h, w, f = X_img.shape
    flat, out = X_img.reshape(-1, f), np.zeros(h * w, dtype=np.uint8)
    idx = np.flatnonzero(valid.ravel())
    for s in range(0, idx.size, chunk):
        i = idx[s: s + chunk]
        out[i] = model.predict(flat[i])
    return out.reshape(h, w)


def predict_year(paths, year: int) -> None:
    model = joblib.load(paths.models / "rf_lulc.joblib")
    X_img, valid, meta = build_feature_stack(paths.s2(year))
    lulc = predict_array(model, X_img, valid)
    write_raster(paths.lulc(year), lulc, meta["crs"], meta["transform"], nodata=0,
                 descriptions=["lulc_class"])
    log.info("saved %s (%.1f%% pixels classified)", paths.lulc(year), valid.mean() * 100)
