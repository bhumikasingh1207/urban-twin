"""Automated checks. Run:  pytest -q
These test the CODE (correct maths, no leakage, consistent areas). They use tiny synthetic data, so
they say nothing about real-world accuracy - that comes from running the pipeline on real data."""
import json

import numpy as np
import pytest

from src import synthetic
from src.features import compute_indices, normalized_difference
from src.gis_stats import build_change, class_area_table, grid_stats, transition_matrix
from src.growth import allocate_growth, train_growth_model
from src.labels import homogeneous_mask, map_worldcover, sample_weak_labels
from src.model import predict_year, spatial_block_split, train_model
from src.pipeline import stage_analyze, stage_check
from src.utils import Paths, load_config, make_grid, read_raster
from rasterio.transform import from_origin


def test_normalized_difference_known_values():
    nir, red = np.array([0.5]), np.array([0.1])
    assert normalized_difference(nir, red)[0] == pytest.approx(0.6666667, abs=1e-6)
    assert np.isnan(normalized_difference(np.array([0.0]), np.array([0.0]))[0])  # 0/0 -> NaN, not inf


def test_indices_water_is_positive_ndwi():
    idx = compute_indices(green=np.array([0.07]), red=np.array([0.05]), nir=np.array([0.03]), swir1=np.array([0.02]))
    assert idx["NDWI"][0] > 0 and idx["MNDWI"][0] > 0 and idx["NDVI"][0] < 0


def test_make_grid_is_metric_and_covers_bbox():
    tf, w, h = make_grid([77.30, 28.40, 77.55, 28.65], "EPSG:32643", 20)
    assert abs(tf.a) == 20 and 1000 < w < 2000 and 1000 < h < 2000


def test_worldcover_mapping_and_homogeneity():
    wc = np.full((7, 7), 40, np.uint8)
    wc[3, 3] = 50
    cls = map_worldcover(wc)
    assert cls[3, 3] == 1 and cls[0, 0] == 3
    hm = homogeneous_mask(cls)
    assert not hm[3, 3] and not hm[2, 2] and bool(hm[0, 0])


def test_sampling_respects_valid_mask():
    cls = np.full((20, 20), 3, np.uint8)
    valid = np.zeros((20, 20), bool)
    valid[:, :10] = True
    r, c, y = sample_weak_labels(cls, valid, 50, seed=0)
    assert (c < 10).all() and (y == 3).all()


def test_spatial_split_has_no_block_shared_between_train_and_test():
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 300, 5000), rng.integers(0, 300, 5000)
    test = spatial_block_split(rows, cols, (300, 300), 50, 0.3, seed=1)
    ids = (rows // 50) * 6 + cols // 50
    assert set(ids[test]).isdisjoint(set(ids[~test])) and test.any() and (~test).any()


def test_area_table_and_change_consistency():
    tf = from_origin(0, 1000, 20, 20)  # 400 m2 pixels
    a = np.array([[1, 1, 2], [3, 4, 0]], np.uint8)
    b = np.array([[1, 3, 1], [3, 4, 5]], np.uint8)
    t = class_area_table(a, tf)
    assert t.pixels.sum() == 5 and t.area_km2.sum() == pytest.approx(5 * 400 / 1e6)
    ch = build_change(a, b)
    assert ch[0, 0] == 2 and ch[0, 1] == 4 and ch[0, 2] == 3 and ch[1, 2] == 0
    tm = transition_matrix(a, b, tf)
    assert tm.values.sum() == pytest.approx(5 * 400 / 1e6)


def test_allocate_growth_respects_protection_and_amount():
    prob = np.random.default_rng(0).random((30, 30)).astype(np.float32)
    cand = np.ones((30, 30), bool)
    prot = np.zeros((30, 30), bool)
    prot[:15] = True
    new = allocate_growth(prob, cand, 100, prot)
    assert new.sum() == 100 and not (new & prot).any()
    assert prob[new].min() >= np.sort(prob[~prot].ravel())[-100] - 1e-6  # took the most susceptible
    assert allocate_growth(prob, cand, 10_000, prot).sum() == (~prot).sum()  # capped by what is allowed


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    cfg = load_config()
    cfg["model"]["n_per_class"] = 1500
    cfg["model"]["rf"]["n_estimators"] = 40
    cfg["growth"]["n_samples"] = 6000
    paths = Paths(tmp_path_factory.mktemp("proj"), "synthetic").ensure()
    synthetic.generate(paths, cfg, size=240)
    return cfg, paths


def test_end_to_end_pipeline_on_synthetic_data(synth):
    cfg, paths = synth
    stage_check(cfg, paths)
    m = train_model(cfg, paths)
    # independent points must not be scored if they overlap training pixels; metrics must be real numbers
    assert 0.5 < m["weak_label_test"]["overall_accuracy"] <= 1.0
    assert m["independent_validation"] is not None
    cm = np.array(m["weak_label_test"]["confusion_matrix"])
    if not m["classes_missing_from_train_or_test"]:
        assert cm.sum() == m["n_test"]  # every held-out sample is scored exactly once
    for y in cfg["years"]:
        predict_year(paths, y)
        arr, _ = read_raster(paths.lulc(y))
        assert arr.dtype == np.uint8 and set(np.unique(arr)) <= {0, 1, 2, 3, 4, 5}
    stage_analyze(cfg, paths)
    summary = json.loads((paths.out / "summary.json").read_text())
    assert "growth" in summary and 0 <= summary["growth"]["roc_auc"] <= 1


def test_metrics_file_is_written_by_code_not_hardcoded(synth):
    cfg, paths = synth
    m = json.loads((paths.out / "metrics.json").read_text())
    assert m["label_source"].startswith("ESA WorldCover") and m["n_train"] > 0


def test_grid_stats_geometry_wgs84(synth):
    cfg, paths = synth
    la, meta = read_raster(paths.lulc(cfg["years"][0]))
    lb, _ = read_raster(paths.lulc(cfg["years"][-1]))
    g = grid_stats(la[0], lb[0], meta["transform"], meta["crs"], 1000)
    assert str(g.crs) == "EPSG:4326" and len(g) > 0 and (g.built_pct_b.between(0, 100)).all()
