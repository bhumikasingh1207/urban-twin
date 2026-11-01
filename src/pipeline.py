"""Pipeline stages: acquire -> train -> predict -> analyze. Used by scripts/run_pipeline.py."""
from __future__ import annotations

import json

import pandas as pd

from . import acquisition, synthetic
from .environment import lst_by_class, ndvi_by_class
from .gis_stats import build_change, class_area_table, grid_stats, transition_matrix
from .growth import train_growth_model
from .model import predict_year, train_model
from .preprocessing import check_alignment, missing_report
from .features import BAND_NAMES
from .utils import get_logger, read_raster, write_raster

log = get_logger("pipeline")


def stage_acquire(cfg, paths, use_synthetic: bool):
    if use_synthetic:
        log.warning("SYNTHETIC mode: generating computer-made test data (not satellite imagery).")
        synthetic.generate(paths, cfg)
        return
    for y in cfg["years"]:
        if paths.s2(y).exists():
            log.info("%s exists - skipping (delete it to re-download)", paths.s2(y))
            continue
        acquisition.build_s2_composite(cfg, paths.s2(y), y)
    if not paths.worldcover().exists():
        acquisition.fetch_worldcover(cfg, paths.worldcover(), 2021)
    if not paths.lst(cfg["lst"]["year"]).exists():
        try:
            acquisition.build_lst_composite(cfg, paths.lst(cfg["lst"]["year"]))
        except Exception as exc:  # LST is optional
            log.warning("LST acquisition failed (%s) - continuing without it.", exc)


def stage_check(cfg, paths):
    files = [paths.s2(y) for y in cfg["years"]] + [paths.worldcover()]
    check_alignment(*files)
    report = {}
    for y in cfg["years"]:
        arr, _ = read_raster(paths.s2(y))
        report[str(y)] = missing_report(arr[: len(BAND_NAMES)], BAND_NAMES)
    (paths.out / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("alignment OK; preprocessing report saved")


def stage_train(cfg, paths):
    train_model(cfg, paths)


def stage_predict(cfg, paths):
    for y in cfg["years"]:
        predict_year(paths, y)


def stage_analyze(cfg, paths):
    years = cfg["years"]
    ya, yb = years[0], years[-1]
    frames = []
    for y in years:
        lulc, meta = read_raster(paths.lulc(y))
        df = class_area_table(lulc[0], meta["transform"])
        df["year"] = y
        frames.append(df)
    area = pd.concat(frames)
    area.to_csv(paths.out / "area_stats.csv", index=False)

    la, meta = read_raster(paths.lulc(ya))
    lb, _ = read_raster(paths.lulc(yb))
    la, lb, tf = la[0], lb[0], meta["transform"]
    chg = build_change(la, lb)
    write_raster(paths.out / "change_map.tif", chg, meta["crs"], tf, nodata=0, descriptions=["change_class"])
    trans = transition_matrix(la, lb, tf)
    trans.to_csv(paths.out / "transition_matrix_km2.csv")
    grid_stats(la, lb, tf, meta["crs"], cfg["growth"]["cell_m"]).to_file(paths.out / "grid_stats.geojson", driver="GeoJSON")
    ndvi_by_class(paths, years).to_csv(paths.out / "ndvi_by_class.csv", index=False)

    summary = {"change_years": [ya, yb]}
    lst_path = paths.lst(cfg["lst"]["year"])
    lulc_for_lst = cfg["lst"]["year"] if cfg["lst"]["year"] in years else yb
    if lst_path.exists():
        df, s = lst_by_class(lst_path, paths.lulc(lulc_for_lst))
        df.to_csv(paths.out / "lst_by_class.csv", index=False)
        summary["lst"] = {**s, "lulc_year_used": lulc_for_lst}
    try:
        summary["growth"] = train_growth_model(cfg, paths, ya, yb)
    except ValueError as exc:
        log.warning("Growth model skipped: %s", exc)
        summary["growth_error"] = str(exc)
    (paths.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("analysis complete -> %s", paths.out)
