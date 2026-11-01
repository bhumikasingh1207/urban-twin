# Remote Sensing, GIS, AI and Digital Twins for Urban Planning

A working prototype that takes **free satellite imagery → GIS processing → machine-learning land-use
classification → change / growth analysis → an interactive "digital twin" dashboard**.

> **Honest scope note.** The "digital twin" here is a *data-driven virtual representation* of a study
> area built from satellite data. It supports exploration and what-if scenarios. It has **no real-time
> synchronisation** with the physical city, and the dashboard says so.

---

## 1. Implementation status (read before submitting anything)

| Component | Status | How it was verified |
|---|---|---|
| Data acquisition (Sentinel-2, WorldCover, Landsat LST via STAC) | **Written, NOT yet run against live servers** | The code was written in an environment with no access to the data servers. Expect to debug the first real run (see Troubleshooting). |
| Preprocessing (common grid, CRS, cloud mask, missing-value report) | Implemented | Unit tests + synthetic run |
| GIS statistics, change detection, grid statistics | Implemented | Unit tests + synthetic run |
| Random Forest LULC classifier + evaluation | Implemented | Unit tests + synthetic run. **No real accuracy exists until you run it on real data.** |
| Growth susceptibility model + scenarios | Implemented (advanced) | Unit tests + synthetic run |
| Land surface temperature / NDVI indicators | Implemented (optional) | Synthetic run |
| Streamlit dashboard (8 pages) | Implemented | Headless test of all pages + browser screenshots on synthetic data |

**Synthetic data** (`--synthetic`) are computer-generated rasters used only to prove the code runs.
Any number computed from them is meaningless as a research result and **must not be put in your report**.
Real results exist only after you run the pipeline on real data (section 4) and record what it prints.

## 2. Install (Windows / macOS / Linux)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # 11 tests should pass
```

Needs Python 3.10+, ~8 GB RAM (16 GB comfortable), ~5 GB disk, no GPU. Internet is needed once for data.

## 3. Quick check with synthetic data (no internet)

```bash
python scripts/run_pipeline.py all --synthetic
streamlit run app/app.py
```
The sidebar shows a red SYNTHETIC banner. Real and synthetic outputs are stored separately
(`outputs/real/` vs `outputs/synthetic/`).

## 4. Run on real data

1. **Approve / edit the study area** in `config.yaml` (`aoi.bbox`, `aoi.crs`, `years`). The default
   (Noida–Greater Noida, ~25 × 25 km, UTM 43N) is a *proposal*. For another city pick the right UTM zone
   (`EPSG:326xx`, `xx` = zone number).
2. **Create independent validation points** (strongly recommended, see section 5).
3. Run:
   ```bash
   python scripts/run_pipeline.py acquire    # downloads only the AOI window; slow the first time
   python scripts/run_pipeline.py all        # check -> train -> predict -> analyze (skips downloads that exist)
   streamlit run app/app.py
   ```
   Stages can be run one at a time: `acquire | check | train | predict | analyze`.
4. Record **the numbers the program prints/saves** (`outputs/real/metrics.json`, etc.) in your report.

## 5. Creating validation points (needed for a scientifically valid accuracy)

Training labels come from ESA WorldCover, itself a classification product. Agreement with it is *not*
ground truth. To get honest accuracy:

1. Open QGIS, add the Esri/Google-style satellite basemap or the true-colour composite.
2. Create a point layer and digitise **≥ 50 points per class** (spread over the whole area, **not** only
   in obvious places), labelling each `class_id`: 1 Built-up, 2 Vegetation, 3 Cropland, 4 Water, 5 Bare/other.
3. Export as CSV with columns `lon,lat,class_id` (EPSG:4326) to `data/real/raw/validation_points.csv`
   (template: `data/validation_points_TEMPLATE.csv`).
4. Re-run `python scripts/run_pipeline.py train`. The dashboard then shows both evaluations.

Points are drawn for the training year's imagery (`train_year`). Points that fall on a training pixel are
automatically excluded.

## 6. Features and technology (why each is used)

| Need | Tool |
|---|---|
| Read only the AOI from cloud-hosted imagery | `pystac-client`, `planetary-computer`, `rasterio` |
| Raster/vector processing | `rasterio`, `geopandas`, `shapely`, `pyproj`, `numpy`, `scipy` |
| Classification & growth model | `scikit-learn` Random Forest (CPU only, explainable) |
| Dashboard & maps | `streamlit`, `folium`/Leaflet, `plotly` |

No deep learning, Docker, cloud services or paid APIs are used.

**ML pipeline.** Features per pixel = 6 Sentinel-2 bands (B02, B03, B04, B08, B11, B12) + NDVI, NDBI,
NDWI, MNDWI. Weak labels = homogeneous WorldCover pixels mapped to 5 classes. Evaluation uses
**spatial-block hold-out** (default 1 km blocks) to avoid leakage between neighbouring pixels, then optional
independent validation points. Outputs: overall accuracy, kappa, per-class precision/recall/F1, confusion matrix.

**Index limitations.** NDVI saturates in dense vegetation and is season-dependent; NDBI confuses bare soil with
built-up; NDWI confuses shadow/some roofs with water (MNDWI is more robust in cities).

## 7. Datasets

| Dataset | Source | Resolution | License | Notes |
|---|---|---|---|---|
| Sentinel-2 L2A | Copernicus / Microsoft Planetary Computer | 10–20 m | Free and open (attribute Copernicus) | UTM CRS; resampled to the common grid |
| ESA WorldCover 2021 | ESA via Planetary Computer | 10 m | CC BY 4.0 | training labels (weak) |
| Landsat 8/9 Collection 2 L2 | USGS via Planetary Computer | 30 m optical; thermal ~100 m resampled | Public domain | optional LST |
| OpenStreetMap / basemaps | OSM, Esri imagery tiles | – | ODbL / Esri terms | display only |

Check each provider's current terms before redistributing data. Acquisition dates are determined by the
cloud-free scenes found; check the STAC item dates in the log and record them in your report.

## 8. Project layout

```
config.yaml            study area, years, model settings
scripts/run_pipeline.py  command-line entry point
src/                   acquisition, preprocessing, features, labels, model, gis_stats, growth,
                       environment, insights, viz, synthetic (test data only), pipeline
app/app.py             Streamlit dashboard (reads saved outputs)
tests/                 pytest unit + end-to-end tests
docs/                  architecture diagrams, evidence checklist, progress-report template
```

## 9. Known limitations

* Change detection compares two classified maps, so classifier errors show up as speckled false change.
  A majority filter or a longer baseline would help (future work).
* WorldCover labels are 2021; applying the model to 2019/2025 assumes similar spectral conditions (same season).
* The growth model learns from one past period and uses only a few spatial drivers (no roads, terrain, zoning,
  population). It ranks susceptibility; it does not forecast.
* LST is a skin temperature from a coarse thermal band; compare only within the same season.
* The basemap needs internet; analysis outputs do not.

## 10. Troubleshooting

* **`No Sentinel-2 scenes found`** – widen `season` or `max_cloud_cover` in `config.yaml`.
* **Slow/hanging download** – shrink `aoi.bbox` or lower `sentinel2.max_scenes`. Re-running resumes:
  existing files are skipped (delete a file to force a re-download).
* **Planetary Computer unreachable** (college firewall) – use another network, or download scenes from
  Copernicus Data Space / USGS EarthExplorer manually, build composites in QGIS, and save them as
  `data/real/raw/s2_<year>.tif` (bands B02,B03,B04,B08,B11,B12 as reflectance 0–1, optionally a 7th band `n_valid_obs`)
  and `worldcover.tif` on the same grid. The rest of the pipeline works unchanged.
* **Map is blank in the dashboard** – the map libraries load from the internet; check your connection.
* **Memory errors** – shrink the AOI or set `resolution_m: 30`.

## 11. Documentation for submission

See `docs/architecture.md` (diagrams), `docs/EVIDENCE_CHECKLIST.md` and `docs/PROGRESS_REPORT_TEMPLATE.md`.
Discuss the submission with your project guide, and only claim what you have actually run.
