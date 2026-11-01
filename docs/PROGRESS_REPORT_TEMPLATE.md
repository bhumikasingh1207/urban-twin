# Project progress report (template - fill with YOUR actual results)

Title: Remote Sensing, GIS, AI and Digital Twins for Urban Planning
Team: ______  Guide: ______  Date: ______  Repository: ______  Demo video: ______

## 1. Work completed
Study area: ______ (approved on ______). Imagery used: ______ (dates, sensor, resolution).

## 2. Modules and status (mark one: Complete / Partial / Planned)
| Module | Status | Evidence |
|---|---|---|
| Satellite data acquisition & preprocessing | | |
| GIS processing & statistics | | |
| ML land-use classification | | |
| Change detection | | |
| Growth susceptibility / scenarios | | |
| Environmental indicators (NDVI, LST) | | |
| Digital twin dashboard | | |

## 3. Technologies used
Python, rasterio, GeoPandas, scikit-learn (Random Forest), Streamlit, Folium/Leaflet, Plotly; Sentinel-2,
ESA WorldCover, (Landsat).

## 4. Testing performed
Unit/integration tests (`pytest`): ___ passed on ______. Model evaluation: spatial-block hold-out and
independent validation points (n = ___).

## 5. Actual results (copy from outputs/real/*.json - do not round up or estimate)
| Metric | Value | Evaluation set |
|---|---|---|
| Overall accuracy | | |
| Kappa | | |
| Macro F1 | | |
Built-up area ____ km2 (year) -> ____ km2 (year). Growth model AUC ____ (distance-only baseline ____).

## 6. Current status, known issues, remaining work
______

## 7. Supporting evidence (links accessible to evaluators)
______

Declaration: all results above were produced by running the code in the repository; nothing is estimated.
