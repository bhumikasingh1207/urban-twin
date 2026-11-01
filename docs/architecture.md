# System architecture

All diagrams are Mermaid (render on GitHub, VS Code with the Mermaid extension, or https://mermaid.live).

## 1. Architecture overview
```mermaid
flowchart LR
  subgraph Sources[Open data]
    S2[Sentinel-2 L2A]:::d
    WC[ESA WorldCover]:::d
    LS[Landsat 8/9 L2]:::d
    VP[Team validation points - QGIS]:::d
  end
  subgraph Pipeline[Python pipeline - scripts/run_pipeline.py]
    ACQ[acquisition.py]
    PRE[preprocessing.py]
    FEA[features.py]
    MOD[model.py - Random Forest]
    GIS[gis_stats.py]
    GRO[growth.py]
    ENV[environment.py]
  end
  STORE[(data/, models/, outputs/)]
  APP[Streamlit digital-twin dashboard\napp/app.py]
  S2 & WC & LS --> ACQ --> PRE --> FEA --> MOD --> GIS --> GRO
  VP --> MOD
  LS --> ENV
  GIS & GRO & ENV & MOD --> STORE --> APP
  classDef d fill:#eef,stroke:#88a
```

## 2. Data flow
```mermaid
flowchart TD
  A[STAC search: AOI + season + cloud filter] --> B[Read AOI window, reproject to common UTM grid]
  B --> C[SCL cloud mask, median composite, n_valid_obs]
  C --> D[Feature stack: 6 bands + NDVI NDBI NDWI MNDWI]
  W[WorldCover] --> L[Map to 5 classes, keep homogeneous pixels]
  D & L --> T[Spatial-block train/test split]
  T --> RF[Random Forest + metrics.json]
  RF --> P[Predict LULC per year -> lulc_YYYY.tif]
  P --> CH[Change map, transition matrix, grid stats]
  P --> GR[Growth model -> growth_prob.tif]
  CH & GR --> UI[Dashboard + downloads]
```

## 3. Module interaction
```mermaid
flowchart LR
  CFG[config.yaml] --> RUN[scripts/run_pipeline.py]
  RUN --> PIPE[src/pipeline.py] --> ACQ[acquisition] & CHK[preprocessing] & MODEL[model] & GIS[gis_stats] & GROW[growth] & ENVM[environment]
  MODEL & GIS & GROW & ENVM --> OUT[(outputs/real | outputs/synthetic)]
  OUT --> APP[app/app.py] --> VIZ[viz.py] & INS[insights.py]
```

## 4. AI/ML pipeline
```mermaid
flowchart LR
  X[Pixel features] --> S[Stratified sample per class]
  Y[Weak labels: WorldCover] --> S
  S --> SPL[Spatial-block split - no neighbouring-pixel leakage]
  SPL --> FIT[RandomForest, class_weight=balanced]
  FIT --> E1[Held-out blocks: accuracy, kappa, F1, confusion matrix]
  FIT --> E2[Independent QGIS points: same metrics - true accuracy]
  FIT --> SAVE[joblib model + metrics.json]
  SAVE --> PRED[Predict full rasters]
```
