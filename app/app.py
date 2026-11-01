"""Digital Twin dashboard (Streamlit).  Launch:  streamlit run app/app.py

The app only READS results produced by scripts/run_pipeline.py, so it starts fast and every number
shown comes from a saved output file.
"""
import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gis_stats import CHANGE_INFO  # noqa: E402
from src.growth import allocate_growth  # noqa: E402
from src.insights import build_html_report, make_insights  # noqa: E402
from src.utils import CLASS_COLORS, CLASS_NAMES, ROOT, get_paths, load_config, pixel_area_km2, read_raster  # noqa: E402
from src.viz import colorize_categorical, colorize_continuous, legend_html, to_wgs84, true_colour  # noqa: E402

st.set_page_config(page_title="Urban Digital Twin", page_icon="🛰️", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}</style>", unsafe_allow_html=True)
CFG = load_config()


# ----------------------------------------------------------------------------- data access
def _ready(synthetic: bool) -> bool:
    return (ROOT / "outputs" / ("synthetic" if synthetic else "real") / "summary.json").exists()


modes = [m for m in (False, True) if _ready(m)]
if not modes:
    st.title("Urban Digital Twin")
    st.error("No results found yet. Run the pipeline first:")
    st.code("python scripts/run_pipeline.py all               # real data (needs internet)\n"
            "python scripts/run_pipeline.py all --synthetic   # code check only (fake data)")
    st.stop()

st.sidebar.title("🛰️ Urban Digital Twin")
if len(modes) == 2:
    pick = st.sidebar.radio("Data mode", ["Real satellite data", "SYNTHETIC test data"])
    SYN = pick.startswith("SYNTHETIC")
else:
    SYN = modes[0]
P = get_paths(synthetic=SYN)
if SYN:
    st.sidebar.error("SYNTHETIC test data: computer-generated, not satellite imagery. Do not report these numbers.")

page = st.sidebar.radio("Navigate", ["Overview", "Interactive map", "Land-use statistics", "Model evaluation",
                                     "Change detection", "Growth scenarios", "Environment",
                                     "Planning insights & export"])
YEARS = CFG["years"]


@st.cache_resource(show_spinner=False)
def _raster(path: str, mtime: float):
    return read_raster(path)


def raster(path: Path):
    return _raster(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _overlay(path: str, mtime: float, kind: str):
    arr, meta = read_raster(path)
    if kind == "lulc":
        a, b = to_wgs84(arr[0], meta, True)
        return colorize_categorical(a, CLASS_COLORS), b
    if kind == "change":
        a, b = to_wgs84(arr[0], meta, True)
        return colorize_categorical(a, {k: v[1] for k, v in CHANGE_INFO.items() if k != 1}), b
    if kind == "growth":
        a, b = to_wgs84(arr[0], meta, False)
        return colorize_continuous(a, "magma", 0, 1), b
    if kind == "rgb":
        return true_colour(arr, meta)
    raise ValueError(kind)


def overlay(path: Path, kind: str):
    return _overlay(str(path), path.stat().st_mtime, kind)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


area = load_csv(P.out / "area_stats.csv")
metrics = load_json(P.out / "metrics.json")
summary = load_json(P.out / "summary.json") or {}
growth_m = load_json(P.out / "growth_metrics.json")
trans = load_csv(P.out / "transition_matrix_km2.csv")
grid = gpd.read_file(P.out / "grid_stats.geojson") if (P.out / "grid_stats.geojson").exists() else None
YA, YB = summary.get("change_years", [YEARS[0], YEARS[-1]])
if trans is not None:
    trans = trans.set_index(trans.columns[0])


def fmt_pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def legend_row(items):
    st.markdown(" &nbsp; ".join(
        f"<span style='display:inline-block;width:12px;height:12px;background:{c};border:1px solid #666;"
        f"margin-right:4px'></span>{n}" for n, c in items), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- pages
if page == "Overview":
    st.title("Remote Sensing, GIS, AI and Digital Twins for Urban Planning")
    st.caption(f"Study area: {CFG['aoi']['name']} · analysis grid {CFG['resolution_m']} m · "
               f"{CFG['aoi']['crs']} · years {', '.join(map(str, YEARS))}")
    st.info("**About this digital twin.** It is a *data-driven virtual representation* of the study area built "
            "from satellite imagery, GIS layers and machine-learning outputs. It supports exploration and "
            "what-if scenarios, but it is **not** synchronised in real time with the physical city.")
    c1, c2, c3, c4 = st.columns(4)
    if area is not None:
        b = area[(area.year == YB) & (area["class"] == "Built-up")].area_km2.iloc[0]
        a = area[(area.year == YA) & (area["class"] == "Built-up")].area_km2.iloc[0]
        c1.metric(f"Built-up area {YB}", f"{b:.1f} km²", f"{b - a:+.1f} km² since {YA}")
    if metrics:
        iv = metrics.get("independent_validation")
        c2.metric("Accuracy: independent points", fmt_pct(iv["overall_accuracy"]) if iv else "not available")
        c3.metric("Accuracy: held-out blocks (weak labels)", fmt_pct(metrics["weak_label_test"]["overall_accuracy"]))
    if growth_m:
        c4.metric("Growth model AUC", f"{growth_m['roc_auc']:.2f}")
    st.subheader("Workflow")
    st.markdown("**Remote sensing** (Sentinel-2 / Landsat) → **GIS** (common grid, area statistics, change maps) → "
                "**AI/ML** (Random Forest classifier + growth-susceptibility model) → **Digital twin** (this "
                "dashboard) → **Planning insights**.")
    st.markdown("Use the sidebar to open the map, statistics, model evaluation, change detection, growth "
                "scenarios and exports.")

elif page == "Interactive map":
    st.title("Interactive map")
    left, right = st.columns([1, 3])
    with left:
        year = st.selectbox("Year", YEARS, index=len(YEARS) - 1)
        show_rgb = st.checkbox("Satellite true-colour composite", value=False)
        show_lulc = st.checkbox("Land-use classification", value=True)
        show_change = st.checkbox(f"Built-up change {YA}→{YB}", value=False)
        show_growth = st.checkbox("Growth susceptibility", value=False)
        show_grid = st.checkbox("Grid: built-up change (pp)", value=False)
        opacity = st.slider("Overlay opacity", 0.1, 1.0, 0.75)
    base_img, bounds = overlay(P.lulc(year), "lulc")
    centre = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=centre, zoom_start=12, tiles="OpenStreetMap")
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Esri satellite basemap", show=False).add_to(m)

    def add(img, b, name, show=True):
        folium.raster_layers.ImageOverlay(image=img, bounds=b, opacity=opacity, name=name, show=show).add_to(m)

    if show_rgb:
        img, b = overlay(P.s2(year), "rgb")
        add(img, b, f"Sentinel-2 true colour {year}")
    if show_lulc:
        add(base_img, bounds, f"Land use {year}")
        m.get_root().html.add_child(folium.Element(
            legend_html("Land use", [(CLASS_NAMES[k], CLASS_COLORS[k]) for k in CLASS_NAMES])))
    if show_change:
        img, b = overlay(P.out / "change_map.tif", "change")
        add(img, b, f"Built-up change {YA}-{YB}")
    if show_growth and (P.out / "growth_prob.tif").exists():
        img, b = overlay(P.out / "growth_prob.tif", "growth")
        add(img, b, "Conversion probability")
    if show_grid and grid is not None:
        folium.GeoJson(grid[["built_change_pp", "gain_km2", "geometry"]].to_json(), name="Grid cells",
                       style_function=lambda f: {"fillOpacity": 0.15, "weight": 0.6, "color": "#444"},
                       tooltip=folium.GeoJsonTooltip(["built_change_pp", "gain_km2"],
                                                     aliases=["Built-up change (pp)", "New built-up (km²)"])).add_to(m)
    m.fit_bounds(bounds)
    folium.LayerControl(collapsed=False).add_to(m)
    with right:
        st_folium(m, height=650, use_container_width=True, returned_objects=[])
    if show_change:
        legend_row([(v[0], v[1]) for k, v in CHANGE_INFO.items() if k != 1])
    if show_growth:
        st.caption("Growth susceptibility: probability (0 dark → 1 bright) that a non-built pixel converts to built-up.")

elif page == "Land-use statistics":
    st.title("Area-wise land-use statistics")
    if area is None:
        st.warning("area_stats.csv missing - run the analyze stage.")
        st.stop()
    year = st.selectbox("Year", YEARS, index=len(YEARS) - 1)
    d = area[area.year == year]
    c1, c2 = st.columns(2)
    c1.dataframe(d[["class", "area_km2", "percent", "pixels"]].round(2), hide_index=True, width="stretch")
    fig = px.pie(d, names="class", values="area_km2", color="class", hole=0.4,
                 color_discrete_map={CLASS_NAMES[k]: v for k, v in CLASS_COLORS.items()})
    c2.plotly_chart(fig, width="stretch")
    fig = px.bar(area, x="class", y="area_km2", color=area.year.astype(str), barmode="group",
                 labels={"area_km2": "Area (km²)", "color": "Year"}, title="Area by class and year")
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download statistics (CSV)", area.to_csv(index=False), "area_stats.csv")

elif page == "Model evaluation":
    st.title("Model evaluation")
    if not metrics:
        st.warning("metrics.json missing - run the train stage.")
        st.stop()
    st.info(f"Model: Random Forest ({metrics['rf_params']['n_estimators']} trees), features: "
            f"{', '.join(metrics['features'])}. Trained on **{metrics['label_source']}**, year {metrics['train_year']}. "
            f"Split: {metrics['split']}. Train/test samples: {metrics['n_train']}/{metrics['n_test']}.")
    sets = {"Held-out blocks vs WorldCover (weak labels)": metrics["weak_label_test"]}
    if metrics.get("independent_validation"):
        sets = {"Independent validation points (team-digitised)": metrics["independent_validation"], **sets}
    else:
        st.warning("No independent validation points were supplied, so only agreement with the weak WorldCover "
                   "labels can be reported. This is NOT true accuracy. See README: 'Creating validation points'.")
    if metrics.get("classes_missing_from_train_or_test"):
        st.warning(f"Classes absent from the evaluation: {metrics['classes_missing_from_train_or_test']}")
    choice = st.selectbox("Evaluation set", list(sets))
    s = sets[choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall accuracy", fmt_pct(s["overall_accuracy"]))
    c2.metric("Kappa", f"{s['kappa']:.3f}")
    c3.metric("Macro F1", f"{s['macro_f1']:.3f}")
    c4.metric("Samples", s["n"])
    cm = np.array(s["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=s["labels"], y=s["labels"], colorscale="Blues", text=cm,
                               texttemplate="%{text}"))
    fig.update_layout(title="Confusion matrix (rows = reference, columns = predicted)", yaxis_autorange="reversed",
                      height=480)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(s["per_class"]).T.round(3), width="stretch")
    fi = pd.Series(metrics["feature_importance"]).sort_values()
    st.plotly_chart(px.bar(fi, orientation="h", labels={"value": "Importance", "index": "Feature"},
                           title="Feature importance"), width="stretch")
    st.download_button("Download metrics (JSON)", json.dumps(metrics, indent=2), "metrics.json")

elif page == "Change detection":
    st.title(f"Change detection {YA} → {YB}")
    if trans is None or area is None:
        st.warning("Run the analyze stage first.")
        st.stop()
    a, b = trans.loc[:, "Built-up"].drop("Built-up").sum(), trans.loc["Built-up"].drop("Built-up").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("New built-up (from other classes)", f"{a:.2f} km²")
    c2.metric("Built-up converted to other classes", f"{b:.2f} km²")
    c3.metric("Net change", f"{a - b:+.2f} km²")
    st.caption("Change is derived by comparing two classified maps, so classification errors create some false "
               "change. Treat small values with caution.")
    fig = go.Figure(go.Heatmap(z=trans.values, x=trans.columns, y=trans.index, colorscale="Reds",
                               text=np.round(trans.values, 2), texttemplate="%{text}"))
    fig.update_layout(title=f"Transition matrix (km²): rows = {YA}, columns = {YB}", yaxis_autorange="reversed",
                      height=480)
    st.plotly_chart(fig, width="stretch")
    if grid is not None:
        st.subheader(f"Grid cells with the largest new built-up area ({CFG['growth']['cell_m']} m cells)")
        st.dataframe(grid.drop(columns="geometry").sort_values("gain_km2", ascending=False).head(10).round(3),
                     hide_index=True, width="stretch")
    st.caption("See the map page for the spatial change layer.")

elif page == "Growth scenarios":
    st.title("Urban growth susceptibility & what-if scenarios")
    gp = P.out / "growth_prob.tif"
    if growth_m is None or not gp.exists():
        st.warning("Growth model not available (" + str(summary.get("growth_error", "run the analyze stage")) + ")")
        st.stop()
    st.info("A Random Forest learned which non-built pixels turned into built-up between "
            f"{growth_m['period'][0]} and {growth_m['period'][1]} using spatial drivers. Scenarios below convert the most "
            "susceptible allowed pixels first. This is a planning what-if tool, **not a guaranteed forecast**.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test ROC-AUC", f"{growth_m['roc_auc']:.3f}")
    c2.metric("Distance-only baseline AUC", f"{growth_m['baseline_auc_distance_only']:.3f}")
    c3.metric("Observed growth", f"{growth_m['observed_gain_km2_per_year']:.2f} km²/yr")

    prob, meta = raster(gp)
    prob = prob[0]
    lulc = raster(P.lulc(YB))[0][0]
    res = abs(meta["transform"].a)
    pa = pixel_area_km2(meta["transform"])
    with st.sidebar:
        st.subheader("Scenario controls")
        horizon = st.slider("Years ahead", 1, 20, 5)
        mult = st.slider("Growth rate vs. past period", 0.25, 3.0, 1.0, 0.25)
        buf = st.slider("Protect buffer around water (m)", 0, 1000, 0, 50)
        prot_veg = st.checkbox("Protect vegetation", value=True)
        prot_crop = st.checkbox("Protect cropland", value=False)
    target_km2 = growth_m["observed_gain_km2_per_year"] * horizon * mult


    @st.cache_resource(show_spinner=False)
    def water_dist(path, mtime):
        return distance_transform_edt(raster(Path(path))[0][0] != 4) * res


    protected = np.zeros(lulc.shape, bool)
    if buf > 0:
        protected |= water_dist(str(P.lulc(YB)), P.lulc(YB).stat().st_mtime) <= buf
    if prot_veg:
        protected |= lulc == 2
    if prot_crop:
        protected |= lulc == 3
    cand = (lulc > 0) & (lulc != 1) & (lulc != 4)
    new = allocate_growth(prob, cand, int(round(target_km2 / pa)), protected)
    got = new.sum() * pa
    base_built = (lulc == 1).sum() * pa
    k1, k2, k3 = st.columns(3)
    k1.metric("Requested new built-up", f"{target_km2:.2f} km²")
    k2.metric("Allocated", f"{got:.2f} km²", None if got >= target_km2 - pa else "limited by protected zones",
              delta_color="off")
    k3.metric("Projected built-up", f"{base_built + got:.1f} km²", f"{100 * got / max(base_built, 1e-9):+.1f}%")
    consumed = pd.DataFrame({"class": [CLASS_NAMES[c] for c in (2, 3, 5)],
                             "km2": [((lulc == c) & new).sum() * pa for c in (2, 3, 5)]})
    st.plotly_chart(px.bar(consumed, x="class", y="km2", title="Land converted by the scenario (km²)",
                           labels={"km2": "km²"}), width="stretch")

    layer = np.zeros(lulc.shape, np.uint8)
    layer[lulc == 1] = 1
    layer[new] = 2
    layer[protected & cand & ~new] = 3
    img, b = to_wgs84(layer, meta, True)
    rgba = colorize_categorical(img, {1: "#7a7a7a", 2: "#ff7f00", 3: "#33a02c"})
    centre = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
    m = folium.Map(location=centre, zoom_start=12)
    folium.raster_layers.ImageOverlay(image=rgba, bounds=b, opacity=0.8, name="Scenario").add_to(m)
    m.fit_bounds(b)
    st_folium(m, height=550, use_container_width=True, returned_objects=[])
    legend_row([("Existing built-up", "#7a7a7a"), ("Scenario new built-up", "#ff7f00"), ("Protected land", "#33a02c")])

elif page == "Environment":
    st.title("Environmental indicators")
    nd = load_csv(P.out / "ndvi_by_class.csv")
    if nd is not None:
        st.plotly_chart(px.line(nd, x="year", y="mean_ndvi", color="class", markers=True,
                                title="Mean NDVI by class and year (same season each year)"), width="stretch")
        st.caption("NDVI saturates over dense vegetation and depends on crop calendar; compare like seasons only.")
    lst = load_csv(P.out / "lst_by_class.csv")
    if lst is not None:
        info = summary.get("lst", {})
        st.subheader("Land surface temperature (Landsat 8/9)")
        st.dataframe(lst.round(2), hide_index=True, width="stretch")
        if info.get("uhi_contrast_c") is not None:
            st.metric("Built-up minus vegetation/cropland LST", f"{info['uhi_contrast_c']:.1f} °C")
        st.caption("Thermal bands are natively ~100 m (resampled), so per-class values at small features are "
                   "approximate. LST is a surface skin temperature, not air temperature.")
    else:
        st.info("No LST layer available (optional Landsat step not run or failed).")

else:
    st.title("Planning insights & export")
    if area is None or trans is None or grid is None:
        st.warning("Run the analyze stage first.")
        st.stop()
    ins = make_insights(area, trans, grid.drop(columns="geometry"), (YA, YB))
    for s in ins:
        st.markdown(f"- {s}")
    note = (f"Weak-label test accuracy {fmt_pct(metrics['weak_label_test']['overall_accuracy'])}; independent validation: "
            + (fmt_pct(metrics['independent_validation']['overall_accuracy']) if metrics.get("independent_validation")
               else "not performed") + ".") if metrics else "No metrics."
    tables = {"Area by class and year": area.round(3), "Transition matrix (km2)": trans.reset_index().round(3)}
    st.download_button("Download HTML report", build_html_report(CFG["project"], ins, tables, note, SYN),
                       "urban_twin_report.html", mime="text/html")
    st.download_button("Download area statistics (CSV)", area.to_csv(index=False), "area_stats.csv")
    st.download_button("Download grid statistics (GeoJSON)", (P.out / "grid_stats.geojson").read_bytes(),
                       "grid_stats.geojson")
    st.download_button(f"Download land-use map {YB} (GeoTIFF)", P.lulc(YB).read_bytes(), f"lulc_{YB}.tif")
