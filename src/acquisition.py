"""Module 1a - Satellite data acquisition from the Microsoft Planetary Computer STAC API.

All data are open: Sentinel-2 L2A (Copernicus, free & open), ESA WorldCover (CC BY 4.0),
Landsat Collection 2 Level-2 (USGS, public domain). Only the pixels inside the study-area
window are read (cloud-optimised GeoTIFFs), so no full-scene downloads are needed.

Requires internet access. Every layer is resampled onto the common grid from config.yaml:
  * continuous data (reflectance, temperature) -> bilinear
  * categorical data (SCL mask, land cover)     -> nearest neighbour
"""
from __future__ import annotations

import calendar
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .utils import get_logger, make_grid, write_raster

log = get_logger("acquisition")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
SCL_VALID = [4, 5, 6, 7]  # vegetation, not-vegetated, water, unclassified (cloud/shadow/snow excluded)


def _catalog():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def _dates(year: int, start_month: int, end_month: int) -> str:
    last = calendar.monthrange(year, end_month)[1]
    return f"{year}-{start_month:02d}-01/{year}-{end_month:02d}-{last:02d}"


def _read_to_grid(href: str, grid, resampling) -> np.ndarray:
    crs, transform, width, height = grid
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                      GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                           resampling=resampling) as vrt:
                return vrt.read(1)


def _grid(cfg):
    transform, w, h = make_grid(cfg["aoi"]["bbox"], cfg["aoi"]["crs"], cfg["resolution_m"])
    return cfg["aoi"]["crs"], transform, w, h


def build_s2_composite(cfg: dict, out_path, year: int) -> None:
    """Cloud-masked median composite of Sentinel-2 L2A reflectance for one year/season."""
    grid = _grid(cfg)
    crs, transform, width, height = grid
    s = cfg["season"]
    search = _catalog().search(
        collections=["sentinel-2-l2a"], bbox=cfg["aoi"]["bbox"],
        datetime=_dates(year, s["start_month"], s["end_month"]),
        query={"eo:cloud_cover": {"lt": cfg["sentinel2"]["max_cloud_cover"]}},
    )
    items = sorted(search.items(), key=lambda i: i.properties.get("eo:cloud_cover", 100))
    items = items[: cfg["sentinel2"]["max_scenes"]]
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {year}. Widen season/cloud limits in config.yaml.")
    log.info("%s: %d Sentinel-2 scenes selected", year, len(items))

    usable = []
    for it in items:
        try:
            scl = _read_to_grid(it.assets["SCL"].href, grid, Resampling.nearest)
        except Exception as exc:  # network hiccup / corrupt asset: skip the scene, keep going
            log.warning("skipping %s (%s)", it.id, exc)
            continue
        ok = np.isin(scl, SCL_VALID)
        if ok.mean() > 0.02:
            usable.append((it, ok))
    if not usable:
        raise RuntimeError("All scenes failed or were fully cloudy.")

    bands, n_obs = [], None
    for band in S2_BANDS:
        stack = np.full((len(usable), height, width), np.nan, dtype=np.float32)
        for k, (it, ok) in enumerate(usable):
            asset = it.assets[band]
            rb = (asset.extra_fields.get("raster:bands") or [{}])[0]
            scale, offset = rb.get("scale", 1e-4), rb.get("offset", 0.0)
            dn = _read_to_grid(asset.href, grid, Resampling.bilinear).astype(np.float32)
            refl = np.clip(dn * scale + offset, 0, 1)
            refl[(dn == 0) | ~ok] = np.nan  # DN 0 = no data
            stack[k] = refl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN, handled later
            bands.append(np.nanmedian(stack, axis=0))
        if n_obs is None:
            n_obs = np.isfinite(stack).sum(axis=0).astype(np.float32)
        log.info("  band %s done", band)
    out = np.concatenate([np.stack(bands), n_obs[None]], axis=0).astype(np.float32)
    write_raster(out_path, out, crs, transform, nodata=np.nan, descriptions=S2_BANDS + ["n_valid_obs"])
    log.info("saved %s", out_path)


def fetch_worldcover(cfg: dict, out_path, year: int = 2021) -> None:
    """ESA WorldCover 10 m land cover (2020 v100 / 2021 v200) resampled to the analysis grid."""
    grid = _grid(cfg)
    crs, transform, width, height = grid
    search = _catalog().search(collections=["esa-worldcover"], bbox=cfg["aoi"]["bbox"],
                               datetime=f"{year}-01-01/{year}-12-31")
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No WorldCover tiles found for {year}.")
    out = np.zeros((height, width), dtype=np.uint8)
    for it in items:  # AOI may straddle several 3x3 degree tiles
        arr = _read_to_grid(it.assets["map"].href, grid, Resampling.nearest)
        out = np.where(out == 0, arr, out)
    write_raster(out_path, out, crs, transform, nodata=0, descriptions=[f"worldcover_{year}"])
    log.info("saved %s", out_path)


def build_lst_composite(cfg: dict, out_path) -> None:
    """Median land-surface-temperature (deg C) from Landsat 8/9 Collection 2 Level-2."""
    grid = _grid(cfg)
    crs, transform, width, height = grid
    c = cfg["lst"]
    search = _catalog().search(
        collections=["landsat-c2-l2"], bbox=cfg["aoi"]["bbox"],
        datetime=_dates(c["year"], c["start_month"], c["end_month"]),
        query={"eo:cloud_cover": {"lt": c["max_cloud_cover"]}, "platform": {"in": ["landsat-8", "landsat-9"]}},
    )
    items = list(search.items())[:12]
    if not items:
        raise RuntimeError("No Landsat 8/9 scenes found for the LST window.")
    stack = np.full((len(items), height, width), np.nan, dtype=np.float32)
    for k, it in enumerate(items):
        try:
            st = _read_to_grid(it.assets["lwir11"].href, grid, Resampling.bilinear).astype(np.float32)
            qa = _read_to_grid(it.assets["qa_pixel"].href, grid, Resampling.nearest)
        except Exception as exc:
            log.warning("skipping %s (%s)", it.id, exc)
            continue
        kelvin = st * 0.00341802 + 149.0  # Collection 2 Level-2 surface temperature scaling
        celsius = kelvin - 273.15
        bad = (st == 0) | ((qa & 0b11111) != 0) | (celsius < -20) | (celsius > 80)  # fill/cloud/shadow
        celsius[bad] = np.nan
        stack[k] = celsius
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lst = np.nanmedian(stack, axis=0).astype(np.float32)
    write_raster(out_path, lst, crs, transform, nodata=np.nan, descriptions=["LST_C"])
    log.info("saved %s", out_path)
