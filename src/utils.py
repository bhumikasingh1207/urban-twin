"""Shared helpers: config, paths, class scheme, raster I/O and grid construction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]

# Simplified land-cover scheme used throughout the project.
CLASS_INFO = {
    1: ("Built-up", "#d7191c"),
    2: ("Vegetation", "#1a9641"),
    3: ("Cropland", "#e8d96b"),
    4: ("Water", "#2b83ba"),
    5: ("Bare / other", "#b8a58c"),
}
CLASS_NAMES = {k: v[0] for k, v in CLASS_INFO.items()}
CLASS_COLORS = {k: v[1] for k, v in CLASS_INFO.items()}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    """All file locations. `mode` keeps real and synthetic-test data fully separate."""

    base: Path
    mode: str  # "real" or "synthetic"

    @property
    def raw(self) -> Path:
        return self.base / "data" / self.mode / "raw"

    @property
    def out(self) -> Path:
        return self.base / "outputs" / self.mode

    @property
    def models(self) -> Path:
        return self.base / "models" / self.mode

    def ensure(self) -> "Paths":
        for d in (self.raw, self.out, self.models):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def s2(self, year: int) -> Path:
        return self.raw / f"s2_{year}.tif"

    def worldcover(self) -> Path:
        return self.raw / "worldcover.tif"

    def lst(self, year: int) -> Path:
        return self.raw / f"lst_{year}.tif"

    def validation_csv(self) -> Path:
        return self.raw / "validation_points.csv"

    def lulc(self, year: int) -> Path:
        return self.out / f"lulc_{year}.tif"


def get_paths(synthetic: bool = False, base: str | Path | None = None) -> Paths:
    return Paths(Path(base) if base else ROOT, "synthetic" if synthetic else "real").ensure()


# ----------------------------------------------------------------------------- raster I/O
def make_grid(bbox, crs: str, res: float):
    """Common analysis grid covering `bbox` (lon/lat) in `crs` at `res` metres.

    Returns (transform, width, height). Every dataset is resampled onto this grid so all
    layers line up pixel-for-pixel.
    """
    w, s, e, n = bbox
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in [(w, s), (w, n), (e, s), (e, n)]])
    xmin = np.floor(min(xs) / res) * res
    ymax = np.ceil(max(ys) / res) * res
    width = int(np.ceil((max(xs) - xmin) / res))
    height = int(np.ceil((ymax - min(ys)) / res))
    return from_origin(xmin, ymax, res, res), width, height


def read_raster(path: str | Path):
    """Return (array[bands, h, w], meta). meta has crs, transform, nodata, descriptions."""
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {"crs": src.crs, "transform": src.transform, "nodata": src.nodata,
                "descriptions": src.descriptions}
    return arr, meta


def write_raster(path, arr, crs, transform, nodata=None, descriptions=None, dtype=None) -> None:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    dtype = np.dtype(dtype or arr.dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype=str(dtype), crs=crs, transform=transform, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(arr.astype(dtype))
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)


def pixel_area_km2(transform) -> float:
    """Pixel area in km2. Only valid for projected (metre) CRS such as UTM."""
    return abs(transform.a * transform.e) / 1e6
