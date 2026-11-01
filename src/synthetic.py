"""SYNTHETIC test data - for checking that the code runs end-to-end WITHOUT internet.

!!! These are computer-generated rasters, not satellite data. Any metric computed on them says
!!! nothing about real-world performance and must NEVER appear in your report as a result.
They are written to data/synthetic/ and outputs/synthetic/, fully separate from real data, and the
dashboard shows a warning banner when synthetic data are displayed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from .features import BAND_NAMES
from .utils import write_raster

# class -> mean reflectance for B02, B03, B04, B08, B11, B12 (rough, illustrative values)
SIGNATURE = {
    1: [0.12, 0.14, 0.16, 0.20, 0.25, 0.22],
    2: [0.03, 0.06, 0.04, 0.40, 0.18, 0.08],
    3: [0.05, 0.09, 0.08, 0.32, 0.25, 0.15],
    4: [0.06, 0.07, 0.05, 0.03, 0.02, 0.01],
    5: [0.14, 0.18, 0.22, 0.27, 0.34, 0.30],
}
CLASS_TO_WC = {1: 50, 2: 10, 3: 40, 4: 80, 5: 60}
LST_OFFSET = {1: 6.0, 2: -3.0, 3: 0.0, 4: -5.0, 5: 3.0}


def _field(rng, size, sigma):
    f = gaussian_filter(rng.normal(size=(size, size)), sigma)
    return (f - f.mean()) / f.std()


def _truth(size, rng_seed, built_threshold):
    rng = np.random.default_rng(rng_seed)  # same fields every year -> consistent geography
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - size * 0.45, xx - size * 0.4) / (size * 0.7)
    urban = (1 - r) + 0.25 * _field(rng, size, 12)
    river = np.abs(xx - (size * 0.75 + 12 * np.sin(yy / 25))) < 4
    veg, bare = _field(rng, size, 10), _field(rng, size, 8)
    t = np.full((size, size), 3, dtype=np.uint8)  # default cropland
    t[veg > 0.5] = 2
    t[bare > 1.4] = 5
    t[urban > built_threshold] = 1
    t[river] = 4
    return t


def generate(paths, cfg: dict, size: int = 300, seed: int = 7) -> None:
    """Create synthetic S2 composites, WorldCover-like labels, LST, and validation points."""
    paths.ensure()
    crs = cfg["aoi"]["crs"]
    res = cfg["resolution_m"]
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(77.4, 28.55)
    transform = from_origin(round(cx / res) * res, round(cy / res) * res, res, res)
    thresholds = {y: t for y, t in zip(sorted(cfg["years"]), np.linspace(0.62, 0.40, len(cfg["years"])))}
    rng = np.random.default_rng(seed)
    truths = {}
    for year in cfg["years"]:
        truth = _truth(size, seed, thresholds[year])
        truths[year] = truth
        img = np.empty((len(BAND_NAMES), size, size), dtype=np.float32)
        noisy = truth.copy()
        flip = rng.random(truth.shape) < 0.05  # spectral confusion between classes
        noisy[flip] = rng.integers(1, 6, flip.sum())
        for c, sig in SIGNATURE.items():
            m = noisy == c
            for b, mu in enumerate(sig):
                img[b][m] = mu
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img, 0.001, 1)
        img[:, rng.random((size, size)) < 0.03] = np.nan  # simulated cloud holes
        n_obs = np.where(np.isfinite(img[0]), 5, 0).astype(np.float32)
        write_raster(paths.s2(year), np.concatenate([img, n_obs[None]]), crs, transform, nodata=np.nan,
                     descriptions=BAND_NAMES + ["n_valid_obs"])

    ty = cfg["train_year"]
    wc = truths[ty].copy()
    wrong = rng.random(wc.shape) < 0.08  # weak labels are imperfect
    wc[wrong] = rng.integers(1, 6, wrong.sum())
    wc_codes = np.zeros_like(wc)
    for c, code in CLASS_TO_WC.items():
        wc_codes[wc == c] = code
    write_raster(paths.worldcover(), wc_codes, crs, transform, nodata=0, descriptions=["worldcover_synthetic"])

    lt = truths[cfg["lst"]["year"]] if cfg["lst"]["year"] in truths else truths[max(truths)]
    lst = 30 + np.vectorize(LST_OFFSET.get)(lt) + gaussian_filter(rng.normal(size=lt.shape), 4) * 2
    write_raster(paths.lst(cfg["lst"]["year"]), lst.astype(np.float32), crs, transform, nodata=np.nan,
                 descriptions=["LST_C"])

    pts = []  # independent "reference" points drawn from the TRUE map (not from the weak labels)
    for c in range(1, 6):
        r, cc = np.nonzero(truths[ty] == c)
        if r.size == 0:
            continue
        sel = rng.choice(r.size, size=min(80, r.size), replace=False)
        xs = transform.c + (cc[sel] + 0.5) * res
        ys = transform.f - (r[sel] + 0.5) * res
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
        pts.append(pd.DataFrame({"lon": lon, "lat": lat, "class_id": c}))
    pd.concat(pts).to_csv(paths.validation_csv(), index=False)
