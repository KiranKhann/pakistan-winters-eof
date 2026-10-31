"""Application layer: use the leading EOF of winter SLP (the Arctic Oscillation)
as a climate index and measure how it shapes winter weather over Pakistan.

Nothing here goes beyond Sections 2-3 of the paper mathematically: the AO index
is the standardised PC1 (eq. 10), and every statistic is a correlation,
a regression or a difference of means between high-amplitude months, which the
paper itself flags as the months where |PC| > 1 (Sec. 3.5, Fig. 3).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from .data import DATA_DIR, data_file
from .eof import EOFResult

CPC_AO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii"
CPC_AO_FILE = "monthly.ao.index.b50.current.ascii"


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
def orient_ao(res: EOFResult, k: int = 0, convention: str = "noaa", polar_lat: float = 70.0) -> EOFResult:
    """Fix the sign of EOF k (in place) from its polar-cap pressure anomaly.

    convention="noaa"   positive index = LOW pressure over the Arctic and high
                        pressure in mid-latitudes (Thompson and Wallace 1998; NOAA CPC)
    convention="paper"  positive pattern = HIGH over the North Pole, as drawn in
                        the paper's Fig. 2. Both are valid: EOFs are sign-free (Sec. 3.4).
    """
    reg = res.regression_map(k)
    polar = np.nanmean(reg[res.lat >= polar_lat, :])
    want_positive = convention == "paper"
    if (polar > 0) != want_positive:
        res.flip_sign(k)
    return res


def index_series(res: EOFResult, k: int = 0, name: str = "AO") -> pd.Series:
    """Standardised PC k as a monthly time series."""
    return pd.Series(res.pcs_std[:, k], index=res.time, name=name)


def load_cpc_ao(data_dir: Path | str | None = None) -> pd.Series | None:
    """The official NOAA CPC monthly AO index (computed from 1000 hPa heights),
    used only to validate our SLP-based PC1. Returns None if the file is absent."""
    path = data_file(CPC_AO_FILE) if data_dir is None else Path(data_dir) / CPC_AO_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["year", "month", "ao"], comment="#")
    df = df.dropna()
    df = df[(df["ao"] > -99) & (df["ao"] < 99)]
    idx = pd.to_datetime(dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1))
    return pd.Series(df["ao"].values, index=idx, name="CPC AO")


# --------------------------------------------------------------------------- #
# Region masks and box averages
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _country_geometry(name: str):
    import cartopy.io.shapereader as shpreader
    path = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    for rec in shpreader.Reader(path).records():
        a = rec.attributes
        if name in (a.get("NAME"), a.get("ADMIN"), a.get("NAME_LONG")):
            return rec.geometry
    raise KeyError(f"country {name!r} not found in Natural Earth")


def country_fraction(lat: np.ndarray, lon: np.ndarray, name: str = "Pakistan") -> np.ndarray:
    """Fraction (0..1) of each grid cell that lies inside the country polygon.
    Cells are assumed centred on (lat, lon) with the grid spacing as width."""
    from shapely.geometry import box
    from shapely.prepared import prep

    geom = _country_geometry(name)
    pg = prep(geom)
    dlat = float(np.median(np.abs(np.diff(lat)))) if lat.size > 1 else 2.5
    dlon = float(np.median(np.abs(np.diff(lon)))) if lon.size > 1 else 2.5
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    frac = np.zeros((lat.size, lon.size))
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon180):
            cell = box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2)
            if pg.intersects(cell):
                frac[i, j] = geom.intersection(cell).area / cell.area
    return frac


def box_fraction(lat: np.ndarray, lon: np.ndarray, lat_rng, lon_rng) -> np.ndarray:
    lon360 = lon % 360.0
    l0, l1 = lon_rng[0] % 360.0, lon_rng[1] % 360.0
    in_lon = (lon360 >= l0) & (lon360 <= l1) if l0 <= l1 else (lon360 >= l0) | (lon360 <= l1)
    in_lat = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    return (in_lat[:, None] & in_lon[None, :]).astype(float)


def region_mean(field: xr.DataArray, frac: np.ndarray) -> pd.Series:
    """Area-weighted mean over a region: weights = coverage fraction x cos(lat)."""
    w = frac * np.cos(np.deg2rad(field.lat.values))[:, None]
    v = np.asarray(field.transpose("time", "lat", "lon").values, dtype=float)
    ok = np.isfinite(v)
    num = np.nansum(v * w, axis=(1, 2))
    den = np.sum(np.where(ok, w, 0.0), axis=(1, 2))
    return pd.Series(num / den, index=pd.DatetimeIndex(field.time.values))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def lag1(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.moveaxis(np.asarray(x, dtype=float), axis, 0)
    x = x - np.nanmean(x, axis=0)
    num = np.nansum(x[:-1] * x[1:], axis=0)
    den = np.nansum(x * x, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / den, -0.99, 0.99)


def correlation_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson r between two 1-D series with a two-sided t-test whose degrees of
    freedom use the Bretherton et al. (1999) effective sample size
    n_eff = n (1 - r1x r1y) / (1 + r1x r1y)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    n = x.size
    r = float(np.corrcoef(x, y)[0, 1])
    a, b = float(lag1(x)), float(lag1(y))
    n_eff = float(np.clip(n * (1 - a * b) / (1 + a * b), 3, n))
    t = r * np.sqrt((n_eff - 2) / max(1e-12, 1 - r * r))
    p = float(2 * stats.t.sf(abs(t), n_eff - 2))
    slope, intercept = np.polyfit(x, y, 1)
    return dict(r=r, p=p, n=n, n_eff=n_eff, slope=float(slope), intercept=float(intercept))


def correlation_map(index: pd.Series, field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-point correlation of a field with an index, plus p-values
    (Bretherton effective sample size at every grid point)."""
    x = index.reindex(pd.DatetimeIndex(field.time.values)).values.astype(float)
    Y = np.asarray(field.transpose("time", "lat", "lon").values, dtype=float)
    ok = np.isfinite(x)
    x, Y = x[ok], Y[ok]
    n = x.size
    xa = (x - x.mean()) / x.std(ddof=1)
    Ya = Y - np.nanmean(Y, axis=0)
    sy = np.nanstd(Y, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nansum(xa[:, None, None] * Ya, axis=0) / ((n - 1) * sy)
        rx = float(lag1(x))
        ry = lag1(Y, axis=0)
        n_eff = np.clip(n * (1 - rx * ry) / (1 + rx * ry), 3, n)
        t = r * np.sqrt((n_eff - 2) / np.clip(1 - r * r, 1e-12, None))
        p = 2 * stats.t.sf(np.abs(t), n_eff - 2)
    return r, p


def composite_difference(index: pd.Series, field: xr.DataArray, threshold: float = 1.0):
    """Mean field in strong-positive months (index > +threshold) minus the mean
    in strong-negative months (index < -threshold), with a Welch t-test p-value.
    The +/-1 threshold is the paper's own definition of extreme events (Sec. 3.5)."""
    x = index.reindex(pd.DatetimeIndex(field.time.values)).values
    Y = np.asarray(field.transpose("time", "lat", "lon").values, dtype=float)
    pos, neg = Y[x > threshold], Y[x < -threshold]
    diff = np.nanmean(pos, axis=0) - np.nanmean(neg, axis=0)
    with np.errstate(invalid="ignore"):
        _, p = stats.ttest_ind(pos, neg, axis=0, equal_var=False, nan_policy="omit")
    return diff, np.asarray(p), int(pos.shape[0]), int(neg.shape[0])


def seasonal_means(series: pd.Series, season_year: np.ndarray) -> pd.Series:
    """Average monthly values into one value per season (e.g. per DJF winter)."""
    return pd.Series(series.values, index=season_year).groupby(level=0).mean()


def extreme_months(index: pd.Series, others: dict[str, pd.Series], threshold: float = 1.0) -> pd.DataFrame:
    """Table of months with |index| > threshold and the matching regional anomalies."""
    df = pd.DataFrame({index.name or "index": index})
    for k, s in others.items():
        df[k] = s.reindex(df.index)
    df = df[np.abs(df.iloc[:, 0]) > threshold].copy()
    df.insert(1, "phase", np.where(df.iloc[:, 0] > 0, "AO+", "AO-"))
    df.index = df.index.strftime("%b %Y")
    df.index.name = "month"
    return df
