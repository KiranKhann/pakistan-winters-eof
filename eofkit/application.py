"""Application layer shared by scripts/application_pakistan.py and app.py.

Two questions, both answered with the paper's Section 2-3 machinery:

  1. Does the hemispheric leading mode (EOF1 of DJF SLP north of 20N, the
     Arctic Oscillation) shape Pakistan's winter temperature and rainfall?
  2. Because EOFs are domain dependent (paper Sec. 3.4), what are the leading
     SLP modes of a Pakistan-centred domain, and do *they* track Pakistan's
     winter weather?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .data import NORTH_PAKISTAN_BOX
from .eof import EOFResult
from .pipeline import load_anomalies, run_eof_analysis
from .teleconnection import (box_fraction, correlation_test, country_fraction, region_mean,
                             seasonal_means)

MAP_BOX = dict(lat=(0.0, 55.0), lon=(30.0, 110.0))     # data domain for the target maps
MAP_EXTENT = [40, 100, 8, 48]                          # map view: Middle East to India
LOCAL_DOMAIN = dict(lat=(10.0, 50.0), lon=(40.0, 100.0))  # Pakistan-centred SLP domain

TARGETS = {
    "Pakistan T": dict(field="air", units="°C", label="Pakistan temperature"),
    "Pakistan P": dict(field="precip", units="mm/day", label="Pakistan precipitation"),
    "N. Pakistan T": dict(field="air", units="°C", label="Northern Pakistan temperature (31-37N, 69-77E)"),
    "N. Pakistan P": dict(field="precip", units="mm/day", label="Northern Pakistan precipitation (31-37N, 69-77E)"),
}


def load_targets(year_start: int, year_end: int, season: str = "DJF", box=MAP_BOX) -> dict:
    """Temperature and precipitation anomalies for the map domain, and the
    Pakistan / northern-Pakistan area means as monthly series."""
    air = load_anomalies("air", box, season, year_start, year_end)
    pr = load_anomalies("precip", box, season, year_start, year_end)
    nb = NORTH_PAKISTAN_BOX
    series = {
        "Pakistan T": region_mean(air, country_fraction(air.lat.values, air.lon.values, "Pakistan")),
        "Pakistan P": region_mean(pr, country_fraction(pr.lat.values, pr.lon.values, "Pakistan")),
        "N. Pakistan T": region_mean(air, box_fraction(air.lat.values, air.lon.values, nb["lat"], nb["lon"])),
        "N. Pakistan P": region_mean(pr, box_fraction(pr.lat.values, pr.lon.values, nb["lat"], nb["lon"])),
    }
    return dict(air=air, pr=pr, series=series)


def detrend(s: pd.Series) -> pd.Series:
    """Remove a least-squares linear trend in time (fractional years)."""
    t = s.index.year + (s.index.month - 0.5) / 12.0
    v = s.values.astype(float)
    ok = np.isfinite(v)
    b, a = np.polyfit(t[ok], v[ok], 1)
    return pd.Series(v - (a + b * t), index=s.index, name=s.name)


def detrend_field(field: xr.DataArray) -> xr.DataArray:
    """Remove a linear trend in time at every grid point."""
    tt = pd.DatetimeIndex(field.time.values)
    t = np.asarray(tt.year + (tt.month - 0.5) / 12.0, dtype=float)
    t = t - t.mean()
    Y = np.asarray(field.transpose("time", "lat", "lon").values, dtype=float)
    slope = np.tensordot(t, Y - Y.mean(axis=0), axes=(0, 0)) / np.dot(t, t)
    out = field.transpose("time", "lat", "lon").copy(data=Y - t[:, None, None] * slope[None])
    return out


def index_vs_targets(index: pd.Series, season_year: np.ndarray, series: dict,
                     do_detrend: bool = False) -> pd.DataFrame:
    """Monthly, winter-mean and one-month-lead correlations of an index with
    each target series (Bretherton effective-sample-size t-tests)."""
    rows = []
    x = detrend(index) if do_detrend else index
    for name, s in series.items():
        y = s.reindex(index.index)
        y = detrend(y) if do_detrend else y
        m = correlation_test(x.values, y.values)
        w = correlation_test(seasonal_means(x, season_year).values, seasonal_means(y, season_year).values)
        lx, ly = [], []
        for i in range(len(x) - 1):
            if season_year[i] == season_year[i + 1]:
                lx.append(x.values[i])
                ly.append(y.values[i + 1])
        lead = correlation_test(np.array(lx), np.array(ly))
        rows.append(dict(target=name, units=TARGETS[name]["units"],
                         r_monthly=m["r"], p_monthly=m["p"], n_monthly=m["n"], slope_per_sd=m["slope"],
                         r_winter_mean=w["r"], p_winter_mean=w["p"], n_winters=w["n"],
                         r_lead_1month=lead["r"], p_lead_1month=lead["p"], n_lead=lead["n"]))
    return pd.DataFrame(rows)


def orient_by_region(res: EOFResult, k: int, frac: np.ndarray) -> None:
    """Fix the sign of EOF k so that a positive PC means HIGHER pressure over the
    region described by `frac` (cell coverage fractions on the EOF grid)."""
    reg = res.regression_map(k)
    w = frac * np.cos(np.deg2rad(res.lat))[:, None]
    if np.nansum(reg * w) < 0:
        res.flip_sign(k)


def local_modes(year_start: int, year_end: int, domain=LOCAL_DOMAIN, season: str = "DJF",
                weighting: str = "sqrtcos", neofs: int = 10) -> EOFResult:
    """EOFs of SLP over a Pakistan-centred domain. Each mode is oriented so that
    a positive PC means higher-than-normal pressure averaged over the domain
    (orienting by Pakistan alone is fragile: EOF1 is close to zero there)."""
    res = run_eof_analysis("slp", domain, season, year_start, year_end, neofs=neofs, weighting=weighting)
    whole = np.ones((res.lat.size, res.lon.size))
    for k in range(res.eigvals.size):
        orient_by_region(res, k, whole)
    return res


def modes_vs_targets(res: EOFResult, series: dict, kmax: int = 5, do_detrend: bool = False) -> pd.DataFrame:
    """Correlation of each of the first kmax PCs with each target (monthly)."""
    rows = []
    for k in range(min(kmax, res.eigvals.size)):
        pc = pd.Series(res.pcs_std[:, k], index=res.time)
        pc = detrend(pc) if do_detrend else pc
        row = dict(mode=f"PC{k + 1}", variance_pct=res.var_pct[k])
        for name, s in series.items():
            y = s.reindex(res.time)
            y = detrend(y) if do_detrend else y
            st = correlation_test(pc.values, y.values)
            row[f"r {name}"] = st["r"]
            row[f"p {name}"] = st["p"]
        rows.append(row)
    return pd.DataFrame(rows)
