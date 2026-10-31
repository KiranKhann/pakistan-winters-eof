"""One-call wrappers shared by the command-line scripts and the Streamlit app."""
from __future__ import annotations

import xarray as xr

from .data import REGIONS, SEASONS, open_field, select_region, select_season, default_years, season_year
from .eof import anomalies, area_weights, compute_eofs, EOFResult


def load_anomalies(var: str, region, season: str = "DJF", year_start: int | None = None,
                   year_end: int | None = None, n_years: int = 30, data_dir=None,
                   time_range: tuple[str, str] | None = None) -> xr.DataArray:
    """Open a dataset, cut the region and season, and remove the seasonal cycle.

    `region` is a key of REGIONS or a dict(lat=(a, b), lon=(c, d)).
    If the years are not given, the last `n_years` complete seasons are used.
    `time_range=("1948-01", "2000-12")` instead keeps every month of the season
    inside a calendar range, which is how the paper's 1948-2000 sample is defined.
    """
    box = REGIONS[region] if isinstance(region, str) else region
    months = SEASONS[season]
    da = open_field(var, data_dir) if data_dir else open_field(var)
    da = select_region(da, **box)
    if time_range is not None:
        da = da.sel(time=slice(*time_range))
        da = da.sel(time=da.time.dt.month.isin(list(months)))
        da = da.assign_coords(season_year=("time", season_year(da.time.values, months)))
        year_start, year_end = int(da.time.dt.year.min()), int(da.time.dt.year.max())
    else:
        if year_start is None or year_end is None:
            ys, ye = default_years(da, months, n_years)
            year_start = ys if year_start is None else year_start
            year_end = ye if year_end is None else year_end
        da = select_season(da, months, year_start, year_end)
    # The files store float32 (about 7 significant digits, i.e. ~1e-4 hPa on
    # 1000 hPa). Work in float64 so the anomalies are exactly centred.
    anom, _ = anomalies(da.load().astype("float64"))
    anom.attrs.update(region=str(region), season=season,
                      year_start=int(year_start), year_end=int(year_end))
    return anom


def run_eof_analysis(var: str = "slp", region="NH north of 20N (paper)", season: str = "DJF",
                     year_start: int | None = None, year_end: int | None = None, n_years: int = 30,
                     neofs: int = 20, weighting: str = "sqrtcos", n_eff: float | None = None,
                     data_dir=None, time_range: tuple[str, str] | None = None) -> EOFResult:
    """The whole of paper Sections 2-3 in one call.

    weighting="sqrtcos" is the default because it reproduces the paper's reported
    variances (21% and 13% for 1948-2000). The literal eq. (5), D = diag(cos phi),
    is available as weighting="cos" (it gives 18.8% and 14.9% on the same data).
    """
    anom = load_anomalies(var, region, season, year_start, year_end, n_years, data_dir, time_range)
    w = area_weights(anom, weighting)
    meta = dict(var=var, label=anom.attrs.get("label", var), units=anom.attrs.get("units", ""),
                source=anom.attrs.get("source", ""), region=str(region), season=season,
                year_start=anom.attrs["year_start"], year_end=anom.attrs["year_end"],
                weighting=weighting)
    return compute_eofs(anom, w, neofs=neofs, n_eff=n_eff, meta=meta)
