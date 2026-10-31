"""Loading and subsetting of gridded monthly climate data (paper Section 2.1).

All datasets are NOAA Physical Sciences Laboratory (PSL) monthly-mean NetCDF files.
They share the layout  variable(time, lat, lon)  with longitude in 0..360 and
latitude stored from 90 down to -90; we sort latitude ascending on load so that
every later step can assume a predictable orientation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
COMPACT_DIR = DATA_DIR / "compact"     # small regional copies committed to the repository


def data_file(name: str) -> Path:
    """The full download in data/ if present, else the compact copy in
    data/compact/ (see scripts/make_compact_data.py). Returns the data/ path
    when neither exists, so error messages point to the download location."""
    full = DATA_DIR / name
    if full.exists():
        return full
    compact = COMPACT_DIR / name
    return compact if compact.exists() else full

DATASETS = {
    "slp": dict(
        file="slp.mon.mean.nc", var="slp", label="Sea level pressure", units="hPa",
        source="NCEP/NCAR Reanalysis 1 (monthly means, 2.5°)",
        url="https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/slp.mon.mean.nc",
    ),
    "air": dict(
        file="air.mon.mean.nc", var="air", label="Surface air temperature", units="°C",
        source="NCEP/NCAR Reanalysis 1 (monthly means, 2.5°, sigma 0.995)",
        url="https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc",
    ),
    "precip": dict(
        file="precip.mon.mean.nc", var="precip", label="Precipitation", units="mm/day",
        source="GPCP v2.3 (monthly means, 2.5°)",
        url="https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
    ),
}

# Season name -> calendar months. DJF wraps around the new year: December is
# counted with the following January and February (see season_year()).
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
    "Annual": tuple(range(1, 13)),
}

REGIONS = {
    "NH north of 20N (paper)": dict(lat=(20.0, 90.0), lon=(0.0, 360.0)),
    "Northern Hemisphere": dict(lat=(0.0, 90.0), lon=(0.0, 360.0)),
    "Global": dict(lat=(-90.0, 90.0), lon=(0.0, 360.0)),
    "North Atlantic / Europe": dict(lat=(20.0, 80.0), lon=(-90.0, 40.0)),
    "Pakistan-centred (10-50N, 40-100E)": dict(lat=(10.0, 50.0), lon=(40.0, 100.0)),
    "Tropical Pacific": dict(lat=(-30.0, 30.0), lon=(120.0, 290.0)),
}

# Boxes used by the application layer (approximate land extent of Pakistan).
PAKISTAN_BOX = dict(lat=(24.0, 37.0), lon=(61.0, 77.0))
NORTH_PAKISTAN_BOX = dict(lat=(31.0, 37.0), lon=(69.0, 77.0))


def open_field(key: str, data_dir: Path | str | None = None) -> xr.DataArray:
    """Open one dataset as a DataArray with dims (time, lat, lon), lat ascending.

    Units are normalised to those listed in DATASETS (hPa, °C, mm/day).
    Without data_dir, the full file in data/ is used if present, otherwise the
    compact regional copy in data/compact/.
    """
    meta = DATASETS[key]
    path = data_file(meta["file"]) if data_dir is None else Path(data_dir) / meta["file"]
    data_dir = path.parent
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run  python scripts/download_data.py  first "
            f"(or download {meta['url']} into {data_dir})."
        )
    # The bounds variables are not needed, and GPCP's time_bnds contains fill
    # values that cannot be decoded as dates, so drop them on read.
    ds = xr.open_dataset(path, drop_variables=["time_bnds", "lat_bnds", "lon_bnds", "climatology_bounds"])
    da = ds[meta["var"]]
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    da = da.squeeze(drop=True)

    units = str(da.attrs.get("units", "")).lower()
    if key == "air" and (units.startswith("k") or float(da.isel(time=0).mean()) > 150.0):
        da = da - 273.15
    if key == "slp" and units in ("pa", "pascals"):
        da = da / 100.0

    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")

    da.attrs.update(key=key, label=meta["label"], units=meta["units"], source=meta["source"])
    return da


def select_region(da: xr.DataArray, lat=(-90.0, 90.0), lon=(0.0, 360.0)) -> xr.DataArray:
    """Subset to a lat/lon box. Longitudes may be given in -180..180 or 0..360
    and may wrap across the date line (e.g. lon=(-90, 40))."""
    la0, la1 = lat
    da = da.sel(lat=da.lat[(da.lat >= la0) & (da.lat <= la1)])
    lo0, lo1 = lon
    if lo1 - lo0 >= 360.0:
        return da
    lonv = da.lon.values % 360.0
    lo0m, lo1m = lo0 % 360.0, lo1 % 360.0
    if lo0m <= lo1m:
        mask = (lonv >= lo0m) & (lonv <= lo1m)
    else:
        mask = (lonv >= lo0m) | (lonv <= lo1m)
    return da.isel(lon=np.where(mask)[0])


def season_year(time, months) -> np.ndarray:
    """Season label for each time step. For a wrap-around season such as DJF,
    December belongs to the *following* year's season (Dec 1996 -> winter 1997)."""
    t = pd.DatetimeIndex(time)
    yr = t.year.values.copy()
    if 12 in months and 1 in months:
        yr[t.month.values == 12] += 1
    return yr


def _season_complete(da: xr.DataArray, year: int, months) -> bool:
    t = pd.DatetimeIndex(da.time.values)
    have = set(zip(t.year.values.tolist(), t.month.values.tolist()))
    wrap = 12 in months and 1 in months
    for m in months:
        y = year - 1 if (wrap and m == 12) else year
        if (y, m) not in have:
            return False
    return True


def latest_complete_season_year(da: xr.DataArray, months) -> int:
    t_last = pd.Timestamp(da.time.values[-1])
    y = t_last.year + (1 if (12 in months and 1 in months and t_last.month == 12) else 0)
    while y > 1800 and not _season_complete(da, y, months):
        y -= 1
    return y


def default_years(da: xr.DataArray, months, n_years: int = 30) -> tuple[int, int]:
    """The professor's rule: the last `n_years` complete seasons in the file."""
    y_end = latest_complete_season_year(da, months)
    return y_end - n_years + 1, y_end


def season_time_range(year_start: int, year_end: int, months):
    wrap = 12 in months and 1 in months
    if wrap:
        start = pd.Timestamp(year=year_start - 1, month=12, day=1)
        last_m = max(m for m in months if m < 12)
        end = pd.Timestamp(year=year_end, month=last_m, day=1) + pd.offsets.MonthEnd(0)
    else:
        start = pd.Timestamp(year=year_start, month=min(months), day=1)
        end = pd.Timestamp(year=year_end, month=max(months), day=1) + pd.offsets.MonthEnd(0)
    return start, end


def select_season(da: xr.DataArray, months, year_start: int, year_end: int) -> xr.DataArray:
    """Keep only the months of one season between two season-years (inclusive)
    and attach a `season_year` coordinate."""
    start, end = season_time_range(year_start, year_end, months)
    da = da.sel(time=slice(start, end))
    da = da.sel(time=da.time.dt.month.isin(list(months)))
    return da.assign_coords(season_year=("time", season_year(da.time.values, months)))
