"""Crop advisor: which crops does the climate of a place suit, and how much water would each need?

Method (documented on the page):

1. Temperature fit (FAO ECOCROP). Each crop has four mean-temperature limits:
   absolute minimum, optimum minimum, optimum maximum, absolute maximum. A month
   scores 1 between the optimum limits, 0 outside the absolute limits, and a
   straight line in between (the ECOCROP "table-mountain" response, as coded in
   Hijmans' Recocrop package). A season scores the minimum over the months the
   crop is in the field, because the worst month limits the crop.
   Project variant: for rabi crops and sugarcane the full range is applied only
   to the sensitive months (establishment, and flowering to grain filling for
   rabi crops; tuber forming for potato; the main growth months for sugarcane). The other months are
   checked only against the crop's survival limits (absolute minimum and
   maximum, softened by 2 °C), because these crops grow slowly but safely
   through a cool January.

2. Water fit.
   * Rain only (barani): ECOCROP's rainfall limits describe the yearly rainfall
     where the crop is grown, so they are compared with the 12 months up to harvest.
     That window includes monsoon rain stored in the soil before rabi sowing.
   * With irrigation: FAO crop water need (Irrigation Water Management manual 3,
     Table 14) is compared with effective rain (FAO: Pe = 0.8P - 25 if P > 75 mm,
     else 0.6P - 10, never negative) plus the irrigation the farmer can give. The
     score is 1 when the need is met and falls linearly to 0 at half the need
     (this last rule is the project's own simple assumption).
     Paddy rice also needs water to saturate and flood the field: 200 mm + 100 mm
     + 6 mm per day of seepage while flooded (FAO manual 3, Sec. 4.4).

3. Overall fit for a season = the smaller of the temperature and water scores.
   Repeating this for each of the last 30 seasons shows how often the crop would
   have done well, and which month or factor usually held it back.

Climate data: GHCN_CAMS 2 m air temperature (0.5 degrees) and PREC/L land
precipitation (0.5 degrees, or the 1 degree version), monthly from NOAA, falling
back to the coarser 2.5 degree files when those are missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import DATA_DIR, data_file

ECOCROP_FILE = DATA_DIR / "ecocrop_pakistan_crops.csv"
TEMP_FILE = DATA_DIR / "ghcncams.air.mon.mean.nc"
RAIN_FILE = DATA_DIR / "precl.precip.mon.mean.0.5x0.5.nc"
TEMP_URL = "https://downloads.psl.noaa.gov/Datasets/ghcncams/air.mon.mean.nc"
RAIN_URL = "https://downloads.psl.noaa.gov/Datasets/precl/0.5deg/precip.mon.mean.0.5x0.5.nc"
RAIN_FILE_1DEG = DATA_DIR / "precl.precip.mon.mean.1x1.nc"
RAIN_URL_1DEG = "https://downloads.psl.noaa.gov/Datasets/precl/1.0deg/precip.mon.mean.1x1.nc"

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
              10: "Oct", 11: "Nov", 12: "Dec"}
MONTH_NAME = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
              8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

# --------------------------------------------------------------------------- #
# Crops: typical Pakistani calendar + FAO water need group
# --------------------------------------------------------------------------- #
# sow = first month in the field, months = number of months in the field.
# water = FAO Table 14 range (mm per growing period) and the group it was taken from.
CROPS = {
    # ---- rabi (winter) ----
    "wheat":     dict(name="Wheat", local="gandum", season="Rabi", sow=11, months=5, critical=[0, 3, 4], water=(450, 650, "barley/oats/wheat")),
    "barley":    dict(name="Barley", local="jau", season="Rabi", sow=11, months=5, critical=[0, 3, 4], water=(450, 650, "barley/oats/wheat")),
    "chickpea":  dict(name="Chickpea", local="chana", season="Rabi", sow=10, months=6, critical=[0, 4, 5], water=(300, 500, "beans")),
    "lentil":    dict(name="Lentil", local="masoor", season="Rabi", sow=11, months=5, critical=[0, 3, 4], water=(300, 500, "beans")),
    "mustard":   dict(name="Mustard (sarson, raya)", local="sarson", season="Rabi", sow=10, months=5, critical=[0, 3, 4],
                      water=(450, 650, "barley/oats/wheat, used as a stand-in")),
    "canola":    dict(name="Canola", local="canola", season="Rabi", sow=10, months=6, critical=[0, 4, 5],
                      water=(450, 650, "barley/oats/wheat, used as a stand-in")),
    "potato":    dict(name="Potato (autumn crop)", local="aloo", season="Rabi", sow=10, months=4, critical=[1, 2, 3],
                      water=(500, 700, "potato")),
    # ---- kharif (summer) ----
    "cotton":    dict(name="Cotton", local="kapas", season="Kharif", sow=5, months=6, water=(700, 1300, "cotton")),
    "rice":      dict(name="Rice (paddy)", local="chawal", season="Kharif", sow=7, months=4, water=(450, 700, "rice (paddy)"),
                      flooded_months=4, note="Scored from transplanting in July; the June nursery is not scored."),
    "maize":     dict(name="Maize", local="makai", season="Kharif", sow=7, months=4, water=(500, 800, "maize")),
    "sorghum":   dict(name="Sorghum", local="jowar", season="Kharif", sow=7, months=4, water=(450, 650, "sorghum/millet")),
    "millet":    dict(name="Pearl millet", local="bajra", season="Kharif", sow=7, months=4, water=(450, 650, "sorghum/millet")),
    "mung":      dict(name="Mung bean", local="moong", season="Kharif", sow=7, months=3, water=(300, 500, "beans")),
    "mash":      dict(name="Black gram", local="maash", season="Kharif", sow=7, months=4, water=(300, 500, "beans")),
    "groundnut": dict(name="Groundnut", local="moongphali", season="Kharif", sow=5, months=5, water=(500, 700, "groundnut")),
    "sesame":    dict(name="Sesame", local="til", season="Kharif", sow=7, months=4,
                      water=(450, 650, "sorghum/millet, used as a stand-in")),
    "guar":      dict(name="Guar", local="guar", season="Kharif", sow=7, months=4, water=(300, 500, "beans")),
    # ---- whole year ----
    "sugarcane": dict(name="Sugarcane", local="ganna", season="Year-round", sow=3, months=8, critical=[2, 3, 4, 5, 6],
                      water=(1500, 2500, "sugarcane"),
                      note="Planted in February or March and harvested from November to March. Scored over its growing "
                           "months, March to October; the mature cane waiting for harvest in winter is not scored."),
}

# Water the farmer can give on top of rain. One watering is taken as about 75 mm
# (3 inches), a common depth for one canal or tubewell irrigation in Pakistan.
WATERING_MM = 75
WATER_OPTIONS = {
    "rain": dict(label="None, rain only (barani)", short="Rain only", mm=0),
    "w3": dict(label="About 3 waterings", short="3 waterings", mm=3 * WATERING_MM),
    "w6": dict(label="About 6 waterings", short="6 waterings", mm=6 * WATERING_MM),
    "w10": dict(label="About 10 waterings", short="10 waterings", mm=10 * WATERING_MM),
    "full": dict(label="As much as the crop needs", short="Unlimited", mm=np.inf),
}

# Places: district towns in farming areas (approximate coordinates).
FARM_PLACES = {
    "Punjab": {
        "Attock": (33.77, 72.36), "Bahawalpur": (29.40, 71.68), "Bhakkar (Thal)": (31.63, 71.06),
        "Chakwal (Potohar)": (32.93, 72.86), "D.G. Khan": (30.05, 70.63), "Faisalabad": (31.42, 73.08),
        "Gujranwala": (32.16, 74.19), "Jhang": (31.27, 72.32), "Lahore": (31.55, 74.34),
        "Layyah (Thal)": (30.96, 70.94), "Mianwali": (32.58, 71.54), "Multan": (30.16, 71.52),
        "Okara": (30.81, 73.45), "Rahim Yar Khan": (28.42, 70.30), "Rawalpindi / Islamabad": (33.60, 73.05),
        "Sahiwal": (30.66, 73.11), "Sargodha": (32.08, 72.67), "Sialkot": (32.49, 74.52), "Vehari": (30.04, 72.35),
    },
    "Sindh": {
        "Badin": (24.66, 68.84), "Hyderabad": (25.40, 68.37), "Jacobabad": (28.28, 68.44),
        "Larkana": (27.56, 68.21), "Mirpur Khas": (25.53, 69.01), "Nawabshah": (26.24, 68.41),
        "Sanghar": (26.05, 68.95), "Sukkur": (27.70, 68.86), "Tharparkar (Mithi)": (24.74, 69.80),
        "Thatta": (24.75, 67.92),
    },
    "Khyber Pakhtunkhwa": {
        "Bannu": (32.99, 70.60), "D.I. Khan": (31.83, 70.90), "Kohat": (33.58, 71.44),
        "Mardan": (34.20, 72.05), "Peshawar": (34.01, 71.58),
    },
    "Balochistan": {
        "Lasbela": (25.81, 66.62), "Nasirabad": (28.55, 68.22), "Quetta": (30.18, 66.98),
    },
}
# Left out on purpose: Gilgit, Skardu, Muzaffarabad, Swat, Zhob, Loralai, Khuzdar
# and Turbat. Checked against their known climate normals, the 0.5 degree data
# there are 3 to 17 °C too cold, because each grid cell averages in the
# surrounding mountains. Advice there would be wrong.

GOOD = 0.6          # ECOCROP "suitable" or better
CLASS_EDGES = [(0.8, "Very good"), (0.6, "Good"), (0.4, "Fair"), (0.2, "Poor"), (0.0, "Very poor")]


def fit_class(x: float) -> str:
    for edge, label in CLASS_EDGES:
        if x >= edge:
            return label
    return "Very poor"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def ecocrop() -> pd.DataFrame:
    return pd.read_csv(ECOCROP_FILE, comment="#").set_index("key")


@lru_cache(maxsize=2)
def _open(path: str, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path, drop_variables=["time_bnds", "lat_bnds", "lon_bnds", "climatology_bounds"])
    da = ds[var]
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")
    return da


def _temp_path():
    return data_file(TEMP_FILE.name)


def _rain_path():
    return data_file(RAIN_FILE.name)


def fine_data_available() -> bool:
    return _temp_path().exists() and _rain_path().exists()


def terrain_spread(lat: float, lon: float) -> float:
    """Spread (°C) of long-term mean temperature across the 4 grid cells around a
    point. A large spread means hills or mountains nearby, where the grid values
    can be several degrees too cold for the fields in the valleys."""
    if not _temp_path().exists():
        return float("nan")
    t = _open(str(_temp_path()), "air")
    lon360 = lon % 360
    lats, lons = t.lat.values, t.lon.values
    i, j = int(np.searchsorted(lats, lat)), int(np.searchsorted(lons, lon360))
    box = t.isel(lat=slice(max(i - 1, 0), i + 1), lon=slice(max(j - 1, 0), j + 1)).sel(time=slice("1991", "2020"))
    m = box.mean("time").values
    return float(np.nanmax(m) - np.nanmin(m)) if np.isfinite(m).any() else float("nan")


def place_climate(lat: float, lon: float) -> pd.DataFrame:
    """Monthly temperature (°C) and rainfall (mm per month) for one point, all
    months both files cover. Uses the 0.5 degree files when present."""
    if _temp_path().exists():
        t, t_src = _open(str(_temp_path()), "air"), "GHCN_CAMS temperature (0.5°)"
    else:
        t, t_src = _open(str(data_file("air.mon.mean.nc")), "air"), "NCEP/NCAR temperature (2.5°)"
    if _rain_path().exists():
        p, p_src = _open(str(_rain_path()), "precip"), "PREC/L rainfall (0.5°)"
    elif RAIN_FILE_1DEG.exists():
        p, p_src = _open(str(RAIN_FILE_1DEG), "precip"), "PREC/L rainfall (1°)"
    else:
        p, p_src = _open(str(data_file("precip.mon.mean.nc")), "precip"), "GPCP rainfall (2.5°)"
    source = f"{t_src} and {p_src}"
    lon360 = lon % 360

    def point(da):
        step = float(abs(da.lat[1] - da.lat[0]))
        pad = 1.5 * step + 0.01
        sub = da.sel(lat=slice(lat - pad, lat + pad), lon=slice(lon360 - pad, lon360 + pad)).load()
        v = sub.interp(lat=lat, lon=lon360)
        if not np.isfinite(v.values).any():        # e.g. a coastal point that falls on sea in a land-only grid
            v = sub.sel(lat=lat, lon=lon360, method="nearest")
        return pd.Series(np.asarray(v.values, dtype=float), index=pd.DatetimeIndex(v.time.values))

    T = point(t)
    if np.nanmean(T.values) > 150:                    # Kelvin -> °C
        T = T - 273.15
    P = point(p)
    df = pd.concat({"t": T, "p_day": P}, axis=1, join="inner").dropna()
    df["p"] = (df["p_day"] * df.index.days_in_month).clip(lower=0)
    df.attrs["source"] = source
    return df[["t", "p"]]


# --------------------------------------------------------------------------- #
# Suitability
# --------------------------------------------------------------------------- #
def approx4(v, x):
    """ECOCROP table-mountain response (vectorised), as in Recocrop's approx4."""
    x = np.asarray(x, dtype=float)
    a, b, c, d = v
    out = np.zeros_like(x)
    out = np.where((x >= b) & (x <= c), 1.0, out)
    up = (x >= a) & (x < b)
    out = np.where(up, (x - a) / (b - a), out)
    dn = (x > c) & (x <= d)
    out = np.where(dn, (d - x) / (d - c), out)
    return out


def effective_rain(p):
    """FAO manual 3, Sec. 4.2."""
    p = np.asarray(p, dtype=float)
    pe = np.where(p > 75, 0.8 * p - 25, 0.6 * p - 10)
    return np.clip(pe, 0, None)


def crop_months(crop: dict) -> list[int]:
    return [((crop["sow"] - 1 + i) % 12) + 1 for i in range(crop["months"])]


def water_need(key: str, months_in_field: int | None = None) -> float:
    """Middle of the FAO range, plus field flooding for paddy rice."""
    c = CROPS[key]
    lo, hi, _ = c["water"]
    need = (lo + hi) / 2
    if c.get("flooded_months"):
        need += 200 + 100 + 6 * 30 * c["flooded_months"]
    return need


@dataclass
class CropResult:
    key: str
    seasons: pd.DataFrame        # one row per season (harvest year)
    summary: dict


def evaluate_crop(clim: pd.DataFrame, key: str, water: str, n_seasons: int = 30) -> CropResult:
    """Score one crop in each of the last n complete seasons at one place."""
    c = CROPS[key]
    par = ecocrop().loc[key]
    tlim = (par.TMIN, par.TOPMN, par.TOPMX, par.TMAX)
    rlim = (par.RMIN, par.ROPMN, par.ROPMX, par.RMAX)
    months = crop_months(c)
    k = len(months)
    need = water_need(key)
    irrig = WATER_OPTIONS[water]["mm"]

    # season windows: every start of the sowing month with all k months present
    idx = clim.index
    starts = idx[idx.month == c["sow"]]
    rows = []
    for s in starts:
        win = clim.loc[s: s + pd.DateOffset(months=k) - pd.Timedelta(days=1)]
        if len(win) != k:
            continue
        year12 = clim.loc[s + pd.DateOffset(months=k) - pd.DateOffset(months=12): s + pd.DateOffset(months=k) - pd.Timedelta(days=1)]
        if len(year12) != 12:
            continue
        crit = c.get("critical", list(range(k)))
        full = approx4(tlim, win["t"].values)
        survive = approx4((tlim[0] - 2, tlim[0] + 2, tlim[3] - 2, tlim[3] + 2), win["t"].values)
        ts = np.array([full[i] if i in crit else survive[i] for i in range(k)])
        t_suit = float(ts.min())
        worst_m = int(win.index.month[int(np.argmin(ts))])
        worst_t = float(win["t"].values[int(np.argmin(ts))])
        rain_season = float(win["p"].sum())
        pe = float(effective_rain(win["p"].values).sum())
        rain_12 = float(year12["p"].sum())
        if water == "rain":
            # Only a shortage counts: extra monsoon rain before rabi sowing does not
            # harm the crop, and waterlogging depends on the field, not the climate.
            w_suit = float(approx4((rlim[0], rlim[1], np.inf, np.inf), [rain_12])[0])
        elif np.isinf(irrig):
            w_suit = 1.0
        else:
            ratio = (pe + irrig) / need
            w_suit = float(np.clip((ratio - 0.5) / 0.5, 0, 1))
        overall = min(t_suit, w_suit)
        if overall >= 0.999:
            limit = "none"
        elif t_suit <= w_suit:
            limit = "cold" if worst_t < tlim[1] else "heat"
        else:
            limit = "water"
        harvest_year = int((s + pd.DateOffset(months=k - 1)).year)
        rows.append(dict(year=harvest_year, start=s, t_suit=t_suit, w_suit=w_suit, fit=overall, limit=limit,
                         worst_month=worst_m, worst_t=worst_t, rain_season=rain_season, eff_rain=pe,
                         rain_12=rain_12, irrigation_needed=max(0.0, need - pe)))
    S = pd.DataFrame(rows).set_index("year").iloc[-n_seasons:]
    if S.empty:
        return CropResult(key, S, dict(key=key, n=0))

    lim_counts = S.loc[S["fit"] < GOOD, "limit"].value_counts()
    main_limit = lim_counts.index[0] if len(lim_counts) else "none"
    lim_months = S.loc[(S["fit"] < GOOD) & S["limit"].isin(["cold", "heat"]), "worst_month"].value_counts()
    half = len(S) // 2
    summary = dict(
        key=key, n=len(S), good=int((S["fit"] >= GOOD).sum()), median_fit=float(S["fit"].median()),
        mean_fit=float(S["fit"].mean()), poor=int((S["fit"] < 0.4).sum()),
        main_limit=main_limit, limit_month=int(lim_months.index[0]) if len(lim_months) else None,
        need=need, eff_rain=float(S["eff_rain"].mean()), rain_season=float(S["rain_season"].mean()),
        irrigation_needed=float(S["irrigation_needed"].mean()),
        early_fit=float(S["fit"].iloc[:half].mean()), late_fit=float(S["fit"].iloc[half:].mean()),
        first=int(S.index[0]), last=int(S.index[-1]),
        t_median=float(S["t_suit"].median()), w_median=float(S["w_suit"].median()),
    )
    return CropResult(key, S, summary)


TIERS = [  # (share of good seasons at least, label, short)
    (0.8, "Suits your land well", "good in at least 8 seasons out of 10"),
    (0.5, "Possible, but some seasons go badly", "good in 5 to 8 seasons out of 10"),
    (0.2, "Risky", "good in 2 to 5 seasons out of 10"),
    (0.0, "Not suitable here", "good in fewer than 2 seasons out of 10"),
]


def tier_of(good: int, n: int) -> int:
    share = good / n if n else 0
    for i, (edge, _, _) in enumerate(TIERS):
        if share >= edge:
            return i
    return len(TIERS) - 1


def rank_crops(clim: pd.DataFrame, water: str, seasons=("Rabi", "Kharif", "Year-round")) -> pd.DataFrame:
    rows = []
    for key, c in CROPS.items():
        if c["season"] not in seasons:
            continue
        r = evaluate_crop(clim, key, water)
        if r.summary.get("n", 0) == 0:
            continue
        rows.append(dict(r.summary, name=c["name"], local=c["local"], season=c["season"],
                         months=crop_months(c)))
    df = pd.DataFrame(rows)
    # Group by how reliably the crop does well; inside a group, crops that need
    # less extra water come first (cheaper and safer), then the better fit.
    df["tier"] = [tier_of(g, n) for g, n in zip(df["good"], df["n"])]
    df["water_rank"] = df["irrigation_needed"].round(-1)
    df = df.sort_values(["tier", "water_rank", "median_fit"], ascending=[True, True, False])
    return df.drop(columns="water_rank").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plain-language helpers
# --------------------------------------------------------------------------- #
def months_text(months: list[int]) -> str:
    return f"{MONTH_ABBR[months[0]]} to {MONTH_ABBR[months[-1]]}"


def limit_text(row) -> str:
    lim, m = row["main_limit"], row["limit_month"]
    if lim == "none" or row["good"] == row["n"]:
        return "no regular problem"
    if lim == "cold":
        return f"too cold in {MONTH_NAME[m]}" if m else "too cold"
    if lim == "heat":
        return f"too hot in {MONTH_NAME[m]}" if m else "too hot"
    return "not enough water"


def trend_text(row) -> str:
    d = row["late_fit"] - row["early_fit"]
    if d >= 0.1:
        return "getting better"
    if d <= -0.1:
        return "getting worse"
    return "steady"


def waterings(mm: float) -> int:
    return int(np.ceil(mm / WATERING_MM)) if mm > 0 else 0
