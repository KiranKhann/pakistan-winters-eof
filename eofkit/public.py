"""Plain-language layer for the public page of the app.

Everything here is built on the same EOF analysis as the rest of the project,
but answers questions a non-specialist asks:

  * How have winters changed where I live?           (trends of city series)
  * Why was a particular month so cold / dry?         (pattern contributions)
  * Which big weather patterns shape our winters?     (named local EOFs)

The attribution of a month to patterns uses one property from the paper
(Sec. 3.2.2): principal components are uncorrelated, so regressing a city's
anomaly on the patterns gives coefficients that can be computed one pattern
at a time, and the pattern contributions simply add up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .application import LOCAL_DOMAIN, local_modes
from .data import SEASONS, default_years, open_field, select_season
from .pipeline import run_eof_analysis
from .teleconnection import country_fraction, orient_ao, region_mean

# --------------------------------------------------------------------------- #
# Places and patterns
# --------------------------------------------------------------------------- #
CITIES = {
    "All of Pakistan": None,
    "Karachi": (24.86, 67.01),
    "Lahore": (31.55, 74.34),
    "Islamabad / Rawalpindi": (33.68, 73.05),
    "Faisalabad": (31.42, 73.08),
    "Chakwal (Potohar)": (32.93, 72.86),
    "D.I. Khan": (31.83, 70.90),
    "Bahawalpur": (29.40, 71.68),
    "Multan": (30.16, 71.52),
    "Peshawar": (34.01, 71.58),
    "Quetta": (30.18, 66.98),
    "Hyderabad": (25.40, 68.37),
    "Sukkur": (27.70, 68.86),
    "Gwadar": (25.13, 62.32),
    "Gilgit": (35.92, 74.31),
    "Muzaffarabad": (34.37, 73.47),
}

# Each named pattern is a local EOF of winter sea level pressure over
# 10-50N, 40-100E. `sign` turns the PC into an index whose positive value
# reads naturally ("more active storms" rather than "higher pressure").
PATTERNS = [
    dict(key="cold", k=0, sign=+1, name="Siberian cold push",
         what="A huge dome of cold, heavy air sits over Central Asia every winter. "
              "When it grows and spreads south, it pushes cold, dry air over the mountains into Pakistan.",
         strong="colder days and nights, frost, clear skies", weak="milder weather",
         up="stronger than usual", down="weaker than usual", adj_up="strong", adj_down="weak", var="T"),
    dict(key="storms", k=1, sign=-1, name="Western storms",
         what="Winter rain on the plains and snow in the mountains come mostly from 'western disturbances', "
              "storms that travel from the Mediterranean along the westerly winds. "
              "This pattern measures how open that storm path is.",
         strong="more rain and snow", weak="a dry spell",
         up="more active than usual", down="quieter than usual", adj_up="active", adj_down="quiet", var="P"),
    dict(key="caspian", k=2, sign=+1, name="Caspian high",
         what="High pressure over the Caspian Sea and Iran. When it is strong, cold north-westerly winds "
              "blow into Balochistan and Sindh.",
         strong="a colder south and west", weak="milder weather in the south and west",
         up="stronger than usual", down="weaker than usual", adj_up="strong", adj_down="weak", var="T"),
]

AO_TEXT = ("The Arctic Oscillation is the famous pattern behind cold snaps in Europe and North America. "
           "It is the leading pattern of the whole Northern Hemisphere, yet it hardly affects Pakistan: "
           "it warms Central Asia and cools the Middle East, and Pakistan sits on the line between the two.")

STRONG = 1.5   # an index of +1.5 is a "strong" month (roughly the top 7%)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class PatternSet:
    """The three named local patterns plus the AO, as monthly indices."""
    time: pd.DatetimeIndex
    season_year: np.ndarray
    index: pd.DataFrame             # columns: pattern keys + "ao" (standardised, unit variance)
    slp_anom: np.ndarray            # (n, lat, lon) local SLP anomalies in hPa
    lat: np.ndarray
    lon: np.ndarray
    maps: dict                      # key -> regression map (hPa per unit index)
    var_pct: dict                   # key -> % of local pressure variance
    year_start: int
    year_end: int


@dataclass
class CityClimate:
    name: str
    latlon: tuple | None
    time: pd.DatetimeIndex
    season_year: np.ndarray
    t_anom: pd.Series               # °C, monthly anomaly
    p_mm: pd.Series                 # mm per month
    p_normal: pd.Series             # mm per month, normal for that calendar month
    winters: pd.DataFrame           # per winter: t_anom, p_mm, p_normal, p_pct
    links: pd.DataFrame             # per pattern: r, p, effect of a strong month on T (°C) and P (mm)
    trend_t: dict = field(default_factory=dict)
    trend_p: dict = field(default_factory=dict)
    tc: np.ndarray | None = None    # centred time in years, for the trend term
    r2_t: float = 0.0               # share of monthly temperature variance the fit explains
    r2_p: float = 0.0


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def latest_window(n_years: int = 30) -> tuple[int, int]:
    return default_years(open_field("slp"), SEASONS["DJF"], n_years)


def build_patterns(year_start: int, year_end: int) -> PatternSet:
    loc = local_modes(year_start, year_end, LOCAL_DOMAIN)
    hemi = run_eof_analysis("slp", n_years=year_end - year_start + 1, year_start=year_start, year_end=year_end)
    orient_ao(hemi, 0, convention="noaa")
    idx, maps, var = {}, {}, {}
    for p in PATTERNS:
        idx[p["key"]] = p["sign"] * loc.pcs_std[:, p["k"]]
        maps[p["key"]] = p["sign"] * loc.regression_map(p["k"])
        var[p["key"]] = float(loc.var_pct[p["k"]])
    idx["ao"] = hemi.pcs_std[:, 0]
    return PatternSet(time=loc.time, season_year=loc.season_year,
                      index=pd.DataFrame(idx, index=loc.time), slp_anom=loc.anom,
                      lat=loc.lat, lon=loc.lon, maps=maps, var_pct=var,
                      year_start=year_start, year_end=year_end)


def _point_or_country(key: str, latlon, months, ys, ye) -> pd.Series:
    da = select_season(open_field(key), months, ys, ye)
    if latlon is None:
        da = da.sel(lat=slice(20, 40), lon=slice(58, 80)).load().astype("float64")
        frac = country_fraction(da.lat.values, da.lon.values, "Pakistan")
        return region_mean(da, frac)
    v = da.interp(lat=latlon[0], lon=latlon[1]).load().astype("float64")
    return pd.Series(v.values, index=pd.DatetimeIndex(v.time.values))


def _trend(years: np.ndarray, values: np.ndarray) -> dict:
    lr = stats.linregress(years, values)
    return dict(per_decade=10 * lr.slope, p=float(lr.pvalue), intercept=lr.intercept, slope=lr.slope)


def build_city(name: str, pats: PatternSet) -> CityClimate:
    months = SEASONS["DJF"]
    ys, ye = pats.year_start, pats.year_end
    latlon = CITIES[name]
    T = _point_or_country("air", latlon, months, ys, ye).reindex(pats.time)
    P = _point_or_country("precip", latlon, months, ys, ye).reindex(pats.time)   # mm/day
    days = pd.Series(pats.time.days_in_month, index=pats.time)
    p_mm = (P * days).clip(lower=0)
    month = pats.time.month
    t_anom = T - T.groupby(month).transform("mean")
    p_normal = p_mm.groupby(month).transform("mean")

    sy = pats.season_year
    w = pd.DataFrame({"t_anom": t_anom.values, "p_mm": p_mm.values, "p_normal": p_normal.values}, index=sy)
    winters = w.groupby(level=0).agg({"t_anom": "mean", "p_mm": "sum", "p_normal": "sum"})
    winters["p_pct"] = 100 * winters["p_mm"] / winters["p_normal"]
    winters.index.name = "winter"

    p_anom = p_mm - p_normal

    # Joint least-squares fit:  anomaly = a + c * (time) + sum_k b_k * pattern_k.
    # The patterns are mutually uncorrelated PCs, so each b_k is close to its
    # one-at-a-time value; the time term separates the long-term trend.
    keys = [p["key"] for p in PATTERNS]
    tc = np.asarray(pats.time.year + (pats.time.month - 0.5) / 12.0, dtype=float)
    tc = tc - tc.mean()
    X = np.column_stack([np.ones_like(tc), tc] + [pats.index[k].values for k in keys])
    coef_t, *_ = np.linalg.lstsq(X, t_anom.values, rcond=None)
    coef_p, *_ = np.linalg.lstsq(X, p_anom.values, rcond=None)

    rows = []
    for j, key in enumerate(keys + ["ao"]):
        x = pats.index[key].values
        rt, pt = stats.pearsonr(x, t_anom.values)
        rp, pp = stats.pearsonr(x, p_anom.values)
        bt = coef_t[2 + j] if key != "ao" else np.cov(x, t_anom.values)[0, 1] / np.var(x, ddof=1)
        bp = coef_p[2 + j] if key != "ao" else np.cov(x, p_anom.values)[0, 1] / np.var(x, ddof=1)
        rows.append(dict(pattern=key, r_t=rt, p_t=pt, b_t=bt, r_p=rp, p_p=pp, b_p=bp))
    rows.append(dict(pattern="trend", r_t=np.nan, p_t=np.nan, b_t=coef_t[1], r_p=np.nan, p_p=np.nan, b_p=coef_p[1]))
    links = pd.DataFrame(rows).set_index("pattern")
    fitted_t = X @ coef_t
    r2_t = 1 - np.sum((t_anom.values - fitted_t) ** 2) / np.sum((t_anom.values - t_anom.values.mean()) ** 2)
    fitted_p = X @ coef_p
    r2_p = 1 - np.sum((p_anom.values - fitted_p) ** 2) / np.sum((p_anom.values - p_anom.values.mean()) ** 2)

    return CityClimate(name=name, latlon=latlon, time=pats.time, season_year=sy, t_anom=t_anom,
                       p_mm=p_mm, p_normal=p_normal, winters=winters, links=links,
                       trend_t=_trend(winters.index.values.astype(float), winters["t_anom"].values),
                       trend_p=_trend(winters.index.values.astype(float), winters["p_pct"].values),
                       tc=tc, r2_t=float(r2_t), r2_p=float(r2_p))


# --------------------------------------------------------------------------- #
# Plain-language helpers
# --------------------------------------------------------------------------- #
def winter_label(year: int) -> str:
    return f"{year - 1}/{str(year)[2:]}"


def temp_words(x: float) -> str:
    if abs(x) < 0.3:
        return "close to normal"
    return f"{abs(x):.1f} °C {'warmer' if x > 0 else 'colder'} than normal"


def rain_words(pct: float) -> str:
    if 85 <= pct <= 115:
        return f"about normal ({pct:.0f}% of normal)"
    if pct >= 200:
        return f"much wetter than normal (about {pct / 100:.0f} times the normal amount)"
    return f"{'wetter' if pct > 100 else 'drier'} than normal ({pct:.0f}% of normal)"


def trend_verdict(p: float) -> str:
    if p < 0.05:
        return "a clear trend, very unlikely to be chance"
    if p < 0.2:
        return "likely real, but 30 winters are not enough to be sure"
    return "not clear, it could be chance"


def sentence_case(s: str) -> str:
    """Capitalise the first letter only (str.capitalize would turn '°C' into '°c')."""
    return s[:1].upper() + s[1:]


def strength_words(value: float, pattern: dict | None = None) -> str:
    up = pattern["up"] if pattern else "stronger than usual"
    down = pattern["down"] if pattern else "weaker than usual"
    a = abs(value)
    if a < 0.5:
        return "about normal"
    level = "slightly " if a < 1.0 else ("" if a < 2.0 else "much ")
    return level + (up if value > 0 else down)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def is_real_link(r: float, p: float) -> bool:
    """A pattern counts as affecting a place only if the link is statistically
    clear (p < 0.05) and explains at least 3% of the month-to-month swings."""
    return p < 0.05 and 100 * r * r >= 3


def link_sentence(place: str, r: float, p: float, what: str) -> str:
    if is_real_link(r, p):
        return f"explains about {100 * r * r:.0f}% of the month-to-month ups and downs in {place}'s winter {what}"
    return f"has little clear effect on {place}'s winter {what}"


def explain_month(city: CityClimate, pats: PatternSet, t: int) -> dict:
    """Numbers and sentences that explain month t for a city."""
    ts = city.time[t]
    month_name = ts.strftime("%B")
    t_an = float(city.t_anom.iloc[t])
    p_mm = float(city.p_mm.iloc[t])
    p_norm = float(city.p_normal.iloc[t])
    p_pct = 100 * p_mm / p_norm if p_norm > 0.5 else np.nan

    same = city.time.month == ts.month
    t_same = city.t_anom[same]
    n_same = int(same.sum())
    if t_an < 0:
        rank = int((t_same < t_an).sum()) + 1
        rank_txt = f"the {ordinal(rank)} coldest {month_name} of the last {n_same}" if rank <= 5 else ""
    else:
        rank = int((t_same > t_an).sum()) + 1
        rank_txt = f"the {ordinal(rank)} warmest {month_name} of the last {n_same}" if rank <= 5 else ""
    rank_txt = rank_txt.replace("the 1st ", "the ")

    contrib_t, contrib_p, strengths = {}, {}, {}
    for p in PATTERNS:
        x = float(pats.index[p["key"]].iloc[t])
        strengths[p["key"]] = x
        contrib_t[p["key"]] = float(city.links.loc[p["key"], "b_t"] * x)
        contrib_p[p["key"]] = float(city.links.loc[p["key"], "b_p"] * x)
    strengths["ao"] = float(pats.index["ao"].iloc[t])
    contrib_t["trend"] = float(city.links.loc["trend", "b_t"] * city.tc[t])
    contrib_p["trend"] = float(city.links.loc["trend", "b_p"] * city.tc[t])
    explained_t = sum(contrib_t.values())
    explained_p = sum(contrib_p.values())

    # the pattern that did most for temperature and for rain this month
    main_t = max(PATTERNS, key=lambda p: abs(contrib_t[p["key"]]))
    main_p = max(PATTERNS, key=lambda p: abs(contrib_p[p["key"]]))

    return dict(
        date=ts, month_label=ts.strftime("%B %Y"), t_anom=t_an, p_mm=p_mm, p_normal=p_norm, p_pct=p_pct,
        rank_text=rank_txt, contrib_t=contrib_t, contrib_p=contrib_p, strengths=strengths,
        explained_t=explained_t, other_t=t_an - explained_t,
        explained_p=explained_p, other_p=(p_mm - p_norm) - explained_p,
        main_t=main_t, main_p=main_p,
    )


def month_option_label(city: CityClimate, t: int) -> str:
    ts = city.time[t]
    p_norm = float(city.p_normal.iloc[t])
    pct = 100 * float(city.p_mm.iloc[t]) / p_norm if p_norm > 0.5 else np.nan
    rain = f"{pct:.0f}% of normal rain" if np.isfinite(pct) else "almost no rain is normal"
    x = float(city.t_anom.iloc[t])
    return f"{ts:%b %Y}  ·  {x:+.1f} °C, {rain}"
