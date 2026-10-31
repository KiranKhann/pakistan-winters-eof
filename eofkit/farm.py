"""Rabi-season (wheat) climate statistics for farmers and agriculture officers.

All numbers are climatology from the last 30 complete seasons: how much rain a
season brings, how much can be counted on, how often seasons are dry, whether
that is changing, what happened after a dry or wet start, and how hot the
grain-filling month has been. Nothing here is a forecast.

Rainfall: GPCP v2.3 monthly (mm/day -> mm per month).
Temperature: NCEP/NCAR Reanalysis 1 monthly surface air temperature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import open_field
from .teleconnection import country_fraction, region_mean

MONTH_NAMES = {10: "October", 11: "November", 12: "December", 1: "January", 2: "February",
               3: "March", 4: "April"}
MONTH_SHORT = {k: v[:3] for k, v in MONTH_NAMES.items()}
RABI_MONTHS = [10, 11, 12, 1, 2, 3, 4]      # in season order

SO_FAR_CATEGORIES = [            # (label, low %, high %)
    ("Much drier than normal (under 60%)", -np.inf, 60.0),
    ("Drier than normal (60 to 85%)", 60.0, 85.0),
    ("About normal (85 to 115%)", 85.0, 115.0),
    ("Wetter than normal (115 to 140%)", 115.0, 140.0),
    ("Much wetter than normal (over 140%)", 140.0, np.inf),
]
DRY, WET = 75.0, 125.0           # season outcome thresholds, % of normal
HOT = 1.5                        # a "hot" month: mean temperature this far above normal (°C)


# --------------------------------------------------------------------------- #
# Series for a place
# --------------------------------------------------------------------------- #
def place_monthly(key: str, latlon) -> pd.Series:
    """Every month in the file for one place: rainfall in mm per month, or
    temperature in °C. latlon=None means the area average over Pakistan."""
    da = open_field(key)
    if latlon is None:
        da = da.sel(lat=slice(20, 40), lon=slice(58, 80)).load().astype("float64")
        s = region_mean(da, country_fraction(da.lat.values, da.lon.values, "Pakistan"))
    else:
        v = da.interp(lat=latlon[0], lon=latlon[1]).load().astype("float64")
        s = pd.Series(v.values, index=pd.DatetimeIndex(v.time.values))
    if key == "precip":
        s = (s * s.index.days_in_month).clip(lower=0)
    return s


def season_months(start: int, end: int) -> list[int]:
    i, j = RABI_MONTHS.index(start), RABI_MONTHS.index(end)
    return RABI_MONTHS[i:j + 1]


def harvest_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Season label = year of harvest: October-December belong to the next year."""
    return np.asarray(index.year + (index.month >= 10))


def season_table(p: pd.Series, months: list[int], n_seasons: int = 30) -> pd.DataFrame:
    """One row per season (indexed by harvest year), one column per month plus
    the total, for the last n complete seasons."""
    s = p[p.index.month.isin(months)]
    df = pd.DataFrame({"year": harvest_year(s.index), "month": s.index.month, "mm": s.values})
    tab = df.pivot_table(index="year", columns="month", values="mm", aggfunc="sum")
    tab = tab.reindex(columns=months).dropna()
    tab = tab.iloc[-n_seasons:]
    tab["total"] = tab[months].sum(axis=1)
    return tab


def season_label(year: int) -> str:
    return f"{year - 1}/{str(year)[2:]}"


# --------------------------------------------------------------------------- #
# Rainfall statistics
# --------------------------------------------------------------------------- #
def rain_summary(tab: pd.DataFrame) -> dict:
    tot = tab["total"]
    normal = float(tot.mean())
    pct = 100 * tot / normal
    years = tot.index.values.astype(float)
    sen = stats.theilslopes(tot.values, years)
    kt = stats.kendalltau(years, tot.values)
    return dict(
        n=len(tot), normal=normal, median=float(tot.median()),
        dependable_3in4=float(np.quantile(tot, 0.25)),     # exceeded in 3 of 4 seasons
        dependable_4in5=float(np.quantile(tot, 0.20)),     # exceeded in 4 of 5 seasons
        dry_n=int((pct < DRY).sum()), severe_n=int((pct < 50).sum()), wet_n=int((pct > WET).sum()),
        trend_pct_per_decade=float(10 * sen[0] / normal * 100), trend_p=float(kt.pvalue),
        driest=tot.idxmin(), wettest=tot.idxmax(), pct=pct,
    )


def monthly_stats(tab: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    rows = []
    for m in months:
        v = tab[m]
        rows.append(dict(month=MONTH_NAMES[m], normal=float(v.mean()),
                         dependable=float(np.quantile(v, 0.25)),
                         chance_almost_dry=float(np.mean(v < 10) * 100),
                         share_of_season=float(v.mean() / tab["total"].mean() * 100)))
    return pd.DataFrame(rows)


def trend_words(p: float, value: float, unit: str) -> str:
    if p < 0.05:
        return f"{'more' if value > 0 else 'less'} rain, a clear trend ({value:+.0f}{unit})"
    if p < 0.2:
        return f"possibly {'more' if value > 0 else 'less'} rain ({value:+.0f}{unit}), but it could be chance"
    return "no clear change"


# --------------------------------------------------------------------------- #
# Season so far: what happened next in past seasons
# --------------------------------------------------------------------------- #
def so_far_analysis(tab: pd.DataFrame, months: list[int], upto: int, category: int) -> dict:
    """Split each past season into 'so far' (start..upto) and 'rest'
    (after upto), and look at what the rest did after starts like the chosen one."""
    k = months.index(upto) + 1
    first, rest = months[:k], months[k:]
    a = tab[first].sum(axis=1)
    b = tab[rest].sum(axis=1)
    a_pct = 100 * a / a.mean()
    b_pct = 100 * b / b.mean()
    lab, lo, hi = SO_FAR_CATEGORIES[category]
    sel = (a_pct >= lo) & (a_pct < hi)
    out = b_pct[sel]
    rho, p = stats.spearmanr(a, b)
    half = len(a) // 2
    rho_old = stats.spearmanr(a.iloc[:half], b.iloc[:half])[0]
    rho_new = stats.spearmanr(a.iloc[half:], b.iloc[half:])[0]
    return dict(
        first=first, rest=rest, so_far_pct=a_pct, rest_pct=b_pct, selected=sel,
        n=int(sel.sum()), dry=int((out < DRY).sum()), normal=int(((out >= DRY) & (out <= WET)).sum()),
        wet=int((out > WET).sum()), base_dry=float(np.mean(b_pct < DRY) * 100),
        base_wet=float(np.mean(b_pct > WET) * 100), rho=float(rho), p=float(p),
        rho_old=float(rho_old), rho_new=float(rho_new),
        old_span=(int(a.index[0]), int(a.index[half - 1])), new_span=(int(a.index[half]), int(a.index[-1])),
        rest_normal_mm=float(b.mean()), first_normal_mm=float(a.mean()),
    )


def link_verdict(res: dict) -> str:
    if res["p"] < 0.05 and res["rho"] > 0:
        txt = "In past seasons, a dry start made a dry finish somewhat more likely, and a wet start a wet finish."
    elif res["p"] < 0.2 and res["rho"] > 0:
        txt = "In past seasons, a dry start made a dry finish only slightly more likely. The link is weak."
    else:
        txt = "In past seasons, the start said almost nothing about the rest. A dry start is not a reason to give up."
    if res["rho"] >= 0.25 and res["rho_new"] < 0.5 * res["rho_old"]:
        txt += (f" The link was stronger in {season_label(res['old_span'][0])} to {season_label(res['old_span'][1])} "
                f"than in recent seasons, so treat it with extra caution.")
    return txt


# --------------------------------------------------------------------------- #
# Heat in the grain-filling month
# --------------------------------------------------------------------------- #
def month_heat(t: pd.Series, month: int, n_years: int = 30) -> dict:
    s = t[t.index.month == month]
    s = s.iloc[-n_years:]
    an = s - s.mean()
    years = s.index.year.values.astype(float)
    lr = stats.linregress(years, s.values)
    an.index = s.index.year
    hot = an[an > HOT]
    return dict(anom=an, trend=float(10 * lr.slope), p=float(lr.pvalue), intercept=float(lr.intercept - s.mean()),
                slope=float(lr.slope), hot_years=list(hot.sort_values(ascending=False).index),
                hottest=list(an.sort_values(ascending=False).index[:3]), first=int(an.index[0]),
                last=int(an.index[-1]), n=len(an))


def heat_verdict(p: float) -> str:
    if p < 0.05:
        return "a clear trend, very unlikely to be chance"
    if p < 0.2:
        return "likely real, but 30 years are not enough to be sure"
    return "not clear, it could be chance"
