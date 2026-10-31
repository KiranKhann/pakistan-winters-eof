"""Application: which large-scale pressure patterns shape winter weather in Pakistan?

Part A  The Arctic Oscillation (PC1 of DJF SLP north of 20N, NOAA sign:
        positive = low pressure over the Arctic) against Pakistan's winter
        temperature (NCEP/NCAR R1) and precipitation (GPCP v2.3).
Part B  The same EOF analysis on a Pakistan-centred domain (paper Sec. 3.4:
        EOFs are domain dependent), and how those local modes track Pakistan.

Uses the project window (last 30 complete DJF winters).

    python scripts/application_pakistan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eofkit import run_eof_analysis  # noqa: E402
from eofkit.application import (MAP_EXTENT, LOCAL_DOMAIN, load_targets, index_vs_targets,  # noqa: E402
                                local_modes, modes_vs_targets, detrend)
from eofkit.plots import (fig_correlation_maps, fig_composite_maps, fig_scatter, fig_winter_bars,  # noqa: E402
                          fig_eof_maps_region, CMAP_DIV, CMAP_RAIN, BLUE, RED, RAIN_COLORS)
from eofkit.teleconnection import (orient_ao, index_series, correlation_test, correlation_map,  # noqa: E402
                                   composite_difference, seasonal_means, extreme_months, _country_geometry)

FIG = ROOT / "figures"
RES = ROOT / "results"
COLS = ["target", "r_monthly", "p_monthly", "slope_per_sd", "r_winter_mean", "p_winter_mean",
        "r_lead_1month", "p_lead_1month"]


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    pak = _country_geometry("Pakistan")

    # ================= Part A: the Arctic Oscillation ==========================
    res = run_eof_analysis("slp", n_years=30)
    orient_ao(res, 0, convention="noaa")
    ao = index_series(res, 0, "AO")
    sy = res.season_year
    ys, ye = res.meta["year_start"], res.meta["year_end"]
    print(f"AO index: {len(ao)} DJF months, winters {ys - 1}/{ys} to {ye - 1}/{ye}")

    tg = load_targets(ys, ye)
    air, pr, series = tg["air"], tg["pr"], tg["series"]
    sT, sP = series["Pakistan T"].reindex(ao.index), series["Pakistan P"].reindex(ao.index)

    stats_raw = index_vs_targets(ao, sy, series)
    stats_dt = index_vs_targets(ao, sy, series, do_detrend=True)
    stats_raw.insert(0, "detrended", False)
    stats_dt.insert(0, "detrended", True)
    stats = pd.concat([stats_raw, stats_dt], ignore_index=True)
    stats.round(4).to_csv(RES / "ao_pakistan_stats.csv", index=False)
    print("\nPart A. AO vs Pakistan winter anomalies (raw)")
    print(stats_raw[COLS].round(3).to_string(index=False))
    print("Part A. same, linearly detrended")
    print(stats_dt[COLS].round(3).to_string(index=False))

    r_t, p_t = correlation_map(ao, air)
    r_p, p_p = correlation_map(ao, pr)
    save(fig_correlation_maps([("AO vs surface air temperature", r_t, p_t)], air.lat.values, air.lon.values,
                              MAP_EXTENT, outline=pak), "app1_corr_temperature.png")
    save(fig_correlation_maps([("AO vs precipitation", r_p, p_p)], pr.lat.values, pr.lon.values,
                              MAP_EXTENT, outline=pak), "app2_corr_precipitation.png")

    d_t, pd_t, npos, nneg = composite_difference(ao, air)
    d_p, pd_p, _, _ = composite_difference(ao, pr)
    save(fig_composite_maps([(f"Temperature: AO+ minus AO- ({npos} vs {nneg} months)", d_t, pd_t, "°C", CMAP_DIV)],
                            air.lat.values, air.lon.values, MAP_EXTENT, outline=pak), "app3_composite_temperature.png")
    save(fig_composite_maps([(f"Precipitation: AO+ minus AO- ({npos} vs {nneg} months)", d_p, pd_p, "mm/day", CMAP_RAIN)],
                            pr.lat.values, pr.lon.values, MAP_EXTENT, outline=pak), "app4_composite_precipitation.png")

    save(fig_scatter([
        (ao.values, sT.values, correlation_test(ao.values, sT.values), "AO index (s.d.)",
         "Temperature anomaly (°C)", "Pakistan temperature vs AO"),
        (ao.values, sP.values, correlation_test(ao.values, sP.values), "AO index (s.d.)",
         "Precipitation anomaly (mm/day)", "Pakistan precipitation vs AO"),
    ]), "app5_scatter_ao.png")

    save(fig_winter_bars([
        (seasonal_means(ao, sy), "AO index", (BLUE, RED)),
        (seasonal_means(sT, sy), "Pakistan T (°C)", (BLUE, RED)),
        (seasonal_means(sP, sy), "Pakistan P (mm/day)", (RAIN_COLORS[2], RAIN_COLORS[8])),
    ], title="Winter (DJF) means"), "app6_winter_bars.png")

    ext = extreme_months(ao, {"Pakistan T (°C)": sT, "Pakistan P (mm/day)": sP})
    ext.round(3).to_csv(RES / "ao_extreme_months.csv")
    print(f"\nExtreme AO months (|AO| > 1): {len(ext)}; mean anomalies by phase:")
    print(ext.groupby("phase")[["Pakistan T (°C)", "Pakistan P (mm/day)"]].mean().round(3).to_string())

    # ================= Part B: Pakistan-centred EOFs ============================
    loc = local_modes(ys, ye, LOCAL_DOMAIN)
    tab = modes_vs_targets(loc, series, kmax=5)
    tab_dt = modes_vs_targets(loc, series, kmax=5, do_detrend=True)
    tab.insert(0, "detrended", False)
    tab_dt.insert(0, "detrended", True)
    pd.concat([tab, tab_dt], ignore_index=True).round(4).to_csv(RES / "local_modes_vs_pakistan.csv", index=False)
    loc.table().round(4).to_csv(RES / "eigen_table_local_domain.csv", index=False)
    print(f"\nPart B. SLP EOFs of {LOCAL_DOMAIN} (positive PC = higher pressure averaged over the domain)")
    show = ["mode", "variance_pct", "r Pakistan T", "p Pakistan T", "r Pakistan P", "p Pakistan P"]
    print(tab[show].round(3).to_string(index=False))
    print("detrended:")
    print(tab_dt[show].round(3).to_string(index=False))

    ext_local = [LOCAL_DOMAIN["lon"][0], LOCAL_DOMAIN["lon"][1], LOCAL_DOMAIN["lat"][0], LOCAL_DOMAIN["lat"][1]]
    save(fig_eof_maps_region(loc, (0, 1), ext_local, outline=pak, title_prefix="Pakistan-centred "),
         "app7_local_eofs.png")

    pc1 = loc.pcs_std[:, 0]
    pc2 = loc.pcs_std[:, 1]
    save(fig_scatter([
        (pc1, sT.values, correlation_test(pc1, sT.values), "Local PC1 (s.d.)",
         "Temperature anomaly (°C)", "Pakistan temperature vs local PC1"),
        (pc2, sP.values, correlation_test(pc2, sP.values), "Local PC2 (s.d.)",
         "Precipitation anomaly (mm/day)", "Pakistan precipitation vs local PC2"),
    ]), "app8_scatter_local.png")

    # variance of Pakistan's winter weather explained (r^2), for the summary
    print("\nShare of monthly variance explained (r^2):")
    for lab, x, y in [("AO -> T", ao.values, sT.values), ("AO -> P", ao.values, sP.values),
                      ("local PC1 -> T", pc1, sT.values), ("local PC2 -> P", pc2, sP.values)]:
        r = correlation_test(x, y)["r"]
        print(f"  {lab:15s} r = {r:+.2f}   r^2 = {100 * r * r:4.1f}%")


if __name__ == "__main__":
    main()
