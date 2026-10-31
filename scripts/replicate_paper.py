"""Replicate Hannachi (2004), Sections 2-3: EOFs of winter (DJF) monthly sea level
pressure over the Northern Hemisphere north of 20N.

Two runs:
  A. "paper period"   every DJF month in 1948-2000, as in the paper (validation)
  B. "project window" the last 30 complete winters in the file (professor's rule)

Outputs go to ./figures and ./results.

    python scripts/replicate_paper.py
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
from eofkit.plots import fig_spectrum, fig_eof_maps, fig_pcs  # noqa: E402
from eofkit.teleconnection import orient_ao, load_cpc_ao  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
REGION = "NH north of 20N (paper)"
PAPER_VAR = {1: 21.0, 2: 13.0}   # explained variance quoted in the paper, Sec. 3.5


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def describe(res, label: str) -> None:
    t = res.table()
    print(f"\n{label}: {res.meta['season']} {res.time[0]:%b %Y} to {res.time[-1]:%b %Y}, "
          f"n = {res.n} months, p = {res.p} grid points ({res.p_used} used)")
    for k in range(5):
        r = t.iloc[k]
        print(f"  EOF{k + 1}: {r['variance_%']:5.1f}% +/- {r['north_error_%']:.1f}  "
              f"(cumulative {r['cumulative_%']:5.1f}%)  separable={bool(r['separable'])}")


def validate_against_cpc(res, label: str) -> float | None:
    cpc = load_cpc_ao()
    if cpc is None:
        print("  (CPC AO index not downloaded; skipping validation)")
        return None
    ours = pd.Series(res.pcs_std[:, 0], index=res.time)
    both = pd.concat([ours, cpc], axis=1, join="inner").dropna()
    r = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
    print(f"  {label}: corr(our PC1, NOAA CPC AO index) = {r:+.3f} over {len(both)} months")
    return r


def main() -> None:
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    # ---------------- Run A: the paper's own sample --------------------------
    paper = run_eof_analysis("slp", REGION, "DJF", time_range=("1948-01", "2000-12"))
    orient_ao(paper, 0, convention="paper")
    describe(paper, "Run A (paper period)")
    for k, v in PAPER_VAR.items():
        print(f"  paper quotes EOF{k} = {v:.0f}%   ours = {paper.var_pct[k - 1]:.1f}%")
    save(fig_spectrum(paper, title="Eigenvalue spectrum, DJF SLP 1948-2000 (paper period)"), "figA1_spectrum_paper_period.png")
    save(fig_eof_maps(paper), "figA2_eofs_paper_period.png")
    save(fig_pcs(paper), "figA3_pcs_paper_period.png")

    # ---------------- Run B: the project window ------------------------------
    res = run_eof_analysis("slp", REGION, "DJF", n_years=30)
    orient_ao(res, 0, convention="paper")
    ys, ye = res.meta["year_start"], res.meta["year_end"]
    describe(res, "Run B (project window)")
    save(fig_spectrum(res, title=f"Eigenvalue spectrum, DJF SLP winters {ys - 1}/{str(ys)[2:]} to {ye - 1}/{str(ye)[2:]}"),
         "fig1_spectrum.png")
    save(fig_eof_maps(res), "fig2_eofs.png")
    save(fig_eof_maps(res, kind="loading"), "fig2b_eofs_loadings.png")
    save(fig_pcs(res), "fig3_pcs.png")

    # ---------------- Validation and tables -----------------------------------
    print("\nValidation against the official NOAA CPC AO index (sign is arbitrary):")
    ra = validate_against_cpc(paper, "Run A")
    rb = validate_against_cpc(res, "Run B")

    res.table().round(4).to_csv(RES / "eigen_table_project_window.csv", index=False)
    paper.table().round(4).to_csv(RES / "eigen_table_paper_period.csv", index=False)
    pcs = pd.DataFrame(res.pcs_std[:, :5], index=res.time, columns=[f"PC{k}" for k in range(1, 6)])
    pcs.insert(0, "winter", res.season_year)
    pcs.round(4).to_csv(RES / "pcs_project_window.csv", index_label="month")

    summary = pd.DataFrame([
        dict(run="paper period (A)", months=paper.n, start=f"{paper.time[0]:%Y-%m}", end=f"{paper.time[-1]:%Y-%m}",
             eof1_pct=paper.var_pct[0], eof2_pct=paper.var_pct[1], eof1_err=paper.north_err_pct[0],
             eof2_err=paper.north_err_pct[1], corr_pc1_cpc_ao=ra),
        dict(run="project window (B)", months=res.n, start=f"{res.time[0]:%Y-%m}", end=f"{res.time[-1]:%Y-%m}",
             eof1_pct=res.var_pct[0], eof2_pct=res.var_pct[1], eof1_err=res.north_err_pct[0],
             eof2_err=res.north_err_pct[1], corr_pc1_cpc_ao=rb),
        dict(run="paper (Hannachi 2004)", months=np.nan, start="1948", end="2000",
             eof1_pct=21.0, eof2_pct=13.0, eof1_err=np.nan, eof2_err=np.nan, corr_pc1_cpc_ao=np.nan),
    ])
    summary.round(3).to_csv(RES / "replication_summary.csv", index=False)
    print("\n" + summary.round(2).to_string(index=False))

    # ---------------- Which weighting did the paper really use? ----------------
    rows = []
    for w, desc in [("cos", "data x cos(lat), eq. (5) as printed"),
                    ("sqrtcos", "data x sqrt(cos(lat)), covariance area-weighted"),
                    ("none", "no weighting")]:
        a = run_eof_analysis("slp", REGION, "DJF", time_range=("1948-01", "2000-12"), weighting=w)
        b = run_eof_analysis("slp", REGION, "DJF", n_years=30, weighting=w)
        rows.append(dict(weighting=w, description=desc,
                         paper_period_eof1=a.var_pct[0], paper_period_eof2=a.var_pct[1],
                         project_window_eof1=b.var_pct[0], project_window_eof2=b.var_pct[1]))
    rows.append(dict(weighting="paper", description="values quoted in Hannachi (2004)",
                     paper_period_eof1=21.0, paper_period_eof2=13.0,
                     project_window_eof1=np.nan, project_window_eof2=np.nan))
    wt = pd.DataFrame(rows)
    wt.round(2).to_csv(RES / "weighting_comparison.csv", index=False)
    print("\nWeighting comparison (% variance):\n" + wt.round(1).to_string(index=False))
    fig = fig_eof_maps(run_eof_analysis("slp", REGION, "DJF", n_years=30, weighting="none"), kind="loading")
    fig.suptitle("No area weighting: loadings pile up near the pole (paper Sec. 2.2)", x=0.02, ha="left",
                 fontsize=11, fontweight="bold")
    save(fig, "figS_unweighted_loadings.png")


if __name__ == "__main__":
    main()
