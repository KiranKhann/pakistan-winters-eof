"""The maths page: Hannachi (2004) Sections 1-3, replicated and applied to Pakistan."""
from __future__ import annotations

import io

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eofkit import run_eof_analysis, open_field, SEASONS, default_years  # noqa: E402
from eofkit.application import (MAP_EXTENT, LOCAL_DOMAIN, TARGETS, load_targets, index_vs_targets,  # noqa: E402
                                local_modes, modes_vs_targets, detrend, detrend_field)
from eofkit.plots import (fig_spectrum, fig_eof_maps, fig_pcs, fig_reconstruction, fig_rmse_curve,  # noqa: E402
                          fig_correlation_maps, fig_composite_maps, fig_winter_bars, fig_eof_maps_region,
                          CMAP_DIV, CMAP_RAIN, BLUE, RED, INK, RAIN_COLORS)
from eofkit.teleconnection import (orient_ao, index_series, load_cpc_ao, correlation_test,  # noqa: E402
                                   correlation_map, composite_difference, seasonal_means, extreme_months,
                                   _country_geometry)


GPCP_FIRST_WINTER = 1980        # GPCP starts Jan 1979, so DJF 1979/80 is the first full winter
REGION = "NH north of 20N (paper)"


# --------------------------------------------------------------------------- #
# Cached computations
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def latest_winter() -> int:
    return default_years(open_field("slp"), SEASONS["DJF"], 30)[1]


@st.cache_data(show_spinner="Computing EOFs by SVD...")
def get_eofs(ys: int, ye: int, weighting: str, time_range, convention: str):
    res = run_eof_analysis("slp", REGION, "DJF", ys, ye, weighting=weighting,
                           time_range=time_range)
    orient_ao(res, 0, convention=convention)
    return res


@st.cache_data(show_spinner="Loading temperature and precipitation...")
def get_targets(ys: int, ye: int):
    return load_targets(ys, ye)


@st.cache_data(show_spinner="Computing Pakistan-centred EOFs...")
def get_local(ys: int, ye: int, lat: tuple, lon: tuple, weighting: str):
    return local_modes(ys, ye, dict(lat=lat, lon=lon), weighting=weighting)


@st.cache_data(show_spinner=False)
def cpc_ao():
    return load_cpc_ao()


@st.cache_resource(show_spinner=False)
def pakistan():
    return _country_geometry("Pakistan")


def show(fig, **kw) -> None:
    st.pyplot(fig, clear_figure=False, **kw)
    plt.close(fig)


def png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def pfmt(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def scatter_chart(df: pd.DataFrame, x: str, y: str, xtitle: str, ytitle: str, st_: dict) -> alt.Chart:
    """Interactive scatter with a least-squares line; hover shows the month."""
    base = alt.Chart(df)
    pts = base.mark_circle(size=60, color=BLUE, opacity=0.85, stroke="#fcfcfb", strokeWidth=1).encode(
        x=alt.X(x, title=xtitle), y=alt.Y(y, title=ytitle),
        tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip(x, format="+.2f", title=xtitle),
                 alt.Tooltip(y, format="+.2f", title=ytitle)])
    xs = np.linspace(df[x].min(), df[x].max(), 20)
    line_df = pd.DataFrame({x: xs, y: st_["intercept"] + st_["slope"] * xs})
    line = alt.Chart(line_df).mark_line(color=INK, strokeWidth=1.5).encode(x=x, y=y)
    title = f"r = {st_['r']:+.2f},  p {pfmt(st_['p']) if st_['p'] < 0.001 else '= ' + pfmt(st_['p'])},  n = {st_['n']}"
    return (pts + line).properties(height=320, title=alt.TitleParams(title, anchor="start", fontSize=12,
                                                                          fontWeight="normal", color="#52514e"))


# --------------------------------------------------------------------------- #
# Sidebar: one set of controls for every tab
# --------------------------------------------------------------------------- #
LATEST = latest_winter()
st.sidebar.subheader("Analysis settings")
st.sidebar.caption("Hannachi (2004), *A primer for EOF analysis of climate data*, Sections 1-3, "
                   "replicated on NOAA monthly NetCDF data and applied to Pakistan's winters.")

period = st.sidebar.radio(
    "Analysis period (DJF winters)",
    ["Last 30 winters (project)", "Paper period, 1948-2000", "Custom"],
    help="The professor's rule: at least 30 years after 1981, or the last 30 years.")
time_range = None
if period == "Last 30 winters (project)":
    ys, ye = LATEST - 29, LATEST
elif period == "Paper period, 1948-2000":
    ys, ye = 1948, 2000
    time_range = ("1948-01", "2000-12")
else:
    ys, ye = st.sidebar.slider("Winters (year of January)", 1949, LATEST, (LATEST - 29, LATEST))
    if ye - ys + 1 < 10:
        st.sidebar.warning("Fewer than 10 winters: the North error bars will be very wide.")

weighting = st.sidebar.selectbox(
    "Area weighting of the data (eq. 5-6)",
    ["sqrtcos", "cos", "none"],
    format_func={"sqrtcos": "√cos(lat), matches the paper",
                 "cos": "cos(lat), eq. (5) as printed",
                 "none": "None (polar artefact)"}.get,
    help="√cos(lat) weights the covariance by grid-cell area and reproduces the paper's 21% / 13%. "
         "cos(lat) is eq. (5) exactly as printed and gives 18.8% / 14.9% on the paper's sample.")
convention = st.sidebar.radio(
    "Sign of EOF1", ["paper", "noaa"],
    format_func={"paper": "As in the paper (high over the pole)",
                 "noaa": "NOAA AO index (low over the pole)"}.get,
    help="EOFs are only defined up to a sign (paper Sec. 3.4). This flips EOF1 and PC1 together.")

res = get_eofs(ys, ye, weighting, time_range, convention)
w_label = f"{res.time[0]:%b %Y} to {res.time[-1]:%b %Y}"
st.sidebar.markdown(f"**Sample:** {res.n} DJF months, {w_label}  \n"
                    f"**Grid:** {res.lat.size} × {res.lon.size} = {res.p} points (2.5°), "
                    f"{res.p_used} used")
st.sidebar.caption("Data: NCEP/NCAR Reanalysis 1 SLP and air temperature; GPCP v2.3 precipitation; "
                   "NOAA CPC monthly AO index (validation only).")

st.title("The maths: EOF analysis of winter pressure")
st.caption(f"Winter (DJF) sea level pressure north of 20°N, {w_label}. "
           "Every number on this page is recomputed from the NetCDF files when you change a control.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Paper replication", "Compression and reconstruction",
                                        "AO and Pakistan", "Pakistan-centred modes", "Method and data"])

# =========================================================================== #
# Tab 1: the paper's Figures 1-3
# =========================================================================== #
with tab1:
    sep = res.separable()
    cpc = cpc_ao()
    r_cpc = None
    if cpc is not None:
        both = pd.concat([pd.Series(res.pcs_std[:, 0], index=res.time), cpc], axis=1, join="inner").dropna()
        if len(both) > 10:
            r_cpc = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])

    c = st.columns(4)
    c[0].metric("EOF1 variance", f"{res.var_pct[0]:.1f}%", f"± {res.north_err_pct[0]:.1f} (North)", delta_color="off")
    c[1].metric("EOF2 variance", f"{res.var_pct[1]:.1f}%", f"± {res.north_err_pct[1]:.1f} (North)", delta_color="off")
    c[2].metric("EOF1 separable from EOF2?", "Yes" if sep[0] else "No",
                "error bars do not overlap" if sep[0] else "error bars overlap", delta_color="off")
    c[3].metric("|corr(PC1, NOAA AO index)|", f"{abs(r_cpc):.2f}" if r_cpc is not None else "n/a",
                f"{len(both)} months" if r_cpc is not None else "", delta_color="off")
    if period == "Paper period, 1948-2000":
        st.info(f"The paper reports EOF1 = 21% and EOF2 = 13% for 1948-2000. This run gives "
                f"{res.var_pct[0]:.1f}% and {res.var_pct[1]:.1f}% with the selected weighting.")

    st.subheader("Figure 1. Eigenvalue spectrum")
    st.caption("Explained variance of the first 50 EOFs with North et al. (1982) error bars, "
               "Δλ ≈ λ·√(2/n) (eq. 12). Filled markers are separated from both neighbours.")
    show(fig_spectrum(res))

    st.subheader("Figure 2. Leading EOFs")
    kind = st.radio("Map units", ["regression", "loading"], horizontal=True,
                    format_func={"regression": "hPa per s.d. of the PC (interpretable)",
                                 "loading": "unit-norm EOF vector (as computed)"}.get)
    show(fig_eof_maps(res, kind=kind))
    st.caption("EOF1 is the Arctic Oscillation: one centre over the Arctic and opposite centres over the "
               "North Atlantic / Mediterranean and the North Pacific. EOF2 has separate centres over the "
               "North Pacific and the North East Atlantic.")

    st.subheader("Figure 3. Leading principal components")
    st.caption("Standardised PCs (eq. 10). Months beyond ±1 are the paper's 'extreme events'. "
               "Lines join the three months of each winter.")
    show(fig_pcs(res))

    with st.expander("Eigenvalue table and PC time series (download)"):
        tab = res.table()
        st.dataframe(tab.style.format({"eigenvalue": "{:.1f}", "variance_%": "{:.2f}",
                                       "cumulative_%": "{:.2f}", "north_error_%": "{:.2f}"}),
                     hide_index=True, width="stretch")
        pcs = pd.DataFrame(res.pcs_std[:, :5], index=res.time.strftime("%Y-%m"),
                           columns=[f"PC{k}" for k in range(1, 6)])
        pcs.index.name = "month"
        a, b = st.columns(2)
        a.download_button("Eigenvalues (CSV)", tab.to_csv(index=False), "eigenvalues.csv", "text/csv")
        b.download_button("PCs 1-5 (CSV)", pcs.to_csv(), "pcs.csv", "text/csv")

# =========================================================================== #
# Tab 2: truncation, reconstruction and compression (eq. 11)
# =========================================================================== #
with tab2:
    st.markdown("Keeping only the first *k* terms of eq. (11), X ≈ Σ c<sub>k</sub>(t) a<sub>k</sub>, gives the "
                "best rank-*k* approximation of the anomaly matrix in the least-squares sense "
                "(Eckart-Young theorem). This tab shows how quickly the winter pressure field "
                "is captured as *k* grows.", unsafe_allow_html=True)
    labels = [f"{t:%b %Y}" for t in res.time]
    pc1 = res.pcs_std[:, 0]
    default_t = int(np.argmax(np.abs(pc1)))
    a, b = st.columns([2, 3])
    t_idx = a.selectbox("Month", range(res.n), index=default_t, format_func=lambda i: labels[i],
                        help="Defaults to the month with the most extreme PC1.")
    k_custom = b.slider("Number of EOFs in the last panel", 1, min(40, res.all_var_pct.size), 10)
    ks = sorted(set([1, 2, 5, k_custom]))
    show(fig_reconstruction(res, t_idx, ks=ks))

    c = st.columns(3)
    cum = np.cumsum(res.all_var_pct)
    c[0].metric(f"Variance kept with {k_custom} EOFs", f"{cum[k_custom - 1]:.1f}%")
    c[1].metric("Storage, full matrix", f"{res.n * res.p_used:,} numbers")
    c[2].metric(f"Storage, rank {k_custom}", f"{k_custom * (res.n + res.p_used):,} numbers",
                f"{res.compression_ratio(k_custom):.0f}× smaller", delta_color="off")
    k80 = int(np.searchsorted(cum, 80.0) + 1)
    st.caption(f"The paper's truncation rule (Sec. 3.2.2): keep enough EOFs to explain e.g. 80% of the variance. "
               f"Here that takes {k80} EOFs.")
    show(fig_rmse_curve(res))

# =========================================================================== #
# Tab 3: Arctic Oscillation vs Pakistan
# =========================================================================== #
with tab3:
    ys_app = max(ys, GPCP_FIRST_WINTER)
    if ys_app != ys or time_range is not None:
        ys_app = max(ys_app, GPCP_FIRST_WINTER)
        ye_app = min(ye, LATEST)
        st.warning(f"GPCP rainfall starts in 1979, so this tab uses winters {ys_app - 1}/{ys_app} to "
                   f"{ye_app - 1}/{ye_app}.")
    else:
        ye_app = ye
    res_app = get_eofs(ys_app, ye_app, weighting, None, "noaa")
    ao = index_series(res_app, 0, "AO")
    sy = res_app.season_year
    tg = get_targets(ys_app, ye_app)

    st.markdown("**Question.** The Arctic Oscillation is the leading pattern of Northern Hemisphere winter "
                "pressure. Does it control how cold or wet Pakistan's winters are?  \n"
                "AO index = standardised PC1 with the NOAA sign (positive = low pressure over the Arctic, "
                "stronger westerlies).")
    a, b = st.columns([2, 1])
    region_choice = a.radio("Area", ["Pakistan", "Northern Pakistan (31-37N, 69-77E)"], horizontal=True)
    do_dt = b.checkbox("Remove linear trends", help="Pakistan has warmed; detrending checks that the trend "
                                                    "is not driving the correlations.")
    pre = "Pakistan" if region_choice == "Pakistan" else "N. Pakistan"
    sT = tg["series"][f"{pre} T"].reindex(ao.index)
    sP = tg["series"][f"{pre} P"].reindex(ao.index)
    x_ao = detrend(ao) if do_dt else ao
    yT = detrend(sT) if do_dt else sT
    yP = detrend(sP) if do_dt else sP
    stT, stP = correlation_test(x_ao.values, yT.values), correlation_test(x_ao.values, yP.values)

    c = st.columns(4)
    c[0].metric("r (AO, temperature)", f"{stT['r']:+.2f}", f"p {pfmt(stT['p'])}", delta_color="off")
    c[1].metric("Temperature variance explained", f"{100 * stT['r'] ** 2:.1f}%")
    c[2].metric("r (AO, precipitation)", f"{stP['r']:+.2f}", f"p {pfmt(stP['p'])}", delta_color="off")
    c[3].metric("Precipitation variance explained", f"{100 * stP['r'] ** 2:.1f}%")

    sig_any = (stT["p"] < 0.05) or (stP["p"] < 0.05)
    if not sig_any:
        st.info("**Finding:** over this period the AO has no statistically significant link with "
                f"{region_choice.split(' (')[0]}'s winter temperature or rainfall. The maps show why: the AO "
                "warms Central Asia and cools the Middle East, and Pakistan lies on the neutral line between "
                "the two.")

    air, pr = tg["air"], tg["pr"]
    if do_dt:
        air, pr = detrend_field(air), detrend_field(pr)
    r_t, p_t = correlation_map(x_ao, air)
    r_p, p_p = correlation_map(x_ao, pr)
    m1, m2 = st.columns(2)
    with m1:
        show(fig_correlation_maps([("AO vs surface air temperature", r_t, p_t)], air.lat.values,
                                  air.lon.values, MAP_EXTENT, outline=pakistan()))
    with m2:
        show(fig_correlation_maps([("AO vs precipitation", r_p, p_p)], pr.lat.values, pr.lon.values,
                                  MAP_EXTENT, outline=pakistan()))

    with st.expander("Composites: strong AO+ months minus strong AO- months (|AO| > 1)"):
        d_t, pd_t, npos, nneg = composite_difference(x_ao, air)
        d_p, pd_p, _, _ = composite_difference(x_ao, pr)
        m1, m2 = st.columns(2)
        with m1:
            show(fig_composite_maps([(f"Temperature ({npos} AO+ vs {nneg} AO- months)", d_t, pd_t, "°C", CMAP_DIV)],
                                    air.lat.values, air.lon.values, MAP_EXTENT, outline=pakistan()))
        with m2:
            show(fig_composite_maps([(f"Precipitation ({npos} AO+ vs {nneg} AO- months)", d_p, pd_p, "mm/day",
                                      CMAP_RAIN)], pr.lat.values, pr.lon.values, MAP_EXTENT, outline=pakistan()))
        ext = extreme_months(ao, {"T anomaly (°C)": sT, "P anomaly (mm/day)": sP})
        st.dataframe(ext.style.format({"AO": "{:+.2f}", "T anomaly (°C)": "{:+.2f}", "P anomaly (mm/day)": "{:+.2f}"}),
                     width="stretch")

    st.markdown("**Month by month** (hover a point to see the month)")
    df = pd.DataFrame({"AO": x_ao.values, "T": yT.values, "P": yP.values, "month": ao.index.strftime("%b %Y")})
    m1, m2 = st.columns(2)
    m1.altair_chart(scatter_chart(df, "AO", "T", "AO index (s.d.)", "Temperature anomaly (°C)", stT),
                    width="stretch")
    m2.altair_chart(scatter_chart(df, "AO", "P", "AO index (s.d.)", "Precipitation anomaly (mm/day)", stP),
                    width="stretch")

    st.markdown("**Winter by winter**")
    show(fig_winter_bars([
        (seasonal_means(x_ao, sy), "AO index", (BLUE, RED)),
        (seasonal_means(yT, sy), f"{pre} T (°C)", (BLUE, RED)),
        (seasonal_means(yP, sy), f"{pre} P (mm/day)", (RAIN_COLORS[2], RAIN_COLORS[8])),
    ]))

    with st.expander("All statistics (monthly, winter-mean, one-month lead)"):
        stats = index_vs_targets(ao, sy, tg["series"], do_detrend=do_dt)
        st.dataframe(stats.style.format({c: "{:+.3f}" for c in stats.columns if c.startswith(("r_", "slope"))}
                                        | {c: pfmt for c in stats.columns if c.startswith("p_")}
                                        | {"n_eff": "{:.0f}"}),
                     hide_index=True, width="stretch")
        st.caption("p-values use a t-test with the Bretherton et al. (1999) effective sample size. "
                   "The one-month lead pairs December with January and January with February.")

# =========================================================================== #
# Tab 4: Pakistan-centred EOFs (domain dependence, paper Sec. 3.4)
# =========================================================================== #
with tab4:
    st.markdown("**Idea.** The paper warns that EOF patterns depend on the domain (Sec. 3.4). So repeat "
                "exactly the same analysis on a domain centred on Pakistan, and ask whether *those* modes "
                "track Pakistan's winter weather. Each PC is oriented so that positive means higher pressure "
                "averaged over the domain.")
    a, b, c = st.columns([2, 2, 1])
    lat_rng = a.slider("Latitude range (°N)", 0, 60, (int(LOCAL_DOMAIN["lat"][0]), int(LOCAL_DOMAIN["lat"][1])), step=5)
    lon_rng = b.slider("Longitude range (°E)", 20, 120, (int(LOCAL_DOMAIN["lon"][0]), int(LOCAL_DOMAIN["lon"][1])), step=5)
    do_dt4 = c.checkbox("Remove linear trends", key="dt4")
    ys4 = max(ys, GPCP_FIRST_WINTER)
    ye4 = ye if time_range is None else min(ye, LATEST)
    loc = get_local(ys4, ye4, tuple(float(v) for v in lat_rng), tuple(float(v) for v in lon_rng), weighting)
    tg4 = get_targets(ys4, ye4)

    tab_m = modes_vs_targets(loc, tg4["series"], kmax=5, do_detrend=do_dt4)
    show(fig_eof_maps_region(loc, (0, 1), [lon_rng[0], lon_rng[1], lat_rng[0], lat_rng[1]],
                             outline=pakistan(), title_prefix="Local "))

    st.markdown("**Which local modes go with Pakistan's weather?** (monthly correlations; "
                "bold when p < 0.05)")
    view = tab_m[["mode", "variance_pct", "r Pakistan T", "p Pakistan T", "r Pakistan P", "p Pakistan P",
                  "r N. Pakistan T", "p N. Pakistan T", "r N. Pakistan P", "p N. Pakistan P"]].copy()

    def bold_sig(row):
        out = [""] * len(row)
        for i, col in enumerate(row.index):
            if col.startswith("r "):
                pcol = "p " + col[2:]
                if row[pcol] < 0.05:
                    out[i] = "font-weight: 700"
        return out

    st.dataframe(view.style.apply(bold_sig, axis=1).format(
        {"variance_pct": "{:.1f}%"} | {c: "{:+.2f}" for c in view.columns if c.startswith("r ")}
        | {c: pfmt for c in view.columns if c.startswith("p ")}),
        hide_index=True, width="stretch",
        column_config={"mode": "Mode", "variance_pct": "Variance"})
    st.caption("Five modes × four targets = 20 tests, so about one 'significant' result is expected by chance "
               "at p < 0.05. Trust the strong, physically interpretable links, not every bold number.")

    a, b = st.columns(2)
    mode = a.selectbox("Mode", [f"PC{k}" for k in range(1, min(5, loc.eigvals.size) + 1)], index=0)
    target = b.selectbox("Target", list(TARGETS), format_func=lambda k: TARGETS[k]["label"])
    k = int(mode[2:]) - 1
    x = pd.Series(loc.pcs_std[:, k], index=loc.time)
    y = tg4["series"][target].reindex(loc.time)
    if do_dt4:
        x, y = detrend(x), detrend(y)
    st4 = correlation_test(x.values, y.values)
    df4 = pd.DataFrame({"PC": x.values, "Y": y.values, "month": loc.time.strftime("%b %Y")})
    st.altair_chart(scatter_chart(df4, "PC", "Y", f"Local {mode} (s.d.)",
                                  f"{TARGETS[target]['label'].split(' (')[0]} anomaly ({TARGETS[target]['units']})", st4),
                    width="stretch")
    st.caption("The local modes describe the same months' weather (diagnosis), not the next month's: "
               "their one-month-lead correlations are not significant.")

# =========================================================================== #
# Tab 5: method and data
# =========================================================================== #
with tab5:
    st.subheader("From NetCDF to EOFs (paper Sections 2-3)")
    st.markdown("1. **Data matrix (eq. 1-2).** The field F(t, φ, λ) is read from NetCDF with `xarray` and reshaped "
                "to X with n rows (months) and p = p₁p₂ columns (grid points).")
    st.markdown("2. **Anomalies (eq. 3-4).** The mean of each calendar month over the window is subtracted, "
                "which removes the seasonal cycle.")
    st.latex(r"X' = X - \mathbf{1}\,\bar{x}, \qquad \bar{x}_{.j} = \frac{1}{n}\sum_{k=1}^{n} x_{kj}")
    st.markdown("3. **Area weighting (eq. 5-6).** Each column is multiplied by a latitude weight. The paper "
                "prints D = diag(cos φ), but its reported variances (21%, 13%) are reproduced only with "
                "D = diag(√cos φ), which weights the covariance matrix by grid-cell area.")
    st.latex(r"X'_w = X' D")
    st.markdown("4. **EOFs by SVD (eq. 7-10, 13).** The covariance matrix is never formed. The thin SVD of the "
                "n × p matrix gives everything at once, at O(n²p) cost instead of O(p³).")
    st.latex(r"X'_w = U\,\Sigma\,V^{T},\quad \text{EOF}_k = v_k,\quad c_k = X'_w v_k = \sigma_k u_k,"
             r"\quad \lambda_k = \frac{\sigma_k^2}{n-1}")
    st.markdown("5. **Sampling error (eq. 12).** North et al. (1982) rule of thumb; an EOF is treated as "
                "separable when its error bar does not overlap its neighbours'.")
    st.latex(r"\delta\lambda_k \approx \lambda_k \sqrt{2/n}")
    st.markdown("6. **Reconstruction (eq. 11).** The truncated sum is the best rank-k approximation.")
    st.latex(r"X'(t,s) \approx \sum_{k=1}^{M} c_k(t)\, a_k(s)")

    st.subheader("Checks built into the code")
    st.markdown("- EOFs are orthonormal and PCs uncorrelated to machine precision (about 10⁻¹⁵ and 10⁻¹¹).\n"
                "- SVD eigenvalues equal a direct eigen-decomposition of the covariance matrix.\n"
                "- On the paper's 1948-2000 sample the variances match the paper (21.6% / 13.2% vs 21% / 13%).\n"
                "- PC1 correlates with NOAA's official monthly AO index at |r| ≈ 0.96-0.98.")

    st.subheader("Data (all monthly NetCDF, free, no login)")
    st.dataframe(pd.DataFrame([
        ["Sea level pressure", "NCEP/NCAR Reanalysis 1", "1948-present", "2.5°", "slp.mon.mean.nc"],
        ["Surface air temperature", "NCEP/NCAR Reanalysis 1", "1948-present", "2.5°", "air.mon.mean.nc"],
        ["Precipitation", "GPCP v2.3", "1979-present", "2.5°", "precip.mon.mean.nc"],
        ["AO index (validation)", "NOAA CPC", "1950-present", "index", "monthly.ao.index.b50.current.ascii"],
    ], columns=["Variable", "Source", "Period", "Grid", "File"]), hide_index=True, width="stretch")
    st.caption("Caveat: reanalysis sea level pressure over the Tibetan Plateau and Himalaya is extrapolated "
               "below high terrain, so local patterns there should be read with care.")
