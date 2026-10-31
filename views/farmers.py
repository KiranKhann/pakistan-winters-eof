"""Farmers' page: rain and heat in the wheat (rabi) season, in plain language."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from eofkit.farm import (MONTH_NAMES, MONTH_SHORT, RABI_MONTHS, SO_FAR_CATEGORIES, DRY, WET, HOT,  # noqa: E402
                         place_monthly, season_months, season_table, season_label, rain_summary, monthly_stats,
                         trend_words, so_far_analysis, link_verdict, month_heat, heat_verdict)
from eofkit.public import CITIES  # noqa: E402
from eofkit.plots import BLUE, RED, RAIN_COLORS  # noqa: E402

WET_C, DRY_C, NEUTRAL = RAIN_COLORS[8], RAIN_COLORS[2], "#c3c2b7"
LIGHT_WET = RAIN_COLORS[6]
INK2 = "#52514e"


@st.cache_data(show_spinner="Reading 30 years of rainfall and temperature records...")
def get_series(name: str):
    return place_monthly("precip", CITIES[name]), place_monthly("air", CITIES[name])


def month_list(ms: list[int]) -> str:
    names = [MONTH_NAMES[m] for m in ms]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


PLURAL = {2: "Februaries", 3: "Marches", 4: "Aprils"}


def span(ms: list[int]) -> str:
    return MONTH_NAMES[ms[0]] if len(ms) == 1 else f"{MONTH_NAMES[ms[0]]} to {MONTH_NAMES[ms[-1]]}"


# --------------------------------------------------------------------------- #
# Header and controls
# --------------------------------------------------------------------------- #
st.title("For farmers: rain and heat in the wheat season")
st.markdown(
    "Wheat and other **rabi** crops are sown in October and November and harvested in April and May. "
    "In **rain-fed (barani)** areas, winter rain is the crop's main water supply. In irrigated areas it decides "
    "how much canal and tubewell water you will need. This page shows, for your area, what a normal season "
    "brings, how much rain you can count on, how often seasons fail, and how hot the grain-filling month has been.")
st.caption("A record of the last 30 seasons, not a forecast. For forecasts, follow the Pakistan Meteorological "
           "Department's agromet advisories and your local agriculture extension office.")
st.page_link("views/crop_advisor.py", label="Deciding what to grow? Compare 18 crops for your farm",
             icon=":material/agriculture:")

names = list(CITIES)
c1, c2 = st.columns([1, 1])
place = c1.selectbox("Choose your area", names, index=names.index("Chakwal (Potohar)"),
                     help="Values describe the area around the place (about 250 km across), not one field or gauge.")
labels = [MONTH_SHORT[m] for m in RABI_MONTHS]
start_l, end_l = c2.select_slider("Season months", options=labels, value=("Nov", "Mar"),
                                  help="Choose the months that matter for your crop, from sowing to grain filling.")
start, end = RABI_MONTHS[labels.index(start_l)], RABI_MONTHS[labels.index(end_l)]
if RABI_MONTHS.index(end) - RABI_MONTHS.index(start) < 1:
    st.warning("Choose at least two months.")
    st.stop()
months = season_months(start, end)

p, t = get_series(place)
tab = season_table(p, months)
R = rain_summary(tab)
where = "Pakistan" if place == "All of Pakistan" else place
first_s, last_s = season_label(int(tab.index[0])), season_label(int(tab.index[-1]))
win = span(months)

# =========================================================================== #
# 1. What a normal season brings
# =========================================================================== #
st.header("1. What a normal season brings")
st.markdown(f"Rain and snow from **{win}**, seasons {first_s} to {last_s}, for **{where}**.")

k = st.columns(4)
k[0].metric("Normal season rain", f"{R['normal']:.0f} mm", help="The average over the 30 seasons.")
k[0].caption(f"{win}")
k[1].metric("Rain you can count on", f"{R['dependable_3in4']:.0f} mm",
            help="The amount reached or beaten in 3 seasons out of 4 (the 25th percentile).")
k[1].caption("Reached in 3 seasons out of 4.")
one_in = R["n"] / R["dry_n"] if R["dry_n"] else np.inf
k[2].metric("Dry seasons", f"{R['dry_n']} of {R['n']}",
            help=f"Seasons with less than {DRY:.0f}% of normal rain.")
k[2].caption((f"About one season in {one_in:.0f}. " if np.isfinite(one_in) else "") +
             f"{R['severe_n']} got less than half.")
k[3].metric("Change over 30 seasons", trend_words(R["trend_p"], R["trend_pct_per_decade"], "% per decade").split(" (")[0].capitalize(),
            help="Theil-Sen trend with a Kendall test, which are not thrown off by one very wet or dry season.")
k[3].caption(f"Trend {R['trend_pct_per_decade']:+.0f}% of normal per decade.")

ms = monthly_stats(tab, months)
ms["short"] = [MONTH_SHORT[m] for m in months]
ms["tip_normal"] = [f"{v:.0f} mm on average" for v in ms["normal"]]
ms["tip_dep"] = [f"{v:.0f} mm in 3 years out of 4" for v in ms["dependable"]]
ms["tip_dry"] = [f"{v:.0f}% of years" for v in ms["chance_almost_dry"]]

left, right = st.columns(2)
with left:
    st.markdown("**Month by month**")
    base = alt.Chart(ms).encode(x=alt.X("short:N", sort=list(ms["short"]), title=None,
                                         axis=alt.Axis(labelAngle=0)))
    tips = [alt.Tooltip("month:N", title="Month"), alt.Tooltip("tip_normal:N", title="Average"),
            alt.Tooltip("tip_dep:N", title="You can count on"),
            alt.Tooltip("tip_dry:N", title="Almost dry (under 10 mm)")]
    bars_n = base.mark_bar(color=LIGHT_WET, size=34, cornerRadiusEnd=3).encode(
        y=alt.Y("normal:Q", title="mm of rain"), tooltip=tips)
    bars_d = base.mark_bar(color=WET_C, size=34, cornerRadiusEnd=3).encode(y="dependable:Q", tooltip=tips)
    st.altair_chart((bars_n + bars_d).properties(height=260), width="stretch")
    st.caption("Light bar: average rain. Dark bar: the amount reached in 3 years out of 4.")
with right:
    st.markdown("**What this means**")
    top = ms.sort_values("share_of_season", ascending=False)
    main_months = []
    share = 0.0
    for _, r in top.iterrows():
        main_months.append(r["month"])
        share += r["share_of_season"]
        if share >= 60:
            break
    order = [MONTH_NAMES[m] for m in months]
    main_months.sort(key=order.index)
    tips_txt = [f"- Most of the season's rain, about **{share:.0f}%**, falls in "
                f"**{month_list([months[order.index(x)] for x in main_months])}**."]
    dryish = ms[ms["chance_almost_dry"] >= 40]
    if len(dryish):
        n10 = int(round(dryish["chance_almost_dry"].mean() / 10))
        tips_txt.append(f"- **{month_list([months[order.index(x)] for x in dryish['month']])}** "
                        f"{'is' if len(dryish) == 1 else 'are'} almost dry "
                        f"(under 10 mm) in about **{n10} years out of 10**. Do not count on rain then; plan on "
                        f"stored soil moisture or irrigation.")
    reliable = ms[(ms["chance_almost_dry"] <= 10) & (ms["dependable"] >= 20)]
    if len(reliable):
        tips_txt.append(f"- **{month_list([months[order.index(x)] for x in reliable['month']])}** "
                        f"{'is' if len(reliable) == 1 else 'are'} the most "
                        f"reliable: at least 20 mm in 3 years out of 4.")
    gap = R["normal"] - R["dependable_3in4"]
    tips_txt.append(f"- In 1 season out of 4 the total falls **{gap:.0f} mm or more short** of normal "
                    f"(below {R['dependable_3in4']:.0f} mm). That is the gap irrigation or soil moisture must cover in a poor season.")
    st.markdown("\n".join(tips_txt))

st.markdown("**Every season since " + first_s + "**")
sdf = pd.DataFrame({"year": tab.index.astype(int), "mm": tab["total"].values, "pct": R["pct"].values})
sdf["season"] = [season_label(y) for y in sdf["year"]]
sdf["kind"] = np.where(sdf["pct"] < DRY, "Dry", np.where(sdf["pct"] > WET, "Wet", "Near normal"))
sdf["tip"] = [f"{m:.0f} mm, {p:.0f}% of normal ({k.lower()})" for m, p, k in zip(sdf["mm"], sdf["pct"], sdf["kind"])]
bars = alt.Chart(sdf).mark_bar(cornerRadiusEnd=3, size=14).encode(
    x=alt.X("year:O", title=None, axis=alt.Axis(labelExpr="datum.value % 5 == 0 ? datum.value : ''", labelAngle=0)),
    y=alt.Y("mm:Q", title="Season rain (mm)"),
    color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Dry", "Near normal", "Wet"],
                                                          range=[DRY_C, NEUTRAL, WET_C]),
                    legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("tip:N", title="Rain")])
rules = pd.DataFrame({"y": [R["normal"], DRY / 100 * R["normal"]], "what": ["normal", f"{DRY:.0f}% of normal"]})
rule = alt.Chart(rules).mark_rule(color=INK2, strokeDash=[5, 3]).encode(y="y:Q", tooltip=["what:N"])
st.altair_chart((bars + rule).properties(height=260), width="stretch")
st.caption(f"Upper dashed line: normal ({R['normal']:.0f} mm). Lower dashed line: {DRY:.0f}% of normal, "
           f"below which a season counts as dry. The axis shows the harvest year.")

# =========================================================================== #
# 2. Season so far
# =========================================================================== #
st.header("2. The season so far: what happened next in past years?")
st.markdown("Partway through a season, it helps to know whether a dry or wet start has tended to continue. "
            "Choose how far into the season you are and how the rain has been, for example from the PMD "
            "bulletin's percentage of normal or your own rain gauge.")
cc1, cc2 = st.columns([1, 2])
upto = cc1.selectbox("Up to the end of", months[:-1], index=min(1, len(months) - 2),
                     format_func=lambda m: MONTH_NAMES[m])
cat = cc2.radio("How has rain been so far?", range(len(SO_FAR_CATEGORIES)), index=0,
                format_func=lambda i: SO_FAR_CATEGORIES[i][0])
A = so_far_analysis(tab, months, upto, cat)
st.markdown(f"Normally **{span(A['first'])}** brings **{A['first_normal_mm']:.0f} mm** here, and the rest of the "
            f"season (**{span(A['rest'])}**) brings **{A['rest_normal_mm']:.0f} mm**.")

if A["n"] == 0:
    st.warning("None of the last 30 seasons started like this, so there is nothing to compare with.")
else:
    m = st.columns(3)
    m[0].metric("Past seasons that started like this", f"{A['n']}")
    m[1].metric("Rest of season was dry", f"{A['dry']} of {A['n']}",
                help=f"Below {DRY:.0f}% of the normal for {span(A['rest'])}.")
    m[1].caption(f"In all 30 seasons: {A['base_dry']:.0f}% were dry.")
    m[2].metric("Rest of season was wet", f"{A['wet']} of {A['n']}",
                help=f"Above {WET:.0f}% of the normal for {span(A['rest'])}.")
    m[2].caption(f"In all 30 seasons: {A['base_wet']:.0f}% were wet.")
    st.markdown(f"After the **{A['n']}** past seasons that started like this, the rest of the season was "
                f"**dry in {A['dry']}**, **near normal in {A['normal']}** and **wet in {A['wet']}**.")
    if A["n"] < 5:
        st.warning(f"Only {A['n']} past season{'s' if A['n'] > 1 else ''} started like this. That is too few to "
                   f"say much.")
st.info(link_verdict(A))

adf = pd.DataFrame({"year": A["so_far_pct"].index.astype(int), "a": A["so_far_pct"].values,
                    "b": A["rest_pct"].values, "sel": A["selected"].values})
adf["season"] = [season_label(y) for y in adf["year"]]
adf["group"] = np.where(adf["sel"], "Started like this", "Other seasons")
pts = alt.Chart(adf).mark_circle(size=80, opacity=0.9, stroke="#fcfcfb", strokeWidth=1).encode(
    x=alt.X("a:Q", title=f"Rain {span(A['first'])}, % of normal"),
    y=alt.Y("b:Q", title=f"Rain {span(A['rest'])}, % of normal"),
    color=alt.Color("group:N", title=None, scale=alt.Scale(domain=["Started like this", "Other seasons"],
                                                           range=[BLUE, NEUTRAL]),
                    legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("a:Q", title="Start, % of normal", format=".0f"),
             alt.Tooltip("b:Q", title="Rest, % of normal", format=".0f")])
ref = alt.Chart(pd.DataFrame({"v": [100]}))
st.altair_chart((pts + ref.mark_rule(color=INK2, strokeDash=[4, 3]).encode(x="v:Q")
                 + ref.mark_rule(color=INK2, strokeDash=[4, 3]).encode(y="v:Q")).properties(height=300),
                width="stretch")
st.caption("Each dot is one past season. Dashed lines mark normal. Points to the lower left had a dry start and a "
           "dry finish; hover a dot to see the season.")

# =========================================================================== #
# 3. Heat at grain filling
# =========================================================================== #
st.header("3. Heat when the grain fills")
st.markdown("Wheat grain fills in **March** in Sindh and southern Punjab, and in **April** further north. "
            "Hot spells at this stage shrivel the grain and cut yields, so a warming trend in these months matters.")
hm = st.radio("Grain-filling month", [2, 3, 4], index=1, horizontal=True, format_func=lambda m: MONTH_NAMES[m])
H = month_heat(t, hm)
mname = MONTH_NAMES[hm]
h = st.columns(3)
h[0].metric(f"Warming of {mname}", f"{H['trend']:+.1f} °C per decade")
h[0].caption(f"{heat_verdict(H['p']).capitalize()}.")
h[1].metric(f"Hot {PLURAL[hm]}", f"{len(H['hot_years'])} of {H['n']}",
            help=f"Months whose average was more than {HOT} °C above the 30-year normal.")
h[1].caption(f"More than {HOT} °C above normal.")
h[2].metric(f"Hottest {PLURAL[hm]}", ", ".join(str(y) for y in H["hottest"]))
recent_hot = [y for y in H["hot_years"] if y >= H["last"] - 9]
h[2].caption(f"{len(recent_hot)} of the hot ones came in the last 10 years.")

hdf = pd.DataFrame({"year": H["anom"].index.astype(int), "an": H["anom"].values})
hdf["trend"] = H["intercept"] + H["slope"] * hdf["year"]
hdf["tip"] = [f"{abs(v):.1f} °C {'warmer' if v > 0 else 'colder'} than normal" for v in hdf["an"]]
hb = alt.Chart(hdf).mark_bar(cornerRadiusEnd=3, size=14).encode(
    x=alt.X("year:O", title=None, axis=alt.Axis(labelExpr="datum.value % 5 == 0 ? datum.value : ''", labelAngle=0)),
    y=alt.Y("an:Q", title=f"{mname}, °C above (+) or below (−) normal"),
    color=alt.condition("datum.an > 0", alt.value(RED), alt.value(BLUE)),
    tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("tip:N", title=mname)])
hl = alt.Chart(hdf).mark_line(color="#0b0b0b", strokeDash=[5, 3], strokeWidth=1.5).encode(x="year:O", y="trend:Q")
hot_rule = alt.Chart(pd.DataFrame({"y": [HOT]})).mark_rule(color=RED, strokeDash=[2, 3]).encode(y="y:Q")
st.altair_chart((hb + hl + hot_rule).properties(height=250), width="stretch")
st.caption(f"Red: warmer than normal. Blue: colder. Black dashed line: the trend. Red dotted line: the 'hot' mark "
           f"(+{HOT} °C). Temperature records here end in February 2026, so {mname} data runs {H['first']} to {H['last']}. "
           "A monthly average hides heat waves of a few days, so a month that looks normal can still contain a damaging hot spell.")

# =========================================================================== #
# 4. Take it with you
# =========================================================================== #
st.header("4. Take it with you")
fact = [
    f"Rabi season fact sheet: {where}, {win}, seasons {first_s} to {last_s}",
    "Source: GPCP v2.3 rainfall and NCEP/NCAR Reanalysis 1 temperature (NOAA). A record, not a forecast.",
    "",
    f"Normal season rain: {R['normal']:.0f} mm",
    f"Rain you can count on (3 seasons in 4): {R['dependable_3in4']:.0f} mm",
    f"Dry seasons (below {DRY:.0f}% of normal): {R['dry_n']} of {R['n']}; below half of normal: {R['severe_n']}",
    f"Wet seasons (above {WET:.0f}% of normal): {R['wet_n']} of {R['n']}",
    f"Change over 30 seasons: {trend_words(R['trend_p'], R['trend_pct_per_decade'], '% per decade')}",
    "",
    "Month by month (average mm / mm in 3 years out of 4 / % of years almost dry):",
] + [f"  {r['month']:<9} {r['normal']:5.0f} / {r['dependable']:5.0f} / {r['chance_almost_dry']:3.0f}%" for _, r in ms.iterrows()] + [
    "",
    f"{mname} temperature: {H['trend']:+.1f} °C per decade ({heat_verdict(H['p'])}); "
    f"hot {PLURAL[hm]} (> +{HOT} °C): {', '.join(str(y) for y in H['hot_years']) or 'none'}",
]
out = tab[months].copy()
out.columns = [MONTH_NAMES[m] for m in months]
out.insert(0, "season", [season_label(int(y)) for y in out.index])
out["total_mm"] = tab["total"].values
out["percent_of_normal"] = R["pct"].values.round(0)
out["season_type"] = sdf["kind"].values
d1, d2 = st.columns(2)
d1.download_button("Download the fact sheet (text)", "\n".join(fact),
                   f"rabi_fact_sheet_{where.split(' (')[0].replace(' ', '_')}.txt", "text/plain")
d2.download_button("Download every season (CSV)", out.round(1).to_csv(index=False),
                   f"rabi_seasons_{where.split(' (')[0].replace(' ', '_')}.csv", "text/csv")
with st.expander("Preview the fact sheet"):
    st.code("\n".join(fact), language=None)

with st.expander("What this page can and cannot tell you"):
    st.markdown(
        "- **It is not a forecast.** It describes the last 30 seasons. The 'season so far' section shows what "
        "happened after similar starts; it does not predict this season.\n"
        "- **Each area is about 250 km across.** Your own field may get more or less rain, especially near hills.\n"
        "- **Rain includes snow**, counted as the water it melts to. Rainfall comes from satellites combined with "
        "rain gauges (GPCP).\n"
        "- **'Normal' is the 30-season average.** Rain is uneven, so most seasons fall a little below the average "
        "and a few very wet seasons pull it up. That is why 'rain you can count on' is the more useful planning number.\n"
        "- **Temperature is a monthly average from a weather model (reanalysis)** on a coarse grid. Use it for "
        "trends and comparisons, not exact values.")
