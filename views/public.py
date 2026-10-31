"""Public page: Pakistan's winters, explained in plain language."""
from __future__ import annotations

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eofkit.public import (CITIES, PATTERNS, AO_TEXT, STRONG, latest_window, build_patterns, build_city,  # noqa: E402
                           explain_month, month_option_label, winter_label, temp_words, rain_words,
                           trend_verdict, strength_words, link_sentence, is_real_link, sentence_case)
from eofkit.plots import fig_public_map, CMAP_DIV, BLUE, RED, RAIN_COLORS  # noqa: E402
from eofkit.teleconnection import _country_geometry  # noqa: E402

WARM, COLD = RED, BLUE
WET, DRY = RAIN_COLORS[8], RAIN_COLORS[2]
GREY = "#898781"
MAP_EXTENT = [40, 100, 10, 50]


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading 30 years of NOAA climate records...")
def get_patterns():
    ys, ye = latest_window()
    return build_patterns(ys, ye)


@st.cache_data(show_spinner="Working out the climate of this place...")
def get_city(name: str):
    return build_city(name, get_patterns())


@st.cache_resource(show_spinner=False)
def pakistan():
    return _country_geometry("Pakistan")


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def short_place(name: str) -> str:
    return "Pakistan" if name == "All of Pakistan" else name


# --------------------------------------------------------------------------- #
# Header and place picker
# --------------------------------------------------------------------------- #
pats = get_patterns()
first_w, last_w = pats.year_start, pats.year_end

st.title("Pakistan's winters, explained")
st.markdown(
    f"Choose a place to see **how its winters have changed**, **why a particular month was unusually "
    f"cold, warm, wet or dry**, and **which big weather patterns are behind it**. Built from NOAA's global "
    f"climate records for the last 30 winters (December to February, {winter_label(first_w)} to "
    f"{winter_label(last_w)}).")

names = list(CITIES)
place = st.selectbox("Choose a place", names, index=names.index("Lahore"),
                     help="Values describe the area around the city (about 250 km across), not a single weather station.")
city = get_city(place)
where = short_place(place)
here = "across Pakistan" if place == "All of Pakistan" else f"around {place}"
W = city.winters

# =========================================================================== #
# 1. How have winters changed?
# =========================================================================== #
st.header("1. How have winters changed here?")

tt, tp = city.trend_t, city.trend_p
last = W.iloc[-1]
dry_n = int((W["p_pct"] < 75).sum())
wet_n = int((W["p_pct"] > 125).sum())
normal_mm = float(W["p_normal"].iloc[0])

c = st.columns(4)
c[0].metric("Winter temperature trend", f"{tt['per_decade']:+.1f} °C per decade",
            help="Straight-line trend through the 30 winter averages.")
c[0].caption(f"{sentence_case(trend_verdict(tt['p']))}.")
c[1].metric(f"Last winter ({winter_label(int(W.index[-1]))})", f"{last['t_anom']:+.1f} °C",
            help="Average of December, January and February, compared with the 30-winter normal.")
c[1].caption(f"{sentence_case(temp_words(last['t_anom']))}, {rain_words(last['p_pct'])}.")
c[2].metric("Normal winter rain and snow", f"{normal_mm:.0f} mm",
            help="Total for December to February. Snow is counted as the water it melts to.")
mon_norm = city.p_normal.groupby(city.time.month).first()
wettest_m = {12: "December", 1: "January", 2: "February"}[int(mon_norm.idxmax())]
c[2].caption(f"Most of it usually falls in {wettest_m} ({mon_norm.max():.0f} mm).")
c[3].metric("Dry winters", f"{dry_n} of {len(W)}",
            help="Winters that got less than 75% of the normal rain and snow.")
c[3].caption(f"{wet_n} winters were wet (more than 125% of normal).")

warm_txt = ("have **warmed**" if tt["per_decade"] > 0 else "have **cooled**")
summary = (f"Winters {here} {warm_txt} by about **{abs(tt['per_decade']):.1f} °C per decade** since the late 1990s "
           f"({trend_verdict(tt['p'])}). ")
if tp["p"] < 0.05:
    summary += (f"Rain and snow have **{'increased' if tp['per_decade'] > 0 else 'decreased'}** by about "
                f"{abs(tp['per_decade']):.0f} percentage points of normal per decade.")
else:
    summary += "Rain and snow show **no clear trend**: wet and dry winters come and go without a steady direction."
st.info(summary)

wdf = pd.DataFrame({
    "winter": [winter_label(int(y)) for y in W.index], "year": W.index.astype(int),
    "t": W["t_anom"].values, "pct": W["p_pct"].values, "mm": W["p_mm"].values,
    "dev": W["p_pct"].values - 100,
})
wdf["t_text"] = [temp_words(v) for v in wdf["t"]]
wdf["p_text"] = [f"{m:.0f} mm, {rain_words(p)}" for m, p in zip(wdf["mm"], wdf["pct"])]
wdf["trend"] = tt["intercept"] + tt["slope"] * wdf["year"]

left, right = st.columns(2)
with left:
    st.markdown("**Each winter's temperature compared with normal**")
    bars = alt.Chart(wdf).mark_bar(cornerRadiusEnd=3, size=12).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelExpr="datum.value % 5 == 0 ? datum.value : ''",
                                                     labelAngle=0)),
        y=alt.Y("t:Q", title="°C warmer (+) or colder (−) than normal"),
        color=alt.condition("datum.t > 0", alt.value(WARM), alt.value(COLD)),
        tooltip=[alt.Tooltip("winter:N", title="Winter"), alt.Tooltip("t_text:N", title="Temperature")])
    line = alt.Chart(wdf).mark_line(color="#0b0b0b", strokeWidth=1.5, strokeDash=[5, 3]).encode(
        x="year:O", y="trend:Q")
    st.altair_chart((bars + line).properties(height=260), width="stretch")
    st.caption("Red: warmer winter. Blue: colder winter. Dashed line: the long-term trend. "
               "The axis shows the year of January.")
with right:
    st.markdown("**Each winter's rain and snow compared with normal**")
    bars = alt.Chart(wdf).mark_bar(cornerRadiusEnd=3, size=12).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelExpr="datum.value % 5 == 0 ? datum.value : ''",
                                                     labelAngle=0)),
        y=alt.Y("dev:Q", title="% above (+) or below (−) normal"),
        color=alt.condition("datum.dev > 0", alt.value(WET), alt.value(DRY)),
        tooltip=[alt.Tooltip("winter:N", title="Winter"), alt.Tooltip("p_text:N", title="Rain and snow")])
    st.altair_chart(bars.properties(height=260), width="stretch")
    st.caption("Green: wetter winter. Brown: drier winter. Hover a bar for the amount in mm.")

with st.expander("See the numbers for every winter"):
    table = pd.DataFrame({
        "Winter": wdf["winter"], "Temperature vs normal (°C)": wdf["t"].round(1),
        "Rain and snow (mm)": wdf["mm"].round(0), "% of normal": wdf["pct"].round(0),
    }).iloc[::-1]
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download as CSV", table.to_csv(index=False), f"winters_{where.replace(' ', '_')}.csv",
                       "text/csv")

# =========================================================================== #
# 2. Why was a month unusual?
# =========================================================================== #
st.header("2. Why was a month unusual?")
st.markdown("Pick a month and get a plain explanation of what the big weather patterns were doing.")

n = len(city.time)
p_dev = (city.p_mm - city.p_normal).values
choices = {
    "Latest month": n - 1,
    "Coldest month": int(np.argmin(city.t_anom.values)),
    "Warmest month": int(np.argmax(city.t_anom.values)),
    "Wettest month": int(np.argmax(p_dev)),
    "Driest month": int(np.argmin(p_dev / np.maximum(city.p_normal.values, 1.0))),
}
pick = st.radio("Show me", list(choices) + ["Pick a month"], horizontal=True)
if pick == "Pick a month":
    t_idx = st.selectbox("Month", range(n), index=n - 1, format_func=lambda i: month_option_label(city, i))
else:
    t_idx = choices[pick]

e = explain_month(city, pats, t_idx)
head = f"**{e['month_label']} {here}** was **{temp_words(e['t_anom'])}**"
head += f", {e['rank_text']}." if e["rank_text"] else "."
if np.isfinite(e["p_pct"]):
    head += (f" Rain and snow: **{e['p_mm']:.0f} mm**, {rain_words(e['p_pct'])}; "
             f"a normal {e['date']:%B} brings {e['p_normal']:.0f} mm.")
else:
    head += f" Rain and snow: {e['p_mm']:.0f} mm ({e['date']:%B} is normally almost dry here)."
st.markdown(head)

st.markdown("**What the weather patterns were doing**")
for p in PATTERNS:
    s = e["strengths"][p["key"]]
    ct, cp = e["contrib_t"][p["key"]], e["contrib_p"][p["key"]]
    lk = city.links.loc[p["key"]]
    effect = []
    if is_real_link(lk["r_t"], lk["p_t"]) and abs(ct) >= 0.15:
        effect.append(f"{abs(ct):.1f} °C {'warmer' if ct > 0 else 'colder'}")
    if is_real_link(lk["r_p"], lk["p_p"]) and abs(cp) >= max(2.0, 0.08 * e["p_normal"]):
        effect.append(f"{abs(cp):.0f} mm {'more' if cp > 0 else 'less'} rain and snow")
    eff = (f" Effect {here}: about {' and '.join(effect)}." if effect
           else f" It made little difference {here} this month.")
    if s >= 1.0:
        meaning = f" When it is this {p['adj_up']} it usually brings {p['strong']}."
    elif s <= -1.0:
        meaning = f" When it is this {p['adj_down']} it usually brings {p['weak']}."
    else:
        meaning = ""
    st.markdown(f"- **{p['name']}**: {strength_words(s, p)} ({s:+.1f}, where ±1 is a typical "
                f"swing).{meaning}{eff}")
trend_c = e["contrib_t"]["trend"]
warming_clear = city.trend_t["p"] < 0.2 and city.trend_t["per_decade"] > 0
trend_name = "Long-term warming" if warming_clear else "Long-term trend"
if abs(trend_c) < 0.1:
    st.markdown(f"- **{trend_name}**: made almost no difference this month.")
elif warming_clear:
    st.markdown(f"- **{trend_name}**: winters {here} are slowly getting warmer, which made this month about "
                f"{abs(trend_c):.1f} °C {'warmer' if trend_c > 0 else 'colder'} than the 30-year average "
                f"{'(it is a recent month)' if trend_c > 0 else '(it is an older month, from before most of the warming)'}.")
else:
    st.markdown(f"- **{trend_name}**: winters {here} show no clear long-term trend, so it changed this month "
                f"by only about {abs(trend_c):.1f} °C.")
other = e["other_t"]
if abs(other) < 0.2:
    st.markdown("- **Other weather**: almost nothing is left over. The patterns and the long-term trend "
                "explain this month well.")
else:
    st.markdown(f"- **Other weather**: shorter-lived things the monthly patterns do not capture, such as fog, "
                f"cloud cover, snow on the ground or a single cold wave, made it another {abs(other):.1f} °C "
                f"{'warmer' if other > 0 else 'colder'}.")

a, b = st.columns([1, 1])
with a:
    keys = [p["key"] for p in PATTERNS] + ["trend"]
    labels = [p["name"] for p in PATTERNS] + [trend_name]
    vals = [e["contrib_t"][k] for k in keys] + [e["other_t"], e["t_anom"]]
    cdf = pd.DataFrame({"factor": labels + ["Other weather", "Total difference"], "value": vals,
                        "kind": ["part"] * (len(labels) + 1) + ["total"]})
    cdf["text"] = [f"{v:+.1f} °C" for v in cdf["value"]]
    cdf["colour"] = ["#52514e" if k == "total" else (WARM if v > 0 else COLD)
                     for k, v in zip(cdf["kind"], cdf["value"])]
    order = list(cdf["factor"])
    ch = alt.Chart(cdf).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("factor:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("value:Q", title="°C compared with normal"),
        color=alt.Color("colour:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("text:N", title="Effect")])
    st.markdown("**Where the temperature difference came from**")
    st.altair_chart(ch.properties(height=230), width="stretch")
    st.caption(f"The patterns plus the long-term trend explain about {100 * city.r2_t:.0f}% of the month-to-month "
               f"temperature ups and downs {here}. The rest is everyday weather.")
with b:
    show(fig_public_map(
        pats.slp_anom[t_idx], pats.lat, pats.lon, MAP_EXTENT,
        f"Air pressure in {e['month_label']} compared with normal",
        "lower: clouds, rain, snow", "higher: clear, dry, cold nights",
        point=city.latlon, point_label=None if city.latlon is None else place, outline=pakistan()))

# =========================================================================== #
# 3. The patterns
# =========================================================================== #
st.header("3. The weather patterns behind Pakistan's winters")
st.markdown("Computers can find the few big, repeating shapes in 30 years of air-pressure maps. These are the "
            "three that matter most for Pakistan, with the share of all winter pressure changes over the "
            "region that each one accounts for.")
cols = st.columns(3)
latest = len(pats.time) - 1
for col, p in zip(cols, PATTERNS):
    with col:
        st.subheader(p["name"])
        st.caption(f"{pats.var_pct[p['key']]:.0f}% of the region's winter pressure changes")
        show(fig_public_map(STRONG * pats.maps[p["key"]], pats.lat, pats.lon, MAP_EXTENT,
                            "Pressure change in a strong month", "lower", "higher",
                            point=city.latlon, outline=pakistan(), width=4.6))
        st.markdown(p["what"])
        st.markdown(f"**When strong:** {p['strong']}.")
        lk = city.links.loc[p["key"]]
        main = "r_t" if p["var"] == "T" else "r_p"
        pv = "p_t" if p["var"] == "T" else "p_p"
        what = "temperature" if p["var"] == "T" else "rain and snow"
        st.markdown(f"**Here:** it {link_sentence(where, lk[main], lk[pv], what)}.")
        s_now = float(pats.index[p["key"]].iloc[latest])
        st.markdown(f"**In {pats.time[latest]:%B %Y}:** {strength_words(s_now, p)}.")

lk = city.links.loc["ao"]
if is_real_link(lk["r_t"], lk["p_t"]) or is_real_link(lk["r_p"], lk["p_p"]):
    ao_here = (f"Here it {link_sentence(where, lk['r_t'], lk['p_t'], 'temperature')}, and it "
               f"{link_sentence(where, lk['r_p'], lk['p_p'], 'rain and snow')}.")
else:
    ao_here = (f"For {where}, it explains under {max(1, round(100 * max(lk['r_t'] ** 2, lk['r_p'] ** 2)))}% of the "
               f"swings in winter temperature or rain, too little to tell apart from chance.")
st.info(f"**What about the Arctic Oscillation?** {AO_TEXT} {ao_here}")

# =========================================================================== #
# 4. Uses and limits
# =========================================================================== #
st.header("4. Who can use this, and how")
u = st.columns(2)
with u[0]:
    st.markdown(
        "- **Farmers and agriculture officers.** Two pages are made for you. One compares 18 common crops "
        "against 30 seasons of temperature and rain at your farm and shows which the climate suits best and how "
        "much irrigation each needs. The other shows how much rabi rain you can count on, how often seasons "
        "fail, what followed a dry start in past years, and how hot March and April have become.")
    st.page_link("views/crop_advisor.py", label="Which crop suits my land?", icon=":material/agriculture:")
    st.page_link("views/farmers.py", label="Rain and heat in the rabi season", icon=":material/water_drop:")
    st.markdown(
        "- **Teachers and students.** Real evidence of climate change for your own city, and a way to see how "
        "large weather systems shape local weather.")
u[1].markdown(
    "- **Journalists and the curious.** Put a cold wave or dry spell in context: was it really the coldest "
    "in decades, and what was the weather doing across Asia at the time?\n"
    "- **Families and planners.** Cold winters raise gas and electricity demand and health risks for the "
    "elderly. Section 1 shows how much milder recent winters have been.")

with st.expander("What this tool can and cannot tell you"):
    st.markdown(
        "- **It is not a forecast.** The patterns explain the weather of the same month. We tested whether "
        "they predict the following month, and they do not.\n"
        "- **Each place is an area about 250 km across.** Mountain towns are smoothed with their surroundings, "
        "so compare changes over time rather than exact temperatures.\n"
        f"- **'Normal' means the average of the last 30 winters** ({winter_label(first_w)} to {winter_label(last_w)}).\n"
        "- **Data:** NOAA NCEP/NCAR Reanalysis (air pressure and temperature) and GPCP satellite-and-gauge "
        "rainfall. Rain includes snow, counted as the water it melts to.\n"
        "- **30 winters is a short record.** Trends marked 'not enough to be sure' could still be chance.\n"
        "- **The method** is Empirical Orthogonal Function (EOF) analysis, from Hannachi (2004). The page "
        "'The maths' in the sidebar shows every step.")
