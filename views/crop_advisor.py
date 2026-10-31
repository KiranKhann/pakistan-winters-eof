"""Crop advisor page: which crops suit the climate of a farm, and how much water each needs."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from eofkit import crops as C
from eofkit.farm import season_label
from eofkit.plots import RED, RAIN_COLORS

WET_C, DRY_C = RAIN_COLORS[8], RAIN_COLORS[2]
LIGHT_WET = RAIN_COLORS[6]
INK2 = "#52514e"
FIT_COLORS = {"Very good": "#1a7a66", "Good": "#3fa98f", "Fair": "#c3c2b7", "Poor": "#e3c08a", "Very poor": "#c48a3a"}


@st.cache_data(show_spinner="Reading 30 years of temperature and rainfall for your area...")
def get_climate(lat: float, lon: float) -> pd.DataFrame:
    df = C.place_climate(lat, lon)
    df.attrs["source"] = df.attrs.get("source", "")
    return df


@st.cache_data(show_spinner="Checking every crop against 30 seasons...")
def get_ranking(lat: float, lon: float, water: str, seasons: tuple) -> pd.DataFrame:
    return C.rank_crops(get_climate(lat, lon), water, seasons)


@st.cache_data(show_spinner=False)
def get_crop(lat: float, lon: float, water: str, key: str):
    r = C.evaluate_crop(get_climate(lat, lon), key, water)
    return r.seasons, r.summary


# =========================================================================== #
# Header and farm location
# =========================================================================== #
st.title("For farmers: which crop suits your land?")
st.markdown(
    "Tell us where your farm is, which season you are planning for, and how much irrigation water you can give. "
    "The page checks every common crop against the **last 30 seasons of temperature and rainfall** at your farm "
    "and shows which crops the climate has suited best, how often each one would have done well, and how much "
    "water each would need.")

how = st.radio("Where is your farm?", ["Choose a district", "Enter my own location"], horizontal=True)
if how == "Choose a district":
    a, b = st.columns(2)
    provinces = list(C.FARM_PLACES)
    prov = a.selectbox("Province", provinces, index=0)
    districts = list(C.FARM_PLACES[prov])
    default = districts.index("Faisalabad") if "Faisalabad" in districts else 0
    place = b.selectbox("District", districts, index=default)
    lat, lon = C.FARM_PLACES[prov][place]
else:
    a, b = st.columns(2)
    lat = a.number_input("Latitude (°N)", min_value=23.6, max_value=37.0, value=31.42, step=0.05, format="%.2f",
                         help="You can read this from Google Maps: press and hold on your farm.")
    lon = b.number_input("Longitude (°E)", min_value=60.9, max_value=77.8, value=73.08, step=0.05, format="%.2f")
    place = f"{lat:.2f}°N, {lon:.2f}°E"
    spread = C.terrain_spread(lat, lon)
    if np.isfinite(spread) and spread > 4:
        st.warning(f"Your location is next to hills or mountains: average temperatures in the surrounding 50 km "
                   f"differ by {spread:.0f} °C. The data may be several degrees too cold for fields in the valley, "
                   f"so treat the advice below with caution.")
    else:
        st.caption("Advice is most reliable on the plains and in broad valleys. Mountain areas (Gilgit-Baltistan, "
                   "AJK, Swat, upland Balochistan) are not well described by this 50 km data.")

clim = get_climate(round(lat, 2), round(lon, 2))
src = clim.attrs.get("source", "")
first_y, last_y = clim.index[0].year, clim.index[-1]
st.divider()

a, b = st.columns([1, 1])
season_pick = a.radio("Season", ["Rabi", "Kharif"],
                      format_func={"Rabi": "Rabi: winter crops, sown October and November",
                                   "Kharif": "Kharif: summer crops, sown April to July"}.get)
water = b.select_slider("Irrigation water you can give the crop", options=list(C.WATER_OPTIONS),
                        value="w6", format_func=lambda k: C.WATER_OPTIONS[k]["short"],
                        help=f"One watering is taken as about {C.WATERING_MM} mm (3 inches) of water.")
b.caption(C.WATER_OPTIONS[water]["label"] +
          ("" if water in ("rain", "full") else f", roughly {C.WATER_OPTIONS[water]['mm']:.0f} mm in the season."))
seasons = ("Rabi",) if season_pick == "Rabi" else ("Kharif", "Year-round")
R = get_ranking(round(lat, 2), round(lon, 2), water, seasons)
R["problem"] = R.apply(C.limit_text, axis=1)
R["trend"] = R.apply(C.trend_text, axis=1)
R["fit_class"] = [C.fit_class(x) for x in R["median_fit"]]
R["extra"] = [C.waterings(x) for x in R["irrigation_needed"]]
n_seasons = int(R["n"].max())

st.subheader(f"{season_pick} crops for {place}")

def crop_card(r) -> None:
    st.markdown(f"#### {r['name']}")
    st.caption(f"{r['local']} · in the field {C.months_text(r['months'])}")
    st.metric("Good seasons", f"{r['good']} of {r['n']}",
              help="Seasons in which both temperature and water suited the crop well "
                   "(ECOCROP score of 0.6 or more).")
    lines = []
    if water == "rain":
        lines.append(f"**Water:** needs about {r['need']:.0f} mm; rain in the season gives about "
                     f"{r['eff_rain']:.0f} mm, and the rest comes from moisture stored in the soil.")
    elif r["extra"] == 0:
        lines.append(f"**Water:** needs about {r['need']:.0f} mm, and rain usually gives enough.")
    else:
        line = (f"**Water:** needs about {r['need']:.0f} mm. Rain gives about {r['eff_rain']:.0f} mm, so "
                f"plan on about **{r['extra']} waterings** ({r['irrigation_needed']:.0f} mm).")
        avail_mm = C.WATER_OPTIONS[water]["mm"]
        if np.isfinite(avail_mm) and r["irrigation_needed"] > avail_mm:
            share = min(100, 100 * (r["eff_rain"] + avail_mm) / r["need"])
            line += (f" With the water you have it gets about **{share:.0f}%** of what it needs, so expect "
                     f"{'somewhat ' if share >= 75 else ''}lower yields.")
        lines.append(line)
    if r["good"] < r["n"]:
        lines.append(f"**In the bad seasons:** {r['problem']}.")
    lines.append(f"**Over 30 years:** {r['trend']}.")
    st.markdown("\n\n".join(lines))
    if C.CROPS[r["key"]].get("note"):
        st.caption(C.CROPS[r["key"]]["note"])

annual_rain = float(clim["p"].groupby(clim.index.year).sum().iloc[-31:-1].median())
if water == "rain" and annual_rain < 350:
    st.info(f"This is very dry country (about {annual_rain:.0f} mm of rain in a typical year). Rain-fed farmers here "
            "use drought-adapted practice, such as sandy soils that hold moisture and sowing only after good rain, "
            "so crops like gram (chana), bajra and guar succeed more often than the generic FAO limits used here "
            "suggest. The drought risk shown is real, but treat 'not suitable' as 'often fails' rather than 'never grows'.")

best_tier = int(R["tier"].min())
if best_tier > 0:
    st.warning("No crop did well in at least 8 seasons out of 10 here with this much water. The crops closest to "
               "suiting your land are shown first. More irrigation, or the other season, may work better.")
best = R[R["tier"] == best_tier]
st.markdown(f"**{C.TIERS[best_tier][1]}** ({C.TIERS[best_tier][2]}). Crops that need less extra water come first.")
for i in range(0, len(best), 3):
    cols = st.columns(3)
    for col, (_, r) in zip(cols, best.iloc[i:i + 3].iterrows()):
        with col:
            crop_card(r)
for ti in range(best_tier + 1, len(C.TIERS)):
    grp = R[R["tier"] == ti]
    if grp.empty:
        continue
    items = [f"{r['name']} ({r['local']}): {r['good']} of {r['n']}" +
             ("" if ti < 3 else f", {r['problem']}") for _, r in grp.iterrows()]
    st.markdown(f"**{C.TIERS[ti][1]}** ({C.TIERS[ti][2]}): " + "; ".join(items) + ".")

st.markdown(f"**All {season_pick.lower()} crops compared**")
table = pd.DataFrame({
    "Crop": [f"{n} ({l})" for n, l in zip(R["name"], R["local"])],
    "Verdict": [C.TIERS[t][1] for t in R["tier"]],
    "In the field": [C.months_text(m) for m in R["months"]],
    "Good seasons": R["good"].astype(int),
    "Fit": R["fit_class"],
    "Problem in bad seasons": R["problem"],
    "Water needed (mm)": R["need"].round(-1).astype(int),
    "From rain (mm)": R["eff_rain"].round(-1).astype(int),
    "Extra waterings": R["extra"].astype(int),
    "Over 30 years": R["trend"],
})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Good seasons": st.column_config.ProgressColumn("Good seasons", min_value=0, max_value=n_seasons,
                                                    format="%d of " + str(n_seasons)),
})
st.download_button("Download this comparison (CSV)", table.to_csv(index=False),
                   f"crop_comparison_{season_pick}_{place.split(' (')[0].replace(' ', '_').replace(',', '')}.csv",
                   "text/csv")

# ---- water budget chart ----
st.markdown("**Extra water each crop needs, beyond the rain**")
avail = C.WATER_OPTIONS[water]["mm"]
wdf = pd.DataFrame({"crop": R["name"], "extra_mm": R["irrigation_needed"].round(0), "waterings": R["extra"]})
if np.isfinite(avail):
    short = R["need"].values - (R["eff_rain"].values + avail)
    wdf["fits"] = np.where(short <= 0, "Within your water",
                           np.where(short <= 0.25 * R["need"].values, "A little short", "Too little water"))
else:
    wdf["fits"] = "Within your water"
wdf["tip"] = [f"about {w} waterings ({m:.0f} mm)" for w, m in zip(wdf["waterings"], wdf["extra_mm"])]
bars = alt.Chart(wdf).mark_bar(cornerRadiusEnd=3).encode(
    y=alt.Y("crop:N", sort=list(wdf["crop"]), title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False),
            scale=alt.Scale(paddingInner=0.35)),
    x=alt.X("extra_mm:Q", title="Extra water needed in the season (mm)"),
    color=alt.Color("fits:N", title=None,
                    scale=alt.Scale(domain=["Within your water", "A little short", "Too little water"],
                                    range=[WET_C, RAIN_COLORS[3], DRY_C]), legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("crop:N", title="Crop"), alt.Tooltip("tip:N", title="Extra water")])
layers = bars
if np.isfinite(avail) and avail > 0:
    layers = bars + alt.Chart(pd.DataFrame({"x": [avail]})).mark_rule(color=INK2, strokeDash=[5, 3],
                                                                     strokeWidth=1.5).encode(x="x:Q")
st.altair_chart(layers.properties(height=34 * len(wdf)), width="stretch", height=34 * len(wdf) + 90)
if np.isfinite(avail):
    st.caption(f"Dashed line: the water you said you can give (about {avail:.0f} mm). 'A little short' means the "
               f"crop would get at least three quarters of what it needs; 'too little water' means less than that. "
               f"Paddy rice includes about 1,000 mm to flood the field and replace seepage.")
else:
    st.caption("Paddy rice includes about 1,000 mm to flood the field and replace seepage.")

# ---- one crop up close ----
st.markdown("**Look closer at one crop**")
pick = st.selectbox("Crop", list(R["key"]), index=0,
                    format_func=lambda k: f"{C.CROPS[k]['name']} ({C.CROPS[k]['local']})")
S, summ = get_crop(round(lat, 2), round(lon, 2), water, pick)
cp = C.CROPS[pick]
par = C.ecocrop().loc[pick]
months = C.crop_months(cp)
tt = pd.DataFrame({"m": clim.index.month, "t": clim["t"].values})
tt = tt[tt["m"].isin(months)]
stats_m = tt.groupby("m")["t"].agg(avg="mean", lo=lambda v: np.percentile(v, 10), hi=lambda v: np.percentile(v, 90))
tdf = pd.DataFrame({"month": [C.MONTH_ABBR[m] for m in months],
                    "avg": [stats_m.loc[m, "avg"] for m in months],
                    "lo": [stats_m.loc[m, "lo"] for m in months],
                    "hi": [stats_m.loc[m, "hi"] for m in months],
                    "ok_lo": par.TOPMN, "ok_hi": par.TOPMX})
crit = cp.get("critical", list(range(len(months))))
tdf["sensitive"] = [i in crit for i in range(len(months))]
tdf["tip"] = [f"{v:.1f} °C on average (usually {l:.0f} to {h:.0f} °C)" for v, l, h in zip(tdf["avg"], tdf["lo"], tdf["hi"])]
order = list(tdf["month"])
x = alt.X("month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
band = alt.Chart(tdf[tdf["sensitive"]]).mark_bar(color=LIGHT_WET, opacity=0.55).encode(
    x=x, y=alt.Y("ok_lo:Q", title="Average temperature (°C)"), y2="ok_hi:Q")
rng = alt.Chart(tdf).mark_rule(color=INK2, strokeWidth=2).encode(x=x, y="lo:Q", y2="hi:Q")
pts = alt.Chart(tdf).mark_point(filled=True, size=90, color="#0b0b0b").encode(
    x=x, y="avg:Q", tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("tip:N", title=place)])
limits = pd.DataFrame({"y": [par.TMIN, par.TMAX], "what": ["too cold below this", "too hot above this"]})
lim_rule = alt.Chart(limits).mark_rule(color=RED, strokeDash=[4, 3]).encode(y="y:Q", tooltip=["what:N"])
left, right = st.columns(2)
with left:
    st.altair_chart((band + lim_rule + rng + pts).properties(height=280), width="stretch")
    all_sensitive = bool(tdf["sensitive"].all())
    st.caption(f"Green boxes: the temperatures {cp['name'].lower()} likes best ({par.TOPMN:.0f} to {par.TOPMX:.0f} °C, "
               f"FAO ECOCROP)" + ("." if all_sensitive else
               ", shown in its sensitive months. In the other months it only needs to stay between the red lines.")
               + f" Red dashed lines: its limits ({par.TMIN:.0f} and {par.TMAX:.0f} °C). Black dots: the average at "
               f"your farm; grey lines: the usual range in 8 years out of 10.")
with right:
    sdf = pd.DataFrame({"year": S.index.astype(int), "fit": S["fit"].values})
    sdf["class"] = [C.fit_class(v) for v in sdf["fit"]]
    sdf["season"] = [season_label(y) if cp["season"] == "Rabi" else str(y) for y in sdf["year"]]
    sdf["why"] = [("no problem" if l == "none" else
                   {"cold": f"too cold in {C.MONTH_NAME[m]}", "heat": f"too hot in {C.MONTH_NAME[m]}",
                    "water": "not enough water"}[l]) for l, m in zip(S["limit"], S["worst_month"])]
    fb = alt.Chart(sdf).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelExpr="datum.value % 5 == 0 ? datum.value : ''",
                                                     labelAngle=0)),
        y=alt.Y("fit:Q", title="How well the season suited the crop", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("class:N", title=None, scale=alt.Scale(domain=list(FIT_COLORS), range=list(FIT_COLORS.values())),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("class:N", title="Fit"),
                 alt.Tooltip("why:N", title="Main limit")])
    st.altair_chart(fb.properties(height=280), width="stretch")
    st.caption("Each bar is one season (harvest year). 1 means everything suited the crop; 0 means it would have failed.")

with st.expander("How the advice is worked out, and what it leaves out"):
    st.markdown(
        "- **Temperature** uses FAO's ECOCROP crop database: each crop's ideal and limiting average temperatures, "
        "checked month by month while the crop is in the field. The worst month decides the season.\n"
        "- **Water, rain only:** ECOCROP's rainfall limits describe the yearly rain where each crop is grown, so "
        "they are compared with the 12 months up to harvest, which includes monsoon rain stored in the soil.\n"
        "- **Water, with irrigation:** FAO's typical crop water need is compared with the useful part of the rain "
        "(FAO's effective-rainfall rule) plus the waterings you can give. Mustard, canola and sesame have no FAO "
        "value, so a similar crop's value is used.\n"
        "- **Good season** means both temperature and water suited the crop well (a score of 0.6 or more on "
        "ECOCROP's 0 to 1 scale).\n"
        f"- **Climate data:** {src}, NOAA, about 50 km across, seasons up to {last_y:%B %Y}.\n"
        "- **Not included:** soil type and salt, water quality, seed variety, pests and disease, input costs, and "
        "market or support prices. These often decide which crop pays best. Please check with your local "
        "agriculture extension office before deciding.\n"
        "- **Very dry rain-fed areas** (the Thal, Tharparkar): checked against real cropping, the advice is too "
        "harsh on drought-adapted crops such as gram, bajra and guar, which farmers grow there on stored moisture.\n"
        "- **Quetta** grows winter-hardy wheat varieties that the generic ECOCROP limits do not describe well, so "
        "rabi results there are too pessimistic. Mountain districts (Gilgit-Baltistan, AJK, Swat, upland "
        "Balochistan) are left out of the list because the 50 km data are several degrees too cold there.")

st.page_link("views/farmers.py", label="Rain and heat in the rabi season: how much rain you can count on",
             icon=":material/water_drop:")


st.caption("A record of past seasons, not a forecast. For forecasts, follow the Pakistan Meteorological Department's "
           "agromet advisories and your local agriculture extension office.")
