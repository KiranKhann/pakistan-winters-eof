# Pakistan's Winters: an EOF climate explorer

A reimplementation of Sections 1 to 3 of **A. Hannachi (2004), "A primer for EOF analysis of climate data"**, on NOAA monthly NetCDF data. The method is then turned into an app that anyone in Pakistan can use to understand their own winters.

The project has four parts:

- a small Python package, `eofkit`, that follows the paper equation by equation
- scripts that reproduce the paper's three figures and run the application
- a verification script for the mathematics
- a four-page Streamlit app: a plain-language page for the public, two pages for farmers (which crop to grow, and rain and heat in the rabi season), and a maths page for the course

## For everyone: what the app does

Open the app and it starts on **Pakistan's winters**. No maths is needed.

1. **Choose a place.** Pick one of 15 cities and farming districts, or all of Pakistan.
2. **See how winters have changed.** Four headline numbers and two charts show the warming trend, last winter, normal rain and snow, and how often winters are dry. Every trend says plainly whether it is clear or could be chance.
3. **Ask why a month was unusual.** Pick the latest, coldest, warmest, wettest or driest month, or any month since December 1996. The page explains what each big weather pattern was doing, how much it changed the temperature or rain at that place, and how much was everyday weather.
4. **Meet the patterns.** Three named patterns shape Pakistan's winters: the *Siberian cold push*, the *Western storms* (western disturbances) and the *Caspian high*. Each comes with a map, a plain description, and how much it matters for the chosen place.

Who it helps:

- **Farmers and agriculture officers** have two pages of their own, described below.
- **Teachers and students** get real evidence of climate change for their own city.
- **Journalists** can check whether a cold wave or dry spell was really record-breaking.
- **Families and planners** can see how much milder recent winters have been.

The page also says clearly what it cannot do. It is not a forecast, because the patterns do not predict the following month. Each place is an area about 250 km across.

Behind the page, the three patterns are local EOFs of winter air pressure. Each month's explanation is a least-squares fit of the place's anomaly on the patterns plus a linear trend. EOF principal components are uncorrelated (paper Sec. 3.2.2), so each pattern's effect can be read separately and the effects add up.

## For farmers: which crop suits your land?

The page **For farmers: which crop?** helps a farmer choose what to grow. The farmer picks:

- **the farm,** from 37 district towns in Punjab, Sindh, Khyber Pakhtunkhwa and Balochistan, or by typing a latitude and longitude
- **the season,** rabi or kharif
- **the irrigation water they can give,** from rain only (barani) up to as much as the crop needs, in steps of 3, 6 and 10 waterings

The page then checks 18 common crops against each of the last 30 seasons at that farm. Each crop is placed in one of four groups: suits your land well, possible but some seasons go badly, risky, or not suitable here. Inside a group, crops that need less extra water come first, because they are cheaper and safer. For each crop it shows:

- how many of the 30 seasons would have gone well, and what went wrong in the bad ones, such as "too hot in March"
- how much water it needs, how much rain gives, and how many waterings to plan, with a chart marking crops that need more water than the farmer has
- whether conditions have been getting better or worse over the 30 years
- a close-up of any crop: the farm's monthly temperatures against the crop's ideal range, and a bar for every season

**How it works.** Temperature limits for every crop come from FAO's ECOCROP database, read from the Recocrop package (R. Hijmans) into [data/ecocrop_pakistan_crops.csv](data/ecocrop_pakistan_crops.csv). Each month the crop is in the field is scored from 0 to 1, and the worst month decides the season. For rabi crops and sugarcane, the full ideal range applies only in the sensitive months: establishment and flowering-to-grain-filling, tuber forming for potato, and the main growth months for sugarcane. The other months need only stay within the crop's survival limits, because wheat and gram grow slowly but safely through a cool January.

Water is scored in two ways:

- **Rain-fed farms:** ECOCROP's rainfall limits are compared with the 12 months up to harvest, which includes stored monsoon moisture.
- **Irrigated farms:** FAO's typical crop water need (Irrigation Water Management manual 3, Table 14) is compared with effective rain, using FAO's rule, plus the waterings the farmer can give. One watering is taken as 75 mm. Paddy rice includes FAO's allowance for flooding the field and replacing seepage.

**Climate data.** Temperature comes from NOAA's gauge-based GHCN_CAMS and rainfall from PREC/L, both monthly at about 50 km. The temperatures were checked against known normals for 20 towns. They agree within about 2 °C on the plains and in broad valleys, but run 3 to 17 °C too cold in mountain towns. Gilgit, Skardu, Muzaffarabad, Swat, Zhob, Loralai, Khuzdar and Turbat are therefore left out of the district list. A typed location next to high ground gets an automatic warning.

**Checked against where crops really grow.** [scripts/validate_crops.py](scripts/validate_crops.py) tests 22 cases against Pakistan's cropping map. Expected crops include wheat in Faisalabad, cotton in Multan and Sanghar, rice in Gujranwala and Larkana, barani wheat and groundnut in Chakwal, and bajra in Tharparkar. It also tests crops that should be rejected, such as rain-fed rice or sugarcane, and rice in Quetta. **19 of 23 cases match.** The four misses are informative. Cotton in Multan with only 10 waterings comes out risky, because cotton needs about 12 there; with enough water it suits the land in 30 of 30 seasons. Rain-fed gram in the Thal, and bajra and guar in Tharparkar, are judged too harshly. These drought-adapted systems grow on sandy soils that store moisture, below ECOCROP's generic rainfall minimums. The page warns rain-fed farmers in very dry areas about this.

**What it leaves out.** Soil, salinity, water quality, seed variety, pests, input costs and crop prices are not included, and these often decide which crop pays best. The page says so and points farmers to their agriculture extension office. It is a record of the last 30 seasons, not a forecast.

## For farmers: the rabi season page

Wheat and other rabi crops are sown in October and November and harvested in April and May. In rain-fed (barani) areas, winter rain is the crop's water supply. The page **For farmers (rabi season)** turns 30 seasons of rainfall and temperature into planning numbers for any area. It defaults to Chakwal in the rain-fed Potohar. Farmers can choose their own season months, from sowing to grain filling.

1. **What a normal season brings.** It shows normal season rain and the **rain you can count on**, the amount reached in 3 seasons out of 4. It also shows how many seasons were dry (below 75% of normal) and whether that is changing. A month-by-month chart and plain advice follow. For example, Chakwal gets 62% of its November–March rain in February and March, while November and December are almost dry in about 4 years out of 10.
2. **The season so far.** Partway through a season, the farmer chooses how far in they are and how the rain has been, for example from the PMD bulletin. The page then counts what happened in past seasons that started the same way. The link is weak and has faded since about 2011, and the page says so plainly. A dry start shifts the odds only slightly.
3. **Heat when the grain fills.** For February, March or April it shows the warming trend, how many months were "hot" (more than 1.5 °C above normal), and the hottest years. Across Pakistan, the hottest Marches were 2010, 2004 and 2022.
4. **Take it with you.** A one-page text fact sheet and a CSV of every season can be downloaded for an extension office or a farmers' meeting.

The page is a record of the last 30 seasons, not a forecast, and it points farmers to the Pakistan Meteorological Department's agromet advisories for forecasts. Rainfall comes from GPCP v2.3 and temperature from NCEP/NCAR Reanalysis 1, both at 2.5°. This page is kept as originally built. Note that GPCP roughly doubles rainfall in the dry south compared with the 0.5° PREC/L data used by the crop advisor: Jacobabad gets 240 mm a year against about 100 mm measured, and Multan 369 mm against about 190 mm. Temperature records end in February 2026, so March and April data run to 2025.

## How the professor's instructions are met

| Instruction | What was done |
|---|---|
| Monthly data category | NOAA PSL monthly-mean NetCDF files only |
| Data after 1981, at least 30 years, or the last 30 years | The last 30 complete winters, December 1996 to February 2026 (90 DJF months) |
| Find out how to load .nc data | `xarray` with the `netCDF4` engine, in `eofkit/data.py` |
| Reimplement the paper up to the 3rd heading | Sections 2.1 to 3.5: data matrix, anomalies, area weighting, EOFs by SVD, North's error bars, truncation and reconstruction |
| Graphs and plots | The paper's Figures 1 to 3, plus validation, application and reconstruction figures in `figures/` |
| Show sir before submission | The app is built for a live walkthrough, and `scripts/verify_math.py` shows the checks |

## Quick start (Windows)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/verify_math.py
.venv/Scripts/python.exe scripts/replicate_paper.py
.venv/Scripts/python.exe scripts/application_pakistan.py
.venv/Scripts/python.exe -m streamlit run app.py
```

The repository already contains everything the app needs. `data/compact/` holds 21 MB of regional copies of the NOAA files: the Northern Hemisphere for pressure and the area around Pakistan for temperature and rainfall. They keep every month, so all results are identical to the full files. To fetch the full global files instead (about 600 MB), run `scripts/download_data.py`; the code then uses them automatically. `scripts/make_compact_data.py` rebuilds the compact copies.

Python 3.12 is used because Cartopy, the map library, has reliable wheels for it. The app opens at http://localhost:8501 on the public page. The two farmers' pages and the maths page are in the sidebar.

## Deploy the app online (Streamlit Community Cloud, free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns this repository.
2. Choose **Create app**, pick this repository, branch `main`, and main file `app.py`.
3. Under **Advanced settings**, choose Python 3.12, then **Deploy**.

The first start takes a few minutes while the packages install. The bundled compact data means no downloads are needed. Cartopy fetches its map outlines from Natural Earth on first use.

The paper itself, Hannachi (2004), *A primer for EOF analysis of climate data*, is not included because it is the author's copyright. It is available from the author's ResearchGate page.

## Results

### 1. Replication of Section 3.5

Winter (DJF) monthly sea level pressure north of 20°N, NCEP/NCAR Reanalysis 1, 2.5° grid (p = 4176 grid points).

| Run | Months | EOF1 | EOF2 | EOF1 and EOF2 separable? | Correlation of PC1 with NOAA's AO index |
|---|---|---|---|---|---|
| Paper, 1948-2000 | 159 | 21% | 13% | yes | not reported |
| This code, 1948-2000 | 159 | 21.6% ± 2.4 | 13.2% ± 1.5 | yes | 0.98 |
| This code, last 30 winters | 90 | 24.1% ± 3.6 | 13.7% ± 2.0 | yes | 0.96 |

The ± values are the North et al. (1982) error bars, eq. (12). The correlation is with NOAA CPC's official monthly AO index. Its sign is negative only because the paper draws EOF1 with high pressure over the pole.

EOF1 is the Arctic Oscillation. It has one centre over the Arctic and opposite centres over the North Atlantic and Mediterranean and over the North Pacific. EOF2 has separate North Pacific and North East Atlantic centres. Both match the paper's description.

### 2. A finding about the paper's weighting

The paper's eq. (5) multiplies the data by cos(latitude). Its reported variances are reproduced only when the data is multiplied by **√cos(latitude)**. That choice weights the covariance matrix by grid-cell area, which is the standard convention.

| Weighting of the data | EOF1, 1948-2000 | EOF2, 1948-2000 |
|---|---|---|
| cos(lat), eq. (5) as printed | 18.8% | 14.9% |
| √cos(lat) | 21.6% | 13.2% |
| none | 35.9% | 10.1% |
| Paper | 21% | 13% |

The project therefore uses √cos(lat) by default. The literal eq. (5) is one click away in the app. The unweighted run shows the polar artefact that Section 2.2 warns about (`figures/figS_unweighted_loadings.png`).

### 3. Application: Pakistan's winters

Temperature comes from NCEP/NCAR Reanalysis 1. Rainfall comes from GPCP v2.3. Pakistan averages use the true national border, with each grid cell weighted by the share of it inside Pakistan.

**The Arctic Oscillation does not control Pakistan's winters.** Over the 30 winters its correlation with Pakistan is small and not significant:

| | r, monthly | p | Variance explained |
|---|---|---|---|
| AO vs temperature | −0.09 | 0.41 | 0.8% |
| AO vs rainfall | −0.02 | 0.86 | 0.0% |

The maps show why. A positive AO warms Central Asia and cools the Middle East, and Pakistan sits on the neutral line between the two. The result does not change with detrending, winter means, a one-month lead, or the northern-Pakistan box.

**Pakistan-centred EOFs do track Pakistan's winters.** Section 3.4 warns that EOFs depend on the domain. Repeating the same analysis on 10-50°N, 40-100°E gives local modes that matter:

| Local mode | Variance | Pattern | Link to Pakistan |
|---|---|---|---|
| EOF1 | 41% | High pressure over Central Asia, a southward push of the Siberian High | Colder months: r = −0.40, p < 0.001, 16% of variance |
| EOF2 | 22% | Centres west and east of Pakistan along the subtropical westerly track | Drier months: r = −0.38, p < 0.001, 14% of variance |

These links survive detrending and changes of domain. The rainfall link moves between PC2 and PC3 as the domain changes, which is the domain dependence the paper describes. The links are diagnostic only: they explain the same month's weather, and their one-month-lead correlations are not significant.

## The maths page of the app

| Tab | Contents |
|---|---|
| Paper replication | Figures 1 to 3, eigenvalue table, downloads; period, weighting and sign controls |
| Compression and reconstruction | Any month rebuilt from k EOFs (eq. 11), RMSE against k, storage saved |
| AO and Pakistan | Correlation and composite maps, hover scatter plots, winter bars, statistics |
| Pakistan-centred modes | Adjustable domain, local EOF maps, mode-by-target correlation table |
| Method and data | The equations, the built-in checks, data sources and caveats |

## Project layout

```text
eofkit/
  data.py            NetCDF loading, region and season selection    (paper Sec. 2.1)
  eof.py             anomalies, weighting, SVD EOFs, North's rule  (paper Sec. 2-3)
  pipeline.py        one-call wrappers
  teleconnection.py  AO index, correlation and composite maps, Pakistan mask
  application.py     Pakistan analysis shared by the script and the app
  public.py          plain-language layer: cities, named patterns, month explanations
  farm.py            rabi-season statistics: dependable rain, dry seasons, season-so-far, heat
  crops.py           crop advisor: ECOCROP temperature fit, FAO water need, 30-season scoring
  plots.py           all figures
scripts/
  download_data.py   fetches the full NetCDF files (optional)
  make_compact_data.py  writes the small regional copies in data/compact/
  verify_math.py     10 numerical checks of the paper's properties
  validate_crops.py  crop advisor checked against where crops are really grown
  replicate_paper.py paper Figures 1-3, validation, weighting comparison
  application_pakistan.py
app.py               Streamlit entry point (navigation)
views/
  public.py          "Pakistan's winters" page, for everyone
  crop_advisor.py    "For farmers: which crop?" page
  farmers.py         "For farmers (rabi season)" page
  maths.py           "The maths" page, for the course
data/
  compact/           21 MB regional copies of the NOAA files (committed)
  ecocrop_pakistan_crops.csv  FAO ECOCROP limits for the 18 crops
figures/  results/   generated outputs
```

## Data

| Variable | Source | Period | File |
|---|---|---|---|
| Sea level pressure | NCEP/NCAR Reanalysis 1 | 1948 to Feb 2026 | slp.mon.mean.nc |
| Surface air temperature | NCEP/NCAR Reanalysis 1 | 1948 to Feb 2026 | air.mon.mean.nc |
| Precipitation | GPCP v2.3 | 1979 to Jun 2026 | precip.mon.mean.nc |
| AO index, validation only | NOAA CPC | 1950 to Sep 2026 | monthly.ao.index.b50.current.ascii |

## Caveats

- With 90 months, North's error bars are about 15% of each eigenvalue, so only EOF1 and EOF2 are cleanly separated.
- p-values use the Bretherton et al. (1999) effective sample size. The local-mode table runs 20 tests, so about one "significant" result is expected by chance.
- Reanalysis sea level pressure over the Tibetan Plateau is extrapolated below high ground, so read local patterns there with care.
- GPCP starts in 1979, so the application tabs cannot use the paper's 1948-2000 period.
