"""Pakistan's Winters: an EOF-based climate explorer.

    .venv/Scripts/python -m streamlit run app.py

Four pages:
  * Pakistan's winters        - plain-language page for anyone (default)
  * For farmers: which crop?  - compares 18 crops against 30 seasons at a farm
  * For farmers (rabi season) - rain you can count on, dry seasons, heat at grain filling
  * The maths                 - Hannachi (2004) Sections 1-3, for the course project
"""
import streamlit as st

st.set_page_config(page_title="Pakistan's Winters", layout="wide")

pages = st.navigation([
    st.Page("views/public.py", title="Pakistan's winters", icon=":material/ac_unit:", default=True),
    st.Page("views/crop_advisor.py", title="For farmers: which crop?", icon=":material/agriculture:"),
    st.Page("views/farmers.py", title="For farmers (rabi season)", icon=":material/water_drop:"),
    st.Page("views/maths.py", title="The maths (EOF analysis)", icon=":material/functions:"),
])
st.sidebar.caption("Built on Hannachi (2004), *A primer for EOF analysis of climate data*, "
                   "with NOAA monthly climate records.")
pages.run()
