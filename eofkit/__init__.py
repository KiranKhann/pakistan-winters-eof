"""eofkit - a small, readable implementation of Hannachi (2004),
"A primer for EOF analysis of climate data", Sections 1-3,
plus an application layer (Arctic Oscillation -> Pakistan winter weather).

Modules
-------
data            loading / subsetting of NOAA PSL monthly NetCDF files   (paper Sec. 2.1)
eof             anomalies, area weighting, SVD-based EOFs, North's rule  (paper Sec. 2-3)
teleconnection  AO index, correlation / composite maps, box averages     (application)
plots           the paper's three figures and the application figures
pipeline        one-call wrappers shared by the scripts and the Streamlit app
"""
from .data import DATASETS, SEASONS, REGIONS, open_field, select_region, select_season, default_years
from .eof import anomalies, area_weights, to_matrix, compute_eofs, EOFResult, effective_sample_size
from .pipeline import run_eof_analysis, load_anomalies

__all__ = [
    "DATASETS", "SEASONS", "REGIONS", "open_field", "select_region", "select_season", "default_years",
    "anomalies", "area_weights", "to_matrix", "compute_eofs", "EOFResult", "effective_sample_size",
    "run_eof_analysis", "load_anomalies",
]
