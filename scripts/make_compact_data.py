"""Write small regional copies of the NetCDF files into data/compact/.

The full NOAA files are about 600 MB (two are over GitHub's 100 MB file limit),
but the app and scripts only use part of the globe. These copies keep every
month and every value inside the regions the project uses, so results are
identical, and they are small enough to commit:

    slp      Northern Hemisphere, 0-90N, all longitudes  (paper Sec. 3.5, AO, local modes)
    air      0-60N, 20-120E                               (public page, application, rabi page)
    precip   0-60N, 20-120E
    ghcncams 22-39N, 59-79E at 0.5 degrees               (crop advisor)
    precl    22-39N, 59-79E at 0.5 degrees               (crop advisor)

The code reads data/<file> when it exists and falls back to data/compact/<file>.

    python scripts/make_compact_data.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xarray as xr  # noqa: E402

from eofkit.data import DATA_DIR, COMPACT_DIR  # noqa: E402

BOXES = {
    "slp.mon.mean.nc": ("slp", (0.0, 90.0), None),
    "air.mon.mean.nc": ("air", (0.0, 60.0), (20.0, 120.0)),
    "precip.mon.mean.nc": ("precip", (0.0, 60.0), (20.0, 120.0)),
    "ghcncams.air.mon.mean.nc": ("air", (22.0, 39.0), (59.0, 79.0)),
    "precl.precip.mon.mean.0.5x0.5.nc": ("precip", (22.0, 39.0), (59.0, 79.0)),
}
COPY = ["monthly.ao.index.b50.current.ascii"]


def subset(name: str, var: str, lat, lon) -> None:
    src = DATA_DIR / name
    if not src.exists():
        print(f"  skip     {name} (not downloaded)")
        return
    ds = xr.open_dataset(src, drop_variables=["time_bnds", "lat_bnds", "lon_bnds", "climatology_bounds"])
    da = ds[var]
    lat_sl = slice(lat[1], lat[0]) if float(da.lat[0]) > float(da.lat[-1]) else slice(lat[0], lat[1])
    da = da.sel(lat=lat_sl)
    if lon is not None:
        da = da.sel(lon=slice(lon[0], lon[1]))
    out = da.load().to_dataset(name=var)
    out.attrs.update(ds.attrs)
    out.attrs["compact_subset"] = f"lat {lat}, lon {lon or 'all'}; made by scripts/make_compact_data.py"
    enc = {var: {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": float("nan")}}
    dest = COMPACT_DIR / name
    out.to_netcdf(dest, encoding=enc)
    print(f"  wrote    compact/{name}: {dict(out.sizes)} -> {dest.stat().st_size / 1e6:.1f} MB "
          f"(from {src.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (var, lat, lon) in BOXES.items():
        subset(name, var, lat, lon)
    for name in COPY:
        if (DATA_DIR / name).exists():
            shutil.copy2(DATA_DIR / name, COMPACT_DIR / name)
            print(f"  copied   compact/{name}")


if __name__ == "__main__":
    main()
