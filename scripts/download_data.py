"""Download the NetCDF files used by the project into ./data (about 520 MB).

    python scripts/download_data.py

All files are free NOAA PSL / CPC products and need no login.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eofkit.data import DATASETS, DATA_DIR  # noqa: E402
from eofkit.teleconnection import CPC_AO_URL, CPC_AO_FILE  # noqa: E402
from eofkit.crops import TEMP_URL, TEMP_FILE, RAIN_URL, RAIN_FILE, RAIN_URL_1DEG, RAIN_FILE_1DEG  # noqa: E402


def fetch(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  exists   {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print(f"  saved    {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    force = "--force" in sys.argv
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for meta in DATASETS.values():
        fetch(meta["url"], DATA_DIR / meta["file"], force)
    fetch(CPC_AO_URL, DATA_DIR / CPC_AO_FILE, force=True)  # small text file, always refresh
    # 0.5 degree temperature and rainfall for the crop advisor (about 450 MB together)
    fetch(TEMP_URL, TEMP_FILE, force)
    fetch(RAIN_URL_1DEG, RAIN_FILE_1DEG, force)   # 80 MB, used if the 0.5 degree file is missing
    fetch(RAIN_URL, RAIN_FILE, force)


if __name__ == "__main__":
    main()
