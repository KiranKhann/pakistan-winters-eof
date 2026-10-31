"""Sanity check of the crop advisor against where crops are actually grown in Pakistan.

Each case names a place, season and water level, and a crop that farmers there
widely grow (should be judged suitable or possible) or that does not grow there
(should be judged risky or not suitable). This checks plausibility; it is not a
statistical validation.

    python scripts/validate_crops.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eofkit import crops as C  # noqa: E402

PLACES = {name: ll for prov in C.FARM_PLACES.values() for name, ll in prov.items()}

# (place, seasons, water, crop, expect)  expect: "yes" = tier 0 or 1, "no" = tier 2 or 3
CASES = [
    ("Faisalabad", ("Rabi",), "w6", "wheat", "yes"),
    ("Lahore", ("Rabi",), "w6", "wheat", "yes"),
    ("Multan", ("Rabi",), "w6", "wheat", "yes"),
    ("Okara", ("Rabi",), "w10", "potato", "yes"),
    ("Multan", ("Kharif",), "w10", "cotton", "yes"),
    ("Multan", ("Kharif",), "full", "cotton", "yes"),
    ("Vehari", ("Kharif",), "w10", "cotton", "yes"),
    ("Sanghar", ("Kharif",), "w10", "cotton", "yes"),
    ("Gujranwala", ("Kharif",), "full", "rice", "yes"),
    ("Larkana", ("Kharif",), "full", "rice", "yes"),
    ("Faisalabad", ("Kharif", "Year-round"), "full", "sugarcane", "yes"),
    ("Mirpur Khas", ("Kharif", "Year-round"), "full", "sugarcane", "yes"),
    ("Mardan", ("Kharif",), "w6", "maize", "yes"),
    ("Bhakkar (Thal)", ("Rabi",), "rain", "chickpea", "yes"),
    ("Chakwal (Potohar)", ("Rabi",), "rain", "wheat", "yes"),
    ("Chakwal (Potohar)", ("Kharif",), "rain", "groundnut", "yes"),
    ("Tharparkar (Mithi)", ("Kharif",), "rain", "millet", "yes"),
    ("Tharparkar (Mithi)", ("Kharif",), "rain", "guar", "yes"),
    # crops that should NOT be judged suitable
    ("Faisalabad", ("Kharif",), "rain", "rice", "no"),
    ("Multan", ("Kharif", "Year-round"), "rain", "sugarcane", "no"),
    ("Quetta", ("Kharif",), "full", "rice", "no"),
    ("Larkana", ("Rabi",), "rain", "chickpea", "no"),
    ("Hyderabad", ("Rabi",), "rain", "wheat", "no"),
]


def main() -> None:
    cache, ok = {}, 0
    for place, seasons, water, crop, expect in CASES:
        lat, lon = PLACES[place]
        if (lat, lon) not in cache:
            cache[(lat, lon)] = C.place_climate(lat, lon)
        R = C.rank_crops(cache[(lat, lon)], water, seasons).set_index("key")
        r = R.loc[crop]
        tier = int(r["tier"])
        passed = (tier <= 1) if expect == "yes" else (tier >= 2)
        ok += passed
        print(f"[{'PASS' if passed else 'FAIL'}] {place:20s} {'/'.join(seasons):17s} {C.WATER_OPTIONS[water]['short']:12s} "
              f"{C.CROPS[crop]['name']:22s} expected {'suitable' if expect == 'yes' else 'unsuitable':10s} -> "
              f"{C.TIERS[tier][1]} ({r['good']}/{r['n']} good, {C.limit_text(r)})")
    src = next(iter(cache.values())).attrs.get("source", "")
    print(f"\n{ok} of {len(CASES)} cases match where crops are really grown. Climate data: {src}.")


if __name__ == "__main__":
    main()
