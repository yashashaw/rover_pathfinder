"""
Download ERA5 hourly data for the Dome C test area (one request per year).
"""

from pathlib import Path

import cdsapi

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "era5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_env(path):
    """Read KEY=VALUE lines from a .env file (no extra packages needed)."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    return values


env = load_env(HERE / ".env")
if "CDSAPI_KEY" in env:
    c = cdsapi.Client(
        url=env.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api"),
        key=env["CDSAPI_KEY"],
    )
else:
    c = cdsapi.Client()  # falls back to ~/.cdsapirc

# ERA5 uses lat/lon, not EPSG:3031. Padded box around Dome C (~75.1S, 123.3E)
# so it includes a few 0.25-degree grid points around the 20 km test area.
AREA = [-74.5, 122.0, -75.75, 125.0]  # [North, West, South, East]

YEARS = range(2010, 2026)
MONTHS = ["11", "12", "01"]           # mission window: Nov-Jan

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "snowfall",
]

for year in YEARS:
    target = OUT_DIR / f"era5_domec_{year}.nc"
    if target.exists():
        print(f"skip {year} (already downloaded)")
        continue
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": MONTHS,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    print(f"done {year}")

# Time-invariant ERA5 surface height (geopotential), for the temperature
# elevation adjustment. Divide by 9.81 to get meters.
geo = OUT_DIR / "era5_domec_geopotential.nc"
if not geo.exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "year": ["2020"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(geo),
    )
print("done geopotential")
