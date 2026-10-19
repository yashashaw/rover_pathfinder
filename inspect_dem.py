"""
Step 1: sanity-check the REMA elevation file before building anything on it.

What it does:
  - loads the .tif in this folder (or the one you pass: python inspect_dem.py myfile.tif)
  - prints basic facts: projection, cell size, grid size, elevation range, holes
  - computes slope for every cell (degrees)
  - saves a picture to outputs/dem_check.png: elevation + hillshade, slope map,
    and a histogram of slopes

Run:  python inspect_dem.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BRAKING_DEG = 8.5   # downhill steeper than this needs braking (Crr ~0.15)
STEEP_DEG = 15.0    # placeholder "too steep" limit for now


def find_tif():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    tifs = sorted(HERE.glob("*.tif"))
    if not tifs:
        sys.exit("No .tif found in this folder. Put the REMA file here or pass its path.")
    if len(tifs) > 1:
        print(f"Found {len(tifs)} .tif files, using: {tifs[0].name}")
    return tifs[0]


def hillshade(dem, res, azimuth=315, altitude=45):
    """Standard hillshade: makes slopes visible as light and shadow."""
    dy, dx = np.gradient(dem, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)


def main():
    path = find_tif()
    with rasterio.open(path) as src:
        dem = src.read(1).astype("float64")
        res_x, res_y = src.res
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds

    # Mark holes as NaN (file's nodata value, plus anything absurd)
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem[(dem < -500) | (dem > 6000)] = np.nan

    n_cells = dem.size
    n_holes = int(np.isnan(dem).sum())

    print(f"\nFile:        {path.name}")
    print(f"Projection:  {crs}")
    print(f"Cell size:   {res_x:.1f} x {res_y:.1f} m")
    print(f"Grid:        {dem.shape[1]} cols x {dem.shape[0]} rows = {n_cells:,} cells")
    print(f"Extent:      x {bounds.left:,.0f} to {bounds.right:,.0f}, "
          f"y {bounds.bottom:,.0f} to {bounds.top:,.0f}")
    print(f"Elevation:   {np.nanmin(dem):,.1f} to {np.nanmax(dem):,.1f} m "
          f"(mean {np.nanmean(dem):,.1f})")
    print(f"Holes:       {n_holes:,} cells ({100 * n_holes / n_cells:.3f}%)")

    # Slope in degrees (fill holes temporarily with the mean so gradient works)
    filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(filled, res_y, res_x)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    slope[np.isnan(dem)] = np.nan

    print(f"\nSlope:       median {np.nanmedian(slope):.3f} deg, "
          f"max {np.nanmax(slope):.2f} deg")
    print(f"  > {BRAKING_DEG} deg (braking): {np.nanmean(slope > BRAKING_DEG) * 100:.3f}% of cells")
    print(f"  > {STEEP_DEG} deg (steep):     {np.nanmean(slope > STEEP_DEG) * 100:.3f}% of cells")

    # Where is the steepest spot? (often buildings or snow piles at a station)
    r, c = np.unravel_index(np.nanargmax(slope), slope.shape)
    print(f"  steepest cell at row {r}, col {c}")

    # ---- Pictures ----
    shade = hillshade(filled, res_x)
    extent = [bounds.left / 1000, bounds.right / 1000,
              bounds.bottom / 1000, bounds.top / 1000]   # km

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.imshow(shade, cmap="gray", extent=extent)
    im = ax.imshow(dem, cmap="terrain", alpha=0.55, extent=extent)
    fig.colorbar(im, ax=ax, label="elevation (m)", shrink=0.8)
    ax.set_title("Elevation + hillshade")
    ax.set_xlabel("x (km, EPSG:3031)")
    ax.set_ylabel("y (km, EPSG:3031)")

    ax = axes[1]
    vmax = max(1.0, np.nanpercentile(slope, 99.5))
    im = ax.imshow(slope, cmap="magma", vmin=0, vmax=vmax, extent=extent)
    fig.colorbar(im, ax=ax, label="slope (deg)", shrink=0.8)
    ax.set_title("Slope (99.5th percentile color cap)")
    ax.set_xlabel("x (km)")

    ax = axes[2]
    ax.hist(slope[~np.isnan(slope)].ravel(), bins=200, log=True)
    ax.axvline(BRAKING_DEG, color="orange", ls="--", label=f"braking {BRAKING_DEG} deg")
    ax.axvline(STEEP_DEG, color="red", ls="--", label=f"steep {STEEP_DEG} deg")
    ax.set_title("Slope histogram (log count)")
    ax.set_xlabel("slope (deg)")
    ax.legend()

    fig.tight_layout()
    out = OUT_DIR / "dem_check.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved picture: {out}")
    plt.show()


if __name__ == "__main__":
    main()