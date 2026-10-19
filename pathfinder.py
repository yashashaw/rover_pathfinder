"""
v1 pathfinder: energy-optimal rover route on the 10 m REMA grid.

Runs A* (compiled with numba, so it's fast) on a directed 8-neighbour graph.
Each edge's cost = battery energy to drive it:

    W     = rolling (Crr*m*g*L) + gravity (m*g*dh) + drag (F_drag*L)
    drive = W / eta_drive          if W > 0
          = -|W| * eta_regen       if W < 0   (0 by default: brakes waste it)
    time  = always-on power * L / speed  (/ (1 - storm fraction) with ERA5)
    edge  = drive + time

Modes
  no_era5 : calm air, one constant outside temperature everywhere
  era5    : drag per direction + heater power + storm waiting, from ERA5 history
  both    : runs both, then scores BOTH routes with the ERA5 model
            -> "regret" = how much energy you'd lose by ignoring ERA5 (ablation)

Examples
  python pathfinder.py                         # both modes, default start/goal
  python pathfinder.py --mode no_era5
  python pathfinder.py --start 1347000 -888000 --goal 1363200 -896050
  python pathfinder.py --months 12             # December weather only
  python pathfinder.py --dijkstra              # also run Dijkstra to verify A*

Coordinates are EPSG:3031 meters (hover in QGIS to read them).
Rover numbers below are placeholders - edit PARAMS.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from numba import njit
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

# ---------------------------------------------------------------- parameters
# Free variables: placeholders until the rover exists. Sweep these.
PARAMS = dict(
    mass_kg=100.0,
    crr=0.15,                    # one value everywhere for v1 (no snow map yet)
    eta_drive=0.70,              # motor + gearbox + controller efficiency
    eta_regen=0.0,               # fraction recovered when braking (0 = none)
    cda_m2=0.5,                  # drag coefficient x frontal area
    speed_ms=1.0,
    p_elec_w=30.0,               # computer, sensors, radio
    t_inside_c=20.0,             # temperature heaters hold inside the rover
    k0_w_per_k=0.5,              # heat leak per degree, calm air
    k1_w_per_k_per_ms=0.05,      # extra heat leak per degree per m/s of wind
    wind_height_factor=0.8,      # 10 m wind -> rover-height wind
    storm_limit_ms=15.0,         # rover parks above this (10 m wind speed)
    max_slope_deg=15.0,          # cells steeper than this are off-limits
    block_buffer_cells=1,        # extra no-go cells around steep cells
    lapse_k_per_m=0.0065,        # temperature drop per metre of height
    use_lapse=True,
    # used only when ERA5 is OFF:
    t_outside_c_default=-30.0,
    p_air_pa_default=65000.0,    # roughly Dome C surface pressure
)

G = 9.81
R_AIR = 287.05
SQ2 = np.sqrt(2.0)

# 8 directions as (d_row, d_col). Row grows downward, so "N" is row - 1.
DIRS = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1],
                 [1, 0], [1, -1], [0, -1], [-1, -1]], dtype=np.int64)
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_LEN = np.array([1, SQ2, 1, SQ2, 1, SQ2, 1, SQ2])          # in cells
# unit vectors in map coordinates (x east, y north)
DIR_XY = np.stack([DIRS[:, 1] / DIR_LEN, -DIRS[:, 0] / DIR_LEN], axis=1)

DEFAULT_START = (1347000.0, -888000.0)
DEFAULT_GOAL = (1363200.0, -896050.0)    # straight line passes through Concordia


# ---------------------------------------------------------------- numba core
@njit(cache=True)
def _heap_push(hk, hv, n, key, val):
    if n == hk.shape[0]:
        nk = np.empty(hk.shape[0] * 2)
        nv = np.empty(hv.shape[0] * 2, dtype=np.int64)
        nk[:n] = hk
        nv[:n] = hv
        hk, hv = nk, nv
    i = n
    hk[i] = key
    hv[i] = val
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return hk, hv, n + 1


@njit(cache=True)
def _heap_pop(hk, hv, n):
    key, val = hk[0], hv[0]
    n -= 1
    hk[0], hv[0] = hk[n], hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        r = l + 1
        m = r if (r < n and hk[r] < hk[l]) else l
        if hk[i] <= hk[m]:
            break
        hk[m], hk[i] = hk[i], hk[m]
        hv[m], hv[i] = hv[i], hv[m]
        i = m
    return key, val, n


@njit(cache=True)
def _edge(z, crr, drag, tcost, r, c, d, nr, nc, L, m, eta_d, eta_r):
    """Returns (total, rolling W, gravity W, drag W, drive E, time E) in J."""
    w_roll = crr[r, c] * m * G * L
    w_grav = m * G * (z[nr, nc] - z[r, c])
    w_drag = drag[d, r, c] * L
    w = w_roll + w_grav + w_drag
    if w > 0.0:
        e_drive = w / eta_d
    else:
        e_drive = w * eta_r          # <= 0: energy recovered (0 if no regen)
    e_time = tcost[r, c] * L
    total = e_drive + e_time
    if total < 0.0:                  # A* needs non-negative edges
        total = 0.0
    return total, w_roll, w_grav, w_drag, e_drive, e_time


@njit(cache=True)
def astar(z, crr, drag, tcost, blocked, start, goal, res,
          m, eta_d, eta_r, tan_max, h_lin, t_min, use_h):
    """A* on the directed 8-neighbour grid. use_h=False -> Dijkstra.

    Heuristic (consistent, so the result is optimal):
      h = max(0, h_lin*octile + m*g*(z_goal - z)) / eta_d + t_min*octile
    where h_lin = min Crr*m*g + min drag force (only used if >= 0 and no regen).
    """
    ny, nx = z.shape
    N = ny * nx
    gscore = np.full(N, np.inf)
    parent = np.full(N, -1, dtype=np.int64)
    closed = np.zeros(N, dtype=np.bool_)
    hk = np.empty(1 << 16)
    hv = np.empty(1 << 16, dtype=np.int64)
    hn = 0

    gr, gc = goal // nx, goal % nx
    zg = z[gr, gc]
    grav_ok = (h_lin >= 0.0) and (eta_r == 0.0)

    def h(r, c):
        dr = abs(r - gr)
        dc = abs(c - gc)
        lo = min(dr, dc)
        hi = max(dr, dc)
        octile = res * ((hi - lo) + SQ2 * lo)
        val = t_min * octile
        if grav_ok:
            a = h_lin * octile + m * G * (zg - z[r, c])
            if a > 0.0:
                val += a / eta_d
        return val

    gscore[start] = 0.0
    sr, sc = start // nx, start % nx
    hk, hv, hn = _heap_push(hk, hv, hn, h(sr, sc) if use_h else 0.0, start)
    expanded = 0

    while hn > 0:
        _, u, hn = _heap_pop(hk, hv, hn)
        if closed[u]:
            continue
        closed[u] = True
        expanded += 1
        if u == goal:
            break
        r, c = u // nx, u % nx
        gu = gscore[u]
        for d in range(8):
            nr = r + DIRS[d, 0]
            nc = c + DIRS[d, 1]
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                continue
            v = nr * nx + nc
            if closed[v] or blocked[nr, nc]:
                continue
            L = res * DIR_LEN[d]
            if abs(z[nr, nc] - z[r, c]) > tan_max * L:      # too steep along edge
                continue
            cost = _edge(z, crr, drag, tcost, r, c, d, nr, nc, L,
                         m, eta_d, eta_r)[0]
            ng = gu + cost
            if ng < gscore[v]:
                gscore[v] = ng
                parent[v] = u
                key = ng + (h(nr, nc) if use_h else 0.0)
                hk, hv, hn = _heap_push(hk, hv, hn, key, v)

    if parent[goal] == -1 and goal != start:
        return np.empty(0, dtype=np.int64), np.inf, expanded
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    out = np.empty(len(path), dtype=np.int64)
    for i in range(len(path)):
        out[i] = path[len(path) - 1 - i]
    return out, gscore[goal], expanded


@njit(cache=True)
def score_path(path, z, crr, drag, tcost, nx, res, m, eta_d, eta_r):
    """Energy breakdown of a path under a given set of layers (J)."""
    tot = np.zeros(6)  # total, rolling W, gravity W, drag W, drive E, time E
    brake = 0.0
    for i in range(len(path) - 1):
        r, c = path[i] // nx, path[i] % nx
        nr, nc = path[i + 1] // nx, path[i + 1] % nx
        dr, dc = nr - r, nc - c
        d = 0
        for k in range(8):
            if DIRS[k, 0] == dr and DIRS[k, 1] == dc:
                d = k
                break
        L = res * DIR_LEN[d]
        e = _edge(z, crr, drag, tcost, r, c, d, nr, nc, L, m, eta_d, eta_r)
        for k in range(6):
            tot[k] += e[k]
        w = e[1] + e[2] + e[3]
        if w < 0.0:
            brake += -w * (1.0 - eta_r)
    return tot, brake


# ---------------------------------------------------------------- inputs
def load_dem(path):
    with rasterio.open(path) as src:
        z = src.read(1).astype(np.float64)
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        tr = src.transform
        res = src.res[0]
        if abs(src.res[0] - src.res[1]) > 1e-6 or tr.b != 0 or tr.d != 0:
            sys.exit("Expected a north-up raster with square cells.")
        info = dict(left=src.bounds.left, top=src.bounds.top,
                    right=src.bounds.right, bottom=src.bounds.bottom,
                    crs=src.crs)
    z[(z < -500) | (z > 6000)] = np.nan
    return z, res, info


def make_blocked(z, res, p):
    filled = np.where(np.isnan(z), np.nanmean(z), z)
    dy, dx = np.gradient(filled, res)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    blocked = (slope > p["max_slope_deg"]) | np.isnan(z)
    if p["block_buffer_cells"] > 0:
        blocked = binary_dilation(blocked, iterations=p["block_buffer_cells"])
    return blocked, filled


def xy_to_index(x, y, info, res, shape):
    col = int((x - info["left"]) // res)
    row = int((info["top"] - y) // res)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        sys.exit(f"Point ({x:.0f}, {y:.0f}) is outside the elevation file.")
    return row, col


def cell_centers(info, res, shape):
    cols = info["left"] + (np.arange(shape[1]) + 0.5) * res
    rows = info["top"] - (np.arange(shape[0]) + 0.5) * res
    return np.meshgrid(cols, rows)


# ---------------------------------------------------------------- layers
def layers_no_era5(z, p):
    """Calm air, one outside temperature everywhere."""
    shape = z.shape
    v = p["speed_ms"]
    t_out = p["t_outside_c_default"]
    rho = p["p_air_pa_default"] / (R_AIR * (t_out + 273.15))
    f_still = 0.5 * rho * p["cda_m2"] * v * v           # drag from own motion
    drag = np.full((8,) + shape, f_still, dtype=np.float32)
    heater = max(0.0, p["k0_w_per_k"] * (p["t_inside_c"] - t_out))
    tcost = np.full(shape, (p["p_elec_w"] + heater) / v, dtype=np.float32)
    crr = np.full(shape, p["crr"], dtype=np.float32)
    summary = dict(t_out_c=t_out, heater_w=heater, storm_frac=0.0,
                   drag_n=[f_still] * 8)
    return crr, drag, tcost, summary


def _normalize_era5(ds):
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for dim in ("number", "expver"):
        if dim in ds.dims:
            ds = ds.isel({dim: 0}, drop=True)
    return ds.drop_vars([v for v in ("number", "expver") if v in ds.variables],
                        errors="ignore")


def _open_era5_file(path):
    """New CDS sometimes returns a zip (instant + accumulated files) even
    when the name ends in .nc. Handle both."""
    import xarray as xr
    if zipfile.is_zipfile(path):
        out = path.parent / "unzipped" / path.stem
        if not out.exists():
            out.mkdir(parents=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(out)
        parts = [_normalize_era5(xr.open_dataset(f)) for f in sorted(out.glob("*.nc"))]
        return xr.merge(parts, compat="override", join="outer")
    return _normalize_era5(xr.open_dataset(path))


def load_era5(era5_dir, months):
    import xarray as xr
    files = sorted(era5_dir.glob("era5_domec_[0-9][0-9][0-9][0-9].nc"))
    if not files:
        sys.exit(f"No ERA5 files found in {era5_dir}. Run era5_download.py first.")
    ds = xr.concat([_open_era5_file(f) for f in files], dim="time",
                   data_vars="all", coords="minimal", compat="override")
    ds = ds.sortby("time")
    if months:
        ds = ds.sel(time=ds["time"].dt.month.isin(months))
    if ds.sizes["time"] == 0:
        sys.exit("No ERA5 hours left after the month filter.")
    need = ["u10", "v10", "t2m", "sp"]
    missing = [v for v in need if v not in ds]
    if missing:
        sys.exit(f"ERA5 files are missing variables: {missing}")
    geo_path = era5_dir / "era5_domec_geopotential.nc"
    z_era = None
    if geo_path.exists():
        g = _open_era5_file(geo_path)["z"]
        if "time" in g.dims:
            g = g.isel(time=0)
        z_era = (g.values / G).astype(np.float64)
    return ds, files, z_era


def layers_era5(z, res, info, p, months, era5_dir):
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    ds, files, z_era = load_era5(era5_dir, months)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    u = ds["u10"].values.astype(np.float64)        # (time, lat, lon)
    vv = ds["v10"].values.astype(np.float64)
    t2m = ds["t2m"].values.astype(np.float64)
    sp = ds["sp"].values.astype(np.float64)
    n_hours = u.shape[0]

    # --- east / north directions in map coordinates at each ERA5 point
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    LON, LAT = np.meshgrid(lon, lat)
    x0, y0 = fwd.transform(LON, LAT)
    xe, ye = fwd.transform(LON + 0.01, LAT)
    xn, yn = fwd.transform(LON, LAT + 0.01)
    e_hat = np.stack([xe - x0, ye - y0], -1)
    e_hat /= np.linalg.norm(e_hat, axis=-1, keepdims=True)
    n_hat = np.stack([xn - x0, yn - y0], -1)
    n_hat /= np.linalg.norm(n_hat, axis=-1, keepdims=True)

    f = p["wind_height_factor"]
    wx = f * (u * e_hat[..., 0] + vv * n_hat[..., 0])   # wind in map x (east)
    wy = f * (u * e_hat[..., 1] + vv * n_hat[..., 1])   # wind in map y (north)

    speed10 = np.hypot(u, vv)
    driving = speed10 <= p["storm_limit_ms"]
    storm_frac = 1.0 - driving.mean(axis=0)
    rho = sp / (R_AIR * t2m)
    V = p["speed_ms"]

    # --- expected drag force per direction, averaged over driving hours only
    drag_pts = np.empty((8,) + lat.shape + lon.shape)
    n_drive = np.maximum(driving.sum(axis=0), 1)
    for d in range(8):
        along = wx * DIR_XY[d, 0] + wy * DIR_XY[d, 1]    # tailwind > 0
        vrel = V - along
        F = 0.5 * rho * p["cda_m2"] * vrel * np.abs(vrel)
        drag_pts[d] = np.where(driving, F, 0.0).sum(axis=0) / n_drive

    # --- heater power, averaged over ALL hours (it runs parked too)
    t_c = t2m - 273.15
    k = p["k0_w_per_k"] + p["k1_w_per_k_per_ms"] * f * speed10
    heater_pts = np.maximum(0.0, k * (p["t_inside_c"] - t_c)).mean(axis=0)
    k_mean_pts = k.mean(axis=0)
    t_mean_pts = t_c.mean(axis=0)

    # --- interpolate point fields onto the 10 m grid (one call, stacked)
    stack = [drag_pts[d] for d in range(8)] + [heater_pts, k_mean_pts, storm_frac]
    if z_era is not None:
        stack.append(z_era)
    vals = np.stack(stack, axis=-1)                     # (lat, lon, k)
    order = np.argsort(lat)
    lat_s, vals = lat[order], vals[order]
    order = np.argsort(lon)
    lon_s, vals = lon[order], vals[:, order]
    interp = RegularGridInterpolator((lat_s, lon_s), vals, method="linear",
                                     bounds_error=False, fill_value=None)

    X, Y = cell_centers(info, res, z.shape)
    inv = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    clon, clat = inv.transform(X.ravel(), Y.ravel())
    grid = interp(np.column_stack([clat, clon])).reshape(z.shape + (vals.shape[-1],))

    drag = np.moveaxis(grid[..., :8], -1, 0).astype(np.float32)
    heater = grid[..., 8]
    k_mean = grid[..., 9]
    storm = np.clip(grid[..., 10], 0.0, 0.95)
    if z_era is not None and p["use_lapse"]:
        z_era_grid = grid[..., 11]
        zf = np.where(np.isnan(z), np.nanmean(z), z)
        # higher than ERA5's ground -> colder -> more heater power
        heater = np.maximum(0.0, heater + k_mean * p["lapse_k_per_m"] * (zf - z_era_grid))
    tcost = ((p["p_elec_w"] + heater) / V / (1.0 - storm)).astype(np.float32)
    crr = np.full(z.shape, p["crr"], dtype=np.float32)

    summary = dict(
        files=len(files), hours=n_hours, months=months,
        t_out_c=float(np.mean(t_mean_pts)),
        heater_w=float(np.mean(heater)),
        storm_frac=float(np.mean(storm)),
        drag_n=[float(np.mean(drag[d])) for d in range(8)],
        mean_wind10=float(np.mean(speed10)),
    )
    return crr, drag, tcost, summary


# ---------------------------------------------------------------- running
def heuristic_terms(crr, drag, tcost, p):
    h_lin = float(np.min(crr)) * p["mass_kg"] * G + float(np.min(drag))
    t_min = float(np.min(tcost))
    return h_lin, t_min


def run(name, z, res, crr, drag, tcost, blocked, s, g, p, use_h=True):
    h_lin, t_min = heuristic_terms(crr, drag, tcost, p)
    t0 = time.perf_counter()
    path, cost, expanded = astar(
        z, crr, drag, tcost, blocked, s, g, res,
        p["mass_kg"], p["eta_drive"], p["eta_regen"],
        np.tan(np.radians(p["max_slope_deg"])), h_lin, t_min, use_h)
    dt = time.perf_counter() - t0
    algo = "A*" if use_h else "Dijkstra"
    if len(path) == 0:
        print(f"[{name}] {algo}: NO ROUTE (start and goal not connected)")
    else:
        print(f"[{name}] {algo}: {cost / 3600:,.1f} Wh, {len(path):,} cells, "
              f"{expanded:,} expanded, {dt:.2f} s")
    return path, cost


def report(label, path, z, crr, drag, tcost, nx, res, p):
    tot, brake = score_path(path, z, crr, drag, tcost, nx, res,
                            p["mass_kg"], p["eta_drive"], p["eta_regen"])
    total, w_roll, w_grav, w_drag, e_drive, e_time = tot / 3600.0
    length_km = 0.0
    for i in range(len(path) - 1):
        r, c = divmod(path[i], nx)
        r2, c2 = divmod(path[i + 1], nx)
        length_km += res * np.hypot(r2 - r, c2 - c) / 1000.0
    print(f"  {label}")
    print(f"    length            {length_km:8.2f} km")
    print(f"    rolling work      {w_roll:8.1f} Wh")
    print(f"    gravity work      {w_grav:8.1f} Wh (net climb)")
    print(f"    drag work         {w_drag:8.1f} Wh")
    print(f"    braked away       {brake / 3600:8.1f} Wh")
    print(f"    -> drive battery  {e_drive:8.1f} Wh (after motor losses)")
    print(f"    -> time battery   {e_time:8.1f} Wh (electronics + heaters)")
    print(f"    TOTAL             {total:8.1f} Wh")
    return total


def plot(z_filled, blocked, res, info, routes, title, out):
    import matplotlib.pyplot as plt
    dy, dx = np.gradient(z_filled, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    shade = np.clip(np.sin(alt) * np.cos(slope)
                    + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    ext = [info["left"] / 1000, info["right"] / 1000,
           info["bottom"] / 1000, info["top"] / 1000]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(shade, cmap="gray", extent=ext)
    ax.imshow(z_filled, cmap="terrain", alpha=0.35, extent=ext)
    mask = np.ma.masked_where(~blocked, blocked)
    ax.imshow(mask, cmap="autumn", alpha=0.9, extent=ext, interpolation="nearest")
    nx = z_filled.shape[1]
    colors = {"no_era5": "#1f77b4", "era5": "#d62728"}
    for name, path, wh in routes:
        r, c = np.divmod(path, nx)
        xs = (info["left"] + (c + 0.5) * res) / 1000
        ys = (info["top"] - (r + 0.5) * res) / 1000
        ax.plot(xs, ys, lw=2.2, color=colors.get(name, "k"),
                label=f"{name}: {wh:,.0f} Wh")
    if routes:
        path = routes[0][1]
        for idx, mk, lab in ((path[0], "o", "start"), (path[-1], "*", "goal")):
            r, c = divmod(int(idx), nx)
            ax.plot((info["left"] + (c + 0.5) * res) / 1000,
                    (info["top"] - (r + 0.5) * res) / 1000,
                    mk, ms=14, color="k", mfc="yellow", label=lab)
    ax.set_title(title)
    ax.set_xlabel("x (km, EPSG:3031)")
    ax.set_ylabel("y (km, EPSG:3031)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.locator_params(nbins=6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved picture: {out}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["no_era5", "era5", "both"], default="both")
    ap.add_argument("--dem", default="REMA_10m_elevation.tif")
    ap.add_argument("--era5-dir", default="data/era5")
    ap.add_argument("--start", nargs=2, type=float, default=DEFAULT_START, metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, default=DEFAULT_GOAL, metavar=("X", "Y"))
    ap.add_argument("--months", default="11,12,1",
                    help="ERA5 months to average over, e.g. '12' or '11,12,1'")
    ap.add_argument("--dijkstra", action="store_true",
                    help="also run Dijkstra and check it matches A* (validation)")
    ap.add_argument("--no-show", action="store_true", help="save pictures without opening them")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    p = PARAMS
    dem_path = Path(args.dem)
    if not dem_path.is_absolute():
        dem_path = HERE / dem_path
    era5_dir = Path(args.era5_dir)
    if not era5_dir.is_absolute():
        era5_dir = HERE / era5_dir
    months = [int(m) for m in args.months.split(",") if m.strip()]

    z, res, info = load_dem(dem_path)
    blocked, z_filled = make_blocked(z, res, p)
    ny, nx = z.shape
    sr, sc = xy_to_index(*args.start, info, res, z.shape)
    gr, gc = xy_to_index(*args.goal, info, res, z.shape)
    for lab, (r, c) in (("start", (sr, sc)), ("goal", (gr, gc))):
        if blocked[r, c]:
            sys.exit(f"The {lab} point is in an off-limits (too steep) cell. Move it.")
    s, g = sr * nx + sc, gr * nx + gc

    print(f"\nGrid {nx} x {ny} at {res:.0f} m, {blocked.mean() * 100:.3f}% off-limits")
    print(f"Start ({args.start[0]:.0f}, {args.start[1]:.0f})  ->  "
          f"Goal ({args.goal[0]:.0f}, {args.goal[1]:.0f})")
    print("(first run compiles the A* code; later runs start instantly)\n")

    layers = {}
    if args.mode in ("no_era5", "both"):
        layers["no_era5"] = layers_no_era5(z_filled, p)
    if args.mode in ("era5", "both"):
        t0 = time.perf_counter()
        layers["era5"] = layers_era5(z, res, info, p, months, era5_dir)
        sm = layers["era5"][3]
        print(f"ERA5: {sm['files']} files, {sm['hours']:,} hours (months {months}), "
              f"built in {time.perf_counter() - t0:.1f} s")
        print(f"  mean outside temp {sm['t_out_c']:.1f} C, mean 10 m wind "
              f"{sm['mean_wind10']:.1f} m/s, storm hours {sm['storm_frac'] * 100:.1f}%")
        print("  mean drag force by direction (N): " + ", ".join(
            f"{n} {f:.2f}" for n, f in zip(DIR_NAMES, sm["drag_n"])))
        print(f"  mean heater power {sm['heater_w']:.1f} W\n")

    results = {}
    for name, (crr, drag, tcost, sm) in layers.items():
        path, cost = run(name, z_filled, res, crr, drag, tcost, blocked, s, g, p)
        if args.dijkstra and len(path):
            _, cost_d = run(name, z_filled, res, crr, drag, tcost, blocked, s, g, p,
                            use_h=False)
            ok = abs(cost_d - cost) <= 1e-6 * max(1.0, cost)
            print(f"  A* vs Dijkstra: {'MATCH' if ok else 'MISMATCH - heuristic bug!'}")
        results[name] = path

    print()
    routes = []
    for name, path in results.items():
        if len(path) == 0:
            continue
        crr, drag, tcost, _ = layers[name]
        wh = report(f"{name} route, scored with the {name} model",
                    path, z_filled, crr, drag, tcost, nx, res, p)
        routes.append((name, path, wh))

    if "no_era5" in results and "era5" in results \
            and len(results["no_era5"]) and len(results["era5"]):
        crr, drag, tcost, _ = layers["era5"]
        args_s = (z_filled, crr, drag, tcost, nx, res,
                  p["mass_kg"], p["eta_drive"], p["eta_regen"])
        e_full = score_path(results["era5"], *args_s)[0][0] / 3600.0
        e_abl = score_path(results["no_era5"], *args_s)[0][0] / 3600.0
        regret = e_abl - e_full
        same = np.array_equal(results["no_era5"], results["era5"])
        print("\n  ABLATION: both routes scored with the ERA5 (more complete) model")
        print(f"    era5 route      {e_full:8.1f} Wh")
        print(f"    no_era5 route   {e_abl:8.1f} Wh")
        print(f"    REGRET of ignoring ERA5: {regret:,.1f} Wh "
              f"({100 * regret / e_full:.2f}%)")
        print(f"    same cells: {same}" + ("" if same else
              "  (different cells but ~0 regret = equal-cost ties on the 8-direction grid)"
              if regret < 0.05 else ""))

    title = f"Routes ({args.mode}), months {months if 'era5' in layers else '-'}"
    fig = plot(z_filled, blocked, res, info, routes, title,
               OUT_DIR / f"route_{args.mode}.png")
    for name, path, _ in routes:
        r, c = np.divmod(path, nx)
        xs = info["left"] + (c + 0.5) * res
        ys = info["top"] - (r + 0.5) * res
        np.savetxt(OUT_DIR / f"route_{name}.csv", np.column_stack([xs, ys]),
                   delimiter=",", header="x_epsg3031,y_epsg3031", comments="", fmt="%.1f")
    print(f"Saved route coordinates: {OUT_DIR / 'route_<mode>.csv'}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()