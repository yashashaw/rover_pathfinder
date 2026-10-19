"""
pathfinder v2: energy-optimal rover route on the 10 m REMA grid.

Search: A* (compiled with numba) on a directed 16-neighbour graph
        (8 compass moves + 8 "knight" moves like 2 east / 1 south), then
        SMOOTHING: stretches of the grid route are replaced by straight lines
        whenever the straight line is allowed and costs no more on the real
        terrain. The smoothed route is what gets reported, scored and plotted.

Each step's cost is the battery energy to drive it:
    W     = rolling (Crr*m*g*L) + gravity (m*g*dh) + drag (F_drag*L)
    drive = W / eta_drive          if W > 0
          = -|W| * eta_regen       if W < 0   (0 by default: brakes waste it)
    time  = always-on power * L / speed  (/ (1 - storm fraction) with weather)

Crr is a MAP:
    Crr(cell) = Crr_snow(cell) + c_rough * max(0, roughness(cell) - noise_floor)

Features (each can be switched on/off; "full" uses every one that has data):
  weather : ERA5 wind drag by direction, heaters (temp + wind chill), storms
  snow    : snow type from ERA5 (mean wind + recent snowfall) -> Crr_snow
  rough   : bumpiness from the 2 m REMA file -> extra Crr

Ablation (default): plans with the full model, then with each feature off,
and scores EVERY route with the full model. Extra energy = that feature's regret.
  weather off -> calm air, one default outside temperature
  snow off    -> every cell gets the area's AVERAGE snow Crr
  rough off   -> every cell gets the area's AVERAGE roughness

Pictures (outputs/):
  routes.png   - the routes on a grey terrain shading
  factors.png  - one map per factor, with plain-language titles and units

Examples
  python pathfinder_v2.py                          # full model + ablation
  python pathfinder_v2.py --features weather,snow  # full = only these
  python pathfinder_v2.py --no-ablate              # just the full model
  python pathfinder_v2.py --months 12              # December weather only
  python pathfinder_v2.py --dijkstra               # verify A* against Dijkstra
  python pathfinder_v2.py --start 1347000 -888000 --goal 1363200 -896050

Files it looks for in this folder:
  REMA_10m_elevation.tif   (required)
  REMA_2m_elevation.tif    (for roughness; computed once, then cached)
  data/era5/               (for weather + snow)

Coordinates are EPSG:3031 meters. Rover/snow numbers below are PLACEHOLDERS.
"""

import argparse
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from numba import njit
from scipy.ndimage import binary_dilation, uniform_filter

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

# ---------------------------------------------------------------- parameters
# Free variables: placeholders until the rover exists. Sweep these.
PARAMS = dict(
    # rover
    mass_kg=100.0,
    eta_drive=0.70,              # motor + gearbox + controller efficiency
    eta_regen=0.0,               # fraction recovered when braking (0 = none)
    cda_m2=0.5,                  # drag coefficient x frontal area
    speed_ms=1.0,
    p_elec_w=30.0,               # computer, sensors, radio
    t_inside_c=20.0,             # temperature heaters hold inside the rover
    k0_w_per_k=0.5,              # heat leak per degree, calm air
    k1_w_per_k_per_ms=0.05,      # extra heat leak per degree per m/s of wind
    max_slope_deg=15.0,          # cells steeper than this are off-limits
    block_buffer_cells=1,        # extra no-go cells around steep cells

    # snow type -> Crr (smooth snow, no bumps)
    crr_soft=0.18,
    crr_medium=0.12,
    crr_hard=0.07,
    crr_default=0.12,            # used when there's no ERA5 at all
    snow_hard_wind_ms=6.0,       # mean 10 m wind >= this -> wind-packed (hard)
    snow_soft_mm=5.0,            # recent snowfall (mm water) >= this -> soft
    snow_recent_days=30,         # "recent" window for snowfall

    # roughness -> extra Crr
    rough_window_m=50.0,         # area around each spot used to measure bumps
    rough_floor_m=None,          # None = auto: smoothest 10% of the area = noise
    rough_floor_pct=10.0,
    c_rough_per_m=0.5,           # +Crr per metre of bumpiness above the floor
    rough_max_m=3.0,             # bigger "bumps" = data glitches/terrain -> ignored
    crr_max=0.5,

    # weather
    wind_height_factor=0.8,      # 10 m wind -> rover-height wind
    storm_limit_ms=15.0,         # rover parks above this (10 m wind speed)
    lapse_k_per_m=0.0065,        # temperature drop per metre of height
    use_lapse=True,
    t_outside_c_default=-30.0,   # weather OFF
    p_air_pa_default=65000.0,    # weather OFF (roughly Dome C pressure)

    # smoothing
    smooth=True,
    smooth_lookahead=60,         # keep trying this many cells past a failed shortcut
)

G = 9.81
R_AIR = 287.05

# 16 directions as (d_row, d_col), clockwise from north. Row grows downward.
DIRS = np.array([[-1, 0], [-2, 1], [-1, 1], [-1, 2], [0, 1], [1, 2], [1, 1], [2, 1],
                 [1, 0], [2, -1], [1, -1], [1, -2], [0, -1], [-1, -2], [-1, -1], [-2, -1]],
                dtype=np.int64)
DIR_NAMES = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
ND = len(DIRS)
DIR_LEN = np.hypot(DIRS[:, 0], DIRS[:, 1]).astype(np.float64)       # in cells
DIR_XY = np.stack([DIRS[:, 1] / DIR_LEN, -DIRS[:, 0] / DIR_LEN], axis=1)  # map x, y
_ang = np.arctan2(DIR_XY[:, 1], DIR_XY[:, 0])                         # map angle
ANG_ORDER = np.argsort(_ang).astype(np.int64)
ANG_SORTED = _ang[ANG_ORDER].astype(np.float64)

DEFAULT_START = (1347000.0, -888000.0)
DEFAULT_GOAL = (1363200.0, -896050.0)    # straight line passes through Concordia
ALL_FEATURES = ("weather", "snow", "rough")


# ================================================================ numba core
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
def _knight_clear(blocked, r, c, dr, dc):
    """A knight move passes between two cells; both must be passable."""
    if abs(dc) == 2:
        h = dc // 2
        return not (blocked[r, c + h] or blocked[r + dr, c + h])
    if abs(dr) == 2:
        h = dr // 2
        return not (blocked[r + h, c] or blocked[r + h, c + dc])
    return True


@njit(cache=True)
def _edge(z, crr, drag, tcost, r, c, d, nr, nc, L, m, eta_d, eta_r):
    """Returns (total, rolling W, gravity W, drag W, drive E, time E) in J.
    Crr and time cost are averaged over the two end cells."""
    cr = 0.5 * (crr[r, c] + crr[nr, nc])
    tc = 0.5 * (tcost[r, c] + tcost[nr, nc])
    w_roll = cr * m * 9.81 * L
    w_grav = m * 9.81 * (z[nr, nc] - z[r, c])
    w_drag = 0.5 * (drag[d, r, c] + drag[d, nr, nc]) * L
    w = w_roll + w_grav + w_drag
    if w > 0.0:
        e_drive = w / eta_d
    else:
        e_drive = w * eta_r
    e_time = tc * L
    total = e_drive + e_time
    if total < 0.0:
        total = 0.0
    return total, w_roll, w_grav, w_drag, e_drive, e_time


@njit(cache=True)
def astar(z, crr, drag, tcost, blocked, dirs, dir_len, start, goal, res,
          m, eta_d, eta_r, tan_max, h_lin, t_min, use_h):
    """A* on the directed 16-neighbour grid. use_h=False -> Dijkstra.

    Heuristic = best-case cost to the goal (consistent -> optimal result):
      h = max(0, h_lin*D + m*g*(z_goal - z)) / eta_d + t_min*D
    D = straight-line distance (no 16-direction route can be shorter),
    h_lin = min(Crr)*m*g + min(drag), t_min = min(time cost per metre).
    """
    ny, nx = z.shape
    N = ny * nx
    nd = dirs.shape[0]
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
        D = res * math.sqrt((r - gr) ** 2 + (c - gc) ** 2)
        val = t_min * D
        if grav_ok:
            a = h_lin * D + m * 9.81 * (zg - z[r, c])
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
        for d in range(nd):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                continue
            v = nr * nx + nc
            if closed[v] or blocked[nr, nc]:
                continue
            if not _knight_clear(blocked, r, c, dr, dc):
                continue
            L = res * dir_len[d]
            if abs(z[nr, nc] - z[r, c]) > tan_max * L:
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
def _dir_index(dirs, dr, dc):
    for k in range(dirs.shape[0]):
        if dirs[k, 0] == dr and dirs[k, 1] == dc:
            return k
    return -1


@njit(cache=True)
def grid_prefix_cost(path, z, crr, drag, tcost, dirs, dir_len, nx, res, m, eta_d, eta_r):
    """Cumulative A* edge cost along a grid path (J)."""
    n = len(path)
    cum = np.zeros(n)
    for i in range(1, n):
        r, c = path[i - 1] // nx, path[i - 1] % nx
        nr, nc = path[i] // nx, path[i] % nx
        d = _dir_index(dirs, nr - r, nc - c)
        L = res * dir_len[d]
        cum[i] = cum[i - 1] + _edge(z, crr, drag, tcost, r, c, d, nr, nc, L,
                                    m, eta_d, eta_r)[0]
    return cum


@njit(cache=True)
def _bilinear(a, y, x):
    ny, nx = a.shape
    if y < 0.0:
        y = 0.0
    if x < 0.0:
        x = 0.0
    if y > ny - 1.0:
        y = ny - 1.0
    if x > nx - 1.0:
        x = nx - 1.0
    r0 = int(y)
    c0 = int(x)
    r1 = min(r0 + 1, ny - 1)
    c1 = min(c0 + 1, nx - 1)
    fy = y - r0
    fx = x - c0
    return ((1 - fy) * ((1 - fx) * a[r0, c0] + fx * a[r0, c1])
            + fy * ((1 - fx) * a[r1, c0] + fx * a[r1, c1]))


@njit(cache=True)
def _drag_heading(drag, ang_sorted, ang_order, r, c, theta):
    """Drag force for any travel heading: interpolate between the two nearest
    of the 16 directions (theta = map angle, x east / y north)."""
    n = ang_sorted.shape[0]
    k = 0
    while k < n and ang_sorted[k] <= theta:
        k += 1
    i1 = k % n
    i0 = (k - 1) % n
    a0 = ang_sorted[i0]
    a1 = ang_sorted[i1]
    if k == 0 or k == n:              # wrap-around across +-pi
        if theta < a1:
            theta += 2 * math.pi
        a1 += 2 * math.pi
    span = a1 - a0
    t = (theta - a0) / span if span > 0 else 0.0
    return (1 - t) * drag[ang_order[i0], r, c] + t * drag[ang_order[i1], r, c]


@njit(cache=True)
def line_cost(y0, x0, y1, x1, z, crr, drag, tcost, blocked, ang_sorted, ang_order,
              res, m, eta_d, eta_r, tan_max):
    """Energy of driving a straight line between two cell centres (row, col),
    sampled every half cell on the real terrain.
    Returns [total, rolling W, gravity W, drag W, drive E, time E, braked] (J);
    total = inf if the line crosses an off-limits cell or a too-steep step."""
    out = np.zeros(7)
    dy = y1 - y0
    dx = x1 - x0
    dist = math.sqrt(dx * dx + dy * dy)
    if dist == 0.0:
        return out
    nseg = max(1, int(math.ceil(dist * 2.0)))
    ds = dist * res / nseg
    theta = math.atan2(-dy, dx)
    ny, nx = z.shape
    zprev = _bilinear(z, y0, x0)
    for k in range(1, nseg + 1):
        t = k / nseg
        tm = (k - 0.5) / nseg
        rm = int(round(y0 + dy * tm))
        cm = int(round(x0 + dx * tm))
        rm = min(max(rm, 0), ny - 1)
        cm = min(max(cm, 0), nx - 1)
        if blocked[rm, cm]:
            out[0] = np.inf
            return out
        zc = _bilinear(z, y0 + dy * t, x0 + dx * t)
        dh = zc - zprev
        zprev = zc
        if abs(dh) > tan_max * ds:
            out[0] = np.inf
            return out
        w_roll = crr[rm, cm] * m * 9.81 * ds
        w_grav = m * 9.81 * dh
        w_drag = _drag_heading(drag, ang_sorted, ang_order, rm, cm, theta) * ds
        w = w_roll + w_grav + w_drag
        if w > 0.0:
            e_drive = w / eta_d
        else:
            e_drive = w * eta_r
            out[6] += -w * (1.0 - eta_r)
        e_time = tcost[rm, cm] * ds
        tot = e_drive + e_time
        if tot < 0.0:
            tot = 0.0
        out[0] += tot
        out[1] += w_roll
        out[2] += w_grav
        out[3] += w_drag
        out[4] += e_drive
        out[5] += e_time
    return out


@njit(cache=True)
def smooth_path(path, cum, nx, z, crr, drag, tcost, blocked, ang_sorted, ang_order,
                res, m, eta_d, eta_r, tan_max, lookahead):
    """Greedy shortcutting: from each anchor, jump to the farthest later point
    whose straight line is allowed and costs no more than the grid route."""
    n = len(path)
    anchors = [0]
    i = 0
    while i < n - 1:
        best = i + 1
        fails = 0
        j = i + 2
        ri, ci = path[i] // nx, path[i] % nx
        while j < n and fails < lookahead:
            rj, cj = path[j] // nx, path[j] % nx
            lc = line_cost(float(ri), float(ci), float(rj), float(cj), z, crr, drag,
                           tcost, blocked, ang_sorted, ang_order, res, m, eta_d,
                           eta_r, tan_max)[0]
            if lc <= (cum[j] - cum[i]) * (1.0 + 1e-9) + 1e-9:
                best = j
                fails = 0
            else:
                fails += 1
            j += 1
        anchors.append(best)
        i = best
    out = np.empty(len(anchors), dtype=np.int64)
    for k in range(len(anchors)):
        out[k] = path[anchors[k]]
    return out


@njit(cache=True)
def polyline_cost(pts, nx, z, crr, drag, tcost, blocked, ang_sorted, ang_order,
                  res, m, eta_d, eta_r, tan_max):
    tot = np.zeros(7)
    for k in range(len(pts) - 1):
        r0, c0 = pts[k] // nx, pts[k] % nx
        r1, c1 = pts[k + 1] // nx, pts[k + 1] % nx
        seg = line_cost(float(r0), float(c0), float(r1), float(c1), z, crr, drag,
                        tcost, blocked, ang_sorted, ang_order, res, m, eta_d,
                        eta_r, tan_max)
        for q in range(7):
            tot[q] += seg[q]
    return tot


# ================================================================ elevation
def load_dem(path):
    if not path.exists():
        sys.exit(f"Can't find {path.name} in {path.parent}")
    with rasterio.open(path) as src:
        z = src.read(1).astype(np.float64)
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        tr = src.transform
        if abs(src.res[0] - src.res[1]) > 1e-6 or tr.b != 0 or tr.d != 0:
            sys.exit("Expected a north-up raster with square cells.")
        info = dict(left=src.bounds.left, top=src.bounds.top,
                    right=src.bounds.right, bottom=src.bounds.bottom,
                    crs=src.crs, transform=tr, res=src.res[0])
    z[(z < -500) | (z > 6000)] = np.nan
    return z, info


def make_blocked(z, res, p):
    filled = np.where(np.isnan(z), np.nanmean(z), z)
    dy, dx = np.gradient(filled, res)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    blocked = (slope > p["max_slope_deg"]) | np.isnan(z)
    if p["block_buffer_cells"] > 0:
        blocked = binary_dilation(blocked, iterations=p["block_buffer_cells"])
    return blocked, filled, slope


def xy_to_index(x, y, info, shape):
    res = info["res"]
    col = int((x - info["left"]) // res)
    row = int((info["top"] - y) // res)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        sys.exit(f"Point ({x:.0f}, {y:.0f}) is outside the elevation file.")
    return row, col


def cell_centers(info, shape, rows=None):
    res = info["res"]
    cols = info["left"] + (np.arange(shape[1]) + 0.5) * res
    r = np.arange(shape[0]) if rows is None else rows
    rr = info["top"] - (r + 0.5) * res
    return np.meshgrid(cols, rr)


# ================================================================ roughness
def compute_roughness(path2m, info, shape, p):
    """Bumpiness (m) for each 10 m cell from the 2 m DEM.

    1. smooth the 2 m surface over rough_window_m  (this is the 'tilt')
    2. bumps = surface - smoothed surface          (slope removed)
    3. RMS of bumps over rough_window_m, then averaged into each 10 m cell
    Done in strips so a 100-million-cell file fits in memory.
    """
    from rasterio.windows import Window
    res10 = info["res"]
    with rasterio.open(path2m) as src:
        r2 = src.res[0]
        ratio = res10 / r2
        if abs(ratio - round(ratio)) > 1e-6:
            sys.exit(f"2 m file resolution {r2} doesn't divide {res10} m evenly.")
        ratio = int(round(ratio))
        col0 = (info["left"] - src.bounds.left) / r2
        row0 = (src.bounds.top - info["top"]) / r2
        if abs(col0 - round(col0)) > 1e-3 or abs(row0 - round(row0)) > 1e-3:
            sys.exit("2 m and 10 m files aren't aligned on the same grid. "
                     "Download both with the same box corners.")
        col0, row0 = int(round(col0)), int(round(row0))
        k = int(round(p["rough_window_m"] / r2)) | 1
        halo = k
        nodata = src.nodata
        ny, nx = shape
        out = np.full(shape, np.nan, dtype=np.float32)
        step = max(1, 4_000_000 // (nx * ratio * ratio))
        t0 = time.perf_counter()
        for tr in range(0, ny, step):
            n10 = min(step, ny - tr)
            win = Window(col0 - halo, row0 + tr * ratio - halo,
                         nx * ratio + 2 * halo, n10 * ratio + 2 * halo)
            z2 = src.read(1, window=win, boundless=True,
                          fill_value=np.nan if nodata is None else nodata).astype(np.float32)
            bad = ~np.isfinite(z2) | (z2 < -500) | (z2 > 6000)
            if nodata is not None:
                bad |= z2 == nodata
            valid = (~bad).astype(np.float32)
            fill = float(np.nanmean(np.where(bad, np.nan, z2))) if valid.any() else 0.0
            z2 = np.where(bad, fill, z2)
            smooth = uniform_filter(z2, k, mode="nearest")
            resid = z2 - smooth
            rms = np.sqrt(np.maximum(uniform_filter(resid * resid, k, mode="nearest"), 0.0))
            rms = rms[halo:halo + n10 * ratio, halo:halo + nx * ratio]
            vfrac = uniform_filter(valid, k, mode="nearest")[halo:halo + n10 * ratio,
                                                             halo:halo + nx * ratio]
            rms10 = np.sqrt((rms * rms).reshape(n10, ratio, nx, ratio).mean(axis=(1, 3)))
            v10 = vfrac.reshape(n10, ratio, nx, ratio).mean(axis=(1, 3))
            rms10[v10 < 0.5] = np.nan
            out[tr:tr + n10] = rms10
            print(f"\r  roughness: {(tr + n10) / ny * 100:5.1f}%  "
                  f"({time.perf_counter() - t0:.0f} s)", end="", flush=True)
        print()
    return out


def load_or_compute_roughness(info, shape, p, recompute=False):
    path2m = HERE / "REMA_2m_elevation.tif"
    cache = OUT_DIR / "roughness_10m.tif"
    if cache.exists() and not recompute:
        with rasterio.open(cache) as src:
            if src.shape == shape and src.transform == info["transform"]:
                print(f"Roughness: using cached {cache.name}")
                return src.read(1).astype(np.float32)
    if not path2m.exists():
        return None
    print(f"Roughness: computing from {path2m.name} (one time, then cached)")
    rough = compute_roughness(path2m, info, shape, p)
    with rasterio.open(cache, "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="float32", crs=info["crs"],
                       transform=info["transform"], nodata=np.nan) as dst:
        dst.write(rough, 1)
    return rough


# ================================================================ ERA5
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
        return None
    ds = xr.concat([_open_era5_file(f) for f in files], dim="time",
                   data_vars="all", coords="minimal", compat="override")
    ds = ds.sortby("time")
    if months:
        ds = ds.sel(time=ds["time"].dt.month.isin(months))
    if ds.sizes["time"] == 0:
        sys.exit("No ERA5 hours left after the month filter.")
    missing = [v for v in ("u10", "v10", "t2m", "sp") if v not in ds]
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


def era5_fields(z, info, p, months, era5_dir):
    """Everything ERA5 gives us, already on the 10 m grid."""
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    loaded = load_era5(era5_dir, months)
    if loaded is None:
        return None
    ds, files, z_era = loaded
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    u = ds["u10"].values.astype(np.float64)        # (time, lat, lon)
    vv = ds["v10"].values.astype(np.float64)
    t2m = ds["t2m"].values.astype(np.float64)
    sp = ds["sp"].values.astype(np.float64)
    has_sf = "sf" in ds
    sf = ds["sf"].values.astype(np.float64) if has_sf else None
    n_hours = u.shape[0]

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
    wx = f * (u * e_hat[..., 0] + vv * n_hat[..., 0])
    wy = f * (u * e_hat[..., 1] + vv * n_hat[..., 1])

    speed10 = np.hypot(u, vv)
    driving = speed10 <= p["storm_limit_ms"]
    storm_frac = 1.0 - driving.mean(axis=0)
    rho = sp / (R_AIR * t2m)
    V = p["speed_ms"]

    drag_pts = np.empty((ND,) + lat.shape + lon.shape)
    n_drive = np.maximum(driving.sum(axis=0), 1)
    for d in range(ND):
        along = wx * DIR_XY[d, 0] + wy * DIR_XY[d, 1]    # tailwind > 0
        vrel = V - along
        F = 0.5 * rho * p["cda_m2"] * vrel * np.abs(vrel)
        drag_pts[d] = np.where(driving, F, 0.0).sum(axis=0) / n_drive

    t_c = t2m - 273.15
    k = p["k0_w_per_k"] + p["k1_w_per_k_per_ms"] * f * speed10
    heater_pts = np.maximum(0.0, k * (p["t_inside_c"] - t_c)).mean(axis=0)
    k_mean_pts = k.mean(axis=0)
    wind_mean_pts = speed10.mean(axis=0)
    if has_sf:
        snow_mm_pts = sf.mean(axis=0) * 24.0 * p["snow_recent_days"] * 1000.0
    else:
        snow_mm_pts = np.zeros_like(wind_mean_pts)

    stack = [drag_pts[d] for d in range(ND)] + [heater_pts, k_mean_pts, storm_frac,
                                                 wind_mean_pts, snow_mm_pts]
    if z_era is not None:
        stack.append(z_era)
    vals = np.stack(stack, axis=-1)
    order = np.argsort(lat)
    lat_s, vals = lat[order], vals[order]
    order = np.argsort(lon)
    lon_s, vals = lon[order], vals[:, order]
    interp = RegularGridInterpolator((lat_s, lon_s), vals, method="linear",
                                     bounds_error=False, fill_value=None)

    # interpolate in row strips to keep memory low
    ny, nx = z.shape
    nl = vals.shape[-1]
    grid = np.empty((nl, ny, nx), dtype=np.float32)
    inv = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    for r0 in range(0, ny, 250):
        rows = np.arange(r0, min(r0 + 250, ny))
        X, Y = cell_centers(info, z.shape, rows)
        clon, clat = inv.transform(X.ravel(), Y.ravel())
        part = interp(np.column_stack([clat, clon])).reshape(len(rows), nx, nl)
        grid[:, r0:r0 + len(rows)] = np.moveaxis(part, -1, 0)

    drag = grid[:ND]
    heater = grid[ND].astype(np.float64)
    k_mean = grid[ND + 1]
    storm = np.clip(grid[ND + 2], 0.0, 0.95)
    wind_mean = grid[ND + 3]
    snow_mm = np.maximum(grid[ND + 4], 0.0)
    if z_era is not None and p["use_lapse"]:
        zf = np.where(np.isnan(z), np.nanmean(z), z)
        heater = np.maximum(0.0, heater + k_mean * p["lapse_k_per_m"] * (zf - grid[ND + 5]))
    power = (p["p_elec_w"] + heater) / (1.0 - storm)       # effective always-on W
    tcost = (power / V).astype(np.float32)

    summary = dict(
        files=len(files), hours=n_hours, months=months, has_sf=has_sf,
        t_out_c=float(np.mean(t_c)), heater_w=float(np.mean(heater)),
        storm_frac=float(np.mean(storm)),
        drag_n=[float(np.mean(drag[d])) for d in range(ND)],
        wind_mean=float(np.mean(wind_mean)),
        snow_mm=float(np.mean(snow_mm)),
    )
    return dict(drag=np.ascontiguousarray(drag), tcost=tcost,
                wind_mean=wind_mean, snow_mm=snow_mm, summary=summary)


def calm_fields(shape, p):
    """Weather OFF: calm air, one outside temperature everywhere."""
    v = p["speed_ms"]
    t_out = p["t_outside_c_default"]
    rho = p["p_air_pa_default"] / (R_AIR * (t_out + 273.15))
    f_still = 0.5 * rho * p["cda_m2"] * v * v           # drag from own motion
    drag = np.full((ND,) + shape, f_still, dtype=np.float32)
    heater = max(0.0, p["k0_w_per_k"] * (p["t_inside_c"] - t_out))
    tcost = np.full(shape, (p["p_elec_w"] + heater) / v, dtype=np.float32)
    return dict(drag=drag, tcost=tcost)


# ================================================================ Crr map
SNOW_NAMES = ["soft", "medium", "hard"]


def snow_crr(wind_mean, snow_mm, p):
    """Snow type from wind + recent snowfall -> Crr for smooth snow.
    0 = soft, 1 = medium, 2 = hard."""
    cls = np.ones(wind_mean.shape, dtype=np.int8)
    cls[snow_mm >= p["snow_soft_mm"]] = 0
    cls[wind_mean >= p["snow_hard_wind_ms"]] = 2
    table = np.array([p["crr_soft"], p["crr_medium"], p["crr_hard"]], dtype=np.float32)
    return table[cls], cls


def rough_crr(rough, p):
    r = np.where(rough > p["rough_max_m"], np.nan, rough)
    r = np.where(np.isnan(r), np.nanmedian(r), r)
    floor = p["rough_floor_m"]
    if floor is None:
        floor = float(np.percentile(r, p["rough_floor_pct"]))
    extra = (p["c_rough_per_m"] * np.maximum(0.0, r - floor)).astype(np.float32)
    return extra, floor, r


# ================================================================ running
def build_variant(on, parts, shape, p):
    """Assemble (crr, drag, tcost) for a set of switched-on features."""
    w = parts["weather"] if "weather" in on else parts["calm"]
    crr = np.full(shape, parts["crr_snow_mean"], dtype=np.float32)
    if "snow" in on:
        crr = parts["crr_snow"].copy()
    if parts.get("rough_extra") is not None:
        if "rough" in on:
            crr = crr + parts["rough_extra"]
        else:
            crr = crr + np.float32(parts["rough_extra"].mean())
    crr = np.minimum(crr, p["crr_max"]).astype(np.float32)
    return crr, w["drag"], w["tcost"]


def _kw(p):
    return (p["mass_kg"], p["eta_drive"], p["eta_regen"],
            float(np.tan(np.radians(p["max_slope_deg"]))))


def plan(z, res, crr, drag, tcost, blocked, s, g, p, use_h=True):
    m, eta_d, eta_r, tan_max = _kw(p)
    h_lin = float(np.min(crr)) * m * G + float(np.min(drag))
    t_min = float(np.min(tcost))
    t0 = time.perf_counter()
    path, cost, expanded = astar(z, crr, drag, tcost, blocked, DIRS, DIR_LEN, s, g, res,
                                 m, eta_d, eta_r, tan_max, h_lin, t_min, use_h)
    return path, cost, expanded, time.perf_counter() - t0


def smooth(path, z, res, crr, drag, tcost, blocked, nx, p):
    m, eta_d, eta_r, tan_max = _kw(p)
    if not p["smooth"]:
        return path
    cum = grid_prefix_cost(path, z, crr, drag, tcost, DIRS, DIR_LEN, nx, res, m, eta_d, eta_r)
    return smooth_path(path, cum, nx, z, crr, drag, tcost, blocked, ANG_SORTED, ANG_ORDER,
                       res, m, eta_d, eta_r, tan_max, p["smooth_lookahead"])


def score(pts, z, res, crr, drag, tcost, blocked, nx, p):
    """Energy breakdown (Wh) of a polyline through cell indices."""
    m, eta_d, eta_r, tan_max = _kw(p)
    t = polyline_cost(pts, nx, z, crr, drag, tcost, blocked, ANG_SORTED, ANG_ORDER,
                      res, m, eta_d, eta_r, tan_max)
    r, c = np.divmod(pts, nx)
    km = float(np.sum(np.hypot(np.diff(r), np.diff(c)))) * res / 1000.0
    return dict(total=t[0] / 3600, roll=t[1] / 3600, grav=t[2] / 3600, drag=t[3] / 3600,
                drive=t[4] / 3600, time=t[5] / 3600, brake=t[6] / 3600, km=km)


def print_breakdown(label, b):
    print(f"  {label}")
    print(f"    length            {b['km']:8.2f} km")
    print(f"    rolling work      {b['roll']:8.1f} Wh")
    print(f"    gravity work      {b['grav']:8.1f} Wh (net climb)")
    print(f"    drag work         {b['drag']:8.1f} Wh")
    print(f"    braked away       {b['brake']:8.1f} Wh")
    print(f"    -> drive battery  {b['drive']:8.1f} Wh (after motor losses)")
    print(f"    -> time battery   {b['time']:8.1f} Wh (electronics + heaters)")
    print(f"    TOTAL             {b['total']:8.1f} Wh")


# ================================================================ pictures
# Categorical slots in fixed order (route identity), status red for off-limits,
# single-hue blue ramp for magnitudes, blue<->red with grey middle for +/-.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
CRITICAL = "#d03b3b"
INK, INK2, MUTED, GRIDC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    return seq, div


def _hillshade(zf, res):
    dy, dx = np.gradient(zf, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    return np.clip(np.sin(alt) * np.cos(slope)
                   + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


def _pts_xy(pts, nx, info):
    r, c = np.divmod(np.asarray(pts), nx)
    res = info["res"]
    return ((info["left"] + (c + 0.5) * res) / 1000,
            (info["top"] - (r + 0.5) * res) / 1000)


def _style(ax, title, sub, width=58):
    import textwrap
    sub = "\n".join(textwrap.wrap(sub, width))
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    try:
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.locator_params(nbins=5)
    except AttributeError:          # categorical axis (bar chart)
        pass


def _cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label, color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(GRIDC)
    return cb


def plot_routes(z_filled, blocked, info, routes, title, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ext = [info["left"] / 1000, info["right"] / 1000,
           info["bottom"] / 1000, info["top"] / 1000]
    nx = z_filled.shape[1]
    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#fcfcfb")
    ax.imshow(_hillshade(z_filled, info["res"]), cmap="gray", vmin=-0.3, vmax=1.2, extent=ext)
    mask = np.ma.masked_where(~blocked, np.ones_like(blocked, dtype=float))
    from matplotlib.colors import ListedColormap
    ax.imshow(mask, cmap=ListedColormap([CRITICAL]), extent=ext, interpolation="nearest")
    handles = []
    for i, (name, pts, grid, wh) in enumerate(routes):
        col = SERIES[i % len(SERIES)]
        if i == 0 and grid is not None:
            gx, gy = _pts_xy(grid, nx, info)
            ax.plot(gx, gy, lw=1, color=col, alpha=0.35, ls=":", zorder=4)
        xs, ys = _pts_xy(pts, nx, info)
        h, = ax.plot(xs, ys, lw=2.4 if i == 0 else 1.6, color=col, zorder=10 - i,
                     label=f"{name}: {wh:,.0f} Wh")
        handles.append(h)
    if routes:
        xs, ys = _pts_xy(routes[0][1], nx, info)
        handles.append(ax.plot(xs[0], ys[0], "o", ms=11, mec=INK, mfc="#fcfcfb",
                               label="start", zorder=20)[0])
        handles.append(ax.plot(xs[-1], ys[-1], "s", ms=10, mec=INK, mfc=INK,
                               label="goal", zorder=20)[0])
    handles.append(Patch(color=CRITICAL, label="off-limits (too steep, e.g. station)"))
    handles.append(plt.Line2D([], [], color=SERIES[0], lw=1, ls=":", alpha=0.6,
                              label="full: raw grid route before smoothing"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True,
              facecolor="#fcfcfb", edgecolor=GRIDC)
    _style(ax, title, "Grey = terrain shading only (light/dark = slope facing the "
           "light). Energy = each route scored with the FULL model.")
    ax.set_xlabel("x (km, EPSG:3031)", color=INK2)
    ax.set_ylabel("y (km, EPSG:3031)", color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved picture: {out}")


def plot_factors(info, z_filled, slope, blocked, snow_cls, rough_r, rough_floor,
                 crr, tcost, drag, route_pts, b, goal_rc, start_rc, p, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    seq, div = _cmaps()
    ext = [info["left"] / 1000, info["right"] / 1000,
           info["bottom"] / 1000, info["top"] / 1000]
    nx = z_filled.shape[1]
    rx, ry = _pts_xy(route_pts, nx, info)
    hs = _hillshade(z_filled, info["res"])

    fig, grid_axes = plt.subplots(3, 3, figsize=(18, 17), facecolor="#fcfcfb",
                                  layout="constrained")
    axes = list(grid_axes.ravel())

    def route(ax):
        ax.plot(rx, ry, color=INK, lw=1.2, alpha=0.8)

    def pct(a, lo=1, hi=99):
        v = np.nanpercentile(a, [lo, hi])
        return v[0], max(v[1], v[0] + 1e-6)

    def spread(lo, hi):
        """Warn when the colors exaggerate a tiny difference."""
        pc = 100 * (hi - lo) / max(abs(lo), 1e-9)
        return (f" NOTE: only {pc:.1f}% difference across the box - the colours "
                "exaggerate it." if pc < 5 else "")

    # 1 elevation
    ax = axes[0]
    lo, hi = pct(z_filled)
    im = ax.imshow(z_filled, cmap=seq, vmin=lo, vmax=hi, extent=ext)
    ax.imshow(hs, cmap="gray", alpha=0.25, extent=ext)
    route(ax)
    _cbar(fig, im, ax, "elevation (m)")
    _style(ax, "1. Elevation  ->  gravity", "darker = higher. Climbing costs energy; "
           "gentle descents give it back.")

    # 2 slope
    ax = axes[1]
    lo, hi = pct(slope, 1, 99.5)
    im = ax.imshow(slope, cmap=seq, vmin=0, vmax=max(hi, 0.5), extent=ext)
    mask = np.ma.masked_where(~blocked, np.ones_like(blocked, dtype=float))
    ax.imshow(mask, cmap=ListedColormap([CRITICAL]), extent=ext, interpolation="nearest")
    route(ax)
    _cbar(fig, im, ax, "slope (degrees)")
    _style(ax, "2. Slope  ->  where the rover may go",
           f"red = off-limits (> {p['max_slope_deg']:.0f} deg + buffer).")

    # 3 snow type
    ax = axes[2]
    if snow_cls is not None:
        cmap = ListedColormap([BLUE_RAMP[1], BLUE_RAMP[3], BLUE_RAMP[6]])
        im = ax.imshow(snow_cls, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                       extent=ext, interpolation="nearest")
        route(ax)
        cb = _cbar(fig, im, ax, "snow type / its Crr")
        cb.set_ticks([0, 1, 2])
        cb.set_ticklabels([f"soft\n{p['crr_soft']}", f"medium\n{p['crr_medium']}",
                           f"hard\n{p['crr_hard']}"])
        frac = [np.mean(snow_cls == k) * 100 for k in range(3)]
        sub = ("one type everywhere: ERA5 (~30 km) is too coarse to vary in this box"
               if max(frac) > 99.9 else
               f"soft {frac[0]:.0f}% / medium {frac[1]:.0f}% / hard {frac[2]:.0f}%")
    else:
        ax.text(0.5, 0.5, "snow feature off\n(no ERA5 files)", ha="center",
                va="center", color=INK2, transform=ax.transAxes)
        sub = "-"
    _style(ax, "3. Snow type  ->  base Crr", sub)

    # 4 roughness
    ax = axes[3]
    if rough_r is not None:
        lo, hi = pct(rough_r, 1, 99)
        im = ax.imshow(rough_r, cmap=seq, vmin=lo, vmax=hi, extent=ext)
        route(ax)
        cb = _cbar(fig, im, ax, "bump size (m, RMS)")
        if lo <= rough_floor <= hi:
            cb.ax.axhline(rough_floor, color=INK, lw=1.2)
        sub = (f"bumps inside each 10 m cell, from the 2 m data. Below {rough_floor:.3f} m "
               "(black line on the scale) = measurement noise, treated as smooth.")
    else:
        ax.text(0.5, 0.5, "roughness off\n(no REMA_2m_elevation.tif)", ha="center",
                va="center", color=INK2, transform=ax.transAxes)
        sub = "-"
    _style(ax, "4. Roughness  ->  extra Crr", sub)

    # 5 total Crr
    ax = axes[4]
    lo, hi = pct(crr)
    im = ax.imshow(crr, cmap=seq, vmin=lo, vmax=hi, extent=ext)
    route(ax)
    _cbar(fig, im, ax, "Crr")
    _style(ax, "5. Total Crr = snow + roughness",
           "rolling resistance = Crr x weight. Darker = harder to roll." + spread(lo, hi))

    # 6 always-on power
    ax = axes[5]
    power = tcost * p["speed_ms"]
    lo, hi = pct(power)
    im = ax.imshow(power, cmap=seq, vmin=lo, vmax=hi, extent=ext)
    route(ax)
    _cbar(fig, im, ax, "always-on power (W)")
    _style(ax, "6. Electronics + heaters  ->  time cost",
           f"electronics {p['p_elec_w']:.0f} W + heaters (colder / windier = more), "
           "incl. storm waiting. Paid every second." + spread(lo, hi))

    # 7 drag by travel direction (bar chart)
    ax = axes[6]
    mean_drag = np.array([float(np.mean(drag[d])) for d in range(ND)])
    xs = np.arange(ND)
    cols = ["#e34948" if v > 0 else "#2a78d6" for v in mean_drag]
    ax.bar(xs, mean_drag, color=cols, width=0.7, edgecolor="#fcfcfb", linewidth=1.5)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(DIR_NAMES, fontsize=8, color=INK2, rotation=90)
    ax.set_ylabel("average drag force (N)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    _style(ax, "7. Wind drag by travel direction",
           "red = headwind (costs energy), blue = tailwind (helps). Same everywhere in a "
           "small box: ERA5 is ~30 km.")
    ax.locator_params(axis="y", nbins=6)

    # 8 cost per metre toward goal on flat ground
    ax = axes[7]
    m, eta_d = p["mass_kg"], p["eta_drive"]
    dy_ = -(goal_rc[0] - start_rc[0])
    dx_ = goal_rc[1] - start_rc[1]
    heading = math.atan2(dy_, dx_)
    dnear = int(np.argmin(np.abs(np.angle(np.exp(1j * (np.arctan2(DIR_XY[:, 1],
                                                                  DIR_XY[:, 0]) - heading))))))
    cpm = (crr * m * G + drag[dnear]) / eta_d + tcost
    lo, hi = pct(cpm)
    im = ax.imshow(cpm, cmap=seq, vmin=lo, vmax=hi, extent=ext)
    route(ax)
    _cbar(fig, im, ax, "energy per metre (J/m)")
    _style(ax, f"8. Combined cost per metre (heading {DIR_NAMES[dnear]}, flat ground)",
           "rolling + wind + electronics/heaters. Darker = more expensive to cross."
           + spread(lo, hi))

    # 9 energy breakdown of the full route
    ax = axes[8]
    labels = ["rolling", "gravity\n(net)", "wind\ndrag", "braked\naway",
              "electronics\n+ heaters"]
    vals = [b["roll"] / eta_d, b["grav"] / eta_d, b["drag"] / eta_d, b["brake"], b["time"]]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=SERIES[0], width=0.6, edgecolor="#fcfcfb", linewidth=2)
    ax.axhline(0, color="#c3c2b7", lw=1)
    top = max(max(vals), 1.0)
    for x, v in zip(xs, vals):
        ax.text(x, max(v, 0) + top * 0.02, f"{v:,.0f}", ha="center", fontsize=9,
                color=INK)
    ax.set_ylim(min(0, min(vals)) - top * 0.05, top * 1.12)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color=INK2)
    ax.set_ylabel("battery energy (Wh)", color=INK2)
    ax.grid(axis="y", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"9. Where the full route's energy goes ({b['total']:,.0f} Wh total)\n"
                 "battery energy per cause (work / motor efficiency)",
                 fontsize=9.5, color=INK, loc="left")

    for ax in axes[:6] + [axes[7]]:
        ax.locator_params(nbins=4)
        ax.set_xlabel("x (km)", color=MUTED, fontsize=8)
        ax.set_ylabel("y (km)", color=MUTED, fontsize=8)
    fig.suptitle("What the planner sees: one map per factor (black line = full route)",
                 fontsize=14, color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=110)
    print(f"Saved picture: {out}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="REMA_10m_elevation.tif")
    ap.add_argument("--era5-dir", default="data/era5")
    ap.add_argument("--features", default=",".join(ALL_FEATURES),
                    help="features in the full model, e.g. 'weather,snow,rough'")
    ap.add_argument("--no-ablate", action="store_true", help="only run the full model")
    ap.add_argument("--no-smooth", action="store_true", help="report raw grid routes")
    ap.add_argument("--start", nargs=2, type=float, default=DEFAULT_START, metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, default=DEFAULT_GOAL, metavar=("X", "Y"))
    ap.add_argument("--months", default="11,12,1",
                    help="ERA5 months to average over, e.g. '12' or '11,12,1'")
    ap.add_argument("--dijkstra", action="store_true",
                    help="also run Dijkstra on the full model and check it matches A*")
    ap.add_argument("--recompute-rough", action="store_true",
                    help="recompute roughness instead of using the cached file")
    ap.add_argument("--no-show", action="store_true", help="save pictures without opening them")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    p = dict(PARAMS)
    if args.no_smooth:
        p["smooth"] = False
    dem_path = Path(args.dem)
    dem_path = dem_path if dem_path.is_absolute() else HERE / dem_path
    era5_dir = Path(args.era5_dir)
    era5_dir = era5_dir if era5_dir.is_absolute() else HERE / era5_dir
    months = [int(m) for m in args.months.split(",") if m.strip()]
    wanted = [f.strip() for f in args.features.split(",") if f.strip()]
    for f in wanted:
        if f not in ALL_FEATURES:
            sys.exit(f"Unknown feature '{f}'. Choose from {ALL_FEATURES}.")

    # ---------------- elevation + constraints
    z, info = load_dem(dem_path)
    res = info["res"]
    blocked, z_filled, slope = make_blocked(z, res, p)
    ny, nx = z.shape
    sr, sc = xy_to_index(*args.start, info, z.shape)
    gr, gc = xy_to_index(*args.goal, info, z.shape)
    for lab, (r, c) in (("start", (sr, sc)), ("goal", (gr, gc))):
        if blocked[r, c]:
            sys.exit(f"The {lab} point is in an off-limits (too steep) cell. Move it.")
    s, g = sr * nx + sc, gr * nx + gc
    straight_km = res * math.hypot(gr - sr, gc - sc) / 1000
    print(f"\nGrid {nx} x {ny} at {res:.0f} m, {blocked.mean() * 100:.3f}% off-limits, "
          f"16 directions{', smoothing on' if p['smooth'] else ''}")
    print(f"Start ({args.start[0]:.0f}, {args.start[1]:.0f})  ->  "
          f"Goal ({args.goal[0]:.0f}, {args.goal[1]:.0f})   straight line {straight_km:.2f} km\n")

    # ---------------- ingredients
    parts = {"calm": calm_fields(z.shape, p)}
    available = []
    era = None
    if "weather" in wanted or "snow" in wanted:
        t0 = time.perf_counter()
        era = era5_fields(z, info, p, months, era5_dir)
        if era is None:
            print(f"ERA5: no files in {era5_dir} -> weather and snow switched OFF")
        else:
            sm = era["summary"]
            print(f"ERA5: {sm['files']} files, {sm['hours']:,} hours (months {months}), "
                  f"built in {time.perf_counter() - t0:.1f} s")
            print(f"  mean outside temp {sm['t_out_c']:.1f} C, mean 10 m wind "
                  f"{sm['wind_mean']:.1f} m/s, storm hours {sm['storm_frac'] * 100:.1f}%")
            print("  mean drag force by direction (N): " + ", ".join(
                f"{n} {f:.2f}" for n, f in zip(DIR_NAMES[::2], sm["drag_n"][::2])))
            print(f"  mean heater power {sm['heater_w']:.1f} W")
            if "weather" in wanted:
                parts["weather"] = era
                available.append("weather")

    snow_cls = None
    if era is not None and "snow" in wanted:
        crr_snow, snow_cls = snow_crr(era["wind_mean"], era["snow_mm"], p)
        parts["crr_snow"] = crr_snow
        frac = [float(np.mean(snow_cls == k)) * 100 for k in range(3)]
        sm = era["summary"]
        note = "" if sm["has_sf"] else "  (no snowfall in ERA5 files: wind only)"
        print(f"Snow type: soft {frac[0]:.1f}%, medium {frac[1]:.1f}%, hard {frac[2]:.1f}% "
              f"of cells{note}")
        print(f"  mean recent snowfall {sm['snow_mm']:.2f} mm water / "
              f"{p['snow_recent_days']} days, mean wind {sm['wind_mean']:.1f} m/s")
        if max(frac) > 99.9:
            print("  -> one snow type everywhere: snow can't change the route in this box")
        available.append("snow")
        parts["crr_snow_mean"] = float(crr_snow.mean())
    else:
        parts["crr_snow_mean"] = p["crr_default"] if era is None else p["crr_medium"]

    parts["rough_extra"] = None
    rough_r, rough_floor = None, None
    if "rough" in wanted:
        rough = load_or_compute_roughness(info, z.shape, p, args.recompute_rough)
        if rough is None:
            print("Roughness: REMA_2m_elevation.tif not found -> rough switched OFF")
        else:
            extra, rough_floor, rough_r = rough_crr(rough, p)
            parts["rough_extra"] = extra
            pc = np.percentile(rough_r, [10, 50, 90, 99])
            print(f"Roughness (m): p10 {pc[0]:.3f}, median {pc[1]:.3f}, p90 {pc[2]:.3f}, "
                  f"p99 {pc[3]:.3f}")
            print(f"  noise floor {rough_floor:.3f} m -> extra Crr: mean {extra.mean():.4f}, "
                  f"max {extra.max():.3f}")
            available.append("rough")

    print(f"\nFull model features: {available if available else 'none (baseline only)'}")

    # ---------------- variants
    variants = [("full", set(available))]
    if not args.no_ablate and available:
        for f in available:
            variants.append((f"no_{f}", set(available) - {f}))
        if len(available) > 1:
            variants.append(("baseline", set()))
    built = {name: build_variant(on, parts, z.shape, p) for name, on in variants}
    full_layers = built["full"]

    print("(first run compiles the code; later runs start much faster)")
    results = {}
    for name, _ in variants:
        crr, drag, tcost = built[name]
        path, cost, expanded, dt = plan(z_filled, res, crr, drag, tcost, blocked, s, g, p)
        if len(path) == 0:
            print(f"[{name}] NO ROUTE (start and goal not connected)")
            continue
        t0 = time.perf_counter()
        pts = smooth(path, z_filled, res, crr, drag, tcost, blocked, nx, p)
        own = score(pts, z_filled, res, crr, drag, tcost, blocked, nx, p)
        print(f"[{name:>11}] A* grid {cost / 3600:8.1f} Wh -> smoothed {own['total']:8.1f} Wh "
              f"({own['km']:.2f} km, {len(pts)} waypoints) | {expanded:,} expanded, "
              f"{dt:.2f} s + smooth {time.perf_counter() - t0:.2f} s")
        results[name] = (path, pts)
        if name == "full" and args.dijkstra:
            _, cost_d, exp_d, dt_d = plan(z_filled, res, crr, drag, tcost, blocked,
                                          s, g, p, use_h=False)
            ok = abs(cost_d - cost) <= 1e-6 * max(1.0, cost)
            print(f"              Dijkstra grid {cost_d / 3600:8.1f} Wh, {exp_d:,} expanded "
                  f"-> {'MATCH' if ok else 'MISMATCH - heuristic bug!'}")
    if "full" not in results:
        sys.exit("Full model found no route.")

    print()
    b_full = score(results["full"][1], z_filled, res, *full_layers, blocked, nx, p)
    print_breakdown("full route (smoothed), scored with the full model", b_full)

    routes = [("full", results["full"][1], results["full"][0], b_full["total"])]
    if len(results) > 1:
        print("\n  ABLATION: every (smoothed) route scored with the FULL model")
        print(f"    {'variant':<12}{'Wh':>10}{'regret Wh':>12}{'regret %':>10}{'km':>8}")
        for name, (path, pts) in results.items():
            b = score(pts, z_filled, res, *full_layers, blocked, nx, p)
            reg = b["total"] - b_full["total"]
            print(f"    {name:<12}{b['total']:>10.1f}{reg:>12.1f}"
                  f"{100 * reg / b_full['total']:>9.2f}%{b['km']:>8.2f}")
            if name != "full":
                routes.append((name, pts, None, b["total"]))
        print("    regret = extra energy a route needs because that feature was ignored")
        print("    (small negative regret = smoothing noise; treat |regret| < ~0.1% as zero)")

    plot_routes(z_filled, blocked, info, routes,
                f"Routes - full model = {sorted(available) if available else 'baseline'}",
                OUT_DIR / "routes.png")
    plot_factors(info, z_filled, slope, blocked, snow_cls, rough_r, rough_floor,
                 full_layers[0], full_layers[2], full_layers[1], results["full"][1],
                 b_full, (gr, gc), (sr, sc), p, OUT_DIR / "factors.png")
    for name, (path, pts) in results.items():
        for tag, arr in (("", pts), ("_grid", path)):
            r, c = np.divmod(arr, nx)
            np.savetxt(OUT_DIR / f"route_{name}{tag}.csv",
                       np.column_stack([info["left"] + (c + 0.5) * res,
                                        info["top"] - (r + 0.5) * res]),
                       delimiter=",", header="x_epsg3031,y_epsg3031", comments="",
                       fmt="%.1f")
    print(f"Saved route coordinates: {OUT_DIR / 'route_<variant>.csv'} "
          "(smoothed waypoints; *_grid.csv = raw A* cells)")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()