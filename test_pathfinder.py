"""
Test the pathfinder on made-up terrain where we KNOW what a good route looks like.

Each scenario builds a small 3 x 3 km world (10 m cells) with ONE thing changed:
an obstacle, a patch of soft snow, a packed-snow road, a hill, a wind zone, a
cold zone, etc. It runs the exact same A* + smoothing code as pathfinder_v2.py,
then checks how the route reacted.

Every scenario is checked three ways:
  1. SAFE      - the route never enters an off-limits cell
  2. NOT WORSE - the route costs no more energy than the plain straight line
                 (when the straight line is allowed at all)
  3. EXPECTED  - the scenario's own expectation (e.g. "uses the road",
                 "goes through the gap", "stays straight in uniform wind")

Pictures (outputs/tests/):
  <scenario>.png - the factor that varies (colour), off-limits (red),
                   straight line (dashed orange), chosen route (black)
  summary.png    - all scenarios on one page

Run:  python test_pathfinder.py                 # opens the summary window at the end
      python test_pathfinder.py --each          # also one window per scenario
      python test_pathfinder.py --only soft_snow_patch,packed_road --each
Needs pathfinder_v2.py in the same folder. Takes ~1 minute (first run compiles).

Some scenarios come in PAIRS (hill_60m_goes_over / hill_100m_bends, cold_pocket_crosses / _avoids):
same obstacle, different strength. The planner should go through the mild one and
around the strong one - avoiding is only worth it when the detour is cheaper.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import pathfinder_v2 as pf

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tests"

N = 300                 # 300 x 300 cells
RES = 10.0              # 10 m cells -> 3 x 3 km world
START = (150, 20)       # (row, col)
GOAL = (150, 280)       # due east, 2.6 km away
BASE_Z = 3000.0
BASE_CRR = 0.12
T_BASE = -30.0
RHO = 0.9               # air density, kg/m^3 (high plateau)


# ------------------------------------------------------------------ helpers
def params(**over):
    p = dict(pf.PARAMS)
    p.update(over)
    return p


def rc_grid():
    return np.mgrid[0:N, 0:N].astype(np.float64)


def disk(cr, cc, radius_cells):
    R, C = rc_grid()
    return (R - cr) ** 2 + (C - cc) ** 2 <= radius_cells ** 2


def drag_from_wind(wx, wy, p):
    """Drag force (N) for each of the 16 travel directions from a wind field
    (wx east, wy north, m/s). Tailwind helps, headwind costs."""
    V = p["speed_ms"]
    out = np.empty((pf.ND, N, N), dtype=np.float32)
    for d in range(pf.ND):
        along = wx * pf.DIR_XY[d, 0] + wy * pf.DIR_XY[d, 1]
        vrel = V - along
        out[d] = 0.5 * RHO * p["cda_m2"] * vrel * np.abs(vrel)
    return out


def tcost_from_temp(T, p, wind_speed=0.0):
    k = p["k0_w_per_k"] + p["k1_w_per_k_per_ms"] * wind_speed
    heater = np.maximum(0.0, k * (p["t_inside_c"] - T))
    return ((p["p_elec_w"] + heater) / p["speed_ms"]).astype(np.float32)


def world(p, **layers):
    """Default flat, calm, uniform world; override any layer."""
    z = layers.get("z", np.full((N, N), BASE_Z))
    crr = layers.get("crr", np.full((N, N), BASE_CRR, dtype=np.float32))
    zero = np.zeros((N, N))
    drag = layers.get("drag", drag_from_wind(zero, zero, p))
    T = layers.get("T", np.full((N, N), T_BASE))
    tcost = layers.get("tcost", tcost_from_temp(T, p))
    steep, z_filled, slope = pf.make_blocked(z, RES, p)
    blocked = steep | layers.get("extra_block", np.zeros((N, N), bool))
    return dict(z=z_filled, crr=crr.astype(np.float32), drag=np.ascontiguousarray(drag),
                tcost=tcost.astype(np.float32), blocked=blocked, slope=slope)


def route_samples(pts):
    """(row, col) every ~half cell along a smoothed route."""
    rr, cc = [], []
    for k in range(len(pts) - 1):
        r0, c0 = divmod(int(pts[k]), N)
        r1, c1 = divmod(int(pts[k + 1]), N)
        n = max(2, int(math.hypot(r1 - r0, c1 - c0) * 2) + 1)
        rr.append(np.linspace(r0, r1, n))
        cc.append(np.linspace(c0, c1, n))
    return np.concatenate(rr), np.concatenate(cc)


def frac_in(mask, pts):
    """Fraction of the route length that lies inside a mask."""
    rr, cc = route_samples(pts)
    ri = np.clip(np.rint(rr).astype(int), 0, N - 1)
    ci = np.clip(np.rint(cc).astype(int), 0, N - 1)
    return float(mask[ri, ci].mean())


STRAIGHT_PTS = None   # set per scenario (start, goal)


# ------------------------------------------------------------------ scenarios
def scenarios():
    S = []

    # 1. flat, uniform: must be a straight line
    p = params()
    S.append(dict(
        name="flat_straight", p=p, w=world(p),
        title="Flat, uniform ground",
        expect="straight line (length within 0.5% of straight)",
        show=("z", "elevation (m)"),
        check=lambda r: r["km"] <= r["straight_km"] * 1.005))

    # 2. flat, awkward angle: 16 directions + smoothing still give a straight line
    p = params()
    S.append(dict(
        name="flat_diagonal", p=p, w=world(p), goal=(40, 280),
        title="Flat, awkward angle (not a multiple of 45 deg)",
        expect="still a straight line (within 0.5%)",
        show=("z", "elevation (m)"),
        check=lambda r: r["km"] <= r["straight_km"] * 1.005))

    # 3. wall with one gap
    p = params()
    wall = np.zeros((N, N), bool)
    wall[:, 148:152] = True
    wall[40:60, 148:152] = False
    w = world(p, extra_block=wall)
    S.append(dict(
        name="wall_with_gap", p=p, w=w,
        title="Wall with one gap (far from the straight line)",
        expect="crosses the wall through the gap (rows 40-60)",
        show=("blocked", "off-limits"),
        check=lambda r: r["crosses_rows"](148, 152, 40, 60)))

    # 4. keep-out circle on the straight line
    p = params()
    circ = disk(150, 150, 40)
    w = world(p, extra_block=circ)
    S.append(dict(
        name="keepout_circle", p=p, w=w,
        title="Keep-out circle (400 m) on the straight line",
        expect="goes around the circle, hugging its edge",
        show=("blocked", "off-limits"),
        check=lambda r: r["safe"] and r["km"] > r["straight_km"]))

    # 5. soft snow patch (high Crr) on the straight line
    p = params()
    crr = np.full((N, N), BASE_CRR, dtype=np.float32)
    patch = disk(150, 150, 50)
    crr[patch] = 0.30
    S.append(dict(
        name="soft_snow_patch", p=p, w=world(p, crr=crr), region=patch,
        title="Soft snow patch (Crr 0.30 vs 0.12) on the straight line",
        expect="avoids the patch (spends less of the route in it than the straight line)",
        show=("crr", "Crr"),
        check=lambda r: r["frac_route"] < r["frac_straight"] * 0.5))

    # 6. packed-snow road slightly off the straight line
    p = params()
    crr = np.full((N, N), BASE_CRR, dtype=np.float32)
    road = np.zeros((N, N), bool)
    road[115:123, 30:270] = True
    crr[road] = 0.04
    S.append(dict(
        name="packed_road", p=p, w=world(p, crr=crr), region=road,
        title="Packed-snow road (Crr 0.04, the light strip) ~300 m north of the straight line",
        expect="detours onto the road for most of the trip (> 50% of length)",
        show=("crr", "Crr"),
        check=lambda r: r["frac_route"] > 0.5))

    # 7. smooth Crr gradient: softer to the north
    p = params()
    R, C = rc_grid()
    crr = (0.08 + 0.12 * (1 - R / (N - 1))).astype(np.float32)   # 0.20 north -> 0.08 south
    S.append(dict(
        name="crr_gradient", p=p, w=world(p, crr=crr),
        title="Crr gets softer to the north (0.20) and harder to the south (0.08)",
        expect="bows south toward harder snow",
        show=("crr", "Crr"),
        check=lambda r: r["mean_row"] > START[0] + 5))

    # 8. gentle hill: going over should be (almost) free
    p = params()
    R, C = rc_grid()
    z = BASE_Z + 8.0 * np.exp(-((R - 150) ** 2 + (C - 150) ** 2) / (2 * 60.0 ** 2))
    S.append(dict(
        name="gentle_hill", p=p, w=world(p, z=z),
        title="Gentle 8 m hill (max ~0.5 deg) on the straight line",
        expect="goes straight over it (climb is paid back on the way down)",
        show=("z", "elevation (m)"),
        check=lambda r: r["km"] <= r["straight_km"] * 1.01))

    # 9a. moderate steep hill: braking loss is SMALLER than a detour -> go over
    p = params()
    hill = 60.0 * np.exp(-((R - 150) ** 2 + (C - 150) ** 2) / (2 * 25.0 ** 2))
    S.append(dict(
        name="hill_60m_goes_over", p=p, w=world(p, z=BASE_Z + hill), hill=hill,
        title="60 m hill (~8 deg): only a small part needs braking",
        expect="goes OVER: the braking loss is smaller than the extra distance around",
        show=("z", "elevation (m)"),
        check=lambda r: r["km"] <= r["straight_km"] * 1.01))

    # 9b. big steep hill: braking loss is LARGER than a detour -> go around
    p = params()
    hill = 100.0 * np.exp(-((R - 150) ** 2 + (C - 150) ** 2) / (2 * 25.0 ** 2))
    S.append(dict(
        name="hill_100m_bends", p=p, w=world(p, z=BASE_Z + hill), hill=hill,
        title="100 m hill (~13 deg): much of the way down needs braking",
        expect="bends off the top to dodge the steepest descent (>1% cheaper than straight over)",
        show=("z", "elevation (m)"),
        check=lambda r: r["km"] > r["straight_km"] and r["wh"] < r["straight_wh"] * 0.99))

    # 10. ridge too steep to cross, except at a saddle
    p = params()
    ridge = 40.0 * np.exp(-((C - 150) ** 2) / (2 * 3.0 ** 2))
    saddle = np.exp(-((R - 230) ** 2) / (2 * 12.0 ** 2))
    z = BASE_Z + ridge * (1 - saddle)
    S.append(dict(
        name="ridge_with_saddle", p=p, w=world(p, z=z),
        title="Ridge too steep to cross (> 15 deg) except a low saddle in the south",
        expect="crosses the ridge at the saddle (rows ~205-255)",
        show=("slope", "slope (deg)"),
        check=lambda r: r["crosses_rows"](147, 153, 205, 255)))

    # 11. uniform headwind: same everywhere, so it can't bend the route
    p = params()
    wx = np.full((N, N), -12.0)
    S.append(dict(
        name="uniform_headwind", p=p,
        w=world(p, drag=drag_from_wind(wx, np.zeros((N, N)), p)),
        title="12 m/s headwind everywhere (blowing from the east)",
        expect="still straight: wind that's the same everywhere can't be dodged",
        show=("drag_e", "drag heading east (N)"),
        check=lambda r: r["km"] <= r["straight_km"] * 1.005))

    # 12. headwind zone: strong wind only in one band
    p = params()
    zone = (C > 100) & (C < 200) & (R > 110) & (R < 190)
    wx = np.where(zone, -15.0, 0.0)
    S.append(dict(
        name="headwind_zone", p=p, region=zone,
        w=world(p, drag=drag_from_wind(wx, np.zeros((N, N)), p)),
        title="Strong headwind (15 m/s) only in a 1 x 0.8 km zone",
        expect="routes around the windy zone",
        show=("drag_e", "drag heading east (N)"),
        check=lambda r: r["frac_route"] < r["frac_straight"] * 0.5))

    # 13a. mild cold pocket: heating extra is SMALLER than a detour -> cross it
    cold = disk(150, 150, 40)
    T = np.where(cold, -80.0, T_BASE)
    p = params(k0_w_per_k=1.0)
    S.append(dict(
        name="cold_pocket_crosses", p=p, w=world(p, T=T), region=cold,
        title="Cold pocket (-80 C vs -30 C), normal insulation",
        expect="CROSSES it: the extra heating is less than the extra distance around",
        show=("power", "always-on power (W)"),
        check=lambda r: r["km"] <= r["straight_km"] * 1.005))

    # 13b. same pocket, leaky insulation: heating extra is LARGER -> go around
    p = params(k0_w_per_k=2.0)
    S.append(dict(
        name="cold_pocket_avoids", p=p, w=world(p, T=T), region=cold,
        title="Same cold pocket, but leaky insulation (heaters work 2x harder)",
        expect="goes AROUND the cold pocket",
        show=("power", "always-on power (W)"),
        check=lambda r: r["frac_route"] < r["frac_straight"] * 0.5))

    # 14. combined: road + soft patch + keep-out + hill
    p = params()
    crr = np.full((N, N), BASE_CRR, dtype=np.float32)
    soft = disk(170, 90, 45)
    crr[soft] = 0.28
    crr[road] = 0.04
    circ = disk(135, 210, 30)
    z = BASE_Z + 60.0 * np.exp(-((R - 175) ** 2 + (C - 200) ** 2) / (2 * 25.0 ** 2))
    S.append(dict(
        name="combined", p=p, w=world(p, crr=crr, z=z, extra_block=circ),
        title="Everything at once: road, soft patch, keep-out, steep hill",
        expect="safe and no worse than any straight line it could take",
        show=("crr", "Crr  (hill + keep-out also present)"),
        check=lambda r: r["safe"]))
    return S


# ------------------------------------------------------------------ running
def run_one(sc):
    p = sc["p"]
    w = sc["w"]
    start = sc.get("start", START)
    goal = sc.get("goal", GOAL)
    s = start[0] * N + start[1]
    g = goal[0] * N + goal[1]
    if w["blocked"][start] or w["blocked"][goal]:
        raise RuntimeError("start/goal inside an off-limits cell")
    path, cost, expanded, dt = pf.plan(w["z"], RES, w["crr"], w["drag"], w["tcost"],
                                       w["blocked"], s, g, p)
    if len(path) == 0:
        return dict(ok=False, why="no route found", pts=np.array([s, g]), path=path)
    pts = pf.smooth(path, w["z"], RES, w["crr"], w["drag"], w["tcost"], w["blocked"], N, p)
    b = pf.score(pts, w["z"], RES, w["crr"], w["drag"], w["tcost"], w["blocked"], N, p)
    straight = np.array([s, g])
    bs = pf.score(straight, w["z"], RES, w["crr"], w["drag"], w["tcost"], w["blocked"], N, p)
    rr, cc = route_samples(pts)
    ri = np.clip(np.rint(rr).astype(int), 0, N - 1)
    ci = np.clip(np.rint(cc).astype(int), 0, N - 1)
    safe = not w["blocked"][ri, ci].any()

    def crosses_rows(c_lo, c_hi, r_lo, r_hi):
        sel = (cc >= c_lo) & (cc <= c_hi)
        return bool(sel.any()) and bool(((rr[sel] >= r_lo) & (rr[sel] <= r_hi)).all())

    res = dict(
        pts=pts, path=path, wh=b["total"], km=b["km"],
        straight_wh=bs["total"], straight_km=RES * math.hypot(goal[0] - start[0],
                                                              goal[1] - start[1]) / 1000,
        safe=safe, mean_row=float(rr.mean()), crosses_rows=crosses_rows,
        expanded=expanded, dt=dt)
    if "region" in sc:
        res["frac_route"] = frac_in(sc["region"], pts)
        res["frac_straight"] = frac_in(sc["region"], straight)
    if "hill" in sc:
        res["max_hill"] = float(sc["hill"][ri, ci].max())

    # the three checks
    not_worse = (not np.isfinite(bs["total"])) or b["total"] <= bs["total"] * (1 + 1e-3)
    expected = bool(sc["check"](res))
    res["checks"] = dict(safe=safe, not_worse=not_worse, expected=expected)
    res["ok"] = safe and not_worse and expected
    return res


# ------------------------------------------------------------------ pictures
def draw(ax, fig, sc, r, compact=False):
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap
    seq, div = pf._cmaps()
    w = sc["w"]
    ext = [0, N * RES / 1000, N * RES / 1000, 0]     # km; row 0 at top
    key, label = sc["show"]
    if key == "blocked":
        img = np.where(w["blocked"], np.nan, w["z"])
        im = ax.imshow(pf._hillshade(w["z"], RES), cmap="gray", vmin=-0.3, vmax=1.2,
                       extent=ext)
        im = None
    else:
        if key == "drag_e":
            img = w["drag"][4]
        elif key == "power":
            img = w["tcost"] * sc["p"]["speed_ms"]
        else:
            img = w[key]
        lo, hi = np.nanpercentile(img, [0.5, 99.5])
        if hi - lo < 1e-6 * max(1.0, abs(lo)):          # same value everywhere
            ax.imshow(np.zeros((N, N)), cmap=seq, vmin=0, vmax=1, extent=ext)
            ax.text(0.98, 0.03, f"{label}: {lo:,.3g} everywhere", transform=ax.transAxes,
                    ha="right", fontsize=7.5, color=pf.INK2)
            im = None
        else:
            im = ax.imshow(img, cmap=seq, vmin=lo, vmax=hi, extent=ext)
    mask = np.ma.masked_where(~w["blocked"], np.ones((N, N)))
    ax.imshow(mask, cmap=ListedColormap([pf.CRITICAL]), extent=ext, interpolation="nearest")

    def xy(pts):
        rr, cc = np.divmod(np.asarray(pts), N)
        return (cc + 0.5) * RES / 1000, (rr + 0.5) * RES / 1000

    sx, sy = xy([sc.get("start", START)[0] * N + sc.get("start", START)[1],
                 sc.get("goal", GOAL)[0] * N + sc.get("goal", GOAL)[1]])
    ax.plot(sx, sy, ls="--", lw=1.4, color=pf.SERIES[1], label="straight line")
    rx, ry = xy(r["pts"])
    ax.plot(rx, ry, lw=2.6, color=pf.INK, label="chosen route",
            path_effects=[pe.Stroke(linewidth=4.5, foreground="#fcfcfb"), pe.Normal()])
    ax.plot(sx[0], sy[0], "o", ms=8, mec=pf.INK, mfc="#fcfcfb", zorder=10)
    ax.plot(sx[1], sy[1], "s", ms=8, mec=pf.INK, mfc=pf.INK, zorder=10)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    status = "PASS" if r["ok"] else "FAIL"
    color = "#006300" if r["ok"] else pf.CRITICAL
    ax.set_title(f"{sc['name']}  [{status}]", fontsize=10 if compact else 12,
                 color=color, loc="left", fontweight="bold")
    info = (f"{sc['title']}\nexpect: {sc['expect']}\n"
            f"route {r['wh']:,.1f} Wh / {r['km']:.2f} km   vs straight "
            + (f"{r['straight_wh']:,.1f} Wh / {r['straight_km']:.2f} km"
               if np.isfinite(r["straight_wh"]) else "line: not allowed"))
    if "frac_route" in r:
        info += (f"\nshare of route inside the zone: {100 * r['frac_route']:.0f}%  "
                 f"(straight line: {100 * r['frac_straight']:.0f}%)")
    ax.text(0.01, -0.07 if compact else -0.08, info, transform=ax.transAxes, va="top",
            fontsize=7.5 if compact else 9, color=pf.INK2)
    ax.tick_params(colors=pf.MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(pf.GRIDC)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, fraction=0.05)
        cb.set_label(label, fontsize=8, color=pf.INK2)
        cb.ax.tick_params(labelsize=7, colors=pf.MUTED)
        cb.outline.set_edgecolor(pf.GRIDC)
    if not compact:
        ax.legend(loc="upper right", fontsize=8, facecolor="#fcfcfb", edgecolor=pf.GRIDC)
        ax.set_xlabel("km east", color=pf.MUTED, fontsize=8)
        ax.set_ylabel("km south", color=pf.MUTED, fontsize=8)


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    ap.add_argument("--no-show", action="store_true", help="save pictures without opening them")
    ap.add_argument("--each", action="store_true",
                    help="also open every scenario's picture in its own window")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    scs = scenarios()
    if args.only:
        keep = {n.strip() for n in args.only.split(",")}
        scs = [s for s in scs if s["name"] in keep]
        if not scs:
            sys.exit(f"No scenarios named {keep}.")

    print(f"\n{len(scs)} scenarios on a {N * RES / 1000:.0f} x {N * RES / 1000:.0f} km world "
          "(first one compiles the planner)\n")
    print(f"{'scenario':<24}{'result':<7}{'safe':<6}{'<=line':<8}{'expected':<10}"
          f"{'route Wh':>10}{'line Wh':>10}{'km':>7}")
    results = []
    t_all = time.perf_counter()
    for sc in scs:
        r = run_one(sc)
        results.append((sc, r))
        ch = r.get("checks", dict(safe=False, not_worse=False, expected=False))
        mark = lambda b: "yes" if b else "NO"
        line = (f"{r['straight_wh']:>10.1f}" if np.isfinite(r.get("straight_wh", np.inf))
                else f"{'blocked':>10}")
        print(f"{sc['name']:<24}{'PASS' if r['ok'] else 'FAIL':<7}{mark(ch['safe']):<6}"
              f"{mark(ch['not_worse']):<8}{mark(ch['expected']):<10}"
              f"{r.get('wh', float('nan')):>10.1f}{line}{r.get('km', float('nan')):>7.2f}")
        fig, ax = plt.subplots(figsize=(8.5, 9.5), facecolor="#fcfcfb", layout="constrained")
        draw(ax, fig, sc, r)
        fig.savefig(OUT / f"{sc['name']}.png", dpi=120)
        if args.no_show or not args.each:
            plt.close(fig)

    n_pass = sum(r["ok"] for _, r in results)
    print(f"\n{n_pass}/{len(results)} passed in {time.perf_counter() - t_all:.1f} s")
    print("columns: safe = never enters off-limits | <=line = no worse than the straight "
          "line | expected = scenario's own check")

    cols = 4
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 6.4 * rows),
                             facecolor="#fcfcfb", layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, (sc, r) in zip(axes, results):
        draw(ax, fig, sc, r, compact=True)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(f"Pathfinder tests: {n_pass}/{len(results)} passed.  Colour = the one thing "
                 "that changes, red = off-limits, dashed orange = straight line, "
                 "black = chosen route", fontsize=13, color=pf.INK, x=0.01, ha="left")
    fig.savefig(OUT / "summary.png", dpi=90)
    print(f"Pictures: {OUT}  (summary.png + one per scenario)")
    if not args.no_show:
        plt.show()          # opens the summary (and each scenario with --each)
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()