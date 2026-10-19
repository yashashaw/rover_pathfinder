"""
Pathfinder tests, part 2: trade-offs, tipping points and hard cases.

Part 1 asked "does the planner react?". Part 2 asks harder questions:

TRADE-OFFS (how far is a detour worth it?)
  1. crr_depth        Harder snow to the south. Would going EVEN LOWER save more?
                      Tries 30 hand-made bows of every depth and plots energy vs depth.
  2. crr_vs_cold      South = harder snow but COLDER. Heaters fight rolling resistance.
                      Sweeps insulation quality and watches the bow shrink / flip.
  3. crr_vs_speed     Same snow gradient, different driving speeds. Slower = every
                      metre costs more heater time = shorter routes win.

TIPPING POINTS (when does the planner change its mind?)
  4. soft_patch_tipping  Soft patch from Crr 0.12 to 0.40: through -> around.
  5. road_offset_tipping Packed road 0.1 -> 1.5 km away: when is it too far to use?

DIRECTION + PHYSICS CHECKS (numbers we can work out by hand)
  6. wind_both_ways   Same windy zone, drive east (headwind) vs west (tailwind).
  7. ramp_both_ways   Gentle ramp: uphill minus downhill energy must equal 2*m*g*dh/eta.
  8. flat_exact       Flat ground: energy must equal (Crr*m*g/eta + P/v) * distance.

HARD SEARCH CASES (A* must still find the exact best route)
  9. maze             Five walls, gaps alternating top/bottom.
 10. crevasse_field   Dense random crevasses with narrow passages.
 11. dead_end_trap    A U-shaped trap opening toward the start.
 12. random_worlds    12 random worlds (snow, hills, wind, cold, obstacles, random
                      start/goal): A* must match brute-force Dijkstra in every one.

Pictures: outputs/tests_pt2/<test>.png (map + chart for each test).
Run:  python test_pathfinder_pt2.py              (opens every picture at the end)
      python test_pathfinder_pt2.py --only crr_depth,crr_vs_cold
      python test_pathfinder_pt2.py --no-show
Needs pathfinder_v2.py and test_pathfinder.py in the same folder.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import pathfinder_v2 as pf
import test_pathfinder as tp
from test_pathfinder import N, RES, START, GOAL, BASE_Z, BASE_CRR, T_BASE

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tests_pt2"

# ordinal blue steps (light -> dark) for "weak -> strong" sweeps
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#104281", "#0d366b", "#0b2a55"]
GOOD, BAD = "#006300", pf.CRITICAL
INK, INK2, MUTED, GRIDC = pf.INK, pf.INK2, pf.MUTED, pf.GRIDC
R, C = tp.rc_grid()


# ------------------------------------------------------------------ helpers
def solve(w, p, start=START, goal=GOAL, dijkstra=False, smooth=True):
    s = start[0] * N + start[1]
    g = goal[0] * N + goal[1]
    path, cost, expanded, dt = pf.plan(w["z"], RES, w["crr"], w["drag"], w["tcost"],
                                       w["blocked"], s, g, p, use_h=not dijkstra)
    if len(path) == 0:
        return None
    pts = pf.smooth(path, w["z"], RES, w["crr"], w["drag"], w["tcost"], w["blocked"], N,
                    p) if smooth else path
    b = pf.score(pts, w["z"], RES, w["crr"], w["drag"], w["tcost"], w["blocked"], N, p)
    return dict(path=path, pts=pts, grid_wh=cost / 3600, wh=b["total"], km=b["km"], b=b,
                expanded=expanded, dt=dt)


def score_pts(pts, w, p):
    return pf.score(np.asarray(pts, dtype=np.int64), w["z"], RES, w["crr"], w["drag"],
                    w["tcost"], w["blocked"], N, p)


def bow(depth, n=27, start=START, goal=GOAL):
    """A smooth bow (parabola) from start to goal, `depth` cells south (+) or north (-)."""
    cols = np.linspace(start[1], goal[1], n)
    mid = 0.5 * (start[1] + goal[1])
    half = 0.5 * (goal[1] - start[1])
    rows = start[0] + depth * (1 - ((cols - mid) / half) ** 2)
    rr = np.clip(np.rint(rows).astype(int), 0, N - 1)
    cc = np.clip(np.rint(cols).astype(int), 0, N - 1)
    pts = rr * N + cc
    keep = np.concatenate([[True], np.diff(pts) != 0])
    return pts[keep]


def signed_depth(pts):
    """How far the route bows from the start row: + south, - north (cells)."""
    rr, _ = tp.route_samples(pts)
    dev = rr - START[0]
    return float(dev[np.argmax(np.abs(dev))])


def map_axes(ax, fig, img, label, w=None, cmap="Greys"):
    ext = [0, N * RES / 1000, N * RES / 1000, 0]
    lo, hi = np.nanpercentile(img, [0.5, 99.5])
    if hi - lo < 1e-9:
        hi = lo + 1e-3
    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=ext, alpha=0.55)
    if w is not None and w["blocked"].any():
        from matplotlib.colors import ListedColormap
        mask = np.ma.masked_where(~w["blocked"], np.ones((N, N)))
        ax.imshow(mask, cmap=ListedColormap([BAD]), extent=ext, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, fraction=0.05)
    cb.set_label(label, fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_edgecolor(GRIDC)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("km east", color=MUTED, fontsize=8)
    ax.set_ylabel("km south", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRIDC)


def plot_route(ax, pts, color, label=None, lw=2.2, ls="-", halo=True, z=5):
    import matplotlib.patheffects as pe
    rr, cc = np.divmod(np.asarray(pts), N)
    x = (cc + 0.5) * RES / 1000
    y = (rr + 0.5) * RES / 1000
    fx = [pe.Stroke(linewidth=lw + 2, foreground="#fcfcfb"), pe.Normal()] if halo else None
    ax.plot(x, y, color=color, lw=lw, ls=ls, label=label, path_effects=fx, zorder=z)


def endpoints(ax, start=START, goal=GOAL):
    for (r, c), mk, fc in ((start, "o", "#fcfcfb"), (goal, "s", INK)):
        ax.plot((c + 0.5) * RES / 1000, (r + 0.5) * RES / 1000, mk, ms=8, mec=INK,
                mfc=fc, zorder=20)


def chart_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def title(fig, name, ok, headline, detail):
    import textwrap
    col = GOOD if ok else BAD
    fig.suptitle(f"{name}  [{'PASS' if ok else 'FAIL'}]   {headline}", x=0.01, ha="left",
                 fontsize=13, color=col, fontweight="bold")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(detail, 190)), fontsize=8.5,
             color=INK2, va="bottom")


def new_fig(ncols=2, w=15, h=7.6):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor="#fcfcfb",
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.91))   # room for title + caption
    return fig, np.atleast_1d(axes)


def gradient_crr():
    """Soft (0.20) at the north edge -> hard (0.08) at the south edge."""
    return (0.08 + 0.12 * (1 - R / (N - 1))).astype(np.float32)


# ================================================================== tests
def t_crr_depth():
    p = tp.params()
    w = tp.world(p, crr=gradient_crr())
    r = solve(w, p)
    depths = np.arange(-20, 141, 5)
    curves = {k: [] for k in ("total", "rolling", "time")}
    for d in depths:
        b = score_pts(bow(d), w, p)
        curves["total"].append(b["total"])
        curves["rolling"].append(b["roll"] / p["eta_drive"])
        curves["time"].append(b["time"])
    tot = np.array(curves["total"])
    i_best = int(np.argmin(tot))
    d_plan = signed_depth(r["pts"])
    interior = 0 < i_best < len(depths) - 1
    ok = interior and r["wh"] <= tot.min() * 1.002

    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, w["crr"], "Crr (darker = softer snow)")
    for k, d in enumerate([0, 40, 80, 120]):
        plot_route(ax1, bow(d), ORD[k * 2], label=f"bow {d * RES / 1000:.1f} km south",
                   lw=1.4, halo=False)
    plot_route(ax1, r["pts"], INK, label=f"planner ({r['wh']:.1f} Wh)", lw=2.6)
    endpoints(ax1)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Map: Crr gets harder to the south", loc="left", fontsize=10, color=INK)

    km = depths * RES / 1000
    ax2.plot(km, tot, color=pf.SERIES[0], lw=2, marker="o", ms=4, label="total")
    ax2.plot(km, curves["rolling"], color=pf.SERIES[1], lw=2, label="rolling resistance")
    ax2.plot(km, curves["time"], color=pf.SERIES[2], lw=2, label="electronics + heaters")
    ax2.axvline(km[i_best], color=MUTED, ls=":", lw=1)
    ax2.plot(d_plan * RES / 1000, r["wh"], "*", ms=16, mfc=INK, mec="#fcfcfb",
             label=f"planner: {d_plan * RES / 1000:.2f} km, {r['wh']:.1f} Wh", zorder=10)
    ax2.annotate(f"best bow {km[i_best]:.2f} km\n{tot[i_best]:.1f} Wh", (km[i_best], tot[i_best]),
                 xytext=(15, 25), textcoords="offset points", fontsize=8, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED))
    chart_axes(ax2, "how far the route bows south (km)", "battery energy (Wh)")
    ax2.legend(fontsize=8, loc="center right")
    ax2.set_title("Energy of 33 hand-made bows: does going lower keep paying?",
                  loc="left", fontsize=10, color=INK)
    title(fig, "crr_depth", ok, "going lower pays - until it doesn't",
          f"Rolling resistance (orange) keeps falling as the route dips into harder snow, but the "
          f"extra distance makes electronics + heaters (green) keep rising. Total (blue) bottoms "
          f"out at {km[i_best]:.2f} km; deeper bows cost MORE. The planner (star) found "
          f"{r['wh']:.1f} Wh vs the best hand bow {tot.min():.1f} Wh and the straight line "
          f"{tot[list(depths).index(0)]:.1f} Wh. PASS = best depth is in the middle of the range "
          "and the planner is at least as good as the best hand bow.")
    return fig, ok, f"best bow {km[i_best]:.2f} km, planner {d_plan * RES / 1000:.2f} km, " \
                    f"{r['wh']:.1f} Wh vs best bow {tot.min():.1f} Wh"


def t_crr_vs_cold():
    ks = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    T = T_BASE - 50.0 * (R - START[0]) / (N - 1 - START[0])     # colder to the south
    T = np.minimum(T, T_BASE + 25.0)
    crr = gradient_crr()
    res = []
    for k in ks:
        p = tp.params(k0_w_per_k=k)
        w = tp.world(p, crr=crr, T=T)
        r = solve(w, p)
        res.append((k, r, signed_depth(r["pts"])))
    depth = np.array([d for _, _, d in res])
    mono = all(depth[i + 1] <= depth[i] + 3 for i in range(len(depth) - 1))
    ok = mono and depth[0] > depth[-1] + 10

    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, T, "outside temp (C)  darker = warmer", cmap="Greys")
    for i, (k, r, d) in enumerate(res):
        plot_route(ax1, r["pts"], ORD[i], label=f"heat leak {k:g} W/K", lw=2, halo=False)
    endpoints(ax1)
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.set_title("South: harder snow (good) but colder (bad). Routes by insulation quality",
                  loc="left", fontsize=10, color=INK)
    ax2.plot(ks, depth * RES / 1000, color=pf.SERIES[0], lw=2, marker="o", ms=7)
    ax2.axhline(0, color="#c3c2b7", lw=1)
    chart_axes(ax2, "heat leak through insulation (W per degree C)  - worse insulation ->",
               "bow (km)   + south / - north")
    ax2.set_title("The worse the insulation, the less the hard southern snow is worth",
                  loc="left", fontsize=10, color=INK)
    title(fig, "crr_vs_cold", ok, "rolling resistance vs heaters",
          "Each metre costs rolling (Crr x weight / efficiency) + time (always-on power / speed). "
          "South has lower Crr but is colder, so heaters draw more. With good insulation the "
          "planner dives south for the hard snow; as insulation gets leakier the cold costs more "
          "than the snow saves and the bow shrinks"
          + (", even flipping north." if depth[-1] < -5 else ".")
          + " PASS = the bow never grows as insulation gets worse, and ends clearly smaller.")
    return fig, ok, "bow by heat leak: " + ", ".join(f"{k:g}->{d * RES / 1000:+.2f} km"
                                                     for k, _, d in res)


def t_crr_vs_speed():
    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
    res = []
    for v in speeds:
        p = tp.params(speed_ms=v)
        w = tp.world(p, crr=gradient_crr())
        r = solve(w, p)
        res.append((v, r, signed_depth(r["pts"]), float(w["tcost"].mean())))
    depth = np.array([d for _, _, d, _ in res])
    ok = all(depth[i + 1] >= depth[i] - 3 for i in range(len(depth) - 1)) \
        and depth[-1] > depth[0] + 5

    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, gradient_crr(), "Crr (darker = softer)")
    for i, (v, r, d, tc) in enumerate(res):
        plot_route(ax1, r["pts"], ORD[i + 1], label=f"{v:g} m/s", lw=2, halo=False)
    endpoints(ax1)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Same snow, different driving speeds", loc="left", fontsize=10, color=INK)
    ax2.plot(speeds, depth * RES / 1000, color=pf.SERIES[0], lw=2, marker="o", ms=7,
             label="bow south (km)")
    chart_axes(ax2, "speed (m/s)", "bow south (km)")
    ax2.set_xscale("log")
    ax2.set_xticks(speeds)
    ax2.set_xticklabels([f"{v:g}" for v in speeds])
    for (v, _, d, tc) in res:
        ax2.annotate(f"time cost {tc:.0f} J/m", (v, d * RES / 1000), xytext=(6, -12),
                     textcoords="offset points", fontsize=7.5, color=INK2)
    ax2.set_title("Slow rover: every metre burns more heater time -> shorter routes win",
                  loc="left", fontsize=10, color=INK)
    title(fig, "crr_vs_speed", ok, "distance gets pricier when you drive slowly",
          "Time cost per metre = always-on power / speed. At 0.25 m/s each metre costs 4x the "
          "electronics+heater energy it costs at 1 m/s, so extra distance is expensive and the "
          "bow shrinks. Fast rovers can afford longer detours onto hard snow (drag is still "
          "small at these speeds). PASS = bow never shrinks as speed goes up, and grows overall.")
    return fig, ok, "bow by speed: " + ", ".join(f"{v:g}->{d * RES / 1000:.2f} km"
                                                 for v, _, d, _ in res)


def t_soft_patch_tipping():
    crrs = [0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40]
    patch = tp.disk(150, 150, 50)
    straight = np.array([START[0] * N + START[1], GOAL[0] * N + GOAL[1]])
    res = []
    for cv in crrs:
        p = tp.params()
        crr = np.full((N, N), BASE_CRR, dtype=np.float32)
        crr[patch] = cv
        w = tp.world(p, crr=crr)
        r = solve(w, p)
        res.append((cv, r, tp.frac_in(patch, r["pts"])))
    frac = np.array([f for _, _, f in res])
    f0 = tp.frac_in(patch, straight)
    ok = all(frac[i + 1] <= frac[i] + 0.02 for i in range(len(frac) - 1)) \
        and frac[0] > 0.9 * f0 and frac[-1] < 0.05
    tip = next((cv for cv, _, f in res if f < 0.5 * f0), None)

    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, patch.astype(float), "soft patch")
    for i, (cv, r, f) in enumerate(res):
        plot_route(ax1, r["pts"], ORD[i], label=f"patch Crr {cv:.2f}", lw=2, halo=False)
    endpoints(ax1)
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.set_title("Same patch, softer and softer snow", loc="left", fontsize=10, color=INK)
    ax2.plot(crrs, 100 * frac, color=pf.SERIES[0], lw=2, marker="o", ms=7)
    ax2.axhline(100 * f0, color=MUTED, ls="--", lw=1)
    ax2.text(crrs[-1], 100 * f0 + 1.5, "straight line", ha="right", fontsize=8, color=MUTED)
    if tip:
        ax2.axvline(tip, color=MUTED, ls=":", lw=1)
        ax2.text(tip, 5, f"  tips at ~{tip:.2f}", fontsize=8, color=INK2)
    chart_axes(ax2, "Crr inside the patch (base snow 0.12)", "% of route inside the patch")
    ax2.set_title("Tipping point: when does going around beat going through?",
                  loc="left", fontsize=10, color=INK)
    title(fig, "soft_patch_tipping", ok, "through -> partly around -> fully around",
          "A little softer snow isn't worth a detour - the planner drives straight through. Past "
          "the tipping point it starts clipping the edge, then avoids the patch completely. "
          "PASS = time in the patch never goes up as the snow gets softer, starts at 'straight "
          "through' and ends at 'fully around'.")
    return fig, ok, f"tips at Crr ~{tip}" if tip else "no tipping found"


def t_road_offset_tipping():
    offsets = [10, 20, 35, 50, 70, 90, 110, 140]
    res = []
    for off in offsets:
        p = tp.params()
        crr = np.full((N, N), BASE_CRR, dtype=np.float32)
        road = np.zeros((N, N), bool)
        r0 = START[0] - off
        road[r0 - 3:r0 + 3, 30:270] = True
        crr[road] = 0.04
        w = tp.world(p, crr=crr)
        r = solve(w, p)
        res.append((off, r, tp.frac_in(road, r["pts"])))
    frac = np.array([f for _, _, f in res])
    ok = frac[0] > 0.5 and all(frac[i + 1] <= frac[i] + 0.05 for i in range(len(frac) - 1))
    last_used = max((off for off, _, f in res if f > 0.3), default=None)

    fig, (ax1, ax2) = new_fig()
    base = np.full((N, N), BASE_CRR)
    for off in offsets:
        base[START[0] - off - 3:START[0] - off + 3, 30:270] = 0.04
    map_axes(ax1, fig, base, "Crr (light strips = road positions tried)")
    for i, (off, r, f) in enumerate(res):
        plot_route(ax1, r["pts"], ORD[i], label=f"road {off * RES / 1000:.1f} km away",
                   lw=2, halo=False)
    endpoints(ax1)
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.set_title("One road at a time, placed farther and farther north",
                  loc="left", fontsize=10, color=INK)
    km = np.array(offsets) * RES / 1000
    ax2.plot(km, 100 * frac, color=pf.SERIES[0], lw=2, marker="o", ms=7)
    chart_axes(ax2, "how far the road is from the straight line (km)", "% of route on the road")
    ax2.set_title("How far out of the way is a road still worth it?", loc="left", fontsize=10,
                  color=INK)
    title(fig, "road_offset_tipping", ok, "a road is worth a detour - up to a point",
          "Packed snow (Crr 0.04) costs about half as much per metre as normal snow (0.12), so "
          "the planner happily detours to use it. The farther away the road, the more of the "
          "saving is eaten by getting there and back"
          + (f"; it stops using it beyond ~{last_used * RES / 1000:.1f} km." if last_used and
             last_used < offsets[-1] else ".")
          + " PASS = uses a nearby road, and road use never increases as the road moves away.")
    return fig, ok, "road use: " + ", ".join(f"{o * RES / 1000:.1f}km->{100 * f:.0f}%"
                                             for o, _, f in res)


def t_wind_both_ways():
    p = tp.params()
    zone = (C > 90) & (C < 210) & (R > 95) & (R < 205)
    wx = np.where(zone, -15.0, 0.0)        # blows WEST inside the zone
    w = tp.world(p, drag=tp.drag_from_wind(wx, np.zeros((N, N)), p))
    east = solve(w, p, START, GOAL)
    west = solve(w, p, GOAL, START)
    fe, fw = tp.frac_in(zone, east["pts"]), tp.frac_in(zone, west["pts"])
    ok = fe < 0.3 and fw > 0.9 * tp.frac_in(zone, np.array([GOAL[0] * N + GOAL[1],
                                                             START[0] * N + START[1]]))
    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, zone.astype(float), "windy zone (15 m/s blowing west)")
    plot_route(ax1, east["pts"], pf.SERIES[0], label=f"driving EAST (headwind) {east['wh']:.1f} Wh")
    plot_route(ax1, west["pts"], pf.SERIES[1], label=f"driving WEST (tailwind) {west['wh']:.1f} Wh")
    endpoints(ax1)
    ax1.annotate("", xy=(1.0, 1.5), xytext=(2.0, 1.5),
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=2))
    ax1.text(1.5, 1.42, "wind", ha="center", fontsize=9, color=INK2)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Same zone, opposite directions", loc="left", fontsize=10, color=INK)
    vals = [100 * fe, 100 * fw]
    ax2.bar([0, 1], vals, color=[pf.SERIES[0], pf.SERIES[1]], width=0.5, edgecolor="#fcfcfb")
    for x, v in zip([0, 1], vals):
        ax2.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["east (headwind)", "west (tailwind)"], color=INK2)
    chart_axes(ax2, "", "% of route inside the windy zone")
    ax2.set_ylim(0, 110)
    ax2.set_title("Directed graph: the cost depends on which way you drive",
                  loc="left", fontsize=10, color=INK)
    title(fig, "wind_both_ways", ok, "avoid the headwind, ride the tailwind",
          "Going east the zone is a 15 m/s headwind, so the planner goes around it. Going west the "
          "same wind is a tailwind that pushes the rover along, so the planner stays in it. This "
          "only works because A->B and B->A are separate edges with separate costs. PASS = east "
          "route avoids the zone (< 30% inside) and the west route uses it.")
    return fig, ok, f"in zone: east {100 * fe:.0f}%, west {100 * fw:.0f}%"


def t_ramp_both_ways():
    p = tp.params()
    z = BASE_Z + 50.0 * C / (N - 1)                    # 50 m rise over 3 km (~1 deg)
    w = tp.world(p, z=z)
    up = solve(w, p, START, GOAL)
    down = solve(w, p, GOAL, START)
    dh = w["z"][GOAL] - w["z"][START]
    expect_wh = 2 * p["mass_kg"] * pf.G * dh / p["eta_drive"] / 3600
    got = up["wh"] - down["wh"]
    ok = abs(got - expect_wh) <= 0.01 * expect_wh

    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, w["z"], "elevation (m)")
    plot_route(ax1, up["pts"], pf.SERIES[0], label=f"uphill east {up['wh']:.1f} Wh")
    plot_route(ax1, down["pts"], pf.SERIES[1], label=f"downhill west {down['wh']:.1f} Wh",
               ls="--")
    endpoints(ax1)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("A gentle ramp, driven both ways", loc="left", fontsize=10, color=INK)
    ax2.bar([0, 1], [got, expect_wh], color=[pf.SERIES[0], "#c3c2b7"], width=0.5)
    for x, v in zip([0, 1], [got, expect_wh]):
        ax2.text(x, v * 1.02, f"{v:.2f} Wh", ha="center", fontsize=10, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["planner: uphill - downhill", "hand calc: 2 x m g dh / eta"],
                        color=INK2)
    chart_axes(ax2, "", "energy difference (Wh)")
    ax2.set_title("Physics check you can do on paper", loc="left", fontsize=10, color=INK)
    title(fig, "ramp_both_ways", ok, "uphill costs exactly what downhill saves",
          f"Climbing {dh:.1f} m costs m*g*dh/eta extra; on a gentle slope the same height is paid "
          f"back going down (no braking), so uphill - downhill = 2*m*g*dh/eta = {expect_wh:.2f} Wh. "
          f"Planner: {got:.2f} Wh. PASS = within 1%.")
    return fig, ok, f"difference {got:.3f} Wh vs hand calc {expect_wh:.3f} Wh"


def t_flat_exact():
    p = tp.params()
    w = tp.world(p)
    r = solve(w, p)
    L = RES * (GOAL[1] - START[1])
    rolling = BASE_CRR * p["mass_kg"] * pf.G
    drag = float(w["drag"][4].mean())
    per_m = (rolling + drag) / p["eta_drive"] + float(w["tcost"].mean())
    expect = per_m * L / 3600
    ok = abs(r["wh"] - expect) <= 0.005 * expect

    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, w["z"] + 0 * R, "elevation (m, flat)")
    plot_route(ax1, r["pts"], INK, label=f"planner {r['wh']:.2f} Wh")
    endpoints(ax1)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Flat, uniform, calm", loc="left", fontsize=10, color=INK)
    parts = [rolling / p["eta_drive"] * L / 3600, drag / p["eta_drive"] * L / 3600,
             float(w["tcost"].mean()) * L / 3600]
    ax2.bar(range(3), parts, color=pf.SERIES[0], width=0.55)
    for x, v in enumerate(parts):
        ax2.text(x, v + 1, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["rolling\nCrr*m*g*L/eta", "still-air drag", "electronics+heaters\nP*L/v"],
                        color=INK2, fontsize=8)
    chart_axes(ax2, "", "Wh")
    ax2.set_title(f"Hand calc total {expect:.2f} Wh vs planner {r['wh']:.2f} Wh",
                  loc="left", fontsize=10, color=INK)
    title(fig, "flat_exact", ok, "the energy formula, checked by hand",
          f"On flat calm ground the whole cost is (Crr*m*g + drag)/eta + P/v per metre = "
          f"{per_m:.1f} J/m over {L / 1000:.2f} km. PASS = planner within 0.5% of the hand number.")
    return fig, ok, f"planner {r['wh']:.2f} Wh vs hand {expect:.2f} Wh"


def _search_case(name, block, headline, detail, start=START, goal=GOAL):
    p = tp.params()
    w = tp.world(p, extra_block=block)
    a = solve(w, p, start, goal, smooth=False)
    d = solve(w, p, start, goal, dijkstra=True, smooth=False)
    sm = solve(w, p, start, goal)
    rr, cc = tp.route_samples(sm["pts"])
    safe = not w["blocked"][np.clip(np.rint(rr).astype(int), 0, N - 1),
                            np.clip(np.rint(cc).astype(int), 0, N - 1)].any()
    match = abs(a["grid_wh"] - d["grid_wh"]) <= 1e-6 * d["grid_wh"]
    ok = safe and match
    fig, (ax1, ax2) = new_fig()
    map_axes(ax1, fig, block.astype(float), "obstacles", w=w)
    plot_route(ax1, a["path"], MUTED, label="A* raw grid route", lw=1.2, halo=False)
    plot_route(ax1, sm["pts"], INK, label=f"smoothed route {sm['wh']:.1f} Wh", lw=2.4)
    endpoints(ax1, start, goal)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Red = off-limits", loc="left", fontsize=10, color=INK)
    ax2.bar([0, 1], [a["expanded"], d["expanded"]], color=[pf.SERIES[0], "#c3c2b7"], width=0.5)
    for x, (v, e) in enumerate([(a["expanded"], a["grid_wh"]), (d["expanded"], d["grid_wh"])]):
        ax2.text(x, v * 1.02, f"{v:,} cells\n{e:.3f} Wh", ha="center", fontsize=9, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["A*", "Dijkstra (checks everything)"], color=INK2)
    chart_axes(ax2, "", "cells explored")
    ax2.set_title("Same answer, less work", loc="left", fontsize=10, color=INK)
    title(fig, name, ok, headline,
          detail + " PASS = A* energy equals brute-force Dijkstra exactly, and the smoothed "
          "route never touches an off-limits cell.")
    return fig, ok, f"A* {a['grid_wh']:.3f} vs Dijkstra {d['grid_wh']:.3f} Wh, " \
                    f"A* explored {a['expanded']:,} vs {d['expanded']:,} cells"


def t_maze():
    block = np.zeros((N, N), bool)
    for i, c in enumerate([60, 105, 150, 195, 240]):
        block[:, c - 2:c + 2] = True
        if i % 2 == 0:
            block[10:40, c - 2:c + 2] = False      # gap at the top
        else:
            block[260:290, c - 2:c + 2] = False    # gap at the bottom
    return _search_case("maze", block, "five walls, gaps alternating top / bottom",
                        "The only way through zig-zags the full height of the map five times.")


def t_crevasse_field():
    from scipy.ndimage import gaussian_filter, label
    rng = np.random.default_rng(7)
    noise = gaussian_filter(rng.normal(size=(N, N)), 4)
    block = noise > np.percentile(noise, 70)          # ~30% of the map is crevasses
    block[START[0] - 6:START[0] + 7, START[1] - 6:START[1] + 7] = False
    block[GOAL[0] - 6:GOAL[0] + 7, GOAL[1] - 6:GOAL[1] + 7] = False
    lab, _ = label(~block)
    if lab[START] != lab[GOAL]:                       # carve a guaranteed passage
        block[np.abs(R - (START[0] + 40 * np.sin(C / 30))) < 3] = False
    return _search_case("crevasse_field", block, "30% of the ground is crevasses",
                        "Random crevasse blobs with narrow passages between them; the route has "
                        "to thread the gaps.")


def t_dead_end_trap():
    block = np.zeros((N, N), bool)
    block[60:64, 90:240] = True          # top arm
    block[236:240, 90:240] = True        # bottom arm
    block[60:240, 236:240] = True        # back wall, between start and goal
    return _search_case("dead_end_trap", block, "a U-shaped trap facing the start",
                        "Greedy 'head straight for the goal' would drive into the U and get "
                        "stuck. A* has to back out and go around the arm.")


def t_random_worlds(n=12):
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(17, 14), facecolor="#fcfcfb",
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.95))
    for i, ax in enumerate(axes.ravel()[:n]):
        p = tp.params(k0_w_per_k=float(rng.uniform(0.2, 2.0)),
                      speed_ms=float(rng.uniform(0.5, 2.0)))
        def field(sigma):
            f = gaussian_filter(rng.normal(size=(N, N)), sigma)
            return f / f.std()
        crr = (0.12 + 0.05 * field(20)).clip(0.04, 0.35)
        z = BASE_Z + 0 * R
        for _ in range(rng.integers(1, 5)):
            cr, cc = rng.uniform(40, 260, 2)
            z = z + rng.uniform(10, 90) * np.exp(-((R - cr) ** 2 + (C - cc) ** 2)
                                                 / (2 * rng.uniform(15, 60) ** 2))
        wx = rng.uniform(2, 10) * field(40)
        wy = rng.uniform(2, 10) * field(40)
        T = T_BASE + 15 * field(30)
        block = field(6) > rng.uniform(1.6, 2.6)
        w = tp.world(p, crr=crr.astype(np.float32), z=z, extra_block=block,
                     drag=tp.drag_from_wind(wx, wy, p), T=T)
        free = np.argwhere(~w["blocked"])
        while True:
            s_ = tuple(free[rng.integers(len(free))])
            g_ = tuple(free[rng.integers(len(free))])
            if math.hypot(s_[0] - g_[0], s_[1] - g_[1]) > 150:
                break
        a = solve(w, p, s_, g_, smooth=False)
        if a is None:
            rows.append((i, True, "not connected (skipped)"))
            ax.axis("off")
            continue
        d = solve(w, p, s_, g_, dijkstra=True, smooth=False)
        sm = solve(w, p, s_, g_)
        match = abs(a["grid_wh"] - d["grid_wh"]) <= 1e-6 * d["grid_wh"]
        rr, cc2 = tp.route_samples(sm["pts"])
        safe = not w["blocked"][np.clip(np.rint(rr).astype(int), 0, N - 1),
                                np.clip(np.rint(cc2).astype(int), 0, N - 1)].any()
        ok = match and safe
        rows.append((i, ok, f"A* {a['grid_wh']:.2f} / Dijkstra {d['grid_wh']:.2f} Wh"))
        ext = [0, 3, 3, 0]
        ax.imshow(w["crr"], cmap="Greys", alpha=0.5, extent=ext)
        ax.imshow(pf._hillshade(w["z"], RES), cmap="gray", alpha=0.25, extent=ext)
        from matplotlib.colors import ListedColormap
        ax.imshow(np.ma.masked_where(~w["blocked"], np.ones((N, N))),
                  cmap=ListedColormap([BAD]), extent=ext, interpolation="nearest")
        plot_route(ax, sm["pts"], pf.SERIES[0], lw=2.2)
        endpoints(ax, s_, g_)
        ax.set_title(f"world {i + 1}  [{'PASS' if ok else 'FAIL'}]  {sm['wh']:.0f} Wh",
                     loc="left", fontsize=9, color=GOOD if ok else BAD, fontweight="bold")
        ax.text(0.01, -0.06, f"A* {a['expanded']:,} vs Dijkstra {d['expanded']:,} cells explored",
                transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
        ax.set_xticks([])
        ax.set_yticks([])
    all_ok = all(o for _, o, _ in rows)
    fig.suptitle(f"random_worlds  [{'PASS' if all_ok else 'FAIL'}]   12 random worlds: grey = "
                 "Crr (darker = softer) + hill shading, red = off-limits, blue = route",
                 x=0.01, ha="left", fontsize=13, color=GOOD if all_ok else BAD,
                 fontweight="bold")
    fig.text(0.01, 0.003, "Each world mixes soft/hard snow, 1-4 hills up to 90 m, swirling "
             "wind, warm/cold patches, random obstacles, random insulation and speed, and random "
             "start/goal. PASS = A* gives exactly the same energy as brute-force Dijkstra and the "
             "route never touches red, in every world.", fontsize=8.5, color=INK2)
    return fig, all_ok, f"{sum(o for _, o, _ in rows)}/{len(rows)} worlds match Dijkstra"


TESTS = [
    ("crr_depth", t_crr_depth),
    ("crr_vs_cold", t_crr_vs_cold),
    ("crr_vs_speed", t_crr_vs_speed),
    ("soft_patch_tipping", t_soft_patch_tipping),
    ("road_offset_tipping", t_road_offset_tipping),
    ("wind_both_ways", t_wind_both_ways),
    ("ramp_both_ways", t_ramp_both_ways),
    ("flat_exact", t_flat_exact),
    ("maze", t_maze),
    ("crevasse_field", t_crevasse_field),
    ("dead_end_trap", t_dead_end_trap),
    ("random_worlds", t_random_worlds),
]


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated test names")
    ap.add_argument("--no-show", action="store_true", help="save pictures without opening them")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tests = TESTS
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        tests = [t for t in TESTS if t[0] in keep]
        if not tests:
            sys.exit(f"No tests named {keep}. Options: {[t[0] for t in TESTS]}")

    print(f"\n{len(tests)} tests (first one compiles the planner)\n")
    passed = 0
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        fig, ok, summary = fn()
        fig.savefig(OUT / f"{name}.png", dpi=110)
        if args.no_show:
            plt.close(fig)
        passed += ok
        print(f"{name:<22}{'PASS' if ok else 'FAIL':<6}{time.perf_counter() - t0:6.1f} s   "
              f"{summary}")
    print(f"\n{passed}/{len(tests)} passed in {time.perf_counter() - t_all:.0f} s")
    print(f"Pictures: {OUT}")
    if not args.no_show:
        plt.show()
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()