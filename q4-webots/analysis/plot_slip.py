"""Q4 (c) - plot the longitudinal slip along the trajectory, wet vs dry.

    python analysis/plot_slip.py            # runs "baseline" and "low_slip"

Reads data/<run>/log.csv + meta.json and writes three figures to figures/:
  slip_map.png       the toy city seen from above, trajectory coloured by s_long
  slip_profile.png   s_long against the distance travelled, both runs
  speeds.png         v_enc (encoders) vs v_x (ground truth): where the slip comes from
s_long is recomputed here from v_enc and v_x, so the filter can change without re-simulating.
Two samples are dropped: |v_x| < V_X_MIN (the quotient is meaningless near zero) and the
steps outside DRIVING, i.e. the coast after the motors stop, where the body still slides at
v_x > 0 with the wheels at a standstill and the formula reads a constant -100 %.
"""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = sys.argv[1:] or ["baseline", "low_slip"]
LABELS = {"baseline": "wet (baseline)", "low_slip": "dry (low slip)"}
V_X_MIN = 0.1            # [m/s] slip is undefined when the robot is (almost) still
DRIVING = ("GOAL", "RECALL", "WANDER", "AVOID")   # states in which the robot drives the mission
SMOOTH = 15              # samples for the rolling median (15 x 16 ms = 0.24 s)

RUN_COLORS = ["#2a78d6", "#eb6834"]
SLIP_CMAP = LinearSegmentedColormap.from_list("slip", ["#2a78d6", "#d9d8d4", "#e34948"])
INK, INK2, GRID, SURFACE = "#2b2440", "#5a5670", "#e6e3ee", "#fcfbff"

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 140, "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def load(run):
    d = pd.read_csv(os.path.join(ROOT, "data", run, "log.csv"))
    with open(os.path.join(ROOT, "data", run, "meta.json")) as f:
        meta = json.load(f)
    valid = (d["v_x"].abs() > V_X_MIN) & d["state"].isin(DRIVING)
    d["s"] = np.where(valid, (d["v_enc"] - d["v_x"]) / d["v_x"], np.nan)
    d["dist"] = np.cumsum(np.hypot(np.diff(d["gt_x"], prepend=d["gt_x"][0]),
                                   np.diff(d["gt_y"], prepend=d["gt_y"][0])))
    d["s_smooth"] = d["s"].rolling(SMOOTH, center=True, min_periods=3).median()
    return d, meta


def friction(meta):
    for c in meta["contact_properties"]:
        if c["material2"] == "wheel":
            return c["coulombFriction"][0], c["forceDependentSlip"][0]
    return float("nan"), float("nan")


def rect(x, y, w, d, yaw, **kw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(x + px * c - py * s, y + px * s + py * c)
           for px, py in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]
    return Polygon(pts, closed=True, **kw)


PROPS = {  # footprint seen from above: ("rect", width, depth) or ("circle", radius), and colour
    "Kiosk": (("rect", 2.0, 1.6), "#9fdbcc"), "MarketStall": (("rect", 2.6, 1.5), "#b3e0fa"),
    "Statue": (("rect", 1.3, 1.3), "#cfc2ec"), "GelatoCart": (("rect", 1.4, 0.9), "#fde5ef"),
    "Bin": (("circle", 0.28), "#99d1d9"), "Bollard": (("circle", 0.12), "#cfc2ec"),
    "Bench": (("rect", 1.8, 0.6), "#f5d79c"), "StreetLamp": (("circle", 0.18), "#5a4e7a"),
}


def draw_city(ax, meta):
    ax.add_patch(Rectangle((-10, -10), 20, 20, color="#f4efe6", zorder=0))          # square
    for cx, cy, w, h in [(0, 20, 5, 20), (0, -20, 5, 20), (20, 0, 20, 5), (-20, 0, 20, 5)]:
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, color="#c9c4d6", zorder=0))
    for o in meta["objects"]:
        t, x, y = o["type"], o["x"], o["y"]
        if t == "ToyBuilding":
            w, d, _ = o["size"]
            ax.add_patch(rect(x, y, w, d, o["yaw"], fc=o["color"], ec="white", lw=1.2, zorder=1))
        elif t == "Fountain":
            ax.add_patch(Circle((x, y), 1.8, fc="#dcd0f2", ec="white", lw=1.2, zorder=2))
            ax.add_patch(Circle((x, y), 1.5, fc="#a8d8ff", zorder=2))
        elif t == "LollipopTree":
            ax.add_patch(Circle((x, y), 1.0, fc=o["color"], ec="white", lw=1, alpha=0.8, zorder=3))
        elif t == "CafeTable":
            ax.add_patch(Circle((x, y), 1.1, fc="#fbd3c4", ec="white", lw=1, alpha=0.75, zorder=3))
            ax.add_patch(Circle((x, y), 0.45, fc="white", zorder=3))
        elif t == "Planter":
            ax.add_patch(rect(x, y, o.get("length", 1.8), 0.7, o["yaw"], fc="#a3d6a3", ec="white", lw=0.8, zorder=2))
        elif t in PROPS:
            (shape, *dims), color = PROPS[t]
            if shape == "rect":
                ax.add_patch(rect(x, y, dims[0], dims[1], o["yaw"], fc=color, ec="white", lw=0.8, zorder=2))
            else:
                ax.add_patch(Circle((x, y), dims[0], fc=color, ec="white", lw=0.5, zorder=2))
        elif t == "Beacon":
            ax.add_patch(Circle((x, y), 0.42, fc=o["color"], ec=INK, lw=1.2, zorder=6))
            if o["is_goal"]:
                ax.annotate("GOAL", (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                            fontsize=9, color=INK, fontweight="bold", zorder=7,
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


if __name__ == "__main__":
    runs = {r: load(r) for r in RUNS}
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    # ---- summary table
    print(f"slip statistics over the driving states {DRIVING}, |v_x| > {V_X_MIN} m/s\n")
    print(f"{'run':<12}{'mu':>6}{'FDS':>8}  {'result':<13}{'time[s]':>8}{'path[m]':>9}"
          f"{'median s':>10}{'mean|s|':>9}{'p95|s|':>8}  decoys ignored")
    for r, (d, m) in runs.items():
        s = d["s"].dropna()
        mu, fds = friction(m)
        print(f"{r:<12}{mu:6g}{fds:8g}  {m['result']:<13}{m['duration_s']:8.1f}{d['dist'].iloc[-1]:9.1f}"
              f"{s.median():10.4f}{s.abs().mean():9.4f}{s.abs().quantile(.95):8.4f}  "
              f"{', '.join(m.get('decoys_ignored', {}))}")

    lim = 0.10               # colour scale +-10 %: the cruise slip (~5 % on wet) sits mid-way, spikes saturate
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)

    # ---- figure 1: the city from above, trajectory coloured by the slip
    fig, axes = plt.subplots(1, len(runs), figsize=(6.4 * len(runs), 6.6), squeeze=False)
    for ax, (r, (d, m)) in zip(axes[0], runs.items()):
        draw_city(ax, m)
        ax.plot(d["gt_x"], d["gt_y"], color="white", lw=6, solid_capstyle="round", zorder=4)
        ok = d["s"].notna()
        sc = ax.scatter(d.loc[ok, "gt_x"], d.loc[ok, "gt_y"], c=d.loc[ok, "s"].clip(-lim, lim),
                        cmap=SLIP_CMAP, norm=norm, s=9, lw=0, zorder=5)
        ax.plot(d["gt_x"].iloc[0], d["gt_y"].iloc[0], "o", ms=9, color=INK, zorder=7)
        ax.annotate("start", (d["gt_x"].iloc[0], d["gt_y"].iloc[0]), xytext=(9, -4),
                    textcoords="offset points", color=INK, zorder=7,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        mu, fds = friction(m)
        ax.set_title(f"{LABELS.get(r, r)}   μ = {mu:g}, FDS = {fds:g}", loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-13, 13)
        ax.set_ylim(-18.5, 12)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
    cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.75, extend="both", pad=0.02)
    cb.set_ticks(np.linspace(-lim, lim, 5))
    cb.set_label("longitudinal slip  s = (v_enc − v_x) / v_x")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:+.0f} %"))
    fig.savefig(os.path.join(ROOT, "figures", "slip_map.png"), bbox_inches="tight")

    # ---- figure 2: slip vs distance travelled
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for color, (r, (d, m)) in zip(RUN_COLORS, runs.items()):
        ax.plot(d["dist"], d["s"], color=color, lw=0.6, alpha=0.3)
        ax.plot(d["dist"], d["s_smooth"], color=color, lw=2.2, label=LABELS.get(r, r))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-0.3, 0.5)           # start/stop spikes go far beyond: they are clipped
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:+.0f} %"))
    ax.set_xlabel("distance travelled [m]")
    ax.set_ylabel("s_long")
    ax.set_title("Longitudinal slip along the way (thin: raw, thick: rolling median)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    fig.savefig(os.path.join(ROOT, "figures", "slip_profile.png"), bbox_inches="tight")

    # ---- figure 3: the two speeds behind the slip
    fig, axes = plt.subplots(len(runs), 1, figsize=(9.5, 2.7 * len(runs)), sharex=True, squeeze=False)
    for ax, (r, (d, m)) in zip(axes[:, 0], runs.items()):
        ax.plot(d["t"], d["v_enc"], color=RUN_COLORS[0], lw=1.6, label="v_enc = ω·r  (wheel encoders)")
        ax.plot(d["t"], d["v_x"], color=RUN_COLORS[1], lw=1.6, label="v_x  (ground truth)")
        ax.set_ylabel("speed [m/s]")
        ax.set_title(LABELS.get(r, r), loc="left")
        ax.legend(loc="lower right", ncol=2)
    axes[-1, 0].set_xlabel("time [s]")
    fig.savefig(os.path.join(ROOT, "figures", "speeds.png"), bbox_inches="tight")
    print("figures written to", os.path.join(ROOT, "figures"))
