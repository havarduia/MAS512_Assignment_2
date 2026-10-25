"""What Toby sees: camera frames with the colour detections drawn on top.

    python analysis/camera_montage.py            # run "baseline" by default
    python analysis/camera_montage.py low_slip

Re-runs the same HSV classification as the controller (only saturated pixels count) on the
frames saved in data/<run>/frames/ and writes figures/what_toby_sees.png:
the red beacon gets a solid box (GOAL), every other colour a dashed box (decoy, ignored).
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_FRAMES = 6

# same classes and thresholds as controllers/slip_nav/slip_nav.py
COLOURS = {"red": [(0, 14), (344, 360)], "orange": [(18, 38)], "yellow": [(40, 70)],
           "green": [(90, 168)], "cyan": [(172, 192)], "blue": [(196, 250)],
           "purple": [(255, 295)], "magenta": [(300, 338)]}
EDGE = {"red": "#e34948", "orange": "#eb6834", "yellow": "#d9a600", "green": "#1f9e4a", "cyan": "#0aa5b0",
        "blue": "#2a78d6", "purple": "#8a3fd1", "magenta": "#d42a9c"}
INK, SURFACE = "#2b2440", "#fcfbff"


def detections(img):
    """[(colour, x_min, y_min, x_max, y_max)] for every pillar in view.

    Same pixel test as the controller; pixels of one colour are then split into separate
    objects wherever there is an empty gap of a few columns (two pillars of the same colour)."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    delta = np.maximum(mx - mn, 1e-6)
    hue = np.select([mx == r, mx == g], [(g - b) / delta % 6, (b - r) / delta + 2], (r - g) / delta + 4) * 60
    vivid = (sat > 0.55) & (mx > 0.25)
    out = []
    for name, ranges in COLOURS.items():
        mask = vivid & np.any([(hue >= lo) & (hue < hi) for lo, hi in ranges], axis=0)
        if mask.mean() <= 0.0003:
            continue
        cols = np.nonzero(mask.any(axis=0))[0]
        runs = np.split(cols, np.nonzero(np.diff(cols) > 3)[0] + 1)
        for run in runs:
            ys = np.nonzero(mask[:, run[0]:run[-1] + 1].any(axis=1))[0]
            if len(run) * len(ys) >= 6:                     # drop specks
                out.append((name, run[0], ys.min(), run[-1], ys.max()))
    return out


if __name__ == "__main__":
    RUN = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    frames = sorted(glob.glob(os.path.join(ROOT, "data", RUN, "frames", "frame_*.png")))
    if not frames:
        sys.exit(f"no camera frames in data/{RUN}/frames - run the simulation first")
    log = pd.read_csv(os.path.join(ROOT, "data", RUN, "log.csv"), usecols=["t", "state"])
    times = [float(os.path.basename(f)[6:-4]) for f in frames]
    state_at = lambda t: log["state"].iloc[min(np.searchsorted(log["t"], t), len(log) - 1)]
    # the story in six frames: start, searching, first time the red beacon is seen, two on the way, arrival
    found = next((i for i, f in enumerate(frames) if any(d[0] == "red" for d in detections(plt.imread(f)[:, :, :3]))),
                 len(frames) // 2)
    searching = [i for i in range(1, found) if state_at(times[i]) == "SEARCH"]
    pick = [0, searching[len(searching) // 2] if searching else 1, found]
    pick += [int(round(v)) for v in np.linspace(found, len(frames) - 1, 4)[1:]]
    pick = sorted(set(min(i, len(frames) - 1) for i in pick))[:N_FRAMES]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(2, (len(pick) + 1) // 2, figsize=(4.2 * ((len(pick) + 1) // 2), 4.6))
    for ax, i in zip(axes.flat, pick):
        img = plt.imread(frames[i])[:, :, :3]
        ax.imshow(img, interpolation="bilinear")
        ax.set_title(f"t = {times[i]:.0f} s  ·  {state_at(times[i])}", loc="left", color=INK, fontsize=10)
        for name, x0, y0, x1, y1 in detections(img):
            goal = name == "red"
            ax.add_patch(Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, fill=False, ec=EDGE[name],
                                   lw=2.2 if goal else 1.4, ls="-" if goal else (0, (3, 2))))
            ax.text(x0 - 2, y0 - 4, "GOAL" if goal else "decoy", color=EDGE[name], fontsize=8,
                    fontweight="bold" if goal else "normal", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in list(axes.flat)[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"What Toby's camera sees ({RUN}): solid box = red goal, dashed = other colours, ignored",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "what_toby_sees.png"), dpi=150, bbox_inches="tight")
    print("written figures/what_toby_sees.png")
