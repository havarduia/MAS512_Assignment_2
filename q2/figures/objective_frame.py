# Presentation figure for the Q2 objective slide: one lidar frame, with the load's cluster and its centre
# (not part of the assignment answers, run separately from q2/: python figures/objective_frame.py)

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q2/, for the lidar code

from lidar_reader import LidarReader
from load_finder import LoadFinder
from main import BOX_MAX, BOX_MIN

HERE = Path(__file__).resolve().parents[1]  # q2/
DATA = HERE / "lidar_data"
SLIDES = HERE.parent / "slides.mplstyle"
FRAME = 24             # the load swung out to one side
LIMITS = [(-2.2, 3.0), (-1.5, 4.0), (-1.2, 1.3)]  # plotted area x, y, z [m]
GREY, ORANGE, RED, INK = "#9a9994", "#eb6834", "#e34948", "#0b0b0b"


def main():
    plt.style.use(SLIDES)
    points = LidarReader(DATA / "swinging_load.pcap", DATA / "swinging_load.json").frames()[FRAME]
    finder = LoadFinder(box_min=BOX_MIN, box_max=BOX_MAX)
    load = finder.find(points)
    centre = finder.centre(load)

    # matplotlib's 3D axes don't clip, so drop the points outside the plotted area
    inside = np.all([(points[:, k] > lo) & (points[:, k] < hi) for k, (lo, hi) in enumerate(LIMITS)], axis=0)
    scene = points[inside]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", computed_zorder=False)  # draw in call order: the centre stays on top
    ax.scatter(*scene.T, s=0.3, color=GREY, alpha=0.4, depthshade=False)
    ax.scatter(*load.T, s=4, color=ORANGE, depthshade=False, label=f"load cluster ({len(load)} points)")
    ax.scatter(*centre, s=350, color=RED, edgecolor="white", linewidth=2, depthshade=False, zorder=10,
               label=f"centre ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}) m")
    ax.scatter(0, 0, 0, s=250, marker="^", color=INK, depthshade=False, label="Ouster lidar")

    # crop box: its 12 edges are the corner pairs that differ in one coordinate
    corners = np.array(list(product(*zip(BOX_MIN, BOX_MAX))))
    edges = [(a, c) for a, c in product(corners, corners) if np.sum(a != c) == 1]
    for k, (a, c) in enumerate(edges):
        ax.plot(*zip(a, c), color=INK, lw=1, ls="--", alpha=0.5, label="search box" if k == 0 else None)

    ax.set(xlim=LIMITS[0], ylim=LIMITS[1], zlim=LIMITS[2], xlabel="x [m]", ylabel="y [m]")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("z [m]", rotation=90)
    ax.set_box_aspect([hi - lo for lo, hi in LIMITS], zoom=1.05)  # true proportions
    ax.view_init(elev=22, azim=-125)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
        axis.labelpad = 14
    ax.tick_params(labelsize=13)
    ax.legend(loc="upper left", fontsize=15, markerscale=1.5)

    fig.savefig(HERE / "result/q2_objective_frame.png")
    print(f"Saved result/q2_objective_frame.png (frame {FRAME})")


if __name__ == "__main__":
    main()
