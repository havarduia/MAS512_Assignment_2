# Presentation figure for the Q2 overview slide: the lidar pipeline as numbered steps, each with a small picture
# (not part of the assignment answers, run separately from q2/: python figures/overview_pipeline.py)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q2/, for results.py and the lidar code
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Assignment2/, for slide_layout.py

import results
from load_finder import LoadFinder
from main import BOX_MAX, BOX_MIN
from slide_layout import BLUE, INK, MUTED, ORANGE, caption, loop_arrow, new_canvas, picture, step_arrows, step_box

HERE = Path(__file__).resolve().parents[1]  # q2/
SLIDES = HERE.parent / "slides.mplstyle"
GREY = "#b5b4af"
SCAN_RANGE = 6.0  # top view of the scan within this distance of the lidar [m]


def search_box(ax):
    # the search box seen from above
    ax.add_patch(Rectangle(BOX_MIN[:2], BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], fill=False,
                           edgecolor=INK, lw=1.2, ls="--"))


def bare(ax):
    # picture without axes: equal scale, no ticks, grid or frame
    ax.set(xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def top_view(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim)
    bare(ax)


def main():
    plt.style.use(SLIDES)
    r = results.load()  # saved by main.py
    scan, centres, poses = r["example_scan"], r["centres"], r["poses"]
    frames = np.arange(len(centres))
    finder = LoadFinder(box_min=BOX_MIN, box_max=BOX_MAX)
    cropped, labels = finder.clusters(scan)
    load_label = np.bincount(labels[labels >= 0]).argmax()

    fig, canvas = new_canvas()
    step_box(canvas, 0, "Read", "Ouster OS1-64:\n64 × 1024 points\nper frame, 10 Hz\n(.pcap + .json)")
    step_box(canvas, 1, "Crop", "keep the points\nin the search box\nwhere the load\nswings")
    step_box(canvas, 2, "Cluster", f"DBSCAN\neps = {finder.eps} m,\nmin {finder.min_points} points;\n"
                                   "keep the largest")
    step_box(canvas, 3, "Centre", "mean of the\ncluster's points\n→ centre per frame\n(a)")
    step_box(canvas, 4, "KISS-ICP", "register the load\nframe to frame\n→ pose: rotation\n+ position  (b)")
    step_arrows(canvas)
    loop_arrow(canvas, 0, 3, f"repeated for every frame ({len(centres)} frames)")

    # 1: one scan from above
    ax = picture(fig, 0)
    near = scan[np.linalg.norm(scan[:, :2], axis=1) < SCAN_RANGE]
    ax.scatter(near[:, 0], near[:, 1], s=0.05, color=GREY)
    ax.plot(0, 0, "^", ms=8, color=INK)
    search_box(ax)
    top_view(ax, (-SCAN_RANGE, SCAN_RANGE), (-SCAN_RANGE, SCAN_RANGE))
    caption(canvas, 0, f"frame {int(r['example_frame'])} from above,\n▲ = lidar, dashed = search box")

    # 2: the points in the search box
    zoom = ((BOX_MIN[0] - 0.8, BOX_MAX[0] + 0.8), (BOX_MIN[1] - 0.5, BOX_MAX[1] + 0.5))
    ax = picture(fig, 1)
    around = scan[(scan[:, 0] > zoom[0][0]) & (scan[:, 0] < zoom[0][1]) &
                  (scan[:, 1] > zoom[1][0]) & (scan[:, 1] < zoom[1][1])]
    ax.scatter(around[:, 0], around[:, 1], s=0.3, color=GREY)
    ax.scatter(cropped[:, 0], cropped[:, 1], s=0.6, color=BLUE)
    search_box(ax)
    top_view(ax, *zoom)
    caption(canvas, 1, f"{len(cropped)} of {len(scan)} points\nare in the box (blue)")

    # 3: the clusters, close up and seen from the lidar (x right, z up)
    ax = picture(fig, 2, width=0.13)
    for label, colour in [(-1, INK), *[(k, GREY) for k in range(labels.max() + 1) if k != load_label],
                          (load_label, ORANGE)]:
        pts = cropped[labels == label]
        ax.scatter(pts[:, 0], pts[:, 2], s=3, color=colour)
    bare(ax)
    n_clusters = labels.max() + 1
    caption(canvas, 2, f"{n_clusters} cluster{'s' if n_clusters != 1 else ''}, seen from the lidar:\n"
                       f"orange = load ({np.sum(labels == load_label)} points)")

    # 4: the centres of all frames, side view of the swing
    ax = picture(fig, 3, width=0.14)
    ax.plot(centres[:, 1], centres[:, 2], ".-", color=ORANGE, lw=1, ms=4)
    ax.set_xlabel("y [m]", fontsize=10)
    ax.set_ylabel("z [m]", fontsize=10)
    caption(canvas, 3, "side view of the centre,\nall frames: swing along y")

    # 5: KISS-ICP position against the centroid
    ax = picture(fig, 4, width=0.14)
    ax.plot(frames, centres[:, 1], "--", color=ORANGE, lw=1.5, label="centroid")
    ax.plot(frames, poses[:, 1, 3], color=BLUE, lw=1.5, label="KISS-ICP")
    ax.set_xlabel("frame", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", handlelength=1.5)
    caption(canvas, 4, "KISS-ICP follows the\ncentroid along the swing")

    fig.savefig(HERE / "result/q2_overview_pipeline.png")
    print("Saved result/q2_overview_pipeline.png")


if __name__ == "__main__":
    main()
