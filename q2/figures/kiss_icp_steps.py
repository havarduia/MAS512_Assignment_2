# Presentation figure for the Q2 "KISS-ICP: how it works" slide: the per-scan loop of KISS-ICP, each step shown
# on our own data (not part of the assignment answers, run separately from q2/: python figures/kiss_icp_steps.py)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q2/, for results.py and the lidar code
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Assignment2/, for slide_layout.py

import results
from load_pose_tracker import LoadPoseTracker
from slide_layout import BLUE, INK, ORANGE, caption, loop_arrow, new_canvas, picture, step_arrows, step_box

HERE = Path(__file__).resolve().parents[1]  # q2/
SLIDES = HERE.parent / "slides.mplstyle"
GREY = "#b5b4af"


def transform(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def run_kiss_icp(loads, centre_0):
    # the same KISS-ICP as main.py, one scan at a time, recording what happens inside each step.
    # KISS-ICP tracks the "sensor"; the load's pose is its inverse (see load_pose_tracker.py)
    icp = LoadPoseTracker().icp
    log = []
    for points in loads:
        sigma = icp.adaptive_threshold.get_threshold()
        guess = icp.last_pose @ icp.last_delta          # constant-velocity prediction
        map_before = icp.local_map.point_cloud()
        _, source = icp.register_frame(points, np.zeros(len(points)))
        centre_guess = transform(np.linalg.inv(guess), centre_0)
        centre_found = transform(np.linalg.inv(icp.last_pose), centre_0)
        log.append(dict(sigma=sigma, guess=guess, pose=icp.last_pose, source=source, map_before=map_before,
                        centre_guess=centre_guess, centre_found=centre_found,
                        centre_error=np.linalg.norm(centre_guess - centre_found)))
    return icp, log


def bare(ax):
    ax.set(xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    plt.style.use(SLIDES)
    r = results.load()  # saved by main.py
    loads = np.split(r["load_points"], np.cumsum(r["load_sizes"])[:-1])
    centres = r["centres"]
    icp, log = run_kiss_icp(loads, centres[:1])
    voxel = icp.config.mapping.voxel_size
    # example scan: where the guess missed the load centre the most (not the first few: the map is still empty)
    k = 3 + int(np.argmax([entry["centre_error"] for entry in log[3:]]))
    step = log[k]

    fig, canvas = new_canvas()
    step_box(canvas, 0, "Downsample", f"voxel grid,\nv = {100 * voxel:.0f} cm:\n{50 * voxel:.0f} cm cells → map\n"
                                      f"{150 * voxel:.0f} cm cells → ICP")
    step_box(canvas, 1, "Predict", "constant velocity:\nguess = last pose\n× last motion")
    step_box(canvas, 2, "Register", "point-to-point ICP\nagainst the map,\nmatches within 3σ,\nrobust kernel")
    step_box(canvas, 3, "Update map", "add the scan at\nits new pose,\n≤ 20 points\nper voxel")
    step_box(canvas, 4, "Adapt σ", "σ from how far\nthe guess was off:\nsets the match\ndistance")
    step_arrows(canvas)
    loop_arrow(canvas, 0, 4, "repeated for every new scan")

    # 1: one scan, raw and downsampled for ICP (seen from the lidar: x right, z up)
    ax = picture(fig, 0, width=0.13)
    raw = loads[k]
    ax.scatter(raw[:, 0], raw[:, 2], s=2, color=GREY)
    ax.scatter(step["source"][:, 0], step["source"][:, 2], s=6, color=ORANGE)
    bare(ax)
    caption(canvas, 0, f"scan {k}: {len(raw)} points (grey)\n→ {len(step['source'])} used for ICP (orange)")

    # 2: prediction of the load centre (y) against what ICP found
    ax = picture(fig, 1, width=0.14)
    window = np.arange(max(k - 5, 0), min(k + 3, len(centres)))
    found = np.array([log[j]["centre_found"][0] for j in window])
    ax.plot(window, found[:, 1], "o-", color=BLUE, ms=5, lw=1.5, label="ICP result")
    ax.plot([k - 1, k], [found[window == k - 1][0, 1], step["centre_guess"][0, 1]], "o--", color=ORANGE, ms=5,
            lw=1.5, markerfacecolor="white", label="guess")
    ax.set_xlabel("scan", fontsize=10)
    ax.set_ylabel("load y [m]", fontsize=10)
    ax.legend(fontsize=9, loc="best", handlelength=1.5)
    caption(canvas, 1, f"guess off by {100 * step['centre_error']:.1f} cm here:\nthe swing turns around")

    # 3: registration, side view (y right, z up) of a 10 cm slice through the middle of the load, so the
    # outlines don't stack up: scan at the guess and after ICP, on the map
    ax = picture(fig, 2, width=0.14)
    at_guess, at_pose = transform(step["guess"], step["source"]), transform(step["pose"], step["source"])
    middle = np.median(step["map_before"][:, 0])
    for points, size, colour, label in [(step["map_before"], 0.5, GREY, "map"), (at_guess, 6, ORANGE, "at guess"),
                                        (at_pose, 6, BLUE, "after ICP")]:
        cut = points[np.abs(points[:, 0] - middle) < 0.05]
        ax.scatter(cut[:, 1], cut[:, 2], s=size, color=colour, label=label)
    bare(ax)
    ax.legend(fontsize=9, loc="center right", markerscale=2, handletextpad=0.2)
    caption(canvas, 2, "ICP slides the scan from the\nguess onto the map (10 cm slice)")

    # 4: the map after all scans (seen from the lidar)
    ax = picture(fig, 3, width=0.13)
    final_map = icp.local_map.point_cloud()
    ax.scatter(final_map[:, 0], final_map[:, 2], s=0.2, color=INK, alpha=0.4)
    bare(ax)
    caption(canvas, 3, f"map after {len(loads)} scans:\n{len(final_map)} points, all stacked\nonto scan 0")

    # 5: the adaptive threshold over the run
    ax = picture(fig, 4, width=0.14)
    ax.plot([entry["sigma"] for entry in log], color=BLUE, lw=1.5)
    ax.axvline(k, color=ORANGE, lw=1, ls="--")
    ax.set_xlabel("scan", fontsize=10)
    ax.set_ylabel("σ [m]", fontsize=10)
    caption(canvas, 4, "σ grows after bad guesses\n(dashed = scan above)")

    fig.savefig(HERE / "result/q2_kiss_icp_steps.png")
    print(f"Saved result/q2_kiss_icp_steps.png (example scan {k})")


if __name__ == "__main__":
    main()
