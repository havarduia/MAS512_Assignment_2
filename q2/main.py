# Assignment 2, Q2 - Lidar
# (a) centre of the swinging load in all frames, (b) pose of the load centre in consecutive frames with KISS-ICP

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation

import results
from lidar_reader import LidarReader
from load_finder import LoadFinder
from load_pose_tracker import LoadPoseTracker

HERE = Path(__file__).resolve().parent
DATA = HERE / "lidar_data"
RESULT = HERE / "result"
SLIDES = HERE.parent / "slides.mplstyle"  # shared slide style: 16:9, large text, thick lines
# the space the load swings in, read off the first frame [m]
BOX_MIN, BOX_MAX = [0.2, 0.8, -0.3], [1.5, 3.5, 1.0]
EXAMPLE_FRAME = 24  # load swung out to one side; its full scan is saved for the slides


def style_3d(ax):
    # fewer ticks and more space for the labels, so a 3D axis stays readable on a slide
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.labelpad = 18
    ax.tick_params(labelsize=14)
    ax.set_box_aspect(None, zoom=0.82)  # leave room for the axis labels inside the figure


def plot_centres(centres):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(centres[:, 0], centres[:, 1], centres[:, 2], ".-", lw=1.5, ms=6)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax.set_title("(a) Centre of the swinging load, all frames", pad=0)
    style_3d(ax)
    fig.savefig(RESULT / "q2_centres.png")


def plot_position(poses, centres):
    # one panel per axis, so the small x and z motion is not flattened by the large swing in y
    frames = np.arange(len(poses))
    fig, axes = plt.subplots(3, 1, sharex=True)
    for k, (ax, axis) in enumerate(zip(axes, "xyz")):
        ax.plot(frames, poses[:, k, 3], label="KISS-ICP")
        ax.plot(frames, centres[:, k], "--", label="cluster centroid (a)")
        ax.set_ylabel(f"{axis} [m]")
        ax.yaxis.set_major_locator(MaxNLocator(3))
    fig.suptitle("(b) Position of the load centre from KISS-ICP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)  # below the panels, off the curves
    axes[-1].set_xlabel("frame")
    fig.savefig(RESULT / "q2_pose_position.png")


def plot_orientation(poses):
    frames = np.arange(len(poses))
    angles = Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz", degrees=True)
    fig, ax = plt.subplots()
    for k, name in enumerate(["roll (about x)", "pitch (about y)", "yaw (about z)"]):
        ax.plot(frames, angles[:, k], label=name)
    ax.set(title="(b) Orientation of the load from KISS-ICP", xlabel="frame", ylabel="angle [deg]")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(top=ax.get_ylim()[1] + 8)  # room for the legend above the curves
    fig.savefig(RESULT / "q2_pose_orientation.png")


def main():
    plt.style.use(SLIDES)
    clouds = LidarReader(DATA / "swinging_load.pcap", DATA / "swinging_load.json").frames()
    print(f"{len(clouds)} lidar frames")

    # (a) centre of the load in every frame
    finder = LoadFinder(box_min=BOX_MIN, box_max=BOX_MAX)
    loads = [finder.find(points) for points in clouds]
    centres = np.array([finder.centre(load) for load in loads])
    print("(a) centre of the load, min:", centres.min(axis=0).round(3), "max:", centres.max(axis=0).round(3), "[m]")
    plot_centres(centres)

    # (b) pose of the load centre in consecutive frames with KISS-ICP
    poses = LoadPoseTracker().track(loads, centres[0])
    np.set_printoptions(precision=3, suppress=True)
    for k in (0, len(poses) // 2, len(poses) - 1):
        print(f"\n(b) pose of the load centre, frame {k}:\n{poses[k]}")
    plot_position(poses, centres)
    plot_orientation(poses)

    # everything the presentation figure scripts need, so they don't rerun the lidar pipeline
    results.save(centres=centres, poses=poses, example_frame=EXAMPLE_FRAME, example_scan=clouds[EXAMPLE_FRAME],
                 load_points=np.vstack(loads), load_sizes=[len(load) for load in loads])
    plt.show()


if __name__ == "__main__":
    main()
