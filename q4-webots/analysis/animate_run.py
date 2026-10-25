"""Animations of the runs for the slides.

    python analysis/animate_run.py                 # run "baseline" -> figures/run_baseline.mp4
    python analysis/animate_run.py low_slip        #                -> figures/run_low_slip.mp4
    python analysis/animate_run.py compare         # wet and dry side by side -> figures/run_compare.mp4
    python analysis/animate_run.py baseline --gif  # GIF instead of MP4 (bigger, but no ffmpeg needed)

One run: on the left the trajectory drawn so far, coloured by s_long, with Toby and its camera
field of view; top right the camera frame of that moment with the colour detections (solid box =
red goal); bottom right s_long against time.
compare: the two runs on the same clock, so you can see the wet one fall behind.
16:9, 1280 x 720, played at 2x real time. Needs data/<run>/ (written by the controller).
"""

import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Polygon, Rectangle, Wedge  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_slip import INK, INK2, LABELS, ROOT, RUN_COLORS, SLIP_CMAP, draw_city, friction, load  # noqa: E402
from camera_montage import EDGE, detections  # noqa: E402

FPS, SPEED = 15, 2.0                     # video frames per second, times faster than real time
CAM_FOV, LIM = 1.2, 0.10                 # camera field of view [rad], slip colour scale (+-10 %)


class RunView:
    """One run drawn on a map (and optionally its s_long curve), updated to any time t."""

    def __init__(self, run, ax_map, ax_s=None, color="#2a78d6"):
        self.run = run
        self.d, self.meta = load(run)
        self.mu, _ = friction(self.meta)
        self.end = self.d["t"].iloc[-1]
        draw_city(ax_map, self.meta)
        ax_map.set_aspect("equal")
        ax_map.set_xlim(-11, 11)
        ax_map.set_ylim(-17, 11)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.grid(False)
        self.trail = ax_map.scatter([], [], c=[], cmap=SLIP_CMAP, vmin=-LIM, vmax=LIM, s=10, lw=0, zorder=8)
        self.fov = Wedge((0, 0), 6, 0, 1, fc="#ffd27a", alpha=0.35, zorder=7)
        ax_map.add_patch(self.fov)
        self.toby = Polygon(np.zeros((3, 2)), closed=True, fc="#ff9a4d", ec=INK, lw=1.2, zorder=9)
        ax_map.add_patch(self.toby)
        self.s_line = None
        if ax_s is not None:
            self.s_line, = ax_s.plot([], [], color=color, lw=1.6, label=f"{LABELS.get(run, run)}, μ = {self.mu:g}")

    def row(self, t):
        return min(len(self.d) - 1, int(np.searchsorted(self.d["t"].to_numpy(), t)))

    def update(self, t):
        r = self.row(t)
        now, past = self.d.iloc[r], self.d.iloc[:r + 1]
        ok = past["s"].notna()
        self.trail.set_offsets(np.c_[past.loc[ok, "gt_x"], past.loc[ok, "gt_y"]])
        self.trail.set_array(past.loc[ok, "s"].clip(-LIM, LIM).to_numpy())
        x, y, yaw = now["gt_x"], now["gt_y"], now["yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        self.toby.set_xy([(x + 0.7 * c, y + 0.7 * s), (x - 0.45 * c - 0.4 * s, y - 0.45 * s + 0.4 * c),
                          (x - 0.45 * c + 0.4 * s, y - 0.45 * s - 0.4 * c)])
        self.fov.set_center((x, y))
        self.fov.set_theta1(math.degrees(yaw - CAM_FOV / 2))
        self.fov.set_theta2(math.degrees(yaw + CAM_FOV / 2))
        if self.s_line is not None:
            self.s_line.set_data(past["t"], past["s_smooth"])
        return now


def s_axes(ax, t_max):
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.3, 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:+.0f} %"))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("s_long")


def single(run):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax_map = fig.add_axes([0.02, 0.05, 0.50, 0.85])
    ax_cam = fig.add_axes([0.56, 0.47, 0.42, 0.42])
    ax_s = fig.add_axes([0.60, 0.08, 0.37, 0.30])
    title = fig.text(0.02, 0.94, "", fontsize=16, color=INK, fontweight="bold")
    view = RunView(run, ax_map, ax_s)
    s_axes(ax_s, view.end)

    frames = sorted(glob.glob(os.path.join(ROOT, "data", run, "frames", "frame_*.png")))
    frame_t = np.array([float(os.path.basename(f)[6:-4]) for f in frames])
    cam_img = ax_cam.imshow(np.ones((128, 256, 3)), interpolation="bilinear")
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_title("Toby's camera", loc="left", color=INK2, fontsize=11)
    boxes = []

    def update(t):
        now = view.update(t)
        while boxes:
            boxes.pop().remove()
        if frames:
            i = max(0, np.searchsorted(frame_t, now["t"], side="right") - 1)
            img = plt.imread(frames[i])[:, :, :3]
            cam_img.set_data(img)
            for name, x0, y0, x1, y1 in detections(img):
                goal = name == "red"
                boxes.append(ax_cam.add_patch(Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, fill=False,
                                                        ec=EDGE[name], lw=2.5 if goal else 1.4,
                                                        ls="-" if goal else (0, (3, 2)))))
                if goal:
                    boxes.append(ax_cam.text(x0 - 2, y0 - 4, "GOAL", color=EDGE[name], fontsize=10,
                                             fontweight="bold", va="bottom"))
        title.set_text(f"Toby · {LABELS.get(run, run)}  (μ = {view.mu:g})    t = {now['t']:4.1f} s    {now['state']}")

    return fig, update, view.end


def compare(runs=("baseline", "low_slip")):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.text(0.02, 0.945, "Same city, same robot, same code: only the floor changes", fontsize=16, color=INK,
             fontweight="bold")
    views, heads = [], []
    for k, (run, color) in enumerate(zip(runs, RUN_COLORS)):
        ax_map = fig.add_axes([0.02 + 0.49 * k, 0.30, 0.47, 0.60])
        views.append(RunView(run, ax_map, None, color))
        heads.append(fig.text(0.255 + 0.49 * k, 0.905, "", ha="center", fontsize=13, color=color,
                              fontweight="bold"))
    ax_s = fig.add_axes([0.08, 0.08, 0.89, 0.17])
    t_max = max(v.end for v in views)
    s_axes(ax_s, t_max)
    for v, color in zip(views, RUN_COLORS):
        v.s_line, = ax_s.plot([], [], color=color, lw=1.6, label=f"{LABELS.get(v.run, v.run)}, μ = {v.mu:g}")
    ax_s.legend(loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2)

    def update(t):
        for v, head in zip(views, heads):
            now = v.update(t)
            state = "goal reached ✓" if t >= v.end else now["state"]
            head.set_text(f"{LABELS.get(v.run, v.run)} · μ = {v.mu:g}    t = {min(t, v.end):4.1f} s    {state}")

    return fig, update, t_max


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "baseline"
    gif = "--gif" in sys.argv or not animation.FFMpegWriter.isAvailable()
    fig, update, t_end = compare() if name == "compare" else single(name)
    times = list(np.arange(0, t_end, SPEED / FPS)) + [t_end] * (2 * FPS)      # hold the end for 2 s
    anim = animation.FuncAnimation(fig, lambda k: update(times[k]), frames=len(times), interval=1000 / FPS)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out = os.path.join(ROOT, "figures", f"run_{name}." + ("gif" if gif else "mp4"))
    writer = animation.PillowWriter(fps=FPS) if gif else animation.FFMpegWriter(fps=FPS, bitrate=4000)
    anim.save(out, writer=writer)
    print("written", os.path.relpath(out, ROOT), f"({len(times)} frames, {len(times) / FPS:.0f} s)")
