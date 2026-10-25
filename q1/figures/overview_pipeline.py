# Presentation figure for the Q1 overview slide: the pipeline as numbered steps, each with a small picture
# (not part of the assignment answers, run separately from q1/: python figures/overview_pipeline.py)

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q1/, for results.py and camera.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Assignment2/, for slide_layout.py

from slide_layout import BLUE, MUTED, ORANGE, caption, loop_arrow, new_canvas, picture, step_arrows, step_box

import results
from camera import Camera

HERE = Path(__file__).resolve().parents[1]  # q1/
DATA = HERE / "camera_data"
SLIDES = HERE.parent / "slides.mplstyle"
FRAME = 30  # same frame as the objective slide figure

def main():
    plt.style.use(SLIDES)

    # data for the pictures: saved by main.py
    r = results.load()
    o1, o2, o2_star, frames = r["o1"], r["o2"], r["o2_star"], r["frames"]
    k = int(np.flatnonzero(frames == FRAME)[0])
    u, v, (x0, y0, x1, y1) = r["u"][k], r["v"][k], r["boxes"][k]
    test = slice(int(r["n_train"] + r["n_val"]), None)
    cam_int = Camera.from_json("M_int", DATA / "depth_scale_json/color_intrinsics_own.json")
    cam_d = Camera("M_d", np.array([[1342, 0, 325], [0, 1342, 235], [0, 0, 1]], float))

    rgb = cv2.cvtColor(cv2.imread(str(DATA / f"rgb_frames/rgb_{FRAME:04d}.png")), cv2.COLOR_BGR2RGB)
    depth_scale = json.loads((DATA / "depth_scale_json/depth_scale.json").read_text())["depth_scale"]
    depth = np.load(DATA / f"depth_raw/depth_{FRAME:04d}_raw.npy") * depth_scale
    depth = np.where(depth > 0, depth, np.nan)

    fig, canvas = new_canvas()

    # steps
    step_box(canvas, 0, "Detect", "YOLO from\nAssignment 1\nfinds the load\n→ centre (u, v)")
    step_box(canvas, 1, "Depth", "depth d at (u, v):\nmedian of 11×11 px,\nskip pixels with\nno reading")
    step_box(canvas, 2, "Back-project", "$p_c = \\tilde{M}^{-1}[du, dv, d, 1]^T$\n"
                                         "$M_{int}$ → $o_1$  (a)\n$M_d$ → $o_2$  (b)")
    step_box(canvas, 3, "Learn $f_\\theta$", "$f_\\theta(o_1) = o_2^*$\none-layer perceptron\n"
                                             "Adam, 80/10/10 split\n(train/val/test)")
    step_box(canvas, 4, "Compare", "(c) field of view\nvs L515 datasheet\n(e) $\\|o_1 - o_2\\|$\n$\\|o_1 - o_2^*\\|$")
    step_arrows(canvas)
    loop_arrow(canvas, 0, 2, f"repeated for every frame ({len(frames)} frames with a valid depth)")

    # 1: camera frame with the box
    ax = picture(fig, 0)
    ax.imshow(rgb)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#eda100", lw=2))
    ax.plot(u, v, "o", ms=6, color="#e34948", markeredgecolor="white")
    ax.set_axis_off()
    caption(canvas, 0, f"frame {FRAME}: box = load,\ndot = centre (u, v)")

    # 2: depth frame
    ax = picture(fig, 1)
    ax.imshow(depth, cmap="plasma_r", interpolation="nearest")
    ax.set_facecolor("#3a3a38")
    ax.plot(u, v, "o", ms=6, color="#e34948", markeredgecolor="white")
    ax.set(xticks=[], yticks=[])
    ax.grid(False)
    caption(canvas, 1, f"bright = near, grey = no\nreading ({100 * np.isfinite(depth).mean():.0f} % of pixels valid)")

    # 3: both trajectories, front view, same scale
    ax = picture(fig, 2, width=0.14)
    ax.plot(o1[:, 0], -o1[:, 1], color=BLUE, lw=1.2)
    ax.plot(o2[:, 0], -o2[:, 1], color=ORANGE, lw=1.2)
    ax.text(0.97, 0.95, "$o_1$  $M_{int}$", transform=ax.transAxes, ha="right", va="top", fontsize=12, color=BLUE)
    ax.text(0.97, 0.83, "$o_2$  $M_d$", transform=ax.transAxes, ha="right", va="top", fontsize=12, color=ORANGE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("−y [m]", fontsize=10)
    caption(canvas, 2, "front view of the swing:\n$o_2$ ~2× smaller than $o_1$")

    # 4: prediction vs truth on the test set
    ax = picture(fig, 3, width=0.13)
    true, pred = o2[test, 0], o2_star[test, 0]
    ax.plot([true.min(), true.max()], [true.min(), true.max()], "--", color=MUTED, lw=1.5)
    ax.plot(true, pred, "o", ms=4, color="#e34948")
    ax.set_xlabel("$x_2$ true [m]", fontsize=10)
    ax.set_ylabel("$x_2^*$ predicted [m]", fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    caption(canvas, 3, f"test set on the diagonal:\nR² = {r['r2']:.4f}")

    # 5: error per frame
    ax = picture(fig, 4, width=0.14)
    ax.plot(frames, 1e3 * np.linalg.norm(o1 - o2, axis=1), color=BLUE, lw=1.5)
    ax.plot(frames, 1e3 * np.linalg.norm(o1 - o2_star, axis=1), "--", color=ORANGE, lw=1.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("frame", fontsize=10)
    ax.set_ylabel("error [mm]", fontsize=10)
    ax.legend(["$\\|o_1 - o_2\\|$", "$\\|o_1 - o_2^*\\|$"], fontsize=9, loc="lower right", handlelength=1.5)
    fov_int, fov_d = cam_int.field_of_view()[0], cam_d.field_of_view()[0]
    caption(canvas, 4, f"FOV $M_{{int}}$ {fov_int:.0f}°, $M_d$ {fov_d:.0f}° (sheet 55°)\n"
                       f"mean error {1e3 * np.linalg.norm(o1 - o2, axis=1).mean():.0f} mm")

    fig.savefig(HERE / "result/q1_overview_pipeline.png")
    print("Saved result/q1_overview_pipeline.png")


if __name__ == "__main__":
    main()
