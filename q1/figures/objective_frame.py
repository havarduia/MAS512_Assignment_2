# Presentation figure for the Q1 objective slide: one camera frame with the YOLO box, its centre and the camera axes
# (not part of the assignment answers, run separately from q1/: python figures/objective_frame.py)

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q1/, for results.py and camera.py

import results

HERE = Path(__file__).resolve().parents[1]  # q1/
SLIDES = HERE.parent / "slides.mplstyle"


def main():
    plt.style.use(SLIDES)

    # YOLO boxes saved by main.py: pick the frame where the load is swung out the most
    r = results.load()
    k = int(np.argmax(np.abs(r["u"] - np.median(r["u"]))))
    x0, y0, x1, y1 = r["boxes"][k]
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    fname = HERE / f"camera_data/rgb_frames/rgb_{r['frames'][k]:04d}.png"

    image = cv2.cvtColor(cv2.imread(str(fname)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#eda100", lw=4))
    ax.plot(u, v, "o", ms=14, color="#e34948", markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"centre (u, v) = ({u:.0f}, {v:.0f}) px", (u, v), xytext=(x0 - 20, y1 - 10), ha="right",
                fontsize=18, color="white", bbox=dict(facecolor="#0b0b0b", alpha=0.75, edgecolor="none"),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))

    # camera frame: x right, y down, z into the image (shown as a dot in a circle)
    ox, oy, length = 50, 50, 70
    ax.add_patch(Rectangle((ox - 35, oy - 40), length + 75, length + 75, color="#0b0b0b", alpha=0.6))  # readable on bright parts
    arrow = dict(arrowstyle="-|>", color="white", lw=3, mutation_scale=25)
    ax.annotate("", (ox + length, oy), xytext=(ox, oy), arrowprops=arrow)
    ax.annotate("", (ox, oy + length), xytext=(ox, oy), arrowprops=arrow)
    ax.text(ox + length + 8, oy + 6, "x", color="white", fontsize=20, fontweight="bold")
    ax.text(ox - 6, oy + length + 26, "y", color="white", fontsize=20, fontweight="bold")
    ax.plot(ox, oy, "o", ms=16, markerfacecolor="none", markeredgecolor="white", markeredgewidth=2.5)
    ax.plot(ox, oy, "o", ms=4, color="white")
    ax.text(ox + 14, oy - 14, "z", color="white", fontsize=20, fontweight="bold")

    ax.set_axis_off()
    fig.savefig(HERE / "result/q1_objective_frame.png", bbox_inches="tight", pad_inches=0.05)
    print(f"Saved result/q1_objective_frame.png from {fname.name}")


if __name__ == "__main__":
    main()
