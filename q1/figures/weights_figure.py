# Presentation figure for the Q1 (d) slide: the learned weights W of f_theta as one annotated 3x3 grid
# (not part of the assignment answers, run separately from q1/: python figures/weights_figure.py)

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q1/, for results.py

import results

HERE = Path(__file__).resolve().parents[1]  # q1/
SLIDES = HERE.parent / "slides.mplstyle"
INK, MUTED = "#0b0b0b", "#52514e"


def main():
    plt.style.use(SLIDES)
    r = results.load()
    W = r["W"]

    # what W should be from the camera geometry: o2 = M_d^-1 M_int o1
    intr = json.loads((HERE / "camera_data/depth_scale_json/color_intrinsics_own.json").read_text())
    M_int = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
    M_d = np.array([[1342, 0, 325], [0, 1342, 235], [0, 0, 1]], float)
    difference = np.abs(W - np.linalg.inv(M_d) @ M_int).max()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(np.abs(W), cmap="Blues", vmin=0, vmax=1.6)
    for i in range(3):
        for j in range(3):
            text = "0" if abs(W[i, j]) < 5e-4 else f"{W[i, j]:.5f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=24,
                    color="white" if abs(W[i, j]) > 0.8 else INK)
    ax.set_xticks(range(3), ["$x_1$", "$y_1$", "$z_1$"], fontsize=24)
    ax.set_yticks(range(3), ["$x_2^*$", "$y_2^*$", "$z_2^*$"], fontsize=24)
    ax.xaxis.tick_top()
    ax.set_xlabel("input $o_1$", fontsize=20, labelpad=12)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("output $o_2^*$", fontsize=20, labelpad=12)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"b ≈ 0  ·  W = $M_d^{{-1}} M_{{int}}$ (max difference {difference:.0e})", y=-0.1,
                 fontsize=18, fontweight="normal", color=MUTED)

    fig.savefig(HERE / "result/q1_weights.png")
    print("Saved result/q1_weights.png")


if __name__ == "__main__":
    main()
