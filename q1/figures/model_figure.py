# Presentation figure for the Q1 (d) slide: how the model f_theta(o1) = o2* is built, in the overview's style
# (not part of the assignment answers, run separately from q1/: python figures/model_figure.py)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q1/, for results.py and camera.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Assignment2/, for slide_layout.py

from slide_layout import (
    BLUE,
    BOX_H,
    BOX_Y,
    INK,
    MUTED,
    ORANGE,
    STEP_COLOURS,
    caption,
    new_canvas,
    picture,
    step_arrows,
    step_box,
)

import results

HERE = Path(__file__).resolve().parents[1]  # q1/
SLIDES = HERE.parent / "slides.mplstyle"
GREEN = STEP_COLOURS[4][0]


def main():
    plt.style.use(SLIDES)

    # data and trained model: saved by main.py
    r = results.load()
    o1, o2, o2_star, frames = r["o1"], r["o2"], r["o2_star"], r["frames"]
    n_train, n_val = int(r["n_train"]), int(r["n_val"])
    n_test = len(o1) - n_train - n_val
    test = slice(n_train + n_val, None)
    W = r["W"]

    fig, canvas = new_canvas()

    step_box(canvas, 0, "Data", f"one pair ($o_1$, $o_2$)\nper frame, {len(o1)} pairs\ninput $o_1$, target $o_2$")
    step_box(canvas, 1, "Split", f"in time order\ntrain {n_train} frames\nvalidation {n_val}\ntest {n_test}")
    step_box(canvas, 2, "Train", "one-layer perceptron,\nno activation:\n$o_2^* = W o_1 + b$\n"
                                 f"Adam, {int(r['epochs'])} epochs")
    step_box(canvas, 3, "Learned W", "the weights after\ntraining, read back\nas a 3×3 matrix")
    step_box(canvas, 4, "Test", f"$o_2^* = f_\\theta(o_1)$ on the\n{n_test} test frames,\ncompared with $o_2$")
    step_arrows(canvas)

    # 1: the two trajectories that make the pairs
    ax = picture(fig, 0, width=0.12)
    ax.plot(o1[:, 0], -o1[:, 1], color=BLUE, lw=1.2)
    ax.plot(o2[:, 0], -o2[:, 1], color=ORANGE, lw=1.2)
    ax.text(0.97, 0.97, "$o_1$ ($M_{int}$)", transform=ax.transAxes, ha="right", va="top", fontsize=12, color=BLUE)
    ax.text(0.97, 0.86, "$o_2$ ($M_d$)", transform=ax.transAxes, ha="right", va="top", fontsize=12, color=ORANGE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("−y [m]", fontsize=10)
    caption(canvas, 0, "front view of both trajectories")

    # 2: split over time, shown on x of the target
    ax = picture(fig, 1, width=0.14)
    edges = [frames[0], frames[n_train], frames[n_train + n_val], frames[-1]]
    for (start, end), colour in zip(zip(edges, edges[1:]), [BLUE, MUTED, GREEN]):
        ax.axvspan(start, end, color=colour, alpha=0.15)
    # val and test are too narrow for a label each: one label over both
    for x, name, colour, ha in [((edges[0] + edges[1]) / 2, "train", BLUE, "center"),
                                (edges[3], "val | test", MUTED, "right")]:
        ax.text(x, 1.02, name, transform=ax.get_xaxis_transform(), ha=ha, va="bottom", fontsize=11, color=colour,
                fontweight="bold")
    ax.plot(frames, o2[:, 0], color=INK, lw=1.2)
    ax.set_xlabel("frame", fontsize=10)
    ax.set_ylabel("$x_2$ [m]", fontsize=10)
    caption(canvas, 1, "80 / 10 / 10 in time order,\nnot random: no near-copies")

    # 3: loss curves
    ax = picture(fig, 2, width=0.13)
    ax.semilogy(r["loss"], color=BLUE, lw=1.5, label="train")
    ax.semilogy(r["val_loss"], "--", color=ORANGE, lw=1.5, label="val")
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("MSE loss [m²]", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", handlelength=1.5)
    caption(canvas, 2, "train and val fall together\nand flatten: no overfitting")

    # 4: the learned W
    ax = picture(fig, 3, width=0.13)
    ax.imshow(np.abs(W), cmap="Blues", vmin=0, vmax=1.6)
    for i in range(3):
        for j in range(3):
            text = "0" if abs(W[i, j]) < 5e-4 else f"{W[i, j]:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=12, color="white" if abs(W[i, j]) > 0.8 else INK)
    ax.set_xticks(range(3), ["$x_1$", "$y_1$", "$z_1$"], fontsize=11)
    ax.set_yticks(range(3), ["$x_2$", "$y_2$", "$z_2$"], fontsize=11)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    caption(canvas, 3, f"{W[0, 0]:.3f} = $f_{{int}}/f_d$ on x, y\n$z_2 = z_1$, the rest ≈ 0")

    # 5: prediction vs truth on the test set, with the KPIs
    ax = picture(fig, 4, width=0.13)
    true, pred = o2[test, 0], o2_star[test, 0]
    ax.plot([true.min(), true.max()], [true.min(), true.max()], "--", color=MUTED, lw=1.5)
    ax.plot(true, pred, "o", ms=4, color=GREEN)
    ax.text(0.04, 0.96, f"RMSE {r['rmse_mm']:.3f} mm\nR²   {r['r2']:.4f}", transform=ax.transAxes,
            va="top", fontsize=11, family="monospace", bbox=dict(facecolor="#fcfcfb", edgecolor="#d9d8d4"))
    ax.set_xlabel("$x_2$ true [m]", fontsize=10)
    ax.set_ylabel("$x_2^*$ predicted [m]", fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    caption(canvas, 4, "every test frame on the diagonal:\n$o_2$ is a linear function of $o_1$")

    # no loop arrow above the boxes as in the overview: crop the empty top of the figure
    width, height = fig.get_size_inches()
    fig.savefig(HERE / "result/q1_model.png", bbox_inches=Bbox([[0, 0], [width, height * (BOX_Y + BOX_H + 0.04)]]))
    print("Saved result/q1_model.png")


if __name__ == "__main__":
    main()
