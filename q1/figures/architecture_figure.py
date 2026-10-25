# Presentation figure for the Q1 (d) slide: architecture of f_theta, a one-layer perceptron without activation
# (not part of the assignment answers, run separately from q1/: python figures/architecture_figure.py)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q1/, for results.py and camera.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Assignment2/, for slide_layout.py

from slide_layout import BLUE, INK, MUTED, ORANGE, STEP_COLOURS

import results

HERE = Path(__file__).resolve().parents[1]  # q1/
SLIDES = HERE.parent / "slides.mplstyle"
GREEN, GREEN_FILL = STEP_COLOURS[4]
GREY_FILL = "#ecebe8"

FIGSIZE = (13.33, 6.6)
LAYER_X = [0.14, 0.42, 0.74]        # input, normalisation, dense output
NODE_Y = [0.80, 0.60, 0.40]         # x, y, z
NODE_R = 0.045                      # node radius as a fraction of the figure height
NODE_RX = NODE_R * FIGSIZE[1] / FIGSIZE[0]   # the same radius in figure-width units, so nodes are round


def node(ax, x, y, text, edge, fill):
    ax.add_patch(Ellipse((x, y), 2 * NODE_RX, 2 * NODE_R, facecolor=fill, edgecolor=edge, linewidth=2.5, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=18, color=INK, zorder=4)


def layer_label(ax, x, title, lines, edge, fill):
    ax.add_patch(FancyBboxPatch((x - 0.12, 0.05), 0.24, 0.2, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=fill, edgecolor=edge, linewidth=2))
    ax.text(x, 0.205, title, ha="center", va="center", fontsize=17, fontweight="bold", color=edge)
    ax.text(x, 0.115, lines, ha="center", va="center", fontsize=13, color=INK, linespacing=1.4)


def main():
    plt.style.use(SLIDES)

    # learned weights and parameter counts: saved by main.py
    r = results.load()
    W, b = r["W"], r["b"]

    fig = plt.figure(figsize=FIGSIZE, layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()

    # connections: input -> normalisation is one-to-one, normalisation -> output is fully connected
    for y in NODE_Y:
        ax.plot([LAYER_X[0] + NODE_RX, LAYER_X[1] - NODE_RX], [y, y], color=MUTED, lw=2, zorder=1)
    for i, y_out in enumerate(NODE_Y):          # i: output x2, y2, z2
        for j, y_in in enumerate(NODE_Y):       # j: input x1, y1, z1
            w = W[i, j]
            strong = abs(w) > 0.01
            ax.plot([LAYER_X[1] + NODE_RX, LAYER_X[2] - NODE_RX], [y_in, y_out], zorder=1,
                    color=BLUE if strong else "#c9c8c4", lw=1 + 6 * abs(w), linestyle="-" if strong else ":")
            if strong:
                # label on the line, nearer the output node so the three labels don't collide
                t = 0.7
                ax.text(LAYER_X[1] + NODE_RX + t * (LAYER_X[2] - LAYER_X[1] - 2 * NODE_RX),
                        y_in + t * (y_out - y_in) + 0.035, f"{w:.3f}", ha="center", va="center", fontsize=14,
                        color=BLUE, fontweight="bold", bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1))

    for (axis, y) in zip("xyz", NODE_Y):
        node(ax, LAYER_X[0], y, f"${axis}_1$", ORANGE, "#fde6d6")
        node(ax, LAYER_X[1], y, "$\\frac{\\cdot - \\mu}{\\sigma}$", MUTED, GREY_FILL)
        node(ax, LAYER_X[2], y, f"${axis}_2^*$", GREEN, GREEN_FILL)
    for k, y in enumerate(NODE_Y):
        bias = "≈ 0" if abs(b[k]) < 5e-4 else f"= {b[k]:+.3f}"
        ax.text(LAYER_X[2] + NODE_RX + 0.015, y, f"+ b {bias}", ha="left", va="center", fontsize=14, color=MUTED)

    ax.text(0.5, 0.94, "dotted = learned weight ≈ 0;   weights shown for the raw input (normalisation folded in)",
            ha="center", va="center", fontsize=13, color=MUTED, style="italic")

    n_norm, n_dense = int(r["n_fixed"]), int(r["n_trainable"])
    layer_label(ax, LAYER_X[0], "Input", "$o_1$ = (x, y, z)\nshape (3,)", ORANGE, "#fde6d6")
    layer_label(ax, LAYER_X[1], "Normalization", f"mean, std of train set\n{n_norm} fixed values", MUTED,
                GREY_FILL)
    layer_label(ax, LAYER_X[2], "Dense(3)", f"no activation: $o_2^* = W o_1 + b$\n{n_dense} trainable (9 W + 3 b)",
                GREEN, GREEN_FILL)

    fig.savefig(HERE / "result/q1_architecture.png")
    print(f"Saved result/q1_architecture.png  ({n_dense} trainable, {n_norm} non-trainable parameters)")


if __name__ == "__main__":
    main()
