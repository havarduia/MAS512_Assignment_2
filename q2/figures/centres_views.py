# Presentation figure for the Q2 (a) results slide: the centre of the load in all frames, seen from above and
# from the side (not part of the assignment answers: main.py makes the 3D plot the assignment asks for)
# Run separately from q2/: python figures/centres_views.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q2/, for results.py

import results

HERE = Path(__file__).resolve().parents[1]  # q2/
SLIDES = HERE.parent / "slides.mplstyle"


def main():
    plt.style.use(SLIDES)
    centres = results.load()["centres"]  # saved by main.py
    x, y, z = centres.T

    fig, (top, side) = plt.subplots(2, 1, sharex=True)
    for ax, other, name in [(top, x, "x"), (side, z, "z")]:
        ax.plot(y, other, ".-", lw=1.5, ms=8)
        ax.plot(y[0], other[0], "o", ms=14, color="#e34948", markeredgecolor="white", markeredgewidth=2,
                label="frame 0")
        ax.set_ylabel(f"{name} [m]")
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    top.set_title(f"From above: swings {y.max() - y.min():.2f} m along y, {x.max() - x.min():.2f} m in x",
                  loc="left", fontsize=20)
    side.set_title(f"From the side: {z.max() - z.min():.2f} m up and down", loc="left", fontsize=20)
    side.set_xlabel("y [m]  (swing direction)")
    top.legend(loc="lower left")
    fig.suptitle(f"(a) Centre of the swinging load, all {len(centres)} frames")

    fig.savefig(HERE / "result/q2_centres_views.png")
    print("Saved result/q2_centres_views.png")


if __name__ == "__main__":
    main()
