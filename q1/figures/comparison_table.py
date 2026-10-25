# Presentation figures for the Q1 (c) slides (not part of the assignment answers,
# run separately from q1/: python figures/comparison_table.py):
#   result/q1_fov_table.png        M_int vs M_d vs reality: field of view and size of the load
#   result/q1_resolution_grid.png  which parts of M_d fit a 640x480 and a 1920x1080 stream

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # q1/, for results.py and camera.py

import results
from camera import Camera

HERE = Path(__file__).resolve().parents[1]  # q1/
DATA = HERE / "camera_data"
SLIDES = HERE.parent / "slides.mplstyle"

DATASHEET_FOV = (55, 43)        # L515 RGB field of view at 4:3 [deg], approximate
DATASHEET_F_MM = 1.88           # OV2740 focal length [mm], L515 datasheet
DATASHEET_PIXEL_MM = 0.0014     # OV2740 pixel size [mm], native 1920x1080
LIDAR_LOAD_WIDTH = 0.5          # width of the load cluster in Q2 [m]

INK, MUTED = "#0b0b0b", "#52514e"
GOOD, BAD = "#2f9e44", "#e34948"


def load_width(r, M):
    # median width of the YOLO box in metres over all frames: width [px] * depth / f_x
    widths = r["boxes"][:, 2] - r["boxes"][:, 0]
    return np.median(widths * r["depth"]) / M[0, 0]


def rounded_box(ax, x, y, w, h, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=fill, edgecolor=edge, linewidth=2))


def canvas(figsize):
    fig = plt.figure(figsize=figsize, layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def fov_table(cam_int, cam_d, width_int, width_d):
    columns_x = [0.11, 0.28, 0.42, 0.57, 0.72, 0.88]
    header = ["", "$f_x$, $f_y$ [px]", "$c_x$, $c_y$ [px]", "field of view\nH × V", "load\nwidth", "matches\nreality?"]
    rows = []
    for cam, name, width, verdict in [(cam_int, "$M_{int}$\nfrom our L515", width_int, "✓  yes"),
                                      (cam_d, "$M_d$\ndefault matrix", width_d, "✗  no\n~2× too small")]:
        M = cam.M
        fov_h, fov_v = cam.field_of_view()
        rows.append([name, f"{M[0, 0]:.0f}, {M[1, 1]:.0f}", f"{M[0, 2]:.0f}, {M[1, 2]:.0f}",
                     f"{fov_h:.0f}° × {fov_v:.0f}°", f"{width:.2f} m", verdict])
    rows.append(["reality", "", "", f"~{DATASHEET_FOV[0]}° × {DATASHEET_FOV[1]}°\n(datasheet)",
                 f"~{LIDAR_LOAD_WIDTH} m\n(lidar, Q2)", "reference"])
    styles = [("#2a78d6", "#dce8f7", GOOD), ("#eb6834", "#fde6d6", BAD), (MUTED, "#ecebe8", MUTED)]

    fig, ax = canvas((13.33, 4.8))
    top, row_h = 0.97, 0.19
    for x, text in zip(columns_x, header):
        ax.text(x, top - 0.065, text, ha="center", va="center", fontsize=16, fontweight="bold", color=MUTED)
    ax.plot([0.02, 0.98], [top - 0.14, top - 0.14], color=MUTED, lw=1.5)

    for r, (row, (edge, fill, verdict_colour)) in enumerate(zip(rows, styles)):
        y = top - 0.17 - (r + 1) * (row_h + 0.03)
        rounded_box(ax, 0.02, y, 0.96, row_h, edge, fill)
        for c, (x, text) in enumerate(zip(columns_x, row)):
            style = dict(fontsize=18, color=INK)
            if c == 0:
                style = dict(fontsize=16, color=edge, fontweight="bold")
            elif c == 5:
                style = dict(fontsize=18, color=verdict_colour, fontweight="bold")
            elif r == 2:
                style = dict(fontsize=16, color=INK)
            ax.text(x, y + row_h / 2, text, ha="center", va="center", linespacing=1.3, **style)

    fig.savefig(HERE / "result/q1_fov_table.png")


def resolution_grid(cam_int, cam_d):
    # does each part of M_d fit the stream? 1080p: focal length from the datasheet, centre = image centre
    f_d, (cx_d, cy_d) = cam_d.M[0, 0], cam_d.M[:2, 2]
    f_1080 = DATASHEET_F_MM / DATASHEET_PIXEL_MM
    columns = ["640 × 480\n(our recording)", "1920 × 1080"]
    rows = ["focal length", "image centre"]
    cells = [  # (ok, M_d value, value the stream needs), [row][column]
        [(False, f"{f_d:.0f} px", f"{cam_int.M[0, 0]:.0f} px"), (True, f"{f_d:.0f} px", f"{f_1080:.0f} px")],
        [(True, f"({cx_d:.0f}, {cy_d:.0f})", f"({cam_int.M[0, 2]:.0f}, {cam_int.M[1, 2]:.0f})"),
         (False, f"({cx_d:.0f}, {cy_d:.0f})", "(960, 540)")],
    ]

    fig, ax = canvas((13.33, 5.6))
    col_x, col_w = [0.24, 0.62], 0.35
    row_y, row_h = [0.45, 0.06], 0.33
    for x, title in zip(col_x, columns):
        ax.text(x + col_w / 2, 0.83, title, ha="center", va="bottom", fontsize=20, fontweight="bold", color=INK,
                linespacing=1.3)
    for y, label, row in zip(row_y, rows, cells):
        ax.text(0.02, y + row_h / 2, label, ha="left", va="center", fontsize=20, fontweight="bold", color=MUTED)
        for x, (ok, has, needs) in zip(col_x, row):
            colour, fill = (GOOD, "#dff1e2") if ok else (BAD, "#fbdddd")
            rounded_box(ax, x, y, col_w, row_h, colour, fill)
            ax.text(x + 0.05, y + row_h / 2, "✓" if ok else "✗", ha="center", va="center", fontsize=40,
                    fontweight="bold", color=colour)
            ax.text(x + 0.10, y + row_h / 2, f"$M_d$:  {has}\nneeds: {needs}", ha="left", va="center",
                    fontsize=18, color=INK, linespacing=1.6)

    fig.savefig(HERE / "result/q1_resolution_grid.png")


def main():
    plt.style.use(SLIDES)
    cam_int = Camera.from_json("M_int", DATA / "depth_scale_json/color_intrinsics_own.json")
    cam_d = Camera("M_d", np.array([[1342, 0, 325], [0, 1342, 235], [0, 0, 1]], float))

    r = results.load()  # YOLO boxes and depths, saved by main.py
    width_int, width_d = load_width(r, cam_int.M), load_width(r, cam_d.M)

    fov_table(cam_int, cam_d, width_int, width_d)
    resolution_grid(cam_int, cam_d)
    print("Saved result/q1_fov_table.png and result/q1_resolution_grid.png")


if __name__ == "__main__":
    main()
