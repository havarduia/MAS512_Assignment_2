"""Pictures for the slides that do not depend on a run: flow chart, state machine, slip formula
and the important pieces of code as images (with their real line numbers).

    python tools/slide_figures.py        ->  figures/slides/*.png

The code images are cut from the source files between two markers, so they stay right if the
code changes. Labels are in English; edit the strings below to translate them.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "figures", "slides")

INK, INK2 = "#2b2440", "#6b6480"
BUILD, NAV, SLIP, ANALYSE = "#ffe3c9", "#dce8fb", "#fde0df", "#e1f2e4"
BUILD_E, NAV_E, SLIP_E, ANALYSE_E = "#eb6834", "#2a78d6", "#e34948", "#2e9e55"
STOP, STOP_E = "#f3dede", "#8b1e3f"          # the mission-failed state
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def canvas():
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 18)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, x1, y1, title, body="", fc="white", ec=INK, size=14, lw=1.8, title_color=INK):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.35",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    lines = body.split("\n") if body else []
    lh = size / 72 * 2 * 1.35                     # line height in data units (2 units per inch)
    top = (y0 + y1) / 2 + (1 + len(lines)) * lh / 2
    ax.text((x0 + x1) / 2, top - lh / 2, title, ha="center", va="center", fontsize=size,
            fontweight="bold", color=title_color, zorder=4)
    for i, line in enumerate(lines):
        ax.text((x0 + x1) / 2, top - lh * (i + 1.5), line, ha="center", va="center",
                fontsize=size - 2, color=INK, zorder=4)


def arrow(ax, a, b, color=INK, rad=0.0, lw=2.0, ls="-", label="", lpos=0.5, loff=(0, 0.35), lsize=12,
          lcolor=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=0.55,head_width=0.28", mutation_scale=20,
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=1, zorder=2))
    if label:
        x = a[0] + (b[0] - a[0]) * lpos + loff[0]
        y = a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=lsize, color=lcolor or INK2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def bus(ax, points, taps, color):
    """A line through `points` with small arrows (x0, y0, x1, y1) branching off it."""
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=2.0, solid_joinstyle="round", zorder=2)
    for x0, y0, x1, y1 in taps:
        arrow(ax, (x0, y0), (x1, y1), color=color)


def stage(ax, x, y, number, name, color):
    ax.text(x, y, f"{number}", fontsize=22, fontweight="bold", color=color, va="center")
    ax.text(x + 0.9, y, name, fontsize=17, fontweight="bold", color=INK, va="center")


# ---------------------------------------------------------------- flow chart
def flowchart():
    fig, ax = canvas()

    stage(ax, 0.6, 16.9, 1, "Build the city", BUILD_E)
    box(ax, 0.9, 12.2, 6.6, 15.4, "make_worlds.py", "one city, one robot,\ntwo floors", BUILD, BUILD_E)
    box(ax, 0.9, 8.0, 6.6, 10.6, "city_baseline.wbt", "wet floor  μ = 0.25", "white", BUILD_E)
    box(ax, 0.9, 4.6, 6.6, 7.2, "city_low_slip.wbt", "dry floor  μ = 1.0", "white", BUILD_E)
    bus(ax, [(0.9, 13.8), (0.4, 13.8), (0.4, 5.9)], [(0.4, 9.3, 0.9, 9.3), (0.4, 5.9, 0.9, 5.9)], BUILD_E)
    ax.text(3.75, 3.7, "only the wheel–floor\nfriction differs", ha="center", va="center", fontsize=12,
            color=INK2, style="italic")

    stage(ax, 7.8, 16.9, 2, "Simulate in Webots · slip_nav.py, every 16 ms", NAV_E)
    ax.add_patch(FancyBboxPatch((7.4, 1.0), 17.6, 14.6, boxstyle="round,pad=0,rounding_size=0.6",
                                fc="#faf9f7", ec="#cfcbd8", lw=1.5, ls=(0, (6, 4)), zorder=1))
    arrow(ax, (6.6, 9.3), (7.4, 9.3), color=BUILD_E)
    arrow(ax, (6.6, 5.9), (7.4, 5.9), color=BUILD_E)
    box(ax, 8.4, 13.0, 24.4, 15.0, "Read every sensor",
        "wheel encoders · camera · lidar · GPS · IMU · gyro · accelerometer · Supervisor", "white", INK)
    bus(ax, [(8.4, 14.0), (7.95, 14.0), (7.95, 2.6)],
        [(7.95, y, 8.4, y) for y in (10.95, 8.55, 5.15, 2.6)], INK)

    ax.text(8.4, 12.35, "NAVIGATE", fontsize=12, fontweight="bold", color=NAV_E, va="center")
    box(ax, 8.4, 10.0, 12.9, 11.9, "Camera", "HSV colour classes\nred = goal, others = decoys", NAV, NAV_E,
        size=13)
    box(ax, 8.4, 7.6, 12.9, 9.5, "Lidar", "nearest obstacle\nfront · left · right", NAV, NAV_E, size=13)
    box(ax, 13.8, 7.6, 18.6, 11.9, "State machine",
        "SEARCH · WANDER\nGOAL · RECALL · AVOID\nRECOVER · DONE · FAILED\n→ speed v, turn ω",
        NAV, NAV_E, size=13)
    box(ax, 19.6, 8.4, 24.4, 11.1, "Wheel speeds", "differential drive\n(v, ω) → ω_L, ω_R", NAV, NAV_E,
        size=13)
    arrow(ax, (12.9, 10.95), (13.8, 10.3), color=NAV_E)
    arrow(ax, (12.9, 8.55), (13.8, 9.2), color=NAV_E)
    arrow(ax, (18.6, 9.75), (19.6, 9.75), color=NAV_E)
    arrow(ax, (22.0, 11.1), (22.0, 13.0), color=INK2, ls=(0, (4, 3)), label="next step", loff=(1.15, 0))

    ax.text(8.4, 6.6, "MEASURE THE SLIP", fontsize=12, fontweight="bold", color=SLIP_E, va="center")
    box(ax, 8.4, 4.2, 12.9, 6.1, "Encoders", "v_enc = r (ω_L + ω_R) / 2", SLIP, SLIP_E, size=13)
    box(ax, 8.4, 1.6, 12.9, 3.6, "Simulator", "v_x = true velocity\nalong the robot's x", SLIP, SLIP_E,
        size=13)
    box(ax, 13.8, 2.4, 18.6, 5.4, "Slip", "s = (v_enc − v_x) / v_x\nonly when |v_x| > 0.1 m/s", SLIP, SLIP_E,
        size=13)
    box(ax, 19.6, 2.4, 24.4, 5.4, "Log", "log.csv · lidar.csv (raw)\ncamera frames", "white", SLIP_E, size=13)
    arrow(ax, (12.9, 5.15), (13.8, 4.5), color=SLIP_E)
    arrow(ax, (12.9, 2.6), (13.8, 3.3), color=SLIP_E)
    arrow(ax, (18.6, 3.9), (19.6, 3.9), color=SLIP_E)

    stage(ax, 25.9, 16.9, 3, "Analyse", ANALYSE_E)
    box(ax, 25.9, 12.4, 31.4, 15.0, "plot_slip.py", "slip on the map,\nslip profile, speeds", ANALYSE,
        ANALYSE_E, size=13)
    box(ax, 25.9, 9.4, 31.4, 11.8, "camera_montage.py", "what Toby saw", ANALYSE, ANALYSE_E, size=13)
    box(ax, 25.9, 6.4, 31.4, 8.8, "animate_run.py", "video of the run", ANALYSE, ANALYSE_E, size=13)
    box(ax, 25.9, 2.4, 31.4, 5.4, "Wet vs dry", "same path, compare\nslip and odometry error", "white",
        ANALYSE_E, size=13, lw=2.6)
    bus(ax, [(24.4, 3.9), (25.45, 3.9), (25.45, 13.7)],
        [(25.45, y, 25.9, y) for y in (13.7, 10.6, 7.6)], ANALYSE_E)
    arrow(ax, (28.65, 6.4), (28.65, 5.4), color=ANALYSE_E)
    return fig


# ---------------------------------------------------------------- state machine
def state_machine():
    fig, ax = canvas()
    W, H = 6.6, 3.0
    nodes = {
        "SEARCH": (7.5, 9.0, "spin on the spot\nuntil 360° (gyro)", NAV, NAV_E),
        "WANDER": (7.5, 3.2, "drive ahead for 8 s,\nlidar steers", NAV, NAV_E),
        "GOAL": (18.0, 9.0, "steer at the red pillar,\nslow down near obstacles", SLIP, SLIP_E),
        "RECALL": (18.0, 3.2, "red lost: turn back to\nwhere it was last seen", SLIP, SLIP_E),
        "DONE": (28.0, 9.0, "red fills 25 % of the\nimage: stop, save", ANALYSE, ANALYSE_E),
        "AVOID": (12.75, 15.4, "back off slowly and\nturn to the freer side", BUILD, BUILD_E),
        "RECOVER": (21.0, 15.4, "reverse 1.5 s, turn 1.5 s,\nat most three times", BUILD, BUILD_E),
        "FAILED": (28.0, 3.2, "give up: stop the motors\nand log the reason", STOP, STOP_E),
    }
    for name, (x, y, body, fc, ec) in nodes.items():
        box(ax, x - W / 2, y - H / 2, x + W / 2, y + H / 2, name, body, fc, ec, size=15, lw=2.2,
            title_color=ec)

    ax.add_patch(plt.Circle((1.2, 9.0), 0.35, color=INK, zorder=3))
    ax.text(1.2, 8.1, "start", ha="center", fontsize=12, color=INK2)
    arrow(ax, (1.55, 9.0), (4.2, 9.0))

    # around the four moving states: AVOID can interrupt any of them
    ax.add_patch(FancyBboxPatch((2.6, 1.1), 20.1, 10.3, boxstyle="round,pad=0,rounding_size=0.6",
                                fc="none", ec="#cfcbd8", lw=1.6, ls=(0, (6, 4)), zorder=1))
    arrow(ax, (10.6, 11.4), (10.6, 13.9), color=BUILD_E, label="obstacle ahead:\n0.9 m, or 0.55 m\nwhen closing in",
          loff=(-2.5, 0), lcolor=BUILD_E)
    arrow(ax, (12.9, 13.9), (12.9, 11.4), color=BUILD_E, label="clear by\n0.5 m more", loff=(1.5, 0),
          lcolor=BUILD_E)

    # AVOID -> RECOVER -> back to exploring
    arrow(ax, (16.05, 15.4), (17.7, 15.4), color=BUILD_E, label="8 s without moving 25 cm",
          loff=(0, 2.0), lcolor=BUILD_E, lsize=11.5)
    arrow(ax, (20.2, 13.9), (20.2, 11.4), color=BUILD_E, label="then explore\nfrom a new side",
          loff=(1.9, 0), lcolor=BUILD_E)

    # anything in the dashed group can end the mission
    arrow(ax, (22.7, 3.2), (24.7, 3.2), color=STOP_E)
    ax.text(14.0, 0.45, "any of these ends the mission:  8 search cycles with no red  ·  "
            "recoveries used up  ·  180 s", ha="center", va="center", fontsize=11.5, color=STOP_E)

    arrow(ax, (5.6, 7.5), (5.6, 4.7), color=NAV_E, label="360° and\nno red", loff=(-1.5, 0))
    arrow(ax, (9.4, 4.7), (9.4, 7.5), color=NAV_E, label="after 8 s", loff=(1.2, 0))
    arrow(ax, (10.8, 9.0), (14.7, 9.0), color=SLIP_E, label="red in view", loff=(0, 0.45), lcolor=SLIP_E)
    arrow(ax, (10.8, 4.0), (14.7, 7.9), color=SLIP_E, label="red in view", lpos=0.45, loff=(0.1, -0.7),
          lcolor=SLIP_E)
    arrow(ax, (16.1, 7.5), (16.1, 4.7), color=SLIP_E, label="red lost", loff=(-1.2, 0), lcolor=SLIP_E)
    arrow(ax, (19.9, 4.7), (19.9, 7.5), color=SLIP_E, label="red again", loff=(1.3, 0), lcolor=SLIP_E)
    arrow(ax, (21.3, 9.0), (24.7, 9.0), color=ANALYSE_E, label="close enough", loff=(0, 0.45),
          lcolor=ANALYSE_E)

    ax.text(25.0, 12.7, "Checked every 16 ms,\nthe first match wins:\nDONE › AVOID › GOAL ›\nRECALL › WANDER › SEARCH\n\n"
            "The memory of the goal expires\nafter 15 s without seeing it:\nToby goes back to searching.",
            fontsize=11.5, color=INK2, va="center", ha="left", linespacing=1.5)
    return fig


# ---------------------------------------------------------------- slip formula
def formula():
    fig = plt.figure(figsize=(10, 3.4), dpi=200)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.72, r"$s_{long} = \dfrac{v_{enc} - v_x}{v_x}$", ha="center", va="center", fontsize=40)
    fig.text(0.27, 0.22, r"$v_{enc} = r\,\dfrac{\omega_L + \omega_R}{2}$", ha="center", va="center", fontsize=24)
    fig.text(0.27, 0.03, "wheel encoders", ha="center", fontsize=13, color=INK2)
    fig.text(0.73, 0.22, r"$v_x = \mathbf{v}_{world} \cdot \hat{x}_{robot}$", ha="center", va="center",
             fontsize=24)
    fig.text(0.73, 0.03, "simulator ground truth", ha="center", fontsize=13, color=INK2)
    return fig


# ---------------------------------------------------------------- code as images
SNIPPETS = [
    # file, first line containing, last line containing, image name, lines to highlight (containing)
    ("controllers/slip_nav/slip_nav.py", "# ---- wheel encoders", "s_long = (v_enc", "code_slip",
     ["v_enc = WHEEL", "v_x = vel", "s_long = (v_enc"]),
    ("controllers/slip_nav/slip_nav.py", "def colour_blobs", "return blobs", "code_vision",
     ["vivid = ", "mask = vivid"]),
    ("controllers/slip_nav/slip_nav.py", "def avoidance_turn", "return float(", "code_avoidance",
     ["push = "]),
    ("controllers/slip_nav/slip_nav.py", "# ---- decide", "state = \"SEARCH\"", "code_decide", []),
    ("controllers/slip_nav/slip_nav.py", "if state == \"DONE\":", "v = CRUISE_SPEED * slow", "code_act",
     ["search_turned += "]),
    ("controllers/slip_nav/slip_nav.py", "# ---- differential drive", "right_motor.setVelocity", "code_wheels",
     []),
    ("tools/make_worlds.py", "FRICTION = {", "\"low_slip\":", "code_friction", []),
]


def cut(path, first, last):
    lines = open(os.path.join(ROOT, path), encoding="utf-8").read().splitlines()
    i = next(k for k, s in enumerate(lines) if first in s)
    j = next(k for k in range(i, len(lines)) if last in lines[k])
    if path.endswith("make_worlds.py"):
        j += 1                                          # include the closing brace of the dict
    block = lines[i:j + 1]
    indent = min(len(s) - len(s.lstrip()) for s in block if s.strip())
    return i + 1, [s[indent:] for s in block]


def code_images():
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import PythonLexer

    for path, first, last, name, marks in SNIPPETS:
        start, block = cut(path, first, last)
        hl = [k + 1 for k, s in enumerate(block) if any(m in s for m in marks)]
        fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=30, style="friendly",
                             line_numbers=True, line_number_start=start,
                             line_number_chars=len(str(start + len(block))) + 1, line_number_bg="#f1eff5",
                             line_number_fg="#9a94ad", line_number_pad=12, line_pad=6, image_pad=24,
                             hl_lines=hl, hl_color="#fff3c4")
        with open(os.path.join(OUT, name + ".png"), "wb") as f:
            f.write(highlight("\n".join(block) + "\n", PythonLexer(), fmt))
        print(f"  {name}.png  ({path} lines {start}-{start + len(block) - 1})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, make in (("flowchart", flowchart), ("state_machine", state_machine), ("slip_formula", formula)):
        fig = make()
        fig.savefig(os.path.join(OUT, name + ".png"), transparent=name == "slip_formula")
        plt.close(fig)
        print(f"  {name}.png")
    code_images()
    print("written in", os.path.relpath(OUT, ROOT))
