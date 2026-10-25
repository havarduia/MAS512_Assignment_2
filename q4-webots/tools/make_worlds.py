"""Generate the Q4 worlds: a candy-coloured toy city made only of solids.

    python tools/make_worlds.py

Writes the wet/dry assignment pair plus dry diagnostic worlds:
``city_no_goal.wbt`` has no red beacon and ``city_hidden_goal.wbt`` places the
goal outside the normal plaza route. The diagnostic worlds exercise explicit
mission-failure reporting and are not part of the wet/dry comparison.
"""
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FRICTION = {
    #  run name     coulombFriction  forceDependentSlip  description
    "baseline": (0.25, 0.03, "wet: low friction, high slip"),
    "low_slip": (1.0, 0.0008, "dry: high friction, low slip"),
}

# Deliberately dry diagnostic worlds. They use the same city and controller as the
# assignment worlds, but exercise terminal failure handling instead of slip.
SPECIAL_WORLDS = {
    "no_goal": {
        "goal": None,
        "description": "dry diagnostic: no red goal (expected goal_not_found)",
    },
    "hidden_goal": {
        # West side of the plaza. The start corridor runs north and its walls hide the whole
        # west half, so the camera cannot see the beacon until the robot has left the corridor
        # and turned: that is the exploration this world is meant to exercise. It also stands
        # 3.1 m clear of every prop. The first spot tried here, (7.0, 6.8), was 0.72 m from a
        # cafe table whose collision radius is 0.45 m, so the beacon could not be reached
        # without pushing into the table, whatever the controller did.
        "goal": (-9.0, -3.7),
        "description": "dry diagnostic: hidden red goal in the plaza (requires exploration)",
    },
}

# ---------------------------------------------------------------- scene
START = (0.0, -16.0, math.pi / 2)            # x, y, yaw: in the south street, looking north
GOAL = (8.7, 5.8)                            # the RED beacon: hidden among the cafe tables, north-east corner
# camera of the 3D view when the world opens (chosen by hand in Webots)
VIEWPOINT = ("0.3429728814149533 0.6260312174478416 -0.7003245800303584 1.3276845932807655",
             "-6.5081438206714495 12.192850653345399 25.420634565699")

COLORS = {  # saturated colours for the beacons; only red is the goal
    "red": (0.9, 0.08, 0.1), "orange": (1.0, 0.5, 0.05), "yellow": (1.0, 0.82, 0.05),
    "green": (0.15, 0.75, 0.3), "cyan": (0.05, 0.75, 0.8), "blue": (0.15, 0.4, 0.95),
    "purple": (0.55, 0.2, 0.85), "magenta": (0.9, 0.1, 0.65),
}
DECOYS = [                                   # same beacon, other colours (orange and magenta are close to red)
    ("magenta", (-5.64, -5.88)),
    ("yellow", (-1.3, -6.5)),
    ("orange", (-7.2, 3.6)),
    ("cyan", (-2.2, 5.6)),
    ("blue", (-5.0, 6.5)),
    ("green", (2.6, -4.2)),
    ("purple", (6.6, 1.0)),
    ("orange", (1.6, 8.0)),
    ("green", (5.4, 6.0)),
    ("blue", (8.0, -4.3)),
]

CANDY = [  # pastel walls (never saturated red: only the goal is red)
    (1.0, 0.72, 0.82), (0.66, 0.9, 0.82), (1.0, 0.93, 0.62), (0.8, 0.74, 0.98),
    (0.65, 0.84, 1.0), (1.0, 0.8, 0.66), (0.86, 0.95, 0.66), (0.98, 0.76, 0.95),
]
DOORS = [(0.45, 0.35, 0.6), (0.3, 0.55, 0.6), (0.6, 0.4, 0.5), (0.35, 0.45, 0.7)]


def building(i, x, y, w, d, h, yaw):
    c = CANDY[i % len(CANDY)]
    dc = DOORS[i % len(DOORS)]
    rc = [round(0.8 * v, 3) for v in c]          # roof: a deeper tone of the walls
    return f"""ToyBuilding {{
  translation {x} {y} 0
  rotation 0 0 1 {yaw:.4f}
  name "building {i}"
  size {w} {d} {h}
  color {c[0]} {c[1]} {c[2]}
  doorColor {dc[0]} {dc[1]} {dc[2]}
  roofColor {rc[0]} {rc[1]} {rc[2]}
  rooftop {"TRUE" if i % 3 else "FALSE"}
}}"""


def city_blocks():
    """Two rings of buildings around the 20x20 m square, leaving 5 m wide streets N, S, E, W."""
    out, i = [], 0
    for ring, (near, depth, hmin) in enumerate([(10.5, 6.5, 7), (17.5, 7.0, 12)]):
        mid = near + depth / 2
        far = near + depth
        # north / south rows: four buildings each, door towards the square
        for x0, x1 in [(-far, -far / 2 - 0.3), (-far / 2 + 0.3, -2.5), (2.5, far / 2 - 0.3), (far / 2 + 0.3, far)]:
            for sign in (1, -1):
                h = hmin + (i * 37 % 7)
                out.append(building(i, round((x0 + x1) / 2, 2), sign * mid, round(x1 - x0, 2), depth, h,
                                    0.0 if sign > 0 else math.pi))
                i += 1
        # east / west columns between the rows
        for y0, y1 in [(-near + 0.3, -2.5), (2.5, near - 0.3)]:
            for sign in (1, -1):
                h = hmin + (i * 37 % 7)
                out.append(building(i, sign * mid, round((y0 + y1) / 2, 2), round(y1 - y0, 2), depth, h,
                                    -math.pi / 2 if sign > 0 else math.pi / 2))
                i += 1
    return out


def plaza_and_streets():
    tiles = []
    colors = [(0.99, 0.95, 0.87), (0.91, 0.87, 0.98)]
    for ix in range(10):
        for iy in range(10):
            c = colors[(ix + iy) % 2]
            tiles.append(f"""      Pose {{
        translation {-9 + 2 * ix} {-9 + 2 * iy} 0.003
        children [
          Shape {{
            appearance PBRAppearance {{
              baseColor {c[0]} {c[1]} {c[2]}
              roughness 0.9
              metalness 0
            }}
            geometry Box {{
              size 2 2 0.006
            }}
          }}
        ]
      }}""")
    road = []
    for (cx, cy, sx, sy) in [(0, 20, 5, 20), (0, -20, 5, 20), (20, 0, 20, 5), (-20, 0, 20, 5)]:
        road.append(f"""      Pose {{
        translation {cx} {cy} 0.002
        children [
          Shape {{
            appearance DEF ROAD PBRAppearance {{
              baseColor 0.47 0.45 0.58
              roughness 0.95
              metalness 0
            }}
            geometry Box {{
              size {sx} {sy} 0.004
            }}
          }}
        ]
      }}""")
        # white dashes on the centre line
        vertical = sx < sy
        for k in range(5):
            off = -8 + 4 * k
            dx, dy = (0, off) if vertical else (off, 0)
            road.append(f"""      Pose {{
        translation {cx + dx} {cy + dy} 0.005
        children [
          Shape {{
            appearance PBRAppearance {{
              baseColor 1 1 1
              roughness 0.6
              metalness 0
            }}
            geometry Box {{
              size {0.2 if vertical else 1.6} {1.6 if vertical else 0.2} 0.004
            }}
          }}
        ]
      }}""")
    body = "\n".join(tiles + road)
    return f"""DEF PLAZA Solid {{
  name "plaza and streets"
  children [
{body}
  ]
  locked TRUE
}}"""


def furniture(goal=GOAL):
    items = ['Fountain {\n  translation 0 0 0\n}']
    for k, (x, y, c) in enumerate([(-7.5, 7.8, (0.6, 0.9, 0.75)), (-7.6, -7.6, (1.0, 0.75, 0.88)),
                                   (7.6, -7.8, (0.78, 0.72, 1.0)), (8.3, 8.3, (1.0, 0.9, 0.6))]):
        items.append(f'LollipopTree {{\n  translation {x} {y} 0\n  name "tree {k}"\n'
                     f'  crownColor {c[0]} {c[1]} {c[2]}\n}}')
    for k, (x, y, yaw) in enumerate([(-4, 8.9, math.pi), (8.9, -1.0, math.pi / 2)]):
        items.append(f'Bench {{\n  translation {x} {y} 0\n  rotation 0 0 1 {yaw:.4f}\n  name "bench {k}"\n}}')
    for k, (x, y) in enumerate([(-9.4, -9.4), (9.4, -9.4), (-9.4, 9.4), (9.4, 9.4),
                                (-3.3, -9.7), (3.3, -9.7), (-3.3, 9.7), (3.3, 9.7)]):
        items.append(f'StreetLamp {{\n  translation {x} {y} 0\n  name "lamp {k}"\n}}')

    # things you find in a piazza (all pastel: the camera must never mistake them for a beacon)
    items.append('Planter {\n  translation -0.17 -8.25 0\n  rotation 0 0 1 2.4527\n  name "planter 0"\n'
                 '  length 2.2\n}')
    # Keep the south approach open: the statue remains a central obstacle,
    # but no longer blocks the robot near its starting corridor.
    items.append('Statue {\n  translation -4.5 -3.5 0\n  rotation 0 0 1 2.4527\n}')
    for k, (x, y, yaw, c) in enumerate([(-9.3, 1.0, math.pi / 2, (0.98, 0.88, 0.78)),
                                        (6.0, -9.3, 0.0, (0.8, 0.9, 0.98)), (-0.5, 9.3, 0.0, (0.98, 0.88, 0.78))]):
        items.append(f'Planter {{\n  translation {x} {y} 0\n  rotation 0 0 1 {yaw:.4f}\n  name "planter {k + 1}"\n'
                     f'  color {c[0]} {c[1]} {c[2]}\n}}')
    for k, ((x, y), c) in enumerate([((4.2, 4.0), (1.0, 0.8, 0.7)), ((7.2, 4.2), (0.8, 0.78, 0.98)),
                                     ((6.6, 7.4), (0.7, 0.9, 0.85)), ((3.4, 7.0), (1.0, 0.86, 0.94))]):
        items.append(f'CafeTable {{\n  translation {x} {y} 0\n  name "cafe table {k}"\n'
                     f'  umbrellaColor {c[0]} {c[1]} {c[2]}\n}}')
    items.append('Kiosk {\n  translation -1.8 7.4 0\n}')
    for k, (x, y, c) in enumerate([(5.6, -5.6, (0.7, 0.88, 0.98)), (5.6, -2.4, (1.0, 0.8, 0.86))]):
        items.append(f'MarketStall {{\n  translation {x} {y} 0\n  rotation 0 0 1 1.5708\n  name "market stall {k}"\n'
                     f'  stripeColor {c[0]} {c[1]} {c[2]}\n}}')
    items.append('GelatoCart {\n  translation 3.2 -7.4 0\n  rotation 0 0 1 0.5\n}')
    for k, (x, y) in enumerate([(-6.05, -8.46), (2.6, 2.6), (-3.0, 2.4), (8.8, 2.8)]):
        items.append(f'Bin {{\n  translation {x} {y} 0\n  name "bin {k}"\n}}')
    for k, (x, y) in enumerate([(-1.6, 10.4), (0, 10.4), (1.6, 10.4), (10.4, -1.6), (10.4, 0), (10.4, 1.6)]):
        items.append(f'Bollard {{\n  translation {x} {y} 0\n  name "bollard {k}"\n}}')

    if goal is not None:
        c = COLORS["red"]
        items.append(f'DEF GOAL Beacon {{\n  translation {goal[0]} {goal[1]} 0\n  name "goal"\n'
                     f'  color {c[0]} {c[1]} {c[2]}\n}}')
    for k, (name, (x, y)) in enumerate(DECOYS):
        c = COLORS[name]
        items.append(f'DEF DECOY_{k} Beacon {{\n  translation {x} {y} 0\n  name "decoy {name} {k}"\n'
                     f'  color {c[0]} {c[1]} {c[2]}\n}}')
    return items


HEADER = """#VRML_SIM R2025a utf8
# Q4 - Autonomous navigation and slip estimation: toy city ({desc}).
# Generated by tools/make_worlds.py - edit that script, not this file.

EXTERNPROTO "../protos/ToyRobot.proto"
EXTERNPROTO "../protos/ToyBuilding.proto"
EXTERNPROTO "../protos/LollipopTree.proto"
EXTERNPROTO "../protos/Beacon.proto"
EXTERNPROTO "../protos/Fountain.proto"
EXTERNPROTO "../protos/Bench.proto"
EXTERNPROTO "../protos/StreetLamp.proto"
EXTERNPROTO "../protos/Kiosk.proto"
EXTERNPROTO "../protos/CafeTable.proto"
EXTERNPROTO "../protos/MarketStall.proto"
EXTERNPROTO "../protos/Planter.proto"
EXTERNPROTO "../protos/Statue.proto"
EXTERNPROTO "../protos/Bin.proto"
EXTERNPROTO "../protos/GelatoCart.proto"
EXTERNPROTO "../protos/Bollard.proto"

WorldInfo {{
  info [
    "Q4 - find the red beacon in the toy city, avoid everything else, measure the wheel slip."
    "Friction: {desc}"
  ]
  title "Q4 toy city - {run}"
  basicTimeStep 16
  contactProperties [
    ContactProperties {{
      material1 "floor"
      material2 "wheel"
      coulombFriction [
        {mu}
      ]
      forceDependentSlip [
        {fds}
      ]
      bumpSound ""
      rollSound ""
      slideSound ""
    }}
    ContactProperties {{
      material1 "floor"
      material2 "caster"
      coulombFriction [
        0.1
      ]
      bumpSound ""
      rollSound ""
      slideSound ""
    }}
  ]
}}
Viewpoint {{
  orientation {vo}
  position {vp}
  near 0.3
}}
Background {{
  skyColor [
    0.72 0.85 1
  ]
}}
DirectionalLight {{
  ambientIntensity 0.8
  direction -0.4 0.6 -1
  intensity 1.8
  castShadows TRUE
}}
DEF FLOOR Solid {{
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.74 0.9 0.72
        roughness 1
        metalness 0
      }}
      geometry Plane {{
        size 90 90
      }}
    }}
  ]
  name "floor"
  contactMaterial "floor"
  boundingObject Plane {{
    size 90 90
  }}
  locked TRUE
}}
"""

ROBOT = """DEF ROBOT ToyRobot {{
  translation {x} {y} 0
  rotation 0 0 1 {yaw:.4f}
  controllerArgs [
    "{run}"
  ]
}}
"""

def write_world(run, mu, fds, desc, goal=GOAL):
    parts = [HEADER.format(run=run, desc=desc, mu=mu, fds=fds, vo=VIEWPOINT[0], vp=VIEWPOINT[1]),
                 plaza_and_streets(), *city_blocks(), *furniture(goal),
                 ROBOT.format(x=START[0], y=START[1], yaw=START[2], run=run)]
    path = os.path.join(ROOT, "worlds", f"city_{run}.wbt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    print("written", path)


if __name__ == "__main__":
    for run, (mu, fds, desc) in FRICTION.items():
        write_world(run, mu, fds, desc)
    for run, config in SPECIAL_WORLDS.items():
        write_world(run, 1.0, 0.0008, config["description"], config["goal"])
