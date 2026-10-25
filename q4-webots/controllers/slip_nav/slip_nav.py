import csv
import json
import math
import os
import sys
from collections import deque

import numpy as np
from controller import Supervisor

# ---------------------------------------------------------------- parameters
WHEEL_RADIUS = 0.12      # [m]   must match protos/ToyRobot.proto
AXLE_TRACK = 0.60        # [m]   distance between the wheels
MAX_WHEEL_SPEED = 14.0   # [rad/s]
CRUISE_SPEED = 1.2       # [m/s]
GOAL_STOP_AREA = 0.25    # stop when the red beacon fills 25 % of the image (~1.5 m away)
OBST_SLOW = 2.5          # [m] start steering away from obstacles
OBST_STOP = 0.9          # [m] something right in front -> turn on the spot
OBST_CLEAR = 0.5         # [m] extra clearance needed to leave AVOID: hysteresis against chattering
GOAL_APPROACH_STOP = 0.55  # [m] stop distance while closing on a visible goal 
AVOID_BACKOFF = 0.15     # [m/s] creep backwards while turning away
SIDE_WEIGHT = 0.2        # smallest weight of a ray in the potential field (at +-90 deg)
HEADING_GAIN = 1.5       # how hard the robot steers onto the goal bearing
MAX_OMEGA = 1.2          # [rad/s] steering limit
FRONT_FILTER = 5         # samples of the median filter on the front distance (5 x 16 ms = 80 ms)
V_X_MIN = 0.1            # [m/s] below this s_long is undefined (division by ~0)
MAX_TIME = 180.0         # [s]
GOAL_MEMORY = 15.0       # [s] how long RECALL keeps trusting the last sighting before searching again
MAX_SEARCH_CYCLES = 8    # rotation/exploration cycles without ever seeing the red goal
WANDER_DURATION = 8.0    # [s] explore the plaza before searching again
STUCK_CHECK_EVERY = 8.0  # [s] progress window while the robot is pursuing the goal
STUCK_MIN_PROGRESS = 0.25  # [m] minimum displacement in the progress window
MAX_RECOVERIES = 3       # obstacle recovery attempts before reporting a failure
REVERSE_DURATION = 1.5   # [s] back away from an obstacle
RECOVERY_TURN_DURATION = 1.5  # [s] turn toward the more open side
START = (0.0, -16.0, math.pi / 2)   # x, y, yaw of the start: must match tools/make_worlds.py
LIDAR_LOG_EVERY = 1      # full lidar scan every N steps (1 = every step)
FRAME_EVERY = 0.25       # [s] save a camera frame every N seconds (enough for an animation)

# colour classes for the beacons (hue in degrees, on saturated pixels only)
COLOURS = {"red": [(0, 14), (344, 360)], "orange": [(18, 38)], "yellow": [(40, 70)],
           "green": [(90, 168)], "cyan": [(172, 192)], "blue": [(196, 250)],
           "purple": [(255, 295)], "magenta": [(300, 338)]}
GOAL_COLOUR = "red"

RUN = sys.argv[1] if len(sys.argv) > 1 else "run"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", RUN))
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
AUTOQUIT = os.environ.get("WEBOTS_AUTOQUIT") == "1"        # used for batch runs
SNAPSHOTS = os.environ.get("Q4_SNAPSHOTS") == "1"         # save 3D-view screenshots

# ---------------------------------------------------------------- devices
robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
dt = dt_ms / 1000.0

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))          # velocity control
    m.setVelocity(0.0)
left_enc = robot.getDevice("left wheel sensor")
right_enc = robot.getDevice("right wheel sensor")
camera = robot.getDevice("camera")
lidar = robot.getDevice("lidar")
gps = robot.getDevice("gps")
imu = robot.getDevice("imu")
acc = robot.getDevice("accelerometer")
gyro = robot.getDevice("gyro")
for d in (left_enc, right_enc, camera, lidar, gps, imu, acc, gyro):
    d.enable(dt_ms)

self_node = robot.getSelf()
cam_w, cam_h, cam_fov = camera.getWidth(), camera.getHeight(), camera.getFov()
n_rays, lidar_fov = lidar.getHorizontalResolution(), lidar.getFov()
ray_angles = np.linspace(lidar_fov / 2, -lidar_fov / 2, n_rays)   # index 0 = leftmost ray
TAN_HALF_FOV = math.tan(cam_fov / 2)
FRONT_HALF = math.radians(25) + 1e-9   # the +-25 deg rays land exactly on the edge: include them
FRONT = np.abs(ray_angles) <= FRONT_HALF
LEFT = (ray_angles > FRONT_HALF) & (ray_angles < math.radians(90))
RIGHT = (ray_angles < -FRONT_HALF) & (ray_angles > -math.radians(90))


def sectors(scan):
    """Nearest echo in front, to the left and to the right."""
    return scan[FRONT].min(), scan[LEFT].min(), scan[RIGHT].min()


def bearing(cx):
    """Image column (cx in [-1, 1], left..right) -> bearing in the robot frame [rad].

    Pinhole model: a linear cx * fov / 2 under-estimates the angle towards the edges,
    which made the goal mask (below) jitter by a degree or two from frame to frame."""
    return -math.atan(cx * TAN_HALF_FOV)

# If the world was saved (Ctrl+S) after a run, Webots stored Toby where the run ended, next to the
# goal. Put him back at the start so every run is the same experiment.
where = self_node.getField("translation").getSFVec3f()
if math.hypot(where[0] - START[0], where[1] - START[1]) > 0.05:
    self_node.getField("translation").setSFVec3f([START[0], START[1], 0.0])
    self_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, START[2]])
    self_node.resetPhysics()
    print(f"[slip_nav] Toby was saved at ({where[0]:.1f}, {where[1]:.1f}): moved back to the start")


# ---------------------------------------------------------------- scene description (for the plots)
def scene_metadata():
    meta = {"run": RUN, "wheel_radius": WHEEL_RADIUS, "axle_track": AXLE_TRACK,
            "time_step_ms": dt_ms, "objects": [], "contact_properties": []}
    children = robot.getRoot().getField("children")
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        kind = node.getTypeName()
        if kind not in ("ToyBuilding", "Beacon", "Fountain", "LollipopTree", "Bench", "StreetLamp", "Kiosk",
                        "CafeTable", "MarketStall", "Planter", "Statue", "Bin", "GelatoCart", "Bollard"):
            continue
        p, R = node.getPosition(), node.getOrientation()
        obj = {"type": kind, "name": node.getField("name").getSFString(),
               "x": p[0], "y": p[1], "yaw": math.atan2(R[3], R[0])}
        if kind == "ToyBuilding":
            obj["size"] = node.getField("size").getSFVec3f()
            obj["color"] = node.getField("color").getSFColor()
        elif kind == "Beacon":
            obj["color"] = node.getField("color").getSFColor()
            obj["is_goal"] = node.getDef() == "GOAL"
        elif kind == "LollipopTree":
            obj["color"] = node.getField("crownColor").getSFColor()
        elif kind == "Planter":
            obj["length"] = node.getField("length").getSFFloat()
        meta["objects"].append(obj)
    cps = children.getMFNode(0).getField("contactProperties")        # WorldInfo is the first node
    for i in range(cps.getCount()):
        c = cps.getMFNode(i)
        meta["contact_properties"].append({
            "material1": c.getField("material1").getSFString(),
            "material2": c.getField("material2").getSFString(),
            "coulombFriction": [c.getField("coulombFriction").getMFFloat(j)
                                for j in range(c.getField("coulombFriction").getCount())],
            "forceDependentSlip": [c.getField("forceDependentSlip").getMFFloat(j)
                                   for j in range(c.getField("forceDependentSlip").getCount())],
        })
    return meta


meta = scene_metadata()

# ---------------------------------------------------------------- perception
def colour_blobs():
    """Split the camera image into colour classes. Returns {colour: (cx, area, x_min, x_max)}.

    cx in [-1, 1] (left .. right), area = fraction of the image. Only saturated pixels
    count, so the pastel buildings never look like a beacon."""
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((cam_h, cam_w, 4))[:, :, :3] / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    delta = np.maximum(mx - mn, 1e-6)
    hue = np.select([mx == r, mx == g], [(g - b) / delta % 6, (b - r) / delta + 2], (r - g) / delta + 4) * 60
    vivid = (sat > 0.55) & (mx > 0.25)
    blobs = {}
    for name, ranges in COLOURS.items():
        mask = vivid & np.any([(hue >= lo) & (hue < hi) for lo, hi in ranges], axis=0)
        area = mask.mean()
        if area > 0.0003:
            xs = np.nonzero(mask)[1]
            to_cx = lambda x: (x - cam_w / 2) / (cam_w / 2)
            blobs[name] = (to_cx(xs.mean()), area, to_cx(xs.min()), to_cx(xs.max()))
    return blobs


def lidar_view(ranges, ignore=None):
    """Return (raw scan, navigation scan, front, left, right) minima.

    The raw scan is what goes in lidar.csv: the sensor as it is, with the "no echo"
    infinities written as maxRange. The navigation scan additionally blanks the bearing
    interval covered by the goal, because the beacon must not repel the robot."""
    raw = np.where(np.isfinite(ranges), ranges, lidar.getMaxRange())
    r = raw
    if ignore is not None:
        lo, hi = ignore
        r = np.where((ray_angles >= lo) & (ray_angles <= hi), lidar.getMaxRange(), raw)
    return (raw, r) + sectors(r)


def avoidance_turn(r):
    """Potential field: every close ray pushes the robot away from its side (+ = turn left).

    Obstacles ahead weigh most, but the weight never reaches zero: a plain cos() cancels
    exactly the rays at +-90 deg, so something right beside the robot (a cafe chair, a bin)
    pushed with zero force and Toby used to scrape along it with its flank."""
    near = r < OBST_SLOW
    if not near.any():
        return 0.0
    weight = np.maximum(np.cos(ray_angles[near]), SIDE_WEIGHT)
    push = (1.0 / r[near] - 1.0 / OBST_SLOW) * weight
    return float(np.clip(-np.sum(np.sign(ray_angles[near]) * push) * 0.25, -MAX_OMEGA, MAX_OMEGA))


# ---------------------------------------------------------------- logging
log_f = open(os.path.join(OUT, "log.csv"), "w", newline="")
log = csv.writer(log_f)
log.writerow([
    "t", "state",
    "theta_l", "theta_r", "omega_l", "omega_r", "cmd_omega_l", "cmd_omega_r",
    "v_enc", "v_x", "s_long",
    "gt_x", "gt_y", "gt_vx_world", "gt_vy_world",
    "gps_x", "gps_y", "gps_z", "gps_speed",
    "roll", "pitch", "yaw",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "lidar_min_front", "lidar_min_left", "lidar_min_right", "nav_front",
    "goal_cx", "goal_area", "decoys_in_view",
])
lidar_f = open(os.path.join(OUT, "lidar.csv"), "w", newline="")
lidar_log = csv.writer(lidar_f)
lidar_log.writerow(["t"] + [f"r{i}" for i in range(n_rays)])


def hud(state, s_long, decoys, failure_reason=None):
    """Text overlay in the 3D view (Supervisor labels)."""
    slip = "  --  " if math.isnan(s_long) else f"{100 * s_long:+5.1f} %"
    robot.setLabel(0, f"Toby  |  {RUN}  |  state: {state}", 0.01, 0.01, 0.08, 0x2B2440, 0, "Arial")
    robot.setLabel(1, f"longitudinal slip: {slip}", 0.01, 0.055, 0.08, 0x2B2440, 0, "Arial")
    if decoys:
        robot.setLabel(2, "decoys seen and ignored: " + ", ".join(sorted(decoys)),
                       0.01, 0.10, 0.07, 0x5A4E7A, 0, "Arial")
    if failure_reason:
        robot.setLabel(3, f"mission stopped: {failure_reason}", 0.01, 0.145, 0.07, 0x8B1E3F, 0, "Arial")


# ---------------------------------------------------------------- main loop
state = "SEARCH"
failure_reason = None
goal_heading = None          # world yaw of the goal when last seen
goal_seen_t = -1e9           # when it was last seen: the memory expires after GOAL_MEMORY
search_turned = 0.0          # how much we rotated while searching [rad]
search_cycles = 0         # complete search/exploration cycles without seeing the goal
wander_until = 0.0
decoys_seen = {}             # colour -> first time seen
prev_theta = None
step = 0
next_frame = 0.0
progress_anchor = None
front_hist = deque(maxlen=FRONT_FILTER)
recovery_until = 0.0
recovery_turn_until = 0.0
recovery_direction = 1.0
recovery_count = 0

while robot.step(dt_ms) != -1:
    t = robot.getTime()
    step += 1

    # ---- wheel encoders -> v_enc
    th_l, th_r = left_enc.getValue(), right_enc.getValue()
    if prev_theta is None:
        prev_theta = (th_l, th_r)
    om_l, om_r = (th_l - prev_theta[0]) / dt, (th_r - prev_theta[1]) / dt
    prev_theta = (th_l, th_r)
    v_enc = WHEEL_RADIUS * (om_l + om_r) / 2.0

    # ---- ground truth -> v_x: world velocity projected on the robot's forward axis
    vel = self_node.getVelocity()
    R = self_node.getOrientation()
    v_x = vel[0] * R[0] + vel[1] * R[3] + vel[2] * R[6]
    pos = self_node.getPosition()
    s_long = (v_enc - v_x) / v_x if abs(v_x) > V_X_MIN else float("nan")

    # ---- other sensors
    g, rpy, a, w = gps.getValues(), imu.getRollPitchYaw(), acc.getValues(), gyro.getValues()
    yaw = rpy[2]
    blobs = colour_blobs()
    goal = blobs.get(GOAL_COLOUR)
    for colour, (_, area, _, _) in blobs.items():
        if colour != GOAL_COLOUR and area > 0.001 and colour not in decoys_seen:
            decoys_seen[colour] = t
            print(f"[slip_nav] t={t:5.1f}s  {colour} beacon in view -> not red, ignored")
    ignore = None
    if goal is not None:
        cx, area, x0, x1 = goal
        goal_heading, goal_seen_t = yaw + bearing(cx), t
        ignore = (bearing(x1) - 0.06, bearing(x0) + 0.06)   # bearings covered by the goal
    elif goal_heading is not None and t - goal_seen_t > GOAL_MEMORY:
        # A remembered heading is only a heading: it does not follow the robot as it moves,
        # so after a while it is worse than no information. Drop it and search again, which
        # also keeps the "goal_not_found" exit reachable after a single stray red blob.
        goal_heading, search_turned, wander_until = None, 0.0, 0.0
        print(f"[slip_nav] t={t:5.1f}s  red beacon lost for {GOAL_MEMORY:.0f}s: back to searching")
    # The mask follows the sighting, not the state, so the beacon is never an obstacle and
    # the scan the controller reacts to no longer depends on the previous step's state.
    raw_scan, r, d_front_now, d_left, d_right = lidar_view(np.array(lidar.getRangeImage()), ignore)
    raw_front, raw_left, raw_right = sectors(raw_scan)      # what the sensor saw, for the log
    front_hist.append(d_front_now)
    d_front = float(np.median(front_hist))   # one flickering ray must not brake the robot

    # ---- decide
    closing = goal is not None and abs(bearing(goal[0])) < 0.35 and goal[1] > 0.05
    stop_at = GOAL_APPROACH_STOP if closing else OBST_STOP
    if state not in ("DONE", "FAILED", "RECOVER"):
        if goal is not None and goal[1] > GOAL_STOP_AREA:
            state = "DONE"
            print(f"[slip_nav] t={t:5.1f}s  red beacon reached")
        elif d_front < stop_at or (state == "AVOID" and d_front < stop_at + OBST_CLEAR):
            # Enter at stop_at, leave only half a metre further out, so the state cannot
            # chatter. Avoidance stays armed all the way in: closing on the goal only
            # shortens the stop distance, it no longer switches avoidance off completely.
            state = "AVOID"
        elif goal is not None:
            state = "GOAL"
        elif goal_heading is not None:
            state = "RECALL"
        elif t < wander_until:
            state = "WANDER"
        elif search_turned > 2 * math.pi:
            search_cycles += 1
            if goal_heading is None and search_cycles >= MAX_SEARCH_CYCLES:
                state = "FAILED"
                failure_reason = "goal_not_found"
                print(f"[slip_nav] t={t:5.1f}s  mission stopped: {failure_reason}")
            else:
                state, search_turned, wander_until = "WANDER", 0.0, t + WANDER_DURATION
        else:
            state = "SEARCH"

    if state == "DONE":
        v, omega = 0.0, 0.0
    elif state == "FAILED":
        v, omega = 0.0, 0.0
    elif state == "RECOVER":
        if t < recovery_until:
            v, omega = -0.8, 0.0
        elif t < recovery_turn_until:
            v, omega = 0.0, recovery_direction * 1.1
        else:
            state = "WANDER"
            wander_until = t + WANDER_DURATION
            v, omega = CRUISE_SPEED, avoidance_turn(r)
    elif state == "SEARCH":
        v, omega = 0.0, 0.8
        search_turned += abs(w[2]) * dt     # measured by the gyro: on a wet floor the wheels slip and
                                            # the robot turns less than commanded
    elif state == "WANDER":
        v, omega = CRUISE_SPEED, avoidance_turn(r)
    elif state == "AVOID":
        v, omega = -AVOID_BACKOFF, 0.9 if d_left > d_right else -0.9
    else:  # GOAL or RECALL
        if state == "GOAL":
            heading_err = bearing(goal[0])
        else:
            heading_err = math.atan2(math.sin(goal_heading - yaw), math.cos(goal_heading - yaw))
        omega = float(np.clip(HEADING_GAIN * heading_err + avoidance_turn(r), -MAX_OMEGA, MAX_OMEGA))
        slow = min(1.0, max(0.3, (d_front - OBST_STOP) / (OBST_SLOW - OBST_STOP)))
        v = CRUISE_SPEED * slow * max(0.2, math.cos(heading_err))

    # A dry world can still contain an intentionally impossible layout. The watchdog
    # runs while the robot is pursuing the goal - driving towards it or manoeuvring
    # around an obstacle - and is off while it spins on the spot to search, because a
    # search rotation is not a stuck driving episode. Turning away from a wall forever
    # (AVOID) is one, which is why AVOID is watched too.
    if state in ("GOAL", "RECALL", "WANDER", "AVOID"):
        if progress_anchor is None:
            progress_anchor = (t, pos[0], pos[1])
        elif t - progress_anchor[0] >= STUCK_CHECK_EVERY:
            if (math.hypot(pos[0] - progress_anchor[1], pos[1] - progress_anchor[2])
                    >= STUCK_MIN_PROGRESS):
                progress_anchor = (t, pos[0], pos[1])
                recovery_count = 0          # it moved again: earlier recoveries worked
            elif d_front < OBST_SLOW and recovery_count < MAX_RECOVERIES:
                recovery_count += 1
                recovery_until = t + REVERSE_DURATION
                recovery_turn_until = recovery_until + RECOVERY_TURN_DURATION
                recovery_direction = 1.0 if d_left > d_right else -1.0
                state = "RECOVER"
                progress_anchor = None
                v, omega = -0.8, 0.0
                print(f"[slip_nav] t={t:5.1f}s  obstacle recovery "
                      f"{recovery_count}/{MAX_RECOVERIES}: reverse and turn")
            else:
                state = "FAILED"
                failure_reason = ("obstacle_recovery_exhausted"
                                  if d_front < OBST_SLOW else "stuck_no_progress")
                print(f"[slip_nav] t={t:5.1f}s  mission stopped: {failure_reason}")
                v, omega = 0.0, 0.0
    else:
        progress_anchor = None

    # ---- differential drive inverse kinematics
    cmd_l = (v - omega * AXLE_TRACK / 2) / WHEEL_RADIUS
    cmd_r = (v + omega * AXLE_TRACK / 2) / WHEEL_RADIUS
    scale = max(1.0, abs(cmd_l) / MAX_WHEEL_SPEED, abs(cmd_r) / MAX_WHEEL_SPEED)
    left_motor.setVelocity(cmd_l / scale)
    right_motor.setVelocity(cmd_r / scale)

    # ---- log + HUD
    log.writerow([
        f"{t:.3f}", state, th_l, th_r, om_l, om_r, cmd_l / scale, cmd_r / scale,
        v_enc, v_x, s_long, pos[0], pos[1], vel[0], vel[1],
        g[0], g[1], g[2], gps.getSpeed(), rpy[0], rpy[1], rpy[2],
        a[0], a[1], a[2], w[0], w[1], w[2],
        raw_front, raw_left, raw_right, d_front,
        "" if goal is None else goal[0], 0.0 if goal is None else goal[1],
        " ".join(sorted(c for c in blobs if c != GOAL_COLOUR)),
    ])
    if step % LIDAR_LOG_EVERY == 0:
        lidar_log.writerow([f"{t:.3f}"] + [f"{x:.3f}" for x in raw_scan])
    if t >= next_frame:
        camera.saveImage(os.path.join(OUT, "frames", f"frame_{t:07.2f}.png"), 85)
        if SNAPSHOTS:
            robot.exportImage(os.path.join(OUT, "frames", f"view_{t:07.2f}.jpg"), 85)
        next_frame += FRAME_EVERY
    hud(state, s_long, decoys_seen, failure_reason)

    if ((state in ("DONE", "FAILED") and abs(v_x) < 0.01) or
            (t > MAX_TIME and state not in ("DONE", "FAILED"))):
        if state not in ("DONE", "FAILED"):
            state = "FAILED"
            failure_reason = "timeout"
            print(f"[slip_nav] t={t:5.1f}s  mission stopped: {failure_reason}")
        break

log_f.close()
lidar_f.close()
meta["result"] = "goal_reached" if state == "DONE" else "failed"
meta["failure_reason"] = failure_reason
meta["duration_s"] = robot.getTime()
meta["decoys_ignored"] = decoys_seen
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"[slip_nav] run '{RUN}' finished: {meta['result']} after {meta['duration_s']:.1f} s -> {OUT}")

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
if SNAPSHOTS:
    robot.step(dt_ms)
    robot.exportImage(os.path.join(OUT, "frames", "view_final.jpg"), 90)
if AUTOQUIT:
    robot.simulationQuit(0)
else:
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    robot.step(dt_ms)
