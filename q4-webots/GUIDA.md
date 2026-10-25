# Q4 guide: what, how and why

This guide explains the whole project from scratch, without assuming you have ever opened
Webots. The README is the shop window, this is the explanation.

## 1. What the question asks for

| item | what you have to deliver | where it lives |
|---|---|---|
| (a) baseline | a robot with camera, lidar, accelerometer, gyro, GPS and IMU that reaches a coloured object while avoiding collisions; sensor logs; computation of `s_long` | `worlds/city_baseline.wbt`, `controllers/slip_nav/`, `data/baseline/` |
| (b) low slip | the same mission with a different friction (higher, hence less slip) and a recomputation of `s_long` | `worlds/city_low_slip.wbt`, `data/low_slip/` |
| (c) plot | the slip along the trajectory for both cases | `analysis/plot_slip.py`, `figures/` |

"Low slip" means *less* slipping, so the second world has **more** friction than the first.
That is why the baseline is deliberately "wet" and the second one "dry".

## 2. What is in the folder

```
q4-webots/
├── worlds/            the two worlds: identical except for the wheel-floor friction
├── protos/            the building blocks of the city and the robot Toby
├── controllers/
│   └── slip_nav/      Toby's brain (Python): vision, lidar, states, logging, slip
├── tools/
│   ├── make_worlds.py      generates the two worlds (the city is drawn from here)
│   └── slide_figures.py    flow chart, state machine and code as images, for the slides
└── analysis/
    ├── plot_slip.py        the slip plots (item c)
    ├── camera_montage.py   what the camera sees, with the detections drawn on top
    └── animate_run.py      video of the run (one run, or wet and dry side by side)
```

`data/` (the logs of the four runs) and `figures/` (plots and videos) are already in the
folder: you regenerate them with your own simulation and with the analysis scripts, and they
should come out the same.

## 3. How to run it

**With the GUI (the normal way).** Install Webots R2025a, open `worlds/city_baseline.wbt`
(File > Open World) and press ▶. Toby starts moving, and the top left corner shows his state,
the instantaneous slip and the decoy beacons he has ignored. When he reaches the goal the
simulation pauses. Then repeat with `worlds/city_low_slip.wbt`. Each run writes (and
overwrites) its own folder in `data/`: `log.csv` (all the sensors at every 16 ms step),
`lidar.csv` (the **raw** scan at every step: the only change is that rays with no echo, which
the sensor reports as infinity, are written as `maxRange`; the mask the controller puts over
the goal never reaches the log), `frames/` (one camera image every 0.25 s) and `meta.json`
(frictions, object positions, outcome). In `log.csv` the columns
`lidar_min_front/left/right` are the minima of the three sectors on the raw scan, while
`nav_front` is the filtered value the controller actually decides on: having both is exactly
what lets you compare the two.

**Careful with Ctrl+S after a run.** If you save the world once the run is over, Webots
writes into the file the position Toby is in at that moment, i.e. next to the goal, and when
you reopen it you see him start from there. The controller notices this and puts him back at
the start on startup (the console prints "moved back to the start"), so the run stays valid.
To get a clean file again, copy the `.wbt` back from the zip or re-run
`python tools/make_worlds.py`. If you want to change the starting point, change `START` both
in `tools/make_worlds.py` and in `controllers/slip_nav/slip_nav.py`.

**The plots and the videos.**

```
pip install -r requirements.txt
python analysis/plot_slip.py               # figures/slip_map.png, slip_profile.png, speeds.png
python analysis/camera_montage.py          # figures/what_toby_sees.png
python analysis/animate_run.py compare     # figures/run_compare.mp4 (wet and dry together)
python analysis/animate_run.py baseline    # figures/run_baseline.mp4 (map + camera + slip)
python tools/slide_figures.py              # figures/slides/ (flow chart, states, code)
```

The videos need `ffmpeg` (Windows: `winget install ffmpeg`, Mac: `brew install ffmpeg`);
without it, `animate_run.py` writes a GIF.

**Headless** (on a Linux server, for example):

```
WEBOTS_AUTOQUIT=1 xvfb-run -a webots --batch --mode=fast --no-rendering --stdout worlds/city_baseline.wbt
```

That same command is what `../MAS512_assignment2_gr1.ipynb` runs: one cell, from
`pip install` to the figures, with no GUI to click through.

## 4. Webots in five minutes

- **World (`.wbt`)**: a text file describing the scene as a tree of *nodes* (floor,
  buildings, lights, robot). You open it in Webots and press ▶.
- **Solid**: a physical object. `children` is how it looks, `boundingObject` is the shape
  used for collisions, `physics` gives it a mass. Without `physics` the object is immovable,
  like our buildings and beacons.
- **PROTO**: a "home-made" node, i.e. a reusable model with parameters.
  `protos/ToyBuilding.proto` for instance is "a building": you pass it sizes and colours and
  it generates walls, windows on every floor, a door, a roof edge. That way the world simply
  says `ToyBuilding { size 6 6 9 color 1 0.7 0.8 }` instead of 200 lines of boxes.
- **Robot**: a special Solid with *devices* (motors, sensors) and a **controller**, i.e. a
  program (Python, for us) that Webots launches on its behalf.
- **The controller loop**: `robot.step(dt)` advances the physics by `dt` (16 ms). On every
  iteration you read the sensors, decide, command the motors. All the code runs inside that
  `while`.
- **Supervisor**: a robot with `supervisor TRUE` can look "behind the scenes" of the
  simulator, for example at its *true* velocity. That is our ground truth for `v_x`.
- **Friction**: in Webots it is not a property of an object but of a **pair of materials**.
  Every Solid has a `contactMaterial` (`"wheel"`, `"floor"`…) and in
  `WorldInfo.contactProperties` you say what happens when those two touch.

## 5. The toy city

The style is that of the cities in *Wreck-It Ralph*: everything made of simple solids, pastel
colours, "pixel" windows.

- **A chequered square**, 20 × 20 m, full of the things you find in a real square: a
  fountain, a newsstand, four café tables with parasols, two market stalls, an ice-cream
  cart, a statue (a big guy with two enormous fists, a nod to Ralph), planters, bins,
  benches, street lamps, bollards and lollipop trees. They are all real obstacles, with
  collisions.
- **Two rings of buildings** (24 in total) with four 5 m wide streets heading north, south,
  east and west. Toby starts from the southern street.
- **The goal**: a **red** beacon with white stripes hidden among the café tables, in the
  north-east corner of the square (position chosen by you). It is not visible from the start:
  Toby has to look around, move forward and search for it, and even once he has seen it the
  stalls and the tables keep covering it.
- **Ten decoy beacons**: same shape, colours **orange, yellow, green, cyan, blue, purple and
  magenta** (some of them repeated). They are there to show that the robot really does
  recognise the *colour* and does not head for "anything beacon-shaped". Orange and magenta
  are placed on purpose because they are the colours closest to red. The blue one and the
  purple one are right on the way to the goal: Toby has to avoid them as obstacles *and*
  ignore them as targets.
- **No red anywhere in the scenery**: buildings and furniture are pastel (low saturation), so
  the camera never mistakes them for a beacon. Even the fruit on the stall is light green.
- **The planters are deliberately low** (0.5 m): the lidar (at 0.3 m above the ground) sees
  them and avoids them, but the camera (at 0.65 m) looks over them.

The two worlds are generated by `tools/make_worlds.py`. Why a script instead of two
hand-written files? Because the experiment is only valid if the two worlds are **identical
except for the friction**: with a generator that is guaranteed. If you want to move the goal,
add a building or change the colours, you edit the script and re-run it
(`python tools/make_worlds.py`).

### Edge cases and controlled termination

The same controller can also be run in the diagnostic worlds `worlds/city_no_goal.wbt` and
`worlds/city_hidden_goal.wbt`. These are dry worlds and they are not part of the slip
comparison.

- `city_no_goal.wbt` does not contain the red beacon. Toby alternates search rotations and
  exploratory sorties; after eight cycles without a single sighting he enters `FAILED` with
  `goal_not_found` (126.2 s in my run). That is the proof that an impossible simulation does
  not wander forever.
- `city_hidden_goal.wbt` puts the goal on the **west** side of the square, where the
  buildings of the starting street hide it completely. Toby spins in place without seeing it,
  leaves the corridor and only finds it once he is in the square (t = 12.4 s):
  `goal_reached` in 22.1 s. That is the proof that the exploration genuinely does something,
  rather than that the robot fails.

The goal of this world used to sit at (7.0, 6.8), in the middle of the tables: 0.72 m from a
table whose collision radius is 0.45 m, i.e. **inside** its footprint. No controller can get
there without pushing against the table, and indeed the robot stayed leaning on it with its
wheels spinning uselessly. When a run "doesn't work", it is always worth checking first that
the geometry of the world is solvable.

The watchdog that declares `FAILED` watches **progress**, not commands: if within an 8 s
window the robot moves less than 0.25 m while it is chasing the goal (`GOAL`, `RECALL`,
`WANDER` or `AVOID`) it enters `RECOVER`, and if three attempts are not enough it writes
`obstacle_recovery_exhausted` or `stuck_no_progress`. The `SEARCH` rotations do not count:
spinning in place is not "being stuck". Spinning in front of a wall in `AVOID`, on the other
hand, is, and that is why `AVOID` is watched too.

If the lack of progress coincides with a nearby obstacle, Toby does not fail straight away:
he enters `RECOVER`, backs up for 1.5 s and rotates for 1.5 s towards whichever side the
lidar sees as freer, then resumes exploring down a different street. This manoeuvre is tried
at most three times; only if it fails to free the robot is `obstacle_recovery_exhausted`
recorded.

## 6. Toby, the robot

I built him from scratch (`protos/ToyRobot.proto`) instead of using an off-the-shelf model,
for three reasons: you know the wheel radius exactly (it is needed in the slip formula), you
can give the wheels your own material (needed in order to change the friction) and the world
does not have to download anything from the internet.

- **Differential drive**: two driven wheels (radius `r = 0.12 m`, wheel separation
  `L = 0.60 m`) and a sliding rear ball (caster). To move at speed `v` while turning at `ω`:
  `ω_wheel = (v ∓ ω·L/2) / r`.
- **Encoders** (a `PositionSensor` on each wheel): they give the wheel angle θ; its angular
  velocity is `Δθ/Δt`, hence `v_enc = r·(ω_L + ω_R)/2`.
- **Camera** (between the eyes, 256 × 128 px): finds the coloured beacons.
- **Lidar** (front bumper, 181 rays over 180°, up to 20 m): distance to obstacles.
- **GPS, IMU, accelerometer, gyro**: required by the question and all recorded in the log.
  The IMU also serves to remember which direction the goal was in when an obstacle covers it.
- Mass 25 kg, 1.2 m/s cruising speed: a small delivery robot.

## 7. How it recognises the goal (and ignores the decoys)

Every camera image goes from RGB to **HSV**: *hue* (the tint, in degrees), *saturation* (how
"vivid" the colour is) and *value* (how bright it is).

1. Only **saturated** pixels (`s > 0.55`) that are not too dark (`v > 0.25`) are kept. That
   alone removes the sky, the pastel buildings, the floor and the streets.
2. The remaining pixels are split by hue: red 0-14° and 344-360°, orange 18-38°, yellow
   40-70°, green 90-168°, cyan 172-192°, blue 196-250°, purple 255-295°, magenta 300-338°.
   Between red and its neighbours (orange and magenta) there is a margin of a few degrees:
   that is what makes the distinction reliable even with shadows and reflections.
3. For each colour the horizontal centre (`cx`, from -1 to +1) and the area (fraction of the
   image occupied) are computed.
4. Only **red** is the goal: Toby steers so as to bring `cx` to zero and stops when the red
   beacon occupies 25 % of the image (about 1.5 m away). The other colours are recorded in
   the log (`decoys_in_view`) and printed to the console as "ignored".

`analysis/camera_montage.py` redoes exactly this on the saved frames and draws the boxes:
solid for the goal, dashed for the decoys. It is the visual proof for the report.

## 8. The navigation (state machine)

| state | when | what it does |
|---|---|---|
| SEARCH | the goal has never been seen | spins in place (the angle turned is measured by the gyro) |
| WANDER | it has done a full turn without seeing it | drives forward for 8 s avoiding obstacles, then searches again |
| GOAL | the red is in the image | steers towards the beacon + lateral push away from nearby obstacles |
| RECALL | the goal is covered by something | heads back towards the last direction it was seen in (IMU yaw); after 15 s without seeing it again the memory expires and it goes back to SEARCH |
| AVOID | obstacle closer than 0.9 m ahead (0.55 m if it is heading for the goal) | backs up slowly and turns towards the freer side; it only exits half a metre further on |
| RECOVER | 8 s without advancing 25 cm with an obstacle nearby | reverse 1.5 s, rotate 1.5 s, then try again (max 3 times) |
| DONE | the red occupies 25 % of the image | stops, saves everything, pauses |
| FAILED | eight fruitless search cycles, recoveries exhausted or 180 s | stops the motors and writes the reason in `meta.json` |

Two details that look like trivia and are not. **The hysteresis on AVOID**: you enter at
0.9 m and only leave at 1.4 m, otherwise the state flickers at every step and the robot keeps
oscillating in front of the obstacle. **Backing up in AVOID**: turning on the spot does not
free a robot that is already touching something with its nose, so while it turns it slowly
reverses.

The avoidance is a **potential field** over the lidar: every ray shorter than 2.5 m "pushes"
the robot the other way, the harder the closer the obstacle is. Rays in front weigh more, but
the weight never drops below 0.2: with a pure `cos()` the rays at ±90° weigh *zero*, and an
object right at the side (a café chair, a bin) pushed not at all — Toby used to scrape along
it with his flank, wheels turning at 1 m/s and the body stuck at 0.17.

The red beacon must not repel Toby (otherwise he would never get there), so when he sees it
the lidar rays falling in its direction are ignored. The direction is computed with the
proper pinhole model, `atan(cx·tan(fov/2))`: the linear approximation `cx·fov/2` is off by a
couple of degrees at the edges of the image, the edge of the mask wobbled from frame to frame
and the commanded speed flickered between 0.4 and 1.2 m/s. For the same reason the frontal
distance goes through a median filter over 5 samples: a single bad ray must not slam the
brakes on. It is all deliberately simple: the point of the question is the slip, not a
sophisticated planner.

One detail that does bear on the slip: in SEARCH Toby has to complete a full turn. If he
counted the angle from the commands given to the wheels, he would get it wrong on the wet
floor, because the wheels slip and the robot turns less than it "thinks" (in the first trial
it stopped after about 280° instead of 360°). That is why the angle is measured with the
**gyro**, which senses the true rotation of the body.

## 9. The longitudinal slip

If the wheel never slipped, the velocity computed from the encoders (`v_enc = ω·r`) would be
equal to the one at which the robot actually advances (`v_x`). On a slippery floor the wheel
partly spins "for nothing", so `v_enc > v_x`.

`s_long = (v_enc − v_x) / v_x`

- **> 0**: traction slip (the wheel turns faster than the robot advances), typical when it is
  accelerating or pushing.
- **< 0**: braking slip (the robot keeps going by inertia while the wheels slow down).
- **undefined when `v_x ≈ 0`** (division by zero): when the robot is standing still or
  spinning in place it makes no sense. The code puts `NaN` if `|v_x| < 0.1 m/s`. This should
  be stated in the report.
- **statistics only over the driving states** (`GOAL`, `RECALL`, `WANDER`, `AVOID`): during
  the final braking the motors are already at zero while the body is still sliding, so
  `v_enc = 0` with `v_x > 0` and the formula gives a constant `−100 %`. That is physically
  correct but it says nothing about the friction, and it would pollute the mean and the
  percentiles; `plot_slip.py` excludes it and says so at the top of the table.

### The two velocities come from independent sources

The slip compares what the wheel *believes* with what the body *does*, so the numerator and
the denominator must come from different instruments: measuring them with the same sensor
would mean one contaminating the other.

**`v_enc`, the wheel's world** (`slip_nav.py`, lines 273-279). The encoders give the
cumulative shaft angle, not the velocity: `om_l` and `om_r` are a numerical derivative
(finite differences over one step `dt`). Then `ω·r` is the velocity of the contact point *if*
the rolling were pure, and averaging the two sides removes the rotation (in a turn one wheel
turns more than the other, but the average remains the advance of the centre).
`WHEEL_RADIUS` must be identical to the one in the proto, otherwise you introduce a scale
error that looks like slip but is not. The point to make in the report: `v_enc` is not a
measurement of the robot's velocity, it is the velocity the robot *would* have if it did not
slip — and it is exactly that assumption that the slip quantifies.

**`v_x`, the body's world** (lines 281-284):

```python
vel = self_node.getVelocity()      # [vx, vy, vz, wx, wy, wz] in the WORLD frame
R   = self_node.getOrientation()   # 3x3 rotation matrix, written row by row
v_x = vel[0] * R[0] + vel[1] * R[3] + vel[2] * R[6]
```

`getVelocity()` gives the velocity along the fixed axes of the city, but the longitudinal
slip only cares about how much the robot advances *along its own nose*, and the nose is
turning all the time. The first column of `R` — the elements `R[0], R[3], R[6]` — is the unit
vector of the robot's x axis written in world coordinates, so that dot product is the
projection `v_x = v⃗ · x̂_robot`.

Why the projection and not the magnitude `|v⃗|` (or `gps.getSpeed()`, which is in the log
too):

- the magnitude is always ≥ 0, so it **loses the sign**: in reverse (AVOID, RECOVER) `v_enc`
  is negative and the magnitude positive, and the slip would come out meaningless;
- the magnitude includes the **lateral slip**: when Toby steers tightly around the stalls the
  body also slides sideways, and that component has nothing to do with `s_long`;
- while spinning in place the magnitude can be small but not zero (the centre of mass does
  not sit exactly on the axis of rotation), whereas `v_x` is correctly ≈ 0 — and that is what
  makes the `|v_x| > 0.1` filter meaningful.

The projection also uses `vel[2]`, the vertical component: on a flat floor its contribution
is negligible, but it is free and it stays correct on a ramp.

## 10. How the friction is changed

Both worlds contain:

```
ContactProperties { material1 "floor" material2 "wheel"
                    coulombFriction [ μ ]  forceDependentSlip [ FDS ] }
```

- `coulombFriction` is the classic μ: the wheel cannot push harder than `μ·N`. On its own, in
  Webots' physics engine, the wheel either grips perfectly or slips entirely: the slip would
  only show up at the start-ups.
- `forceDependentSlip` makes the wheel slip at a velocity proportional to the force it
  transmits, like a real tyre. That way there is a small but continuous slip even at constant
  speed, which shows up nicely in the plots.
- The caster slides with a bit of friction (`"caster"`, μ = 0.1): it acts as rolling
  resistance, so the wheels always have to push a little and the slip while driving is not
  zero.

| world | μ | FDS | idea |
|---|---|---|---|
| `city_baseline.wbt` (a) | 0.25 | 0.03 | wet square |
| `city_low_slip.wbt` (b) | 1.0 | 0.0008 | dry square |

These are the "knob" of the experiment: you can change them in `FRICTION` inside
`tools/make_worlds.py`.

## 11. Reading the results

Numbers from my reference runs (Webots R2025a), printed by `python analysis/plot_slip.py`.
Yours should come out the same or nearly so:

| | wet (a) | dry (b) |
|---|---|---|
| goal | found and reached in 46.8 s | found and reached in 40.8 s |
| first search rotation (360°) | 10.2 s | 8.1 s |
| first sighting of the red | t = 13.0 s, at the entrance of the square | t = 10.9 s |
| median slip | +7.1 % | −0.06 % (i.e. zero) |
| slip, 95th percentile of \|s\| | 51 % (start-ups and brakings) | 1.0 % |
| distance from the encoders vs true | 28.7 m vs 26.8 m (+7.2 %) | 25.9 m vs 25.8 m (+0.5 %) |
| decoy colours ignored | yellow, green, blue, orange, cyan, magenta, purple | the same |

The run has three phases, visible in the video and in the `state` column of the log:

1. **SEARCH** at the start: the goal is hidden, Toby does a full turn in place and does not
   find it (he does see a few decoy beacons at the end of the street, though, and discards
   them).
2. **WANDER**: he drives towards the square. After about 3 s, level with the entrance
   (t = 13.0 s, at (−0.5, −13.3)), he glimpses a distant red dot, on the right, among the
   café tables: at first it occupies 0.03 % of the image.
3. **GOAL / RECALL**: he heads for the red, which however keeps disappearing behind the
   ice-cream cart, the stalls and the tables. Each time Toby switches to RECALL and goes back
   towards the direction saved with the IMU; he goes around the stalls on the east side,
   passes between the blue beacon and the purple one and stops about 1.5 m from the goal
   (DONE).

The number to quote for the slip is the **median**: the mean of |s| and the 95th percentile
that the script prints are inflated by the instants in which `v_x` is nearly zero (see
below).

- **`figures/slip_map.png`**: the city seen from above with the trajectory coloured by the
  slip (red = the wheel turns faster than the robot, blue = slower, grey = nothing). Same
  scale for the two cases: the wet one is red along almost the whole path, with blue stretches
  where Toby brakes to steer around the stalls; the dry one is grey.
- **`figures/slip_profile.png`**: the slip as a function of the distance travelled (thin
  line: raw value, thick: moving median). The enormous peaks are real but "degenerate": at
  the start the wheels pick up speed immediately while the body still has to accelerate (a
  tiny `v_x` in the denominator), while braking the wheels stop while the robot is still
  sliding (−100 %), and in tight turns `v_x` drops a lot. They should be commented on in the
  report, not hidden.
- **`figures/speeds.png`**: `v_enc` and `v_x` superimposed. On the wet floor the encoder line
  sits above, on the dry one they coincide. It is the most intuitive figure for explaining
  what slip is. The stretches at zero speed are the search rotations in place.
- **`figures/what_toby_sees.png`**: the story in six camera frames: the start (no red, only
  decoys), the search, the first sighting (a small distant red rectangle next to the purple
  beacon), the approach among the decoys, the arrival.
- **`figures/run_compare.mp4`**: the two runs on the same clock, with the trail coloured by
  the slip. You can see the wet one falling behind.

Two consequences to discuss. On the wet floor Toby takes 6 s longer: he starts more slowly,
for the same number of wheel revolutions he advances less, and spinning in place also takes
more time. And a robot that estimated distance from the encoders alone would be off by 7 %
(against 0.4 % on the dry floor, due only to the start-ups and the brakings): that is exactly
why in reality wheel odometry is corrected with an IMU, GPS or lidar. Toby himself does it
when searching for the goal: he counts the degrees turned with the gyro, not with the wheels.

## 12. What to put in the report

1. **Setup**: Toby (r, L, mass, sensors), the city, the table of the two frictions.
2. **How you compute `v_enc` and `v_x`** (section 9) and why the Supervisor as ground truth
   (below).
3. **The two filters on the samples**: `|v_x| > 0.1 m/s` (division by nearly zero) and the
   driving states only (during the final braking the wheels are stopped and the body is
   sliding: a constant `−100 %`, true but useless). Say how many samples are left.
4. **The perception**: HSV + saturation threshold, and the figure `what_toby_sees.png` as
   proof that the decoy beacons are seen but ignored.
5. **The slip figures** and the table printed by `plot_slip.py`.
6. **Discussion**: positive slip under acceleration and when the robot pushes, negative while
   braking, order of magnitude in the two cases, effect on the trajectory and on the odometry
   (a robot trusting the encoders alone on the wet floor would get the distance travelled
   wrong by about 7 %).

### Why the Supervisor as ground truth

A `Supervisor` in Webots is not a sensor: it is a controller with privileged access to the
scene tree, and `getVelocity()` reads the value the physics engine is using in that
integration step. No conversion, no noise, no delay, no drift: it is the true value by
construction, hence "ground truth".

The argument to put in the report is the comparison with the alternatives:

| source for `v_x` | error it introduces |
|---|---|
| **Supervisor** (the choice made here) | none: it is the simulator's own value |
| encoders | **circular**: it is affected by the very slip you want to measure |
| GPS (`gps.getSpeed()`) | in Webots it is exact by default, but with noise configured it becomes an estimate, and differentiating the position amplifies it |
| integrated IMU | drift: the accelerometer has a bias, and integrating it makes the error grow like t² |
| lidar (scan matching between successive scans) | a registration error of its own, dependent on the geometry of the scene |

The logical point is that **you cannot measure slip with an instrument that is affected by
slip**. The text of the exercise lets you choose between the simulator and the lidar, but
estimating `v_x` from the lidar would mean that `s_long = (v_enc − v̂_x) / v̂_x` contains both
the physical slip and the lidar's estimation error; and in the dry case, where the true slip
is 0.06 %, that error would be far larger than the signal. You would have a number that does
not distinguish the two worlds, i.e. precisely what the experiment is supposed to do. The
lidar remains a legitimate extension only as a *second* estimator, to be validated against
the ground truth.

The methodological corollary is worth stating: the ground truth enters **only the
measurement, not the control**. The navigation uses the camera, the lidar, the gyro and the
IMU — in SEARCH the degrees turned are counted by the gyro, not by the wheels and not by the
Supervisor — so the robot remains a realistic system and the Supervisor acts as an external
laboratory instrument, like a motion capture system in a real experiment.

One exception to declare, if you want to be precise: the stall detection that triggers
RECOVER uses the Supervisor's `pos` (`slip_nav.py`, lines 379-385), while the GPS is only
logged. It does not distort the slip measurements — the recovery changes the trajectory, not
the relation between `v_enc` and `v_x` — but if you write "the control uses onboard sensors
only" it is a detail an attentive reviewer may call you on. Either you put it in a footnote,
or you replace `pos` with `g[0], g[1]`: it is three lines, and the GPS is a sensor Toby
already has onboard.

For the presentation, `python tools/slide_figures.py` draws into `figures/slides/` the flow
chart, the state machine, the slip formula and the important pieces of code as images with
the real line numbers.

## 13. GitHub

It makes sense, even though you are the only one responsible for Q4: the group can see and
reproduce the work, and you get a version history (useful when you change parameters and want
to go back).

- **One repo for the whole assignment**, with one folder per question (`q1/`, `q2/`, …,
  `q4-webots/`) and a README at the root saying who does what. If the group already has a
  repo, add this folder there; if not, create it yourself, private, and invite your
  teammates.
- **To be versioned**: everything in this folder. Once you have the final runs you can also
  add `data/` and `figures/` (about 15 MB): that way whoever writes the report can redo the
  plots without installing Webots.
- **Not to be versioned** (already in `.gitignore`): the `.wbproj` files Webots creates for
  each user, and `__pycache__`.
- **How to work**: since you are the only one on Q4, you can work on `main` inside
  `q4-webots/` without conflicts; use a branch + pull request only if you want someone to
  review first. Commit in small steps with clear messages ("tune baseline friction",
  "add decoy beacons").
