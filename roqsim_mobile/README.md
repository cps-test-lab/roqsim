# roqsim_mobile

Wheeled mobile **bases** for [roqsim](../README.md): the drive plugins, the room they drive in, and the
robot models themselves. ROS-free — ROS coupling lives in `roqsim_ros_bridge`.

```sh
roqsim sim roqsim_mobile:turtlebot4_demo                     # each model has a demo world
roqsim sim roqsim_mobile:turtlebot3_demo --manual-control    # ... and can be hand-driven
roqsim sim roqsim_mobile:warthog_terrain_demo               # the outdoor base on ground that is not flat
python -m pytest roqsim_mobile/tests
```

## Models

One folder per model — `models/<name>/<name>.xml` with its meshes in `meshes/` beside it, plus a
`<model>.manifest.yaml` that brings the robot's own controller and sensors, the vendor's licence, and a
**port log** recording where every number came from. Spawn one with `spawn_robot: {model: <name>}`; the
manifest does the rest, so a world is ~15 lines.

| model | platform | drive | scanner | port log |
|---|---|---|---|---|
| `turtlebot4` | iRobot Create 3 + TurtleBot 4 | differential (front caster) | RPLIDAR A1 + OAK-D (RGB-D) | — |
| `turtlebot3_waffle` | ROBOTIS TurtleBot3 Waffle | differential (passive casters) | LDS-01 | |
| `husky_a200` | Clearpath Husky A200 | skid-steer, 4 driven wheels (`slip_factor` 3.0) | planar, 1440 rays @ 30 Hz | |
| `clearpath_jackal` | Clearpath Jackal | skid-steer, 4 driven wheels (`slip_factor` 1.7) | VLP-16, planar cast @ 10 Hz | |
| `piracer` | Waveshare PiRacer AI Kit | **Ackermann** — two steered front wheels, rear pair driven; cannot turn in place | none (a `camera` site, unpopulated) | |

`turtlebot4` has no port log yet — it predates the convention. Its provenance
(`nav2_minimal_tb4_description`, Apache-2.0) is in `turtlebot4_LICENSE` and its MJCF comments, and
`tests/test_turtlebot4_scene.py` pins the numbers; `tests/test_model_layout.py` carries the gap as a strict
xfail so writing the log is what clears it.

**Read the port log before changing a model.** Several of the numbers are load-bearing calibrations,
not defaults: a wheel `armature` that keeps the velocity servo integrable at a 2 ms step, a
`slip_factor` that compensates skid-steer scrub, a caster `priority="1"` that is the only way to make
a caster frictionless (MuJoCo combines contact params by `max()`, so `condim="1"` alone loses to the
floor and the caster drags — worth 48% of the robot's yaw rate). Each log states the sensitivity.

## Plugins

| plugin | role |
|---|---|
| `spawn_robot` | Attach a robot model at a pose, pulling in the plugins its manifest declares. |
| `diff_drive` | Differential **and** skid-steer kinematics + wheel odometry. Any number of driven wheels per side; `slip_factor` inflates the commanded yaw term and divides back out of odometry, as vendor skid-steer drivers do. Declares `cmd_vel` in, `odom` / `joint_states` out. |
| `omni_drive` | Holonomic (mecanum) kinematics + odometry — a body-frame `vx, vy, wz` twist, with a cap on the resultant planar speed as real holonomic controllers have. Rollers are not modelled: the twist goes through three velocity actuators on the base's free joint, so contacts still win and a wall still stops the robot. Wheel spin is observational. |
| `ackermann_drive` | Car-like (**Ackermann**) kinematics + odometry — the geometry whose *constraint* is the point: a car cannot turn in place, so `cmd_vel` with `v = 0` and a yaw rate moves it nowhere. The two front wheels are steered by *different* angles and the two rear wheels driven at *different* speeds, both from the same curve; a shared value would scrub the tyres. No `slip_factor` and none wanted — a car turns by steering, not by scrubbing sideways. Declares `cmd_vel` **and** `ackermann_cmd` in (`ackermann_msgs/AckermannDriveStamped`, which states the steering angle rather than a curvature, so a stopped car can still turn its wheels), `odom` / `joint_states` out. |
| `floorplan` | Build the environment from walls: a room, corridors, obstacles, with textured floor/walls. Provides its own ground and overrides `sim.world`. |

All three drive plugins integrate odometry from the **achieved** twist, so a robot held against a
wall reports no progress. Note the consequence for experiment design: `omni_drive` has no wheel slip
in the model at all, so encoder odometry and ground truth coincide by construction — it cannot be
used to study odometry drift. `diff_drive`'s skid-steer models do drift (~0.3 m over a 5 s arc), and
`ackermann_drive` drifts on a curve where the tyres slip, deliberately uncorrected: a skid-steer's
scrub is systematic enough for a `slip_factor`, while a tyre's slip angle varies with speed and load,
so a constant would only make the estimate look better than the sensor it stands for.

## Layering

Depends on `roqsim` + `roqsim_sensors` only. **Wheeled bases only** — a robot that is also an arm belongs in
a package that depends on both this and `roqsim_manipulation`, which is
[`roqsim_mobile_manipulation`](../roqsim_mobile_manipulation). Keeping two such robots here is what once
forced this package to declare `roqsim_manipulation`: it inverted this package's own contract, made
anyone who wanted a TurtleBot install every arm and gripper, and reordered the wheel install graph
for deployments (which broke a campaign image build). `roqsim_assets` is not a dependency either —
`floorplan` resolves a texture only from an explicit `<package>:<name>`, so a world that wants a prop
library's texture depends on that library itself.

`omni_drive` lives here rather than with the robot it was written for (the TIAGo Pro, a mobile
manipulator): it is base kinematics, and the composite package depends on this one.
