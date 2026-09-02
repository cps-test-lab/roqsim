# roqsim_aerial

Aerial-vehicle plugins and models for [roqsim](../README.md).

| | |
| --- | --- |
| Models | `crazyflie_2` — Bitcraze Crazyflie 2 nano quadrotor (MIT, from MuJoCo Menagerie)<br>`x500` — Holybro X500 V2 class 2 kg quad-X, four per-rotor actuators |
| Plugins | `quadrotor_controller` — cascaded position + attitude control over collective thrust and three body moments<br>`multirotor_motors` — normalized motor commands to rotor forces and yaw reaction torque<br>`px4_sitl` — MAVLink HIL bridge: roqsim as the physics behind PX4 SITL<br>`gnss` — local ENU to WGS84 fix, with bias, noise and a denial switch<br>`wind_field` — steady flow, 1-cosine gust, Dryden turbulence |
| Worlds | `crazyflie_2_demo.yaml` — takes off from the floor and holds 1 m<br>`x500_px4_demo.yaml` — an X500 waiting on TCP 4560 for PX4 SITL to fly it |

```bash
roqsim sim roqsim_aerial/src/roqsim_aerial/worlds/crazyflie_2_demo.yaml
```

## Two things an aerial world needs that a ground world does not

**Air.** MuJoCo defaults `density` and `viscosity` to 0 — a vacuum. A drone still hovers there, but
nothing damps it, so a lateral step rings forever and the run reads as badly tuned gains rather than
as missing air. Every aerial world should carry:

```yaml
sim:
  integrator: rk4     # a quadrotor's attitude loop is stiff relative to the timestep
  density: 1.225      # kg/m^3, air at sea level
  viscosity: 1.8e-5   # Pa*s
```

`quadrotor_controller` warns rather than silently flying in vacuum.

**A stabiliser, always — but not always ours.** A quadrotor MJCF has no stabiliser: an uncommanded
drone is not a robot standing still, it is a falling brick. `crazyflie_2`'s manifest therefore pulls
`quadrotor_controller` in automatically, and unlike a ground robot's manifest that entry is not a
convenience.

`x500`'s manifest deliberately does not. Its stabiliser is a real flight stack running outside the
simulator, so its manifest supplies only `multirotor_motors` — the actuation model — and a world
that spawns an `x500` without either an external autopilot or a controller of its own gets a falling
brick on purpose. Which airframe to reach for:

| | `crazyflie_2` | `x500` |
| --- | --- | --- |
| Actuation | collective thrust + 3 body moments | 4 per-rotor forces |
| Stabilised by | `quadrotor_controller`, in-process | an external flight stack (PX4) |
| Mass / span | 27 g / 0.1 m | 2.0 kg / 0.5 m |
| Under test | the experiment's own control law | a shipped autopilot, estimator and mixer |
| Room needed | a 6 m room | a 20 m room |

Neither supersedes the other. A control-law study wants the loop it is studying to be the loop it
can edit; a study of an autopilot wants the autopilot, and the whole point is that we did not write
it.

## The flight envelope is the experiment

The Crazyflie carries 0.35 N of collective thrust against a 27 g airframe — a thrust-to-weight ratio
of **1.32**. That margin, not the controller, is what an aerial campaign is usually about. The two
two plugins that vary it are `wind_field`, below, and core roqsim's `payload` (carried mass — see
[docs/plugins.rst](../docs/plugins.rst)).

```yaml
components:
  - spawn_robot: {model: crazyflie_2, prefix: "cf2_", pos: [0.0, 0.0]}
    name: drone
    components:
      - quadrotor_controller: {target: [0.0, 0.0, 1.0]}
      - payload: {mass: 0.005}        # 5 g -> T/W 1.11
  - wind_field:
      steady: [3.0, 0.0, 0.0]
      gust: {magnitude: 6.0, onset: 4.0, duration: 1.5}
      turbulence: {intensity: 0.8, length_scale: 4.0}
```

Measured against a 1 m altitude hold, the boundary sits where physics says it does:

| payload | total | T/W | altitude held |
| --- | --- | --- | --- |
| 0 g | 27 g | 1.32 | 1.00 m |
| 3 g | 30 g | 1.19 | 0.91 m |
| 6 g | 33 g | 1.08 | 0.82 m |
| 9 g | 36 g | 0.99 | never leaves the floor |

The graded sag before the collapse is worth knowing: `quadrotor_controller` has no integral term, so
added weight buys steady-state altitude error rather than a cliff. The cliff arrives at T/W = 1.

## `wind_field`

`sim.wind` states a *constant*, which is the one wind a controller never has to reject: it trims
against it once and the error goes to zero. Rejection is tested by wind that changes, so this plugin
writes `model.opt.wind` each tick — steady flow, a MIL-F-8785C 1-cosine gust, and Dryden turbulence.
MuJoCo's own drag terms do the work; nothing here applies a force of its own. Three consequences:

- **One owner per knob.** Declaring both `sim.wind` and `wind_field` is refused rather than merged:
  the compiled model would say one thing and the first tick another, and the run's provenance would
  record the value that was immediately overwritten. State the mean flow in `steady:`.
- **Turbulence draws from the run's seed** through `ctx.rng_for`, so there is no `seed` key here and
  the whole world reproduces together. Because the episode is part of the key, repetitions of one
  configuration see *different* turbulence — samples of the weather, not copies of it.
- **Wind is inert in a vacuum**, for the same reason the drone is undamped in one: it acts through
  the density and viscosity drag terms. With both at 0 the plugin has no effect at all, and warns.

## Flying a real autopilot: `px4_sitl`

PX4 is the flight stack most ROS 2 drone work is built on, so "can this substrate fly a drone" is
really "can it fly PX4". It can, and the way in is worth understanding, because the obvious route is
the one that does not generalise.

**What PX4's own default simulator provides.** Since v1.14 PX4's default SITL simulator is Gazebo
(gz-sim). It ships the airframes (`x500` and its camera/lidar/gimbal variants, plane, VTOL, rover),
a handful of worlds, rendered camera and lidar streams, an IMU/barometer/magnetometer/GNSS sensor
suite with realistic noise, wind, and lockstep. PX4 drives it through a **native `gz_bridge`
module** speaking gz-transport — a private, gz-shaped interface that nothing which is not gz can
use.

**What every other simulator uses instead.** jMAVSim, Gazebo Classic, AirSim and FlightGear all
attach through PX4's documented, simulator-agnostic **Simulator MAVLink API**: the simulator listens
on **TCP 4560** and PX4 connects out to it. That is the interface `px4_sitl` implements, and
choosing it rather than reimplementing gz-transport is the reason this bridge is ~one file instead
of a subsystem.

| direction | messages | carrying |
| --- | --- | --- |
| roqsim → PX4 | `HIL_SENSOR` | accelerometer, gyro, magnetometer, barometer (body FRD, SI) |
| roqsim → PX4 | `HIL_GPS` | the `gnss` plugin's fix |
| roqsim → PX4 | `HIL_STATE_QUATERNION` | ground truth, for analysis — not for the estimator |
| PX4 → roqsim | `HIL_ACTUATOR_CONTROLS` | up to 16 normalized outputs; for a quad-X, four motors |

**Lockstep is the point, not a detail.** PX4 blocks until the next `HIL_SENSOR`; the simulator
blocks until the matching `HIL_ACTUATOR_CONTROLS`. Neither side can run ahead, so the run is
deterministic and can go faster or slower than realtime without the estimator noticing — which is
what makes a campaign of repetitions comparable rather than a sample of the host's load.

**Two frame conventions meet here.** MuJoCo is ENU/FLU; MAVLink HIL is NED/FRD. Altitude is a
*negative* z on the PX4 side. Every conversion lives in one named helper pair in `px4_sitl.py`,
because a sign error here does not crash — it flies the aircraft confidently in the wrong direction.

**What this bridge does not carry**, honestly, against what gz does: camera and lidar streams into
PX4, optical flow (`HIL_OPTICAL_FLOW`), rangefinders (`DISTANCE_SENSOR`) and RC input
(`RC_CHANNELS`). Those are the messages that would carry them, and their absence bounds what a PX4
experiment on this substrate can currently ask.

**This is verified, not aspirational.** PX4 v1.18.0-beta2 has been flown against this bridge:
`Simulator connected on TCP port 4560`, EKF2 reaching attitude, local and global position on the
synthesised IMU/barometer/magnetometer plus the `gnss` fix, `commander takeoff` arming with no
preflight refusal, and the aircraft climbing and holding a hover with all four rotor commands
settling at 0.60 — exactly the airframe's `MPC_THR_HOVER`, and an independent check that the motor
order and spin directions are right.

**Mixing, control, estimation and failsafes are PX4's job, not ours.** The substrate supplies
physics and sensors; that division is what makes the result mean something about PX4.

```yaml
components:
  - spawn_robot: {model: x500, prefix: "x500_", pos: [0.0, 0.0], namespace: drone}
    name: drone
    components:
      - multirotor_motors: {}
      - gnss: {datum: {lat: 47.397742, lon: 8.545594, alt: 488.0}}
      - px4_sitl: {port: 4560}
  - ros2_bridge: {}
```

ROS 2 then talks to **PX4**, not to roqsim: a uXRCE-DDS agent bridges PX4's uORB topics onto
`/fmu/in/*` and `/fmu/out/*` as `px4_msgs`, and a node commands offboard flight there. The roqsim
`ros2_bridge` above still publishes the simulator's own ground-truth odometry, which is what lets an
experiment measure the estimator's error rather than trusting it.

`pymavlink` carries the wire format and is an extra, not a hard dependency:

```bash
pip install 'roqsim_aerial[px4]'
```

## Licensing

Package code is Apache-2.0. The Crazyflie 2 model is MIT — see
`src/roqsim_aerial/models/crazyflie_2/CRAZYFLIE_2_LICENSE`, and the substrate's `THIRD_PARTY`
convention in the root README.
