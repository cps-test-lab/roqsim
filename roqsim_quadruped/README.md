# roqsim_quadruped

Quadruped family for [roqsim](../): the **Boston Dynamics Spot** robot and an RL **locomotion
controller** that turns a body-frame velocity command into smooth physically-simulated walking.

It is the quadruped analogue of `roqsim_humanoid`'s `g1_locomotion` (and, at the interface, of
`roqsim_mobile`'s `diff_drive`): the world spawns the robot with the generic `spawn_robot` plugin,
and the model's manifest injects the `spot_locomotion` controller, a `lidar` and Spot's five body
camera stations. The controller declares
the same backend-neutral endpoints (`cmd_vel` in, `odom` / `joint_states` out), so the ROS 2 bridge
and nav2 drive Spot with no robot-specific wiring.

## What's here

- `models/spot.xml` — 12-DoF Spot MJCF, adapted from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot)
  `boston_dynamics_spot/spot.xml` (BSD-3-Clause). The base body/joint are renamed
  `base_link`/`base_free` and a `lidar` site added (framework spawn/odom/lidar convention); the
  position-actuator gains are set to the policy's training PD (`kp=60 kv=1.5`). Meshes under
  `models/meshes/` (decimated — see below). Uses **position** actuators (the controller writes joint
  targets; MuJoCo closes the PD loop), unlike the G1's torque `<motor>`s.
- `policy/spot.yaml` — our deploy config (joint order, default pose, action scale, obs layout, PD
  gains, 50 Hz / 0.002 s timing), transcribed from NVIDIA's Isaac `Isaac-Velocity-Flat-Spot-v0` env.
- `policy/fetch_policy.py` — fetches the pretrained `spot_policy.pt` (see licensing below); a thin
  wrapper around the repo's external-resources system (`spot_locomotion_policy` in
  `external/external_assets.yaml`).
- `plugins/spot_locomotion.py` — the controller: builds the 48-dim observation, runs the policy at
  50 Hz, and writes leg position targets to the 12 actuators.

## Body cameras

`spot.manifest.yaml` mounts the five stations of a real Spot -- `frontleft`, `frontright`, `left`,
`right`, `back` -- as `spot_body_camera` devices (`roqsim_sensors`), each publishing what the
`spot_ros2` driver publishes for it: a greyscale `camera/<station>/image` with its `camera_info`,
`depth/<station>/image` in millimetres, and `depth_registered/<station>/image`, stamped in Spot's own
frames (`body` → `head` → `frontleft`/`frontright`, `body` → `left`/`right`/`back`, each station →
`<station>_fisheye`). `body` is published as an identity child of `base_link`, so a consumer that
looks up a camera from `body`, as it would on the real robot, finds it.

The front pair looks across: `frontleft` sits on the left and looks to the right, 20° down. Every
image arrives rotated as the real one does -- rotating it by the Spot SDK's `ROTATION_ANGLE`
(frontleft −78°, frontright −102°, right 180°) levels it. Where each pose and lens comes from is in
the manifest and in `THIRD_PARTY.md`; `tests/test_spot_cameras.py` holds every value as a fixture
naming its source.

The stations render on demand: only for a ROS subscriber or an in-process claim, so a run that
reads no image pays nothing. A world that wants colour bodies sets `color_encoding: bgr8` on a
station (`components: [{spawn_sensor: {}, name: frontleft, components: [{spot_body_camera:
{color_encoding: bgr8}}]}]` under the `spawn_robot`); one that wants none switches a station off with
`enabled: false`.

The state interface stays roqsim's own (`odom`, `fl_hx` …, nav2's `/scan`), which the nav2 example
reads. A world that wants the driver's -- `odometry`, `front_left_hip_x` …, its QoS and its 0.125 s
command lapse -- renames them through `spot_locomotion`'s `topics:`, `qos:`, `joint_names:` and
`cmd_vel_timeout:`, as `ros2_ws/src/roqsim_nav2_example/worlds/spot_ros2_interface.yaml` does.

## Policy weights — fetched, not committed (licensing)

The pretrained policy `spot_policy.pt` comes from NVIDIA's public Isaac asset server. It is
**downloadable anonymously (no login) but NVIDIA-asset-licensed: local/internal R&D use only, and may
not be modified or redistributed.** So — unlike the G1's BSD-3 `motion.pt`, which ships in-repo — it
is **not committed** here. Fetch it once (also wired into `make venv`) with either of:

```
python -m roqsim_quadruped.policy.fetch_policy          # thin wrapper
make external-resources RESOURCE=spot_locomotion_policy    # directly, from the roqsim tree
```

or point the controller at your own copy via the `policy_path` config or the `SPOT_POLICY_PATH` env
var. If the policy is missing at load time, `spot_locomotion` raises an error pointing back here. See
`THIRD_PARTY.md` for full provenance and a fully-redistributable retraining fallback.

## Requirements

Needs `torch` for policy inference (declared in `pyproject.toml`). CPU torch is enough — the policy
is a small 48→12 MLP evaluated at 50 Hz. The world must set `sim.timestep: 0.002` — Isaac trains Spot
at a 500 Hz sim / 50 Hz policy (decimation 10), and the plugin refuses any other step; see
`ros2_ws/src/roqsim_nav2_example/worlds/spot_nav2.yaml`.

## Decimated meshes

The visual link meshes in `models/meshes/` are **decimated** copies of the upstream Menagerie `.obj`
meshes, not the originals. Each was reduced with Blender's Decimate (Collapse) modifier to a cap of
**2000 triangles** (meshes already under the cap — including all collision meshes — are unchanged),
taking the visual set from **~129k → ~34k triangles (~26%) and 43 MB → 12 MB**. The body shell drove
the cost (`body_1` 27k→2k; each hip/leg ~7–8k→2k).

Why: the native MuJoCo viewer syncs the scene at the physics rate, so high-poly meshes drop the
interactive view below real-time and inflate offscreen camera-render cost. The reduction is visually
indistinguishable at realistic distances.

Dynamics are unaffected: MuJoCo builds the **convex hull** of a mesh for collision, so triangle count
does not change contact behaviour — the locomotion policy walks identically. To regenerate at a
different cap, re-run `external/convert/decimate_spot_meshes.py` over the upstream assets (see `THIRD_PARTY.md`):

```
blender --background --python external/convert/decimate_spot_meshes.py -- <menagerie>/boston_dynamics_spot/assets \
    src/roqsim_quadruped/models/meshes 2000
```

## Files not in git

`spot_policy.pt` is **not** in git (fetched locally; git-ignored under `policy/`).
