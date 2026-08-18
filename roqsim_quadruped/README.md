# roqsim_quadruped

Quadruped family for [roqsim](../): the **Boston Dynamics Spot** robot and an RL **locomotion
controller** that turns a body-frame velocity command into smooth physically-simulated walking.

It is the quadruped analogue of `roqsim_humanoid`'s `g1_locomotion` (and, at the interface, of
`roqsim_mobile`'s `diff_drive`): the world spawns the robot with the generic `spawn_robot` plugin,
and the model's manifest injects the `spot_locomotion` controller + `lidar`. The controller declares
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
  `roqsim/external/external_assets.yaml`).
- `plugins/spot_locomotion.py` — the controller: builds the 48-dim observation, runs the policy at
  50 Hz, and writes leg position targets to the 12 actuators.

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
at a 500 Hz sim / 50 Hz policy (decimation 10); see `worlds/*spot*.yaml`.

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

## Large files (git-lfs)

The decimated `.obj` meshes are tracked with git-lfs (see the repo `.gitattributes`). Run
`git lfs install` once before cloning/pulling this package's assets. `spot_policy.pt` is **not** in
git at all (fetched locally; git-ignored under `policy/`).
