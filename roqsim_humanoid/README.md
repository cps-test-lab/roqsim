# roqsim_humanoid

Humanoid family for [roqsim](../): the **Unitree G1** and **LimX Oli** legged robots (each with an
RL **locomotion controller**), and the **AgiBot Genie G2** wheeled dual-arm mobile manipulator.

The legged robots are the legged analogue of `roqsim_mobile`'s `diff_drive`: the world spawns the
robot with the generic `spawn_robot` plugin, and the model's manifest injects the locomotion
controller + `lidar` (+ depth cameras for the Oli). Each controller declares the same backend-neutral
endpoints (`cmd_vel` in, `odom` / `joint_states` out), so the ROS 2 bridge and nav2 drive either
humanoid with no humanoid-specific wiring.

## AgiBot Genie G2 (`agibot_g2`)

A **wheeled dual-arm mobile manipulator** (contrast the legged G1/Oli): a 4-wheel swerve chassis,
5-DoF torso lift, 3-DoF head, two 7-DoF arms and two omnipicker grippers. It reuses the mobile
`diff_drive` for its base and adds `agibot_g2_controller` for the upper body — so it spawns and is
driven exactly like the other robots (`spawn_robot`, `cmd_vel`/`odom`/`joint_states`, `lidar`).

- `models/agibot_g2.xml` — **built** from AgiBot's `genie_sim` URDF (MPL-2.0; see `THIRD_PARTY.md` /
  `THIRD_PARTY.md`) by `external/convert/build_g2_mjcf.py`. The 4 swerve **steer** joints are
  welded straight and the base is driven as a **planar skid-steer diff-drive** ("wheels welded as
  caster"); wheel + torso/head/arm/gripper actuators, gripper mimic→equality couplings, anti-tip
  casters, `lidar`/`base_imu` sites and a head camera are added; a home keyframe is baked.
- `models/meshes/agibot_g2/` — visual `.obj` (converted from Collada with `external/convert/dae2obj.py`, full-res,
  split per source material to recover colours: whites/greys/blacks + an orange accent) + 17 convex
  `.STL` collision hulls, all MPL-2.0 from the source repo.
- **Controllers:** `diff_drive` (base) + `agibot_g2_controller` (torso/head/arms/grippers) + `lidar`,
  wired in `models/agibot_g2.manifest.yaml`. No `torch` needed (analytic controllers).

> **Known limitation:** welding the swerve steer joints makes G2 a rigid 4-wheel skid-steer, so
> **in-place rotation is scrub-limited** (~0.2–0.3 of commanded yaw); straight-line/arc driving is
> clean. Precise torso-lift pose-holding also needs gravity compensation (holds home + modest
> offsets today). Both are known follow-ups.

## LimX Oli (`oli`)

- `models/oli.xml` — 31-DoF whole-body LimX Oli (HU_D04_01), **built** from the vendor URDF by
  `external/convert/build_oli.py` (Apache-2.0; see `THIRD_PARTY.md`). The parallel
  ankle/waist linkages are approximated as serial joints so the pretrained PR-space policy drives
  the 31 actuators 1:1 with no closed loops. `base_link`/`base_free` + `lidar`/`imu` sites and head +
  chest RealSense-D435 depth cameras, per the framework conventions.
- `policy/oli/policy.onnx` + `policy/oli/walk_param.yaml` — pretrained **ONNX** whole-body walk
  policy + deploy config, vendored verbatim from LimX `humanoid-rl-deploy-python`.
- `plugins/oli_locomotion.py` — the controller: builds the 102-dim observation, keeps a 5-deep
  history (510-dim policy input), runs the ONNX policy at 100 Hz, and applies a 1000 Hz PD torque
  loop to the 31 actuators. **The world must set `sim.timestep: 0.001`.** Needs `onnxruntime` (CPU).

## Unitree G1 (`unitree_g1`)

## What's here

- `models/unitree_g1.xml` — 12-DoF leg-only G1 MJCF, adapted from
  [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)'s
  `resources/robots/g1_description/g1_12dof.xml` (BSD-3-Clause). Head/torso/arms are rigid
  decorative geoms on the base; only the 12 leg joints are actuated (`<motor>` torque actuators).
  The base body/joint are renamed `base_link`/`base_free` and a `lidar` site added, to match the
  framework's spawn/odom/lidar conventions. Meshes under `models/meshes/` (decimated — see below).
- `policy/motion.pt` — pretrained TorchScript walking policy (unitree_rl_gym `deploy/pre_train/g1`).
- `policy/g1.yaml` — the deploy config it was trained with (PD gains, default angles, obs/action
  scales, 50 Hz policy / 500 Hz PD timing). Vendored verbatim.
- `plugins/g1_locomotion.py` — the controller: builds the 47-dim observation, runs the policy at
  50 Hz, and applies a 500 Hz PD torque loop to the 12 leg actuators.

## Requirements

Needs `torch` for policy inference (declared in `pyproject.toml`). CPU torch is enough — the policy
is a small 47→12 MLP evaluated at 50 Hz. The world must set `sim.timestep: 0.002` — the policy's PD
loop is tuned for a 500 Hz step (see `worlds/*g1*.yaml`).

**Optional follow-up:** convert `policy/motion.pt` to ONNX and switch `g1_locomotion` to
`onnxruntime` to drop the `torch` dependency. The observation/action math is unchanged; only the
inference call swaps.

## Decimated meshes

The visual link meshes in `models/meshes/` are **decimated** copies of the upstream unitree_rl_gym
STLs, not the originals. Each mesh was reduced with Blender's Decimate (Collapse) modifier to a cap
of **2000 triangles** (meshes already under the cap are unchanged), taking the set from **503k → 52k
triangles (~10%) and 25 MB → 2.6 MB**. The heavy meshes drove the cost — torso 156k→2k, each rubber
hand 70k→2k, `pelvis_contour` 36k→2k.

Why: the native MuJoCo viewer syncs the scene at the physics rate, so a high-poly G1 made the
interactive view drop below real-time; lighter meshes also cut offscreen camera-render cost. The
reduction is visually indistinguishable at realistic distances (verified by side-by-side renders; the
per-pixel difference is sub-perceptual and confined to curved-surface highlights).

Dynamics are unaffected: MuJoCo builds the **convex hull** of a mesh for collision, so triangle count
does not change contact behaviour — the locomotion policy walks identically (re-verified after
decimation). To regenerate at a different cap, re-run the Decimate pass over the upstream STLs from
the source in `THIRD_PARTY.md`.

## Large files (git-lfs)

The STL meshes and `motion.pt` are tracked with git-lfs (see `.gitattributes`). Run
`git lfs install` once before cloning/pulling this package's assets.
