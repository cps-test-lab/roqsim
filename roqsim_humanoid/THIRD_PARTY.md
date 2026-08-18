# Third-party assets & provenance

This package vendors robot-description and policy files for three platforms: the **Unitree G1** (from
unitree_rl_gym), the **LimX Oli / HU_D04_01** (from LimX Dynamics' `humanoid-*` repos), and the
**AgiBot Genie G2** (from AgiBot's `genie_sim`).

## AgiBot Genie G2 (`agibot_g2`)

A wheeled dual-arm mobile manipulator (not a legged humanoid). Both the robot description **and its
meshes** come from the in-repo ROS model package, which is MPL-2.0 — no non-commercial assets are used
(the separate `GenieSimAssets` scene-prop dataset is CC BY-NC-SA and is **not** vendored).

- **Upstream:** https://github.com/AgibotTech/genie_sim
- **Commit pinned:** `da424345f3a2e851b5f342aeed8e5616fc210f0e`
- **License:** **Mozilla Public License 2.0** — the Genie robot assets carry their own MPL-2.0 notice
  at `source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/genie/LICENSE`, vendored here as
  `src/roqsim_humanoid/models/LICENSE.genie_sim`. © AgiBot (agibot.com). MPL is file-level copyleft:
  the notice is retained on the vendored files; combining with the rest of the tree is permitted.

| Vendored file | Upstream source (in genie_sim) |
| --- | --- |
| `models/agibot_g2.xml` | **Built** by `external/convert/build_g2_mjcf.py` from the expanded URDF of `genie_sim_robot_model/robots/genie/g2/g2_crs_omnipicker.urdf.xacro` (crs arm + omnipicker gripper, coarse collision). Adaptations: root `base_link` + `base_free`; the 4 swerve **steer** joints welded straight (planar diff-drive approximation); wheel velocity + body/head/arm/gripper position actuators added; the 14 URDF gripper `<mimic>` couplings reproduced as MJCF joint equalities; anti-tip caster contacts, a chassis footprint box, `lidar`/`base_imu` sites, a head camera, and a home keyframe added. |
| `models/meshes/agibot_g2/*.obj` | **visual** meshes converted from `.../g2/meshes/**/*.dae` (Collada) by `external/convert/dae2obj.py` (pycollada; MuJoCo cannot load Collada). Full-res (~309k tris, git-LFS); faces carry no normals so MuJoCo computes them from winding (a Blender obj_export was tried first but its axis remap reflected the meshes → flipped winding → culled geometry). The DAEs have **no textures**, only per-material solid colours; since MuJoCo ignores OBJ/MTL materials, `dae2obj.py` splits each mesh into one sub-`.obj` per material and `build_g2_mjcf.py` emits MJCF `<material>`s (7 colours: whites/greys/blacks + an AgiBot-orange accent) — hence the 34 source meshes expand to ~56 sub-meshes. Geometry + orientation preserved. |
| `models/meshes/agibot_g2/*.STL` | 17 **collision** convex hulls, copied verbatim from `.../g2/meshes/**/convex/*.STL` (MuJoCo loads STL directly). |

The base is driven by the generic `diff_drive` (from `roqsim_mobile`); the upper body by
`plugins/agibot_g2_controller.py` (authored here). Neither reuses any AgiBot code — only the
MPL-licensed geometry/meshes are vendored.

## LimX Oli (HU_D04_01)

- **Description upstream:** https://github.com/limxdynamics/humanoid-description
  — commit pinned `a90f734c153aa3ecffc8b674af1e0a323cb55d1a` (`1.0.0.20260706`), Apache-2.0.
- **Deploy/policy upstream:** https://github.com/limxdynamics/humanoid-rl-deploy-python
  — commit pinned `6d8771cd2b5599e90e7598cfad3623dce66d1218` (`1.0.0.20260330`), Apache-2.0.
- **License:** Apache-2.0 — full text in `src/roqsim_humanoid/models/LICENSE.limx_oli` and
  `src/roqsim_humanoid/policy/LICENSE.limx_oli`. © LimX Dynamics Technology Co., Ltd.

| Vendored file | Upstream source |
| --- | --- |
| `models/oli.xml` | **Built** from `HU_D04_description/urdf/HU_D04_01.urdf` (the serial / PR-space tree) by `external/convert/build_oli.py`. Armatures from `urdf/HU_D04_01.srdf` (`rotor_mass × gear_ratio²`); collision primitives transplanted from `xml/HU_D04_01.xml`; base renamed to `base_link`/`base_free`, `lidar`/`imu` sites + head/chest D435 cameras added, home keyframe baked. The parallel ankle/waist linkages are approximated as serial joints. |
| `models/meshes/oli/*.STL` | `HU_D04_description/meshes/HU_D04_01/*.STL` — the 32 meshes the serial model references, **verbatim / full-res** (359k tris, 18 MB; git-LFS). Not decimated: these are CAD parts with thin-walled limb shells (forearm/thigh/shin covers) that ratio-Collapse destroys and planar-Dissolve barely reduces. Collision is primitive geoms, so mesh detail never affects dynamics; `external/convert/decimate_oli_meshes.py` (planar) is available but opt-in. |
| `policy/oli/policy.onnx` | `controllers/HU_D04_01/walk_controller/policy/default/policy.onnx` — pretrained ONNX whole-body walk policy (verbatim). |
| `policy/oli/walk_param.yaml` | `controllers/HU_D04_01/walk_controller/walk_param.yaml` — deploy config: PD gains, default angles, action/obs scales, torque limits, timing (verbatim). |

The observation/action math and PD loop in `plugins/oli_locomotion.py` are re-implemented from
`walk_controller.py` (same commit) so the policy runs on exactly the conventions it was trained on.

## Unitree G1

This package vendors robot-description and policy files from **unitree_rl_gym** by Unitree Robotics.

- **Upstream:** https://github.com/unitreerobotics/unitree_rl_gym
- **Commit pinned:** `276801e46c5d433564f24658bac64f254b7d2d4b` (branch `main`)
- **License:** BSD 3-Clause — Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD.
  ("Unitree Robotics"). Full text in `src/roqsim_humanoid/models/LICENSE.unitree_rl_gym` and
  `src/roqsim_humanoid/policy/LICENSE.unitree_rl_gym`.

## Files and their upstream sources

| Vendored file | Upstream path (in unitree_rl_gym) |
| --- | --- |
| `models/unitree_g1.xml` | `resources/robots/g1_description/g1_12dof.xml` — **adapted**: base body/joint renamed `pelvis`→`base_link`, `floating_base_joint`→`base_free`; added a `lidar` site; dropped the standalone `scene.xml` include (the roqsim world provides ground + lighting). Bodies, joints, inertials, geoms and `<motor>` actuators are otherwise verbatim. |
| `models/meshes/*.STL` | `resources/robots/g1_description/meshes/*.STL` — the 27 meshes referenced by the 12-DoF model, **decimated** (Blender Collapse, 2000-tri cap; 503k→52k tris) for viewer/render performance. Shape is preserved; see "Decimated meshes" in `README.md`. |
| `policy/motion.pt` | `deploy/pre_train/g1/motion.pt` — pretrained TorchScript walking policy (verbatim). |
| `policy/g1.yaml` | `deploy/deploy_mujoco/configs/g1.yaml` — deploy config: PD gains, default angles, obs/action scales, timing (verbatim). |

The observation/action math and PD loop in `plugins/g1_locomotion.py` are re-implemented from
`deploy/deploy_mujoco/deploy_mujoco.py` (same upstream commit) so the policy runs on exactly the
conventions it was trained with.

## Unitree G1 29-DoF with Dex1 grippers (`unitree_g1_dex1`)

The manipulation variant. A **second** Unitree upstream, because `unitree_rl_gym` ships only the
12-DoF leg model and no gripper: the arms, waist and Dex1 gripper come from the official description
repository.

- **Upstream:** https://github.com/unitreerobotics/unitree_ros — `robots/g1_description`
- **Commit pinned:** `f3772ce54c56ef2d34c6aee8100bc768896c7d19` (branch `master`, 2026-07-29)
- **License:** BSD 3-Clause — Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
  ("Unitree Robotics"). Full text in `src/roqsim_humanoid/models/LICENSE.unitree_ros`.

| Vendored file | Upstream path (in unitree_ros) |
| --- | --- |
| `models/unitree_g1_dex1.xml` | **Built** by `external/convert/build_g1_dex1.py` from `robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf`. Built from the URDF and not from the sibling `g1_29dof_rev_1_0.xml` MJCF because the two disagree on the wrist (`wrist_yaw` at x=0.051 vs 0.046 — different `_5010` hardware). Adaptations: `pelvis`→`base_link`, `floating_base_joint`→`base_free`, rubber hands dropped (upstream carries them *and* the gripper on the same mount), finger range clamped to the non-crossed half, four-sphere foot contacts transplanted from `unitree_g1.xml`, `lidar` site added, position actuators + gripper tendon/equality added, upstream `<light>`/floor dropped. |
| `models/meshes/unitree_g1_dex1/*.STL`, `*.stl` | `robots/g1_description/meshes/` — the 37 meshes the model references, **verbatim / full-res** (389k tris, 19 MB; git-LFS). Not decimated: 278k of those triangles back collision geoms which MuJoCo collides by convex hull, so reduction would perturb contact geometry for no performance gain (compile 0.19 s, 30.8× realtime). Upstream already ships reduced finger collision meshes (`dex1_col_*.stl`). |

The arms and grippers are driven by the generic `arm_controller` from `roqsim_manipulation` (one instance
per side, each scoped with `joints:`), the legs by this package's `plugins/g1_locomotion.py` with the
policy from the `unitree_rl_gym` pin above. No Unitree code is vendored — only the BSD-licensed
description and meshes.
