# Third-party assets & provenance

This package vendors a robot-description from **MuJoCo Menagerie** and consumes (without committing) a
pretrained locomotion policy from **NVIDIA Isaac**.

## Robot model — MuJoCo Menagerie (committed, BSD-3-Clause)

- **Upstream:** https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot
- **Commit pinned:** `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- **License:** BSD 3-Clause — Copyright (c) 2021, Clearpath Robotics Inc. Full text in
  `src/roqsim_quadruped/models/LICENSE.mujoco_menagerie`. (Menagerie derived the model from the
  publicly available `bdaiinstitute/spot_ros2` URDF.)

| Vendored file | Upstream path (in mujoco_menagerie) |
| --- | --- |
| `models/spot.xml` | `boston_dynamics_spot/spot.xml` — **adapted** for roqsim: base body `body`→`base_link` and freejoint `freejoint`→`base_free`; added a `lidar` site; dropped the tracking/spot `<light>`s, `<option>`, `<visual>` and the `keyframe` (the roqsim world provides ground, lighting, timestep); base `pos` z set to the standing rest height at the policy's default pose. **Position actuator gains changed from Menagerie's untuned `kp=500 kv=40` to `kp=60 kv=1.5`** to match the policy's training PD (Isaac stiffness=60, damping=1.5). **Masses + inertias scaled ~0.65× to ~32.5 kg total** (real Spot): Menagerie's 50.34 kg is ~1.5× too heavy for the soft PD, which sagged the default stance. **Foot contact retuned** (friction 0.8→2.0, `solimp` stiffened) for the policy's near-no-slip assumption. Bodies, joints, geoms and joint ordering are otherwise verbatim. |
| `models/meshes/*.obj` | `boston_dynamics_spot/assets/*.obj` — the 23 meshes referenced by the leg-only model, with the 18 heavy **visual** meshes **decimated** (Blender Collapse, 2000-tri cap) for viewer/render performance; the 5 already-tiny **collision** meshes are verbatim. Shape is preserved; see "Decimated meshes" in `README.md`. |

## Locomotion policy — NVIDIA Isaac (NOT committed, fetched locally)

- **Upstream weights:** `spot_policy.pt` + `spot_env.yaml` from NVIDIA's public asset bucket
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Samples/Policies/Spot_Policies/`
  (also under `4.5/`). Trained on Isaac Lab's `Isaac-Velocity-Flat-Spot-v0`.
- **License:** **NVIDIA asset license** — downloadable anonymously (no login) but **local/internal R&D
  use only; may not be modified or redistributed.** Therefore `spot_policy.pt` (and `spot_env.yaml`)
  are **not committed** (`.gitignore`); `fetch_policy.py` downloads them on demand. This differs from
  the G1's `motion.pt`, which is BSD-3 and shipped in-repo.
- `policy/spot.yaml` is **our own** deploy config, *transcribed* from the fetched `spot_env.yaml` +
  the open-source `isaacsim.robot.policy.examples` Spot class (Apache-2.0 / BSD-3). It records the
  joint order, default pose, action scale, obs layout and timing — read, not guessed — and IS
  committed.

The observation/action convention in `plugins/spot_locomotion.py` (48-d obs: base lin/ang vel,
projected gravity, velocity command, joint pos/vel relative to default, last action; action =
`default_pose + a * 0.2` written to the position actuators) is re-implemented from that env config so
the policy runs on exactly the conventions it was trained on.

### Fully-redistributable fallback

If a redistributable checkpoint is ever needed, train `Isaac-Velocity-Flat-Spot-v0` (Isaac Lab,
BSD-3) or MuJoCo Playground's Spot joystick task (Apache-2.0) and export a TorchScript/ONNX policy
with the same 48→12 convention. Only `policy/spot_policy.pt` and its provenance change; the package
is otherwise unchanged.
