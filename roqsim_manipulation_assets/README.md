# roqsim_manipulation_assets

Arm and gripper models for [roqsim](../README.md). The **asset** half of what used to be a single
`roqsim_manipulation`; the plugins stayed in [`roqsim_manipulation`](../roqsim_manipulation).

Real robots only. Workpieces — a bored block, a pipe weldment — are one experiment's geometry and
ship with that experiment, so what is here is what any user of the substrate can pick up and mount.

| kind | models |
|---|---|
| arms | `ur10e`, `ur5e`, `panda`, `gen3` (7-DOF + integrated Robotiq 2F-85), `open_manipulator_x` |
| grippers | `robotiq_2f85` (85 mm), `schunk_pg70` (61 mm) — standalone and attachable to any flange |

```sh
roqsim sim roqsim_manipulation_assets:ur10e_demo
roqsim sim roqsim_manipulation_assets:gen3_demo
roqsim sim roqsim_manipulation_assets:open_manipulator_x_demo
python -m pytest roqsim_manipulation_assets/tests
```

Each demo shows an arm holding its home pose. For a worked *contact* cell — peg, FT sensor, bored
block, and a choice of position or admittance control — see a downstream insertion experiment: the
reusable halves (`ur5e`, `force_torque`, `cartesian_admittance`) are here and in
`roqsim_sensors`/`roqsim_manipulation`, while the bored block and the world that assembles them ship with
that experiment, because a fit swept at 0.1 mm is one paper's workpiece.

## Layout: one folder per model

```
src/roqsim_manipulation_assets/models/
  ur10e/
    ur10e.xml              # MJCF; <compiler meshdir="meshes">, so mesh refs are bare filenames
    ur10e.manifest.yaml    # the plugins intrinsic to this model (its arm_controller)
    UR10e_LICENSE          # provenance for the vendored geometry
    ur10e.thumb.png        # make thumbnails, in roqsim/
    meshes/*.obj           # this model's meshes only
  ur5e/ …  panda/ …  gen3/ …  robotiq_2f85/ …  schunk_pg70/ …  open_manipulator_x/ …
```

Same shape `roqsim_assets` uses for props, and `resolve_model` accepts it directly. It replaced a flat
`models/*.xml` over one shared `models/meshes/`, where a model's files were spread across four globs
and Menagerie link names (`base_0.obj`, `link1.stl`) had to be kept apart by a per-model mesh
subdirectory anyway. Two consequences worth knowing:

- `gen3` is an arm **plus** the 2F-85, and reuses that model's meshes rather than copying 3 MB of them:
  it is the one MJCF here with `meshdir=".."`, so both halves are named `<model>/meshes/<file>`.
- The provider's `MESHES_DIR` is the models root, so another package borrowing these meshes through its
  manifest's `assets:` key names them `panda/meshes/link0.stl` (Frankie does exactly this).

## Why it is split from the plugins

Every robot family with an actuated limb needs `arm_controller` — `roqsim_humanoid` for the G1's arms,
`roqsim_mobile_manipulation` for Frankie's and TIAGo Pro's. Almost none of them needs these *models*.
While plugins and geometry shared one package, wanting the shared limb controller meant installing
**115 MB** of arms and grippers you never load.

The dependency runs `roqsim_manipulation_assets` → `roqsim_manipulation`, never the reverse: a model's
manifest names the plugins intrinsic to it, while the plugins know nothing about any particular model.
That asymmetry is what keeps it acyclic.

## Adding a model from outside

Nothing here is privileged. A downstream package — including an experiment's own — registers its
models with an `roqsim.models` entry point pointing at a module that exposes `MODELS_DIR`, and its
models then resolve by bare name exactly as these do. That is how the two workpieces that used to
live here now ship with their experiments without either of them being a special case.

## Grippers are interchangeable

`spawn_arm`'s `end_effector:` welds any gripper here onto any arm's `attachment_site`, and the
gripper's own manifest supplies the gripper half of `arm_controller`'s config — so swapping hands is
one line of world YAML. See the model's MJCF for the measurements behind the PG+70's
constants, and note both grippers are driven through a **tendon**: a non-joint transmission is what
makes `arm_controller` expose a `GripperCommand` action instead of two more arm joints.
