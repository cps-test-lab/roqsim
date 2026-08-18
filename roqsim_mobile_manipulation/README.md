# roqsim_mobile_manipulation

Robot models that are a **mobile base and an arm**: composite platforms which belong to neither
[`roqsim_mobile`](../roqsim_mobile) (wheeled bases) nor [`roqsim_manipulation`](../roqsim_manipulation) (arms).

| model | platform | base | arm(s) |
|---|---|---|---|
| `frankie` | Franka Emika Panda on an Omron LD-60 AGV (QUT "Frankie") | differential drive (`diff_drive`) | 1 x Panda 7-DOF + Franka Hand |
| `tiago_pro` | PAL Robotics TIAGo Pro | holonomic (`omni_drive`) | 2 x 7-DOF + grippers, lifting torso |

```sh
roqsim sim roqsim_mobile_manipulation:frankie_demo
roqsim sim roqsim_mobile_manipulation:tiago_pro_ros2
python -m pytest roqsim_mobile_manipulation/tests
```

Each model ships one folder — `models/<name>/` holds its MJCF, `manifest.yaml`, licence, port log,
thumbnail and its own `meshes/`. The port log beside a model is where every number in it comes from.

## Why this package exists

`roqsim_mobile`'s contract is *"depends on `roqsim` + `roqsim_sensors`; add arms/manipulators as sibling
packages."* Keeping these two robots there broke it: their manifests declare `arm_controller`, so
`roqsim_mobile` had to declare `roqsim_manipulation`, and every consumer of a plain TurtleBot then pulled in
every arm, gripper and their meshes. It also made the sibling family packages a dependency mesh rather
than a tree, and silently reordered the wheel install graph for anyone deploying them (which broke a
container image build with `No matching distribution found for roqsim_manipulation`).

Depending on both is correct *here*, because these robots genuinely are both, and it is not circular —
neither base nor arm package depends on the other or on this one. `roqsim_assets` is *not* declared for
the same reason one layer down: no model, manifest or demo world here names a prop or texture, so a
robot model does not drag the prop library into anyone's wheel graph.

Nothing in this package knows about any downstream user of it. A model, its controllers and their
provenance are all that ships here; what an experiment measures with them belongs to that experiment.

## No new plugins, on purpose

Both platforms are assembled from plugins that already existed: `spawn_robot`, `diff_drive` /
`omni_drive`, `arm_controller`, `lidar`. Each port log carries a "substrate extensions" table, and a
composite robot that needed a *new* plugin would be evidence the composition mechanism is missing
something rather than evidence of a hard robot.

`tests/test_mounted_arm_composition.py` is what that claim is measured by: it bolts a stock `ur10e`
onto a stock `husky_a200` from world YAML alone and checks the arm really joins the base's kinematic
subtree, that the two controllers keep disjoint actuators, and that a gripper swap is one line. It is
here rather than with the arm plugins because it needs a base and an arm at once, which only this
package may depend on.

## The failure mode to know about

**Actuator ownership.** `arm_controller`'s default prefix scan claims every actuator sharing the
entity's prefix — which on a mobile manipulator includes the wheel motors. It then writes arm position
targets into wheel drives that `diff_drive`/`omni_drive` own, and the robot simply will not drive.
Both manifests name their `joints:` explicitly to prevent it, and both test suites assert the
controllers hold disjoint actuator sets. Read the port logs before changing a manifest's `joints:`.
