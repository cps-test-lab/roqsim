# roqsim_manipulation

Manipulator **plugins** for [roqsim](../README.md): fixed-base and rail-mounted robot arms with a
position-hold controller, a Cartesian/force-control layer, and backend-neutral `joint_states` +
`follow_joint_trajectory` endpoints, so a ROS 2 bridge can serve everything MoveIt2 needs. Sibling of
`roqsim_mobile` (ROS-free; ROS coupling lives in `roqsim_ros_bridge`).

**No arm models live here.** They are in [`roqsim_manipulation_assets`](../roqsim_manipulation_assets)
(`ur10e`, `ur5e`, `panda`, `gen3`, `open_manipulator_x`, the `robotiq_2f85` / `schunk_pg70` grippers,
and the demo worlds). The split exists so that a package needing only the shared limb controller — a
humanoid's arms, a mobile manipulator's arm, a gantry — does not install 115 MB of geometry it never
loads. The dependency runs assets → plugins and never the reverse: a model's manifest names the
plugins intrinsic to it, while the plugins know nothing about any particular model.

## Plugins

| plugin | role |
|--------|------|
| `spawn_arm` | Attach an arm MJCF at a mount pose (`pos`/`quat`/`yaw`, optional `pedestal`); apply its home pose; optionally weld an `end_effector:` gripper onto its flange. Use a distinct `prefix` per arm. With `rail:` the arm instead rides a prismatic carriage (gantry / ceiling track / seventh axis) — a 6-DOF arm on a rail is a redundant 7-DOF system, and the rail is joint 0 everywhere downstream. With `mount: {robot: ...}` the arm joins a mobile base's kinematic subtree and rides it. |
| `arm_controller` | Resolve the arm's position actuators, hold a target joint vector, and declare the arm's endpoints: a `joint_states` output and a `follow_joint_trajectory` action (plus an `ArmHandle` on the blackboard for in-process drivers), and — with `stream_commands: true` — a high-rate `<controller_name>/joint_trajectory` topic input for streaming controllers like `moveit_servo`. If the arm has a non-joint (tendon) actuator — a parallel gripper — it also declares a `GripperCommand` action at `<gripper_controller_name>/gripper_cmd` and a `gripper:<arm>` state reader on the blackboard. |
| `cartesian_admittance` | The Cartesian layer on top of `arm_controller`'s joint servo. Two laws: `position` (a Cartesian P controller, blind to contact — the honest baseline) and `admittance` (`M ẍ = (w_d − w_a) − D ẋ − C (x − x₀)`, closing a loop around a measured wrench). Both resolve to joint velocities through a damped least-squares Jacobian inverse and write *targets* through the `ArmHandle`, never `data.ctrl`. Publishes a `CartesianHandle` at `cartesian:<arm>`. |

**Three plugins, and that is deliberate.** The line is reuse: what is here is arm-agnostic
*mechanism* — mount an arm, hold a joint vector, close a Cartesian loop — and none of it knows what
the arm is doing or what counts as doing it well. Anything that answered those questions has moved
to the experiment that was asking: `peg_in_hole` (a bored block with a swept clearance) and
`insertion_task` (one paper's trial protocol, right down to its default `approach_height`) to the
a downstream insertion experiment, `pick_place_metrics` to a downstream pick experiment.

The test for a new plugin here is whether a *second* arm experiment would use it unchanged.

## Run standalone (no ROS)

The demo worlds ship with the models, in `roqsim_manipulation_assets`:

```bash
roqsim sim roqsim_manipulation_assets:ur10e_demo   # arm holding its home pose
```

`ur10e_demo` has a `test_target` that drives the arm so the scene shows motion. For a worked
*contact* cell, see a downstream insertion experiment: its world composes `spawn_arm` + `force_torque` +
`cartesian_admittance` from here and `roqsim_sensors`, and adds its own `peg_in_hole` and
`insertion_task` alongside them. That mix — substrate by name, experiment by path — is the intended
shape, and `docs/plugins.rst` walks through it.

## ROS 2 / MoveIt2

Add the `ros2_bridge` plugin (from `roqsim_ros_bridge`) to a world. From the endpoints `arm_controller`
declares it publishes `<ns>/joint_states` and runs a `FollowJointTrajectory` action server at
`<ns>/<controller_name>/follow_joint_trajectory` — what `moveit_simple_controller_manager` executes
against. Namespace each arm on its spawn (`spawn_arm` config `namespace: ur10e`) so several arms
coexist under the one bridge. TF for the arm links comes from `robot_state_publisher` + the arm
URDF (run alongside), not the bridge. Namespace a second arm the same way for a two-arm cell.

Set `stream_commands: true` to additionally expose `<ns>/<controller_name>/joint_trajectory` — a
high-rate `trajectory_msgs/JointTrajectory` **topic** input, the same interface a ros2_control
`JointTrajectoryController` offers. This is the reusable path for streaming controllers: point a
`moveit_servo` node (or any streaming position controller) at it and each inbound single-point message
sets the held target, so a stream of positions servos the arm. Because it mirrors the real driver's
topic, one servo/controller config drives sim and hardware unchanged. Arm-agnostic — any arm using
this plugin gets it; off by default.

For a gripper-equipped arm (e.g. `gen3`), `arm_controller` additionally serves a `GripperCommand`
action at `<ns>/<gripper_controller_name>/gripper_cmd` — a MoveIt `moveit_simple_controller_manager`
`GripperCommand` controller executes against it to open/close the hand. The commanded position (the
`gripper_joint` angle, 0 open .. 0.8 closed for the 2F-85) is mapped onto the tendon actuator's
ctrlrange; the bridge reports `reached_goal`/`stalled` from the live finger state (a stall = a grasp).

## Test

```bash
python -m pytest roqsim_manipulation/tests
```

The tests drive the plugins against a real arm rather than a synthetic one, so they need the model
package too — it is declared in the `test` extra:

```bash
pip install -e "roqsim_manipulation[test]"
```

Mounting an arm on a wheeled base is tested in `roqsim_mobile_manipulation`, which is the package
allowed to depend on both families.
