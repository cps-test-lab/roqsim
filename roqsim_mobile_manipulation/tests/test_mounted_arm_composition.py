"""Composing a mobile manipulator in the world YAML: an arm welded onto a mobile base, with a gripper.

This is the mechanism that makes a base x arm x gripper matrix possible without an MJCF per
combination, so the tests cover the three things that silently go wrong:

* the arm really becomes part of the base's kinematic subtree (it *rides* the base) rather than being
  welded to the world next to it -- which looks identical until the base drives away;
* the two subsystems' actuators stay separated, because ``arm_controller`` claims joints by prefix
  scan and would otherwise write arm position targets into the base's wheel drives;
* the gripper's own manifest reaches ``arm_controller``, since that config is what turns a tendon
  actuator into a served ``GripperCommand`` action.

It uses a stock base and a stock arm (``husky_a200`` + ``ur10e``) rather than either shipped
composite, because what is under test is the *composition mechanism* — the claim in this package's
"no new plugins, on purpose" that ``frankie`` and ``tiago_pro`` needed nothing invented. It lives here
and not with the arm plugins for the reason the package exists: it is the one place that may depend on
a wheeled base and an arm at once.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

# Husky A200 top plate, in base_link coordinates: the base collision box top is at
# 0.12498 + 0.13375 = 0.25873. Forward of centre (x=+0.25) to clear the lidar mast at x=0.
MOUNT_POS = [0.25, 0.0, 0.2587]


def _world(tmp_path, arm_extra=None, robot_first=True, prefix="ur10e_"):
    robot = {
        "spawn_robot": {"model": "husky_a200", "pose": {"position": {"x": 0.0, "y": 0.0}}},
        "name": "husky",
    }
    # The arm stays a top-level entry: it PROVIDES an entity rather than attaching to one, and
    # `mount:` is a build-time attachment naming a body, not ownership. Keeping it a sibling is
    # also what lets this file test declaration order in both directions.
    arm = {
        "spawn_arm": {
            "model": "ur10e",
            "prefix": prefix,
            "mount": {"robot": "husky", "body": "base_link"},
            "pos": MOUNT_POS,
            **(arm_extra or {}),
        },
        "name": "arm",
    }
    plugins = [robot, arm] if robot_first else [arm, robot]
    return load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path)


GRIPPER = {
    "end_effector": {"model": "robotiq_2f85", "pos": [0, 0, 0.011], "replaces": ["ee_plate"]}
}


def test_mounted_arm_is_in_the_base_subtree(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    m = engine.ctx.model
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    arm_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur10e_base")
    assert arm_root > 0, "the arm's root body is missing"
    assert m.body_parentid[arm_root] == base, "the arm must hang off the base, not off the world"
    # The base keeps exactly one free joint; mounting must not add a second one for the arm.
    assert list(m.jnt_type).count(mujoco.mjtJoint.mjJNT_FREE) == 1


def test_actuators_stay_separated(tmp_path):
    """Wheels stay the base's, arm joints stay the arm's -- the failure mode prefixes exist to stop."""
    engine = Engine(_world(tmp_path, arm_extra=GRIPPER))
    engine.setup()
    m = engine.ctx.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
    wheels = [n for n in names if "wheel" in n]
    arm = [n for n in names if n.startswith("ur10e_")]
    assert len(wheels) == 4
    assert len(arm) == 7, "six UR10e joint servos plus the gripper's tendon actuator"
    assert set(wheels).isdisjoint(arm)

    # arm_controller must own the six arm joints and NOT the wheels.
    handle = engine.ctx.blackboard.require("arm:arm")
    assert handle.joint_names == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]


def test_mounted_arm_rides_the_base(tmp_path):
    """Drive the base; the arm has to travel with it. This is the whole point of mounting."""
    config = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_robot": {
                        "model": "husky_a200",
                        "pose": {"position": {"x": 0.0, "y": 0.0}},
                    },
                    "name": "husky",
                    # A scripted forward command, so the test needs no ROS transport.
                    "components": [{"diff_drive": {"test_cmd": [0.5, 0.0]}}],
                },
                {
                    "spawn_arm": {
                        "model": "ur10e",
                        "prefix": "ur10e_",
                        "mount": {"robot": "husky", "body": "base_link"},
                        "pos": MOUNT_POS,
                        **GRIPPER,
                    },
                    "name": "arm",
                },
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(config)
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ur10e_pinch")

    for _ in range(1000):  # settle
        engine.step()
    base0, pinch0 = d.xpos[base].copy(), d.site_xpos[pinch].copy()
    for _ in range(2000):  # 4 s of driving
        engine.step()

    base_moved = float(np.linalg.norm(d.xpos[base][:2] - base0[:2]))
    assert base_moved > 0.3, f"the base did not drive (moved {base_moved:.3f} m)"
    # The gripper travelled with the base: same displacement, since the arm held its pose.
    offset0 = pinch0 - base0
    offset1 = d.site_xpos[pinch] - d.xpos[base]
    assert np.allclose(offset0, offset1, atol=0.02), (
        "the gripper's offset from the base changed while driving: the arm is not riding the base"
    )


def test_mount_requires_a_prefix(tmp_path):
    with pytest.raises(PluginError, match="non-empty 'prefix'"):
        Engine(_world(tmp_path, prefix="")).setup()


def test_mount_before_the_base_is_a_clear_error(tmp_path):
    """Build order is declaration order, so the base must come first -- say so, don't KeyError."""
    engine = Engine(_world(tmp_path, robot_first=False))
    with pytest.raises(RuntimeError, match="Declare the base's spawn_robot BEFORE"):
        engine.setup()


def test_end_effector_manifest_reaches_arm_controller(tmp_path):
    """The gripper model's manifest supplies the gripper half of arm_controller's config."""
    engine = Engine(_world(tmp_path, arm_extra=GRIPPER))
    engine.setup()
    endpoints = {e.name: e for e in engine.ctx.interface.all()}
    grip = endpoints["gripper_cmd"]
    assert grip.backend["ros2"]["action"] == "control_msgs.action.GripperCommand"
    assert grip.backend["ros2"]["name"] == "gripper_controller/gripper_cmd"
    # The reader the bridge watches to report reached/stalled must exist.
    assert engine.ctx.blackboard.get(grip.backend["ros2"]["state_key"]) is not None


def test_end_effector_replaces_the_shipped_tool(tmp_path):
    """The ur10e's conveyor pushing plate is removed, or the gripper is welded straight into it."""
    engine = Engine(_world(tmp_path, arm_extra=GRIPPER))
    engine.setup()
    m = engine.ctx.model
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur10e_ee_plate") == -1
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur10e_robotiq_85_base_link") > 0


def test_replaces_an_absent_body_is_refused(tmp_path):
    """A stale `replaces` entry means the arm model was renamed under us -- fail, don't shrug."""
    extra = {"end_effector": {"model": "robotiq_2f85", "replaces": ["no_such_body"]}}
    with pytest.raises(RuntimeError, match="does not have"):
        Engine(_world(tmp_path, arm_extra=extra)).setup()


def test_schunk_gripper_is_interchangeable(tmp_path):
    """Swapping the gripper is one line of world YAML -- same arm, same controller, other hand."""
    extra = {"end_effector": {"model": "schunk_pg70", "replaces": ["ee_plate"]}}
    engine = Engine(_world(tmp_path, arm_extra=extra))
    engine.setup()
    m = engine.ctx.model
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur10e_palm_link") > 0
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ur10e_finger_actuator")
    assert m.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_TENDON
    endpoints = {e.name: e for e in engine.ctx.interface.all()}
    assert "gripper_cmd" in endpoints


def test_reach_envelope_is_statically_stable(tmp_path):
    """A tabletop-height grasp inside the documented reach does not tip the base.

    Husky A200 (44 kg) + UR10e + 2F-85 (33.8 kg) is top-heavy: the arm is three quarters of the base's
    mass. Measured, it stays level out to ~1.05 m of horizontal reach from base_link and tips beyond
    ~1.5 m, so this pins the working end of that envelope. The port log carries the full sweep and the
    parking constraint it implies.
    """
    engine = Engine(_world(tmp_path, arm_extra=GRIPPER))
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ur10e_pinch")
    handle = engine.ctx.blackboard.require("arm:arm")

    for _ in range(1500):
        engine.step()
    # Elbow-up reach over a table, well inside the stable envelope.
    handle.set_targets(handle.joint_names, [0.0, -1.6, 1.4, -(np.pi / 2 - 1.6 + 1.4), -1.5708, 0.0])
    for _ in range(3000):
        engine.step()

    tilt = np.degrees(np.arccos(np.clip(d.xmat[base].reshape(3, 3)[2, 2], -1.0, 1.0)))
    reach = float(np.linalg.norm((d.site_xpos[pinch] - d.xpos[base])[:2]))
    assert reach < 1.1, f"test pose reaches {reach:.2f} m, outside the envelope it means to check"
    assert tilt < 5.0, f"base tilted {tilt:.1f} deg at {reach:.2f} m reach"
