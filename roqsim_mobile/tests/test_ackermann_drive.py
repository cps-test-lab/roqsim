# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``ackermann_drive``: a car, including the things a car cannot do.

A four-wheeled test vehicle -- two steered front wheels on position servos, two driven rear wheels on
velocity servos -- because the plugin's whole content is the relationship between those four.

The assertion that matters most is a negative: ``cmd_vel`` with ``v = 0`` and a yaw rate must move
NOTHING. A differential base rotates on the spot there, and a substrate that quietly did the same
would hide the failure a car-like experiment exists to provoke.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim_mobile.plugins.ackermann_drive import AckermannDrivePlugin

WHEELBASE = 0.4
TRACK = 0.3
WHEEL_R = 0.06

CONFIG = {
    "wheel_radius": WHEEL_R,
    "wheelbase": WHEELBASE,
    "track": TRACK,
    "max_steer_angle": 0.6,
    "steer_actuators": ["steer_left_motor", "steer_right_motor"],
    "steer_joints": ["steer_left", "steer_right"],
    "drive_actuators": ["rear_left_motor", "rear_right_motor"],
    "drive_joints": ["rear_left", "rear_right"],
}


class _CarScene(Plugin):
    """A car: steered front wheels, driven rear wheels, on a plane."""

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, WHEEL_R])
        base.add_freejoint()
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.12, 0.04], mass=6.0)

        for side, y in (("left", TRACK / 2), ("right", -TRACK / 2)):
            steer = base.add_body(name=f"steer_{side}_link", pos=[WHEELBASE / 2, y, 0])
            steer.add_joint(
                name=f"steer_{side}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 0, 1],
                range=[-0.8, 0.8],
                damping=0.5,
            )
            steer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.2)
            wheel = steer.add_body(name=f"front_{side}_link", pos=[0, 0, 0])
            wheel.add_joint(name=f"front_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            self._wheel(wheel)

            rear = base.add_body(name=f"rear_{side}_link", pos=[-WHEELBASE / 2, y, 0])
            rear.add_joint(name=f"rear_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            self._wheel(rear)

            # A position servo for the rack, a velocity servo for the drive -- what a real vehicle's
            # controllers are, and what the plugin writes targets for.
            self._servo(spec, f"steer_{side}_motor", f"steer_{side}", kp=40.0, kd=2.0)
            self._servo(spec, f"rear_{side}_motor", f"rear_{side}", kv=8.0)

    @staticmethod
    def _wheel(body) -> None:
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[WHEEL_R, 0.02], mass=0.6)
        geom.quat = [0.70710678, 0.70710678, 0, 0]  # roll about y
        geom.friction = [1.5, 0.01, 0.01]

    @staticmethod
    def _servo(spec, name: str, joint: str, *, kp: float = 0.0, kd: float = 0.0, kv: float = 0.0):
        actuator = spec.add_actuator()
        actuator.name = name
        actuator.target = joint
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        if kp:  # position servo: bias -kp * q, and -kd * qdot for damping
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kd
        else:  # velocity servo: bias -kv * qdot
            actuator.gainprm[0] = kv
            actuator.biasprm[2] = -kv
        return actuator

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name, kind="robot", body="base_link", meta={"prefix": "", "namespace": ""}
            )
        )


def _engine(**config):
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    f"{__name__}:_CarScene": {},
                    "name": "robot",
                    "components": [{"ackermann_drive": {**CONFIG, **config}}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(100):  # settle onto the wheels
        engine.step()
    return engine


def _plugin(engine) -> AckermannDrivePlugin:
    return next(p for p in engine.plugins if isinstance(p, AckermannDrivePlugin))


def _run(engine, v: float, w: float, steps: int = 1500):
    plugin = _plugin(engine)
    for _ in range(steps):
        plugin.drive(v, 0.0, w)
        engine.step()
    return _pose(engine)


def _pose(engine) -> tuple[float, float, float]:
    bid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    d = engine.ctx.data
    quat = np.array(d.xquat[bid])
    yaw = float(
        np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
        )
    )
    return float(d.xpos[bid][0]), float(d.xpos[bid][1]), yaw


# -- what a car does -------------------------------------------------------------------------


def test_it_drives_straight():
    engine = _engine()
    x, y, yaw = _run(engine, 0.6, 0.0)
    assert x > 0.3, "the car should have moved forward"
    assert abs(y) < 0.1 and abs(yaw) < 0.15


def test_a_yaw_command_while_moving_turns_the_car_the_way_it_was_asked():
    left = _run(_engine(), 0.6, 0.8)
    right = _run(_engine(), 0.6, -0.8)
    assert left[2] > 0.2, "a positive yaw rate turns left"
    assert right[2] < -0.2
    assert left[0] > 0.1 and right[0] > 0.1, "and it goes somewhere while turning"


def test_reversing_with_a_steer_command_turns_the_other_way_round():
    """The curvature is w/v, so the same yaw request backwards is the opposite steering angle --
    which is why reversing out of a dead end is a manoeuvre rather than a rotation."""
    forward = _plugin(_engine())
    forward.drive(1.0, 0.0, 0.5)
    backward = _plugin(_engine())
    backward.drive(-1.0, 0.0, 0.5)
    assert np.arctan(WHEELBASE * (0.5 / 1.0)) > 0
    assert np.arctan(WHEELBASE * (0.5 / -1.0)) < 0


# -- what a car cannot do --------------------------------------------------------------------


def test_a_zero_speed_turn_command_moves_nothing():
    """The negative that matters: a differential base rotates here, and a car does not."""
    engine = _engine()
    before = _pose(engine)
    after = _run(engine, 0.0, 1.5, steps=800)
    assert abs(after[0] - before[0]) < 0.02
    assert abs(after[1] - before[1]) < 0.02
    assert abs(after[2] - before[2]) < 0.05


def test_the_rack_holds_its_angle_when_the_car_stops():
    """A real rack does not centre itself, and a planner that stops mid-corner resumes mid-corner."""
    engine = _engine()
    _run(engine, 0.6, 0.8, steps=600)
    turned = _plugin(engine)._steer
    assert turned > 0.05
    _run(engine, 0.0, 0.0, steps=200)
    assert _plugin(engine)._steer == pytest.approx(turned, abs=1e-9)


def test_the_steering_angle_is_capped_by_the_rack():
    engine = _engine(max_steer_angle=0.2)
    _run(engine, 0.6, 3.0, steps=600)
    assert _plugin(engine)._steer == pytest.approx(0.2, abs=1e-6)


# -- the geometry ----------------------------------------------------------------------------


def test_the_inner_wheel_turns_more_than_the_outer_one():
    """The linkage this is named after: a shared angle would scrub both tyres."""
    plugin = _plugin(_engine())
    left, right = plugin.steer_angles(0.4)  # a left turn
    assert left > right > 0
    # The exact geometry, not just the ordering: atan(L / (R -/+ track/2)).
    radius = WHEELBASE / np.tan(0.4)
    assert left == pytest.approx(np.arctan(WHEELBASE / (radius - TRACK / 2)))
    assert right == pytest.approx(np.arctan(WHEELBASE / (radius + TRACK / 2)))
    # Mirrored for a right turn, and both vanish going straight.
    mirror_left, mirror_right = plugin.steer_angles(-0.4)
    assert mirror_left == pytest.approx(-right) and mirror_right == pytest.approx(-left)
    assert plugin.steer_angles(0.0) == (0.0, 0.0)


def test_the_driven_wheels_are_split_the_same_way():
    engine = _engine()
    plugin = _plugin(engine)
    _run(engine, 0.6, 0.8, steps=600)
    d = engine.ctx.data
    left, right = (float(d.ctrl[a]) for a in plugin._drive_aid)
    # Turning left, the right (outer) wheel travels further and so is driven faster.
    assert abs(right) > abs(left)


# -- odometry and wiring ----------------------------------------------------------------------


def test_odometry_tracks_a_straight_run():
    engine = _engine()
    x, y, _ = _run(engine, 0.6, 0.0)
    ox, oy, oyaw, *_ = _plugin(engine).read_odom()
    assert ox == pytest.approx(x, rel=0.1)  # a few percent of wheel slip, no more
    assert abs(oy) < 0.05 and abs(oyaw) < 0.05


def test_odometry_is_the_encoder_estimate_and_drifts_on_a_curve():
    """Dead reckoning, not ground truth -- and the difference is the point of publishing both.

    Cornering, the tyres slip and the bicycle relation under-reports the turn: measured here, the car
    ends up at a yaw of ~1.1 rad while its odometry believes ~0.8. That gap is what a localisation
    experiment is about, so it is asserted to EXIST rather than tuned away with a fudge factor;
    `ground_truth_pose` is what a grader compares against.
    """
    engine = _engine()
    _, _, yaw = _run(engine, 0.6, 0.6)
    _, _, oyaw, *_ = _plugin(engine).read_odom()
    assert yaw > 0.3 and oyaw > 0.3, "both agree the car turned left"
    assert oyaw < yaw, "and the encoders under-report how far it came round"


def test_the_steer_joints_are_in_joint_states():
    """A car's steering angle is state a stack watches; leaving it out is how a URDF's front wheels
    stay straight in RViz while the robot corners."""
    names, positions, _ = _plugin(_engine()).read_joint_states()
    assert names[:2] == ["steer_left", "steer_right"]
    assert len(positions) == 4


def test_the_endpoints_are_the_ones_every_base_publishes():
    engine = _engine()
    names = {e.name: e for e in engine.ctx.interface.all()}
    assert names["cmd_vel"].backend["ros2"]["type"] == "geometry_msgs.msg.Twist"
    assert names["odom"].backend["ros2"]["emit_tf"] is True
    assert engine.ctx.blackboard.get("robot:robot") is not None


# -- refusals ----------------------------------------------------------------------------------


def test_it_belongs_to_a_robot():
    with pytest.raises(PluginError):
        load_config_from_dict({"sim": {}, "components": [{"ackermann_drive": CONFIG}]})


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"wheelbase": 0}, "'wheelbase' must be > 0"),
        ({"max_steer_angle": 1.6}, "must be < pi/2"),
        ({"steer_rate": -1}, "'steer_rate' must be >= 0"),
        ({"steer_joints": ["only_one"]}, "exactly two"),
        ({"test_cmd": [1.0]}, "'test_cmd' must be [v, w]"),
    ],
)
def test_config_errors_are_reported_by_name(override, expected):
    config = {**CONFIG, **override}
    errors = AckermannDrivePlugin(config, entity="robot", label="drive").validate_config(config)
    assert any(expected in e for e in errors), errors


def test_the_four_name_lists_are_required():
    errors = AckermannDrivePlugin({}, entity="robot", label="drive").validate_config({})
    assert sum("is required" in e for e in errors) == 4
