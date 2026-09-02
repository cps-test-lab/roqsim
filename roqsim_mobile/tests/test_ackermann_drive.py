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


def test_the_steer_split_is_measured_across_the_steering_axes():
    """``steer_track`` is the kingpin separation, and it is the width the steer split pivots about.

    On most vehicles the steering axes sit inboard of the wheels, so the two widths differ; using the
    wider one overstates the split at every radius.
    """
    kingpin = TRACK * 0.7
    plugin = _plugin(_engine(steer_track=kingpin))
    assert plugin.track == TRACK  # the drive split still uses the driven axle
    left, right = plugin.steer_angles(0.4)
    radius = WHEELBASE / np.tan(0.4)
    assert left == pytest.approx(np.arctan(WHEELBASE / (radius - kingpin / 2)))
    assert right == pytest.approx(np.arctan(WHEELBASE / (radius + kingpin / 2)))
    # Narrower axes, smaller split -- the default (track) would have claimed a wider one.
    wide_left, wide_right = _plugin(_engine()).steer_angles(0.4)
    assert left - right < wide_left - wide_right


def test_steer_track_defaults_to_track():
    """A model that states one width gets it for both splits, as before the key existed."""
    assert _plugin(_engine()).steer_track == TRACK


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
        ({"steer_track": 0}, "'steer_track' must be > 0"),
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


# -- the Ackermann command interface -------------------------------------------------------------
#
# `steer` is the direct form of what `drive` infers, and it exists for the one thing a twist cannot
# say. A twist states a CURVATURE, w/v, which is undefined at rest; an Ackermann command states the
# angle, so the rack can be turned while the car stands still -- what a real car does while parking,
# and what a car-like stack sends when lining up before it moves off.


def _steer_qpos(engine) -> tuple[float, float]:
    """What the steer joints actually reached."""
    m, d = engine.ctx.model, engine.ctx.data
    return tuple(
        float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]])
        for j in ("steer_left", "steer_right")
    )


def _steer_ctrl(engine) -> tuple[float, float]:
    """What the plugin ASKED the two steering servos for.

    The angles to assert geometry against. This test vehicle's servos are deliberately soft -- it
    exists to exercise the plugin, not to be a calibrated car -- so its joints reach maybe a third of
    what they are told, and asserting on `qpos` would measure the fixture's gains rather than the
    plugin's arithmetic. That is why the suite tests `steer_angles` as a pure function; this is the
    same discipline one layer out.
    """
    m, d = engine.ctx.model, engine.ctx.data
    return tuple(
        float(d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)])
        for a in ("steer_left_motor", "steer_right_motor")
    )


def test_a_stationary_car_can_turn_its_wheels_but_still_does_not_move():
    """The whole reason for the second interface, stated as one test.

    `test_a_zero_speed_turn_command_does_nothing` pins that a TWIST cannot do this. Both are true at
    once and neither is a contradiction: the car may point its wheels anywhere while stopped, and it
    still goes nowhere until something drives the rear wheels.
    """
    engine = _engine()
    plugin = _plugin(engine)
    before = _pose(engine)
    for _ in range(1500):
        plugin.steer(0.5, 0.0)
        engine.step()
    asked_left, asked_right = _steer_ctrl(engine)
    want_left, want_right = plugin.steer_angles(0.5)
    assert asked_left == pytest.approx(want_left) and asked_right == pytest.approx(want_right)
    # And the rack physically followed, which is the claim: not merely commanded while stopped.
    left, right = _steer_qpos(engine)
    assert left > 0.1 and right > 0.1, "the rack must actually turn with the car stopped"
    # Same tolerance as `test_a_zero_speed_turn_command_moves_nothing`, and for the same reason: the
    # claim is that it does not drive off, not that it is nailed down. Turning a loaded rack under a
    # stopped car scrubs the tyres and nudges the chassis a millimetre or two -- a real one does that
    # too, which is why dry steering is hard on tyres.
    after = _pose(engine)
    assert abs(after[0] - before[0]) < 0.02
    assert abs(after[1] - before[1]) < 0.02
    assert abs(after[2] - before[2]) < 0.05


def test_a_twist_cannot_do_the_same():
    """The contrast that makes the interface worth having, rather than an alias for `drive`."""
    engine = _engine()
    plugin = _plugin(engine)
    for _ in range(1500):
        plugin.drive(0.0, 0.0, 1.0)  # all the yaw rate in the world, at zero speed
        engine.step()
    left, right = _steer_ctrl(engine)
    assert left == 0.0 and right == 0.0, "a curvature says nothing at rest"


def test_a_commanded_angle_is_split_between_the_wheels_like_any_other():
    """`steer` feeds the same geometry `drive` does -- it skips the curvature, not the linkage."""
    engine = _engine()
    plugin = _plugin(engine)
    delta = 0.4
    for _ in range(1500):
        plugin.steer(delta, 0.5)
        engine.step()
    left, right = _steer_ctrl(engine)
    want_left, want_right = plugin.steer_angles(delta)
    assert left == pytest.approx(want_left)
    assert right == pytest.approx(want_right)
    assert left > right, "left is the inner wheel of a left turn"


def test_the_commanded_angle_is_clipped_to_the_rack():
    engine = _engine()
    plugin = _plugin(engine)
    plugin.steer(10.0, 0.0)
    assert plugin._steer_cmd == pytest.approx(CONFIG["max_steer_angle"])


def test_whichever_command_arrived_last_owns_the_angle():
    """The two forms are not merged: a twist after an angle goes back to deriving one.

    Averaging a stated angle with a curvature-derived one would obey neither, so `drive` drops the
    direct command and `steer` replaces it.
    """
    engine = _engine()
    plugin = _plugin(engine)
    plugin.steer(0.5, 0.0)
    assert plugin._steer_cmd is not None
    plugin.drive(0.5, 0.0, 0.0)
    assert plugin._steer_cmd is None
    for _ in range(1500):
        plugin.drive(0.5, 0.0, 0.0)
        engine.step()
    left, right = _steer_ctrl(engine)
    assert left == pytest.approx(0.0, abs=1e-6) and right == pytest.approx(0.0, abs=1e-6), (
        "a straight twist must re-centre the rack"
    )
