# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``tricycle_drive``: one steered wheel, a fixed axle, and what that can and cannot do.

Two test vehicles, because the plugin's two drive modes ask different actuators for different things:
a rear-steered truck with a driven axle (a three-wheel counterbalance forklift) and a front-steered
one whose steered wheel is also the driven one (a tugger).

The assertion that matters most is a negative, as for ``ackermann_drive``: ``cmd_vel`` with ``v = 0``
and a yaw rate must move NOTHING. Counter-rotating the axle wheels would pivot the base, which the
real vehicle -- its wheel short of 90 degrees -- cannot do.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.types import JointState, Odometry, Twist
from roqsim_mobile.plugins.tricycle_drive import TricycleDrivePlugin

WHEELBASE = 0.8
TRACK = 0.5
WHEEL_R = 0.1
STEER_R = 0.08
LOCK = 1.2

REAR_AXLE = {
    "drive": "axle",
    "steer_offset": -WHEELBASE,
    "wheel_radius": WHEEL_R,
    "track": TRACK,
    "max_linear_vel": 1.5,
    "max_steer_angle": LOCK,
    "steer_rate": 3.0,
    "accel_limit": 2.0,
    "steer_actuator": "steer_motor",
    "steer_joint": "steer",
    "drive_actuators": ["left_motor", "right_motor"],
    "drive_joints": ["left", "right"],
    "passive_joints": ["steer_wheel"],
}

FRONT_STEER_WHEEL = {
    **REAR_AXLE,
    "drive": "steer_wheel",
    "steer_offset": WHEELBASE,
    "steer_wheel_radius": STEER_R,
    "drive_actuators": ["steer_wheel_motor"],
    "drive_joints": ["steer_wheel"],
    "passive_joints": ["left", "right"],
}


class _Tricycle(Plugin):
    """A fixed axle through base_link and one steered wheel at ``steer_x`` on the centre line."""

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        steer_x = float(self.config.get("steer_x", -WHEELBASE))
        # Only the driven wheels get a motor: a velocity servo held at zero is a brake.
        driven_steer_wheel = bool(self.config.get("driven_steer_wheel", False))
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0])
        base.add_freejoint()
        base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[steer_x / 2, 0, 0.25],
            size=[WHEELBASE / 2 + 0.1, 0.2, 0.08],
            mass=60.0,
        )
        for side, y in (("left", TRACK / 2), ("right", -TRACK / 2)):
            wheel = base.add_body(name=f"{side}_link", pos=[0, y, WHEEL_R])
            wheel.add_joint(
                name=side, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], armature=0.2
            )
            self._wheel(wheel, WHEEL_R)
            if not driven_steer_wheel:
                self._servo(spec, f"{side}_motor", side, kv=15.0)

        steer = base.add_body(name="steer_link", pos=[steer_x, 0, STEER_R])
        steer.add_joint(
            name="steer",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            # In degrees: an MjSpec built in code compiles angles as degrees unless told otherwise.
            range=[-np.degrees(LOCK + 0.1), np.degrees(LOCK + 0.1)],
            damping=2.0,
            armature=0.2,
        )
        steer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.5)
        roll = steer.add_body(name="steer_wheel_link")
        roll.add_joint(
            name="steer_wheel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], armature=0.2
        )
        self._wheel(roll, STEER_R)
        self._servo(spec, "steer_motor", "steer", kp=1000.0, kd=50.0)
        if driven_steer_wheel:
            self._servo(spec, "steer_wheel_motor", "steer_wheel", kv=15.0)
        spec.add_exclude(bodyname1="base_link", bodyname2="steer_wheel_link")

    @staticmethod
    def _wheel(body, radius: float) -> None:
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0.03], mass=1.0)
        geom.quat = [0.70710678, 0.70710678, 0, 0]  # roll about y
        geom.friction = [1.2, 0.01, 0.001]

    @staticmethod
    def _servo(spec, name: str, joint: str, *, kp: float = 0.0, kd: float = 0.0, kv: float = 0.0):
        actuator = spec.add_actuator()
        actuator.name = name
        actuator.target = joint
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        if kp:
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kd
        else:
            actuator.gainprm[0] = kv
            actuator.biasprm[2] = -kv

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name, kind="robot", body="base_link", meta={"prefix": "", "namespace": ""}
            )
        )


def _engine(base: dict | None = None, steer_x: float = -WHEELBASE, **config):
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    f"{__name__}:_Tricycle": {
                        "steer_x": steer_x,
                        "driven_steer_wheel": (base or REAR_AXLE)["drive"] == "steer_wheel",
                    },
                    "name": "robot",
                    "components": [{"tricycle_drive": {**(base or REAR_AXLE), **config}}],
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


def _front(**config):
    return _engine(FRONT_STEER_WHEEL, steer_x=WHEELBASE, **config)


def _plugin(engine) -> TricycleDrivePlugin:
    return next(p for p in engine.plugins if isinstance(p, TricycleDrivePlugin))


def _run(engine, v: float, w: float, steps: int = 1500):
    plugin = _plugin(engine)
    for _ in range(steps):
        plugin.drive(v, 0.0, w)
        engine.step()
    return _pose(engine)


def _pose(engine) -> tuple[float, float, float]:
    bid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    d = engine.ctx.data
    q = np.array(d.xquat[bid])
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return float(d.xpos[bid][0]), float(d.xpos[bid][1]), yaw


def _steer_qpos(engine) -> float:
    m, d = engine.ctx.model, engine.ctx.data
    return float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "steer")]])


# -- what it does ----------------------------------------------------------------------------


def test_it_drives_straight():
    x, y, yaw = _run(_engine(), 0.6, 0.0)
    assert x > 1.0
    assert abs(y) < 0.05 and abs(yaw) < 0.05


def test_a_rear_steered_truck_turns_left_by_steering_its_wheel_right():
    """The rear swings out: a positive yaw rate going forward is a NEGATIVE rear-wheel angle."""
    engine = _engine()
    x, _, yaw = _run(engine, 0.6, 0.5)
    assert yaw > 0.4, "a positive yaw rate turns left"
    assert x > 0.2
    assert _steer_qpos(engine) < -0.2, "and the rear wheel points right to do it"


def test_the_angle_is_the_one_the_geometry_asks_for():
    plugin = _plugin(_engine())
    assert plugin.steer_angle_for(0.6, 0.5) == pytest.approx(np.arctan(0.5 * -WHEELBASE / 0.6))
    # Reversing with the same yaw rate steers the other way.
    assert plugin.steer_angle_for(-0.6, 0.5) == pytest.approx(-plugin.steer_angle_for(0.6, 0.5))


def test_reversing_turns_the_way_the_twist_says():
    engine = _engine()
    x, _, yaw = _run(engine, -0.6, 0.5)
    assert x < -0.2
    assert yaw > 0.4, "w > 0 is counter-clockwise in either direction of travel"
    assert _steer_qpos(engine) > 0.2


def test_a_front_steered_driven_wheel_turns_the_way_asked():
    engine = _front()
    x, _, yaw = _run(engine, 0.5, 0.5)
    assert yaw > 0.4 and x > 0.2
    assert _steer_qpos(engine) > 0.2, "a front wheel points INTO the turn"


# -- what it cannot do -----------------------------------------------------------------------


@pytest.mark.parametrize("make", [_engine, _front], ids=["rear_axle", "front_steer_wheel"])
def test_a_zero_speed_turn_command_moves_nothing(make):
    engine = make()
    before = _pose(engine)
    after = _run(engine, 0.0, 1.5, steps=800)
    assert abs(after[0] - before[0]) < 0.02
    assert abs(after[1] - before[1]) < 0.02
    assert abs(after[2] - before[2]) < 0.05


def test_the_wheel_holds_its_angle_when_the_truck_stops():
    engine = _engine()
    _run(engine, 0.6, 0.5, steps=600)
    turned = _plugin(engine)._steer
    assert turned < -0.1
    _run(engine, 0.0, 0.0, steps=200)
    assert _plugin(engine)._steer == pytest.approx(turned, abs=1e-9)


def test_the_steering_angle_is_capped_by_the_lock():
    engine = _engine(max_steer_angle=0.4)
    _run(engine, 0.6, 5.0, steps=600)
    assert _plugin(engine)._steer == pytest.approx(-0.4, abs=1e-6)


# -- the split ---------------------------------------------------------------------------------


def test_the_axle_is_split_like_a_differential():
    plugin = _plugin(_engine())
    delta = -0.5  # rear wheel right: a left turn going forward
    left, right = plugin.wheel_speeds(0.5, delta)
    w = 0.5 * np.tan(delta) / -WHEELBASE
    assert left == pytest.approx(0.5 - w * TRACK / 2)
    assert right == pytest.approx(0.5 + w * TRACK / 2)
    assert right > left > 0


def test_near_full_lock_the_inner_wheel_runs_backwards():
    """At 1.4 rad the turning radius |a| / tan(delta) is 0.14 m, inside the 0.25 m half-track, so
    the centre of the turn lies between the wheels and the inner one reverses."""
    plugin = _plugin(_engine())
    left, right = plugin.wheel_speeds(0.3, -1.4)
    assert left < 0 < right
    # At 1.2 rad the radius (0.31 m) is still outside the half-track: both roll forward.
    left, right = plugin.wheel_speeds(0.3, -1.2)
    assert 0 < left < right


def test_no_driven_wheel_exceeds_its_speed_cap():
    plugin = _plugin(_engine(max_wheel_speed=0.5))
    speeds = plugin.wheel_speeds(1.0, -LOCK)
    assert max(abs(s) for s in speeds) == pytest.approx(0.5)
    front = _plugin(_front(max_wheel_speed=0.5))
    assert front.wheel_speeds(0.4, 1.0) == [pytest.approx(0.5)]


def test_the_steered_wheel_is_driven_faster_than_the_base_moves():
    plugin = _plugin(_front())
    (s,) = plugin.wheel_speeds(0.3, 0.8)
    assert s == pytest.approx(0.3 / np.cos(0.8))


# -- odometry and wiring ----------------------------------------------------------------------


@pytest.mark.parametrize("make", [_engine, _front], ids=["rear_axle", "front_steer_wheel"])
def test_odometry_tracks_a_straight_run(make):
    engine = make()
    x, _, _ = _run(engine, 0.6, 0.0)
    ox, oy, oyaw, *_ = _plugin(engine).read_odom()
    assert ox == pytest.approx(x, rel=0.05)
    assert abs(oy) < 0.05 and abs(oyaw) < 0.05


@pytest.mark.parametrize("make", [_engine, _front], ids=["rear_axle", "front_steer_wheel"])
def test_odometry_follows_a_curve(make):
    engine = make()
    _, _, yaw = _run(engine, 0.5, 0.5)
    _, _, oyaw, *_ = _plugin(engine).read_odom()
    assert yaw > 0.4 and oyaw > 0.4
    assert oyaw == pytest.approx(yaw, rel=0.25)


def test_joint_states_carry_steering_then_driven_then_passive():
    state = _plugin(_engine()).joint_states()
    assert isinstance(state, JointState)
    assert state.names == ["steer", "left", "right", "steer_wheel"]
    assert len(state.positions) == len(state.velocities) == 4


def test_the_endpoints_are_the_ones_every_base_publishes():
    engine = _engine(stamped_cmd_vel=True, odom_rate_hz=62.0)
    names = {e.name: e for e in engine.ctx.interface.all() if e.owner == "robot"}
    assert names["cmd_vel"].direction == "in"
    assert names["cmd_vel"].payload_type.cls is Twist
    assert names["cmd_vel"].backend["ros2"] == {"stamped": True}
    assert names["odom"].payload_type.cls is Odometry
    assert names["odom"].backend["ros2"] == {"child_frame_id": "base_link", "emit_tf": True}
    assert names["odom"].rate_hz == names["joint_states"].rate_hz == pytest.approx(62.0)
    assert names["joint_states"].payload_type.cls is JointState
    assert engine.ctx.blackboard.get("robot:robot").kinematics == "ackermann"


def test_a_named_twist_on_cmd_vel_drives_the_truck():
    engine = _engine()
    (cmd_vel,) = [e for e in engine.ctx.interface.all() if e.name == "cmd_vel"]
    for _ in range(1500):
        cmd_vel.write({"vx": 0.6, "wz": 0.0})
        engine.step()
    x, y, _ = _pose(engine)
    assert x > 1.0 and abs(y) < 0.05


def test_joint_states_can_be_left_to_a_joint_state_publisher():
    engine = _engine(publish_joint_states=False)
    assert "joint_states" not in {e.name for e in engine.ctx.interface.all()}


def test_the_watchdog_stops_a_truck_whose_stack_went_quiet():
    engine = _engine(cmd_vel_timeout=0.2)
    plugin = _plugin(engine)
    plugin.drive(0.6, 0.0, 0.0)
    for _ in range(1500):
        engine.step()
    assert plugin._cmd_v == pytest.approx(0.0)


# -- refusals ----------------------------------------------------------------------------------


def test_it_belongs_to_a_robot():
    with pytest.raises(PluginError):
        load_config_from_dict({"sim": {}, "components": [{"tricycle_drive": REAR_AXLE}]})


def test_a_steer_offset_the_model_disagrees_with_is_refused():
    with pytest.raises(RuntimeError, match="steer_offset"):
        _engine(steer_offset=-0.6)


def test_a_track_the_model_disagrees_with_is_refused():
    with pytest.raises(RuntimeError, match="track"):
        _engine(track=0.4)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"drive": "tracks"}, "'drive' must be one of"),
        ({"steer_offset": 0.0}, "must be non-zero"),
        ({"max_steer_angle": 1.6}, "must be < pi/2"),
        ({"steer_rate": -1}, "'steer_rate' must be >= 0"),
        ({"drive_joints": ["only_one"]}, "exactly 2"),
        ({"steer_joint": ["a", "b"]}, "names ONE"),
        ({"test_cmd": [1.0]}, "'test_cmd' must be [v, w]"),
    ],
)
def test_config_errors_are_reported_by_name(override, expected):
    config = {**REAR_AXLE, **override}
    errors = TricycleDrivePlugin(config, entity="robot", label="drive").validate_config(config)
    assert any(expected in e for e in errors), errors


def test_the_geometry_and_the_names_are_required():
    errors = TricycleDrivePlugin({}, entity="robot", label="drive").validate_config({})
    for key in ("steer_offset", "steer_actuator", "steer_joint", "drive_actuators", "drive_joints"):
        assert any(f"'{key}' is required" in e for e in errors), (key, errors)


def test_a_driven_steered_wheel_needs_its_radius():
    config = {k: v for k, v in FRONT_STEER_WHEEL.items() if k != "steer_wheel_radius"}
    errors = TricycleDrivePlugin(config, entity="robot", label="drive").validate_config(config)
    assert any("'steer_wheel_radius' is required" in e for e in errors), errors
