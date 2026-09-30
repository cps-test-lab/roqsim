"""An inbound message reaches a typed endpoint as named parameters, an untyped one as before.

An endpoint declared with :mod:`roqsim.endpoint` gets the fields of its payload type it names, by
name; one built by hand keeps the positional payload its ``write`` was written for. The ROS side of
a migrated plugin -- topic, type, service -- is the same.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, WrenchStamped
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import roqsim  # noqa: F401, I001
from roqsim import endpoint
from roqsim.config import load_config_from_dict
from roqsim.context import Endpoint, SimContext
from roqsim.endpoint import build
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim_ros_bridge.params import payload_for
from roqsim_ros_bridge.registry import get_decoder
from roqsim_ros_bridge.services import set_bool, trigger
from roqsim_ros_bridge.typemap import resolve

UNTYPED = Endpoint(name="legacy", direction="in")


def _twist():
    msg = Twist()
    msg.linear.x, msg.linear.y, msg.angular.z = 0.3, 0.1, -0.2
    return msg


def _pose():
    msg = PoseStamped()
    msg.header.frame_id = "odom"
    msg.pose.position.x = 1.0
    msg.pose.orientation.w = 1.0
    return msg


def _wrench():
    msg = WrenchStamped()
    msg.wrench.force.z, msg.wrench.torque.x = -8.0, 0.5
    return msg


def _trajectory():
    return JointTrajectory(joint_names=["a"], points=[JointTrajectoryPoint(positions=[0.1])])


@pytest.mark.parametrize(
    ("type_path", "msg", "positional"),
    [
        ("geometry_msgs.msg.Twist", _twist(), (0.3, 0.1, -0.2)),
        ("geometry_msgs.msg.TwistStamped", TwistStamped(twist=_twist()), (0.3, 0.1, -0.2)),
        (
            "ackermann_msgs.msg.AckermannDrive",
            AckermannDrive(steering_angle=0.4, speed=1.5),
            (pytest.approx(0.4), 1.5),
        ),
        ("geometry_msgs.msg.PoseStamped", _pose(), ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), "odom")),
        ("geometry_msgs.msg.WrenchStamped", _wrench(), ((0.0, 0.0, -8.0), (0.5, 0.0, 0.0))),
        ("trajectory_msgs.msg.JointTrajectory", _trajectory(), (["a"], [0.1])),
        ("std_msgs.msg.Float64", Float64(data=2.5), 2.5),
    ],
)
def test_an_untyped_endpoint_gets_the_positional_payload_it_always_got(type_path, msg, positional):
    assert payload_for(UNTYPED, get_decoder(type_path)(msg)) == positional


def _drain(ctx):
    stop = threading.Event()

    def physics():
        while not stop.is_set():
            ctx.drain_commands()
            stop.wait(0.001)

    thread = threading.Thread(target=physics, daemon=True)
    thread.start()
    return stop, thread


class _Response:
    success = False
    message = ""


class _Request:
    def __init__(self, data=False):
        self.data = data


class Switch(Plugin):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    @endpoint.command(ros2={"service": "std_srvs.srv.SetBool"})
    def enable(self, data: bool) -> None:
        self.seen.append(("enable", data))

    @endpoint.command
    def zero(self) -> None:
        self.seen.append(("zero",))


def test_services_hand_a_typed_command_its_named_parameter():
    switch = Switch({}, label="thing")
    ctx = SimContext(config={})
    eps = {e.name: e for e in build(switch, ctx)}
    stop, thread = _drain(ctx)
    try:
        on = eps["enable"].write
        assert set_bool(_Request(True), _Response(), ctx, on, eps["enable"]).success
        assert trigger(_Request(), _Response(), ctx, eps["zero"].write, eps["zero"]).success
    finally:
        stop.set()
        thread.join(timeout=1.0)
    assert switch.seen == [("enable", True), ("zero",)]


def test_diff_drives_ros_interface_is_unchanged_and_a_twist_drives_it():
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {"model": "turtlebot3_waffle", "prefix": "z_"},
                "name": "z",
                "components": [{"diff_drive": {"publish_joint_states": True}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    try:
        eps = {e.name: e for e in engine.ctx.interface.all() if e.owner == "z"}
        default = {
            "reliability": "reliable",
            "durability": "volatile",
            "history": "keep_last",
            "depth": 10,
        }
        ros = {name: resolve(ep) for name, ep in eps.items()}
        assert ros["cmd_vel"].hints == {
            "type": "geometry_msgs.msg.Twist",
            "topic": "cmd_vel",
            "qos": default,
        }
        assert ros["odom"].hints == {
            "type": "nav_msgs.msg.Odometry",
            "topic": "odom",
            "frame_id": "odom",
            "child_frame_id": "base_footprint",
            "emit_tf": True,
            "qos": default,
        }
        assert ros["joint_states"].hints == {
            "type": "sensor_msgs.msg.JointState",
            "topic": "joint_states",
            "qos": default,
        }
        msg = Twist()
        msg.linear.x = 0.2
        eps["cmd_vel"].write(ros["cmd_vel"].decode(msg))
        for _ in range(750):
            engine.step()
        odom = eps["odom"].read()
        assert odom.linear[0] == pytest.approx(0.2, abs=0.03) and odom.position[0] > 0.05
    finally:
        engine.shutdown()


def test_an_ackermann_drive_stamped_reaches_the_car_by_field_name_and_turns_it_at_rest():
    from ackermann_msgs.msg import AckermannDriveStamped

    from roqsim_ros_bridge.registry import resolve_type

    world = {"sim": {}, "components": [{"spawn_robot": {"model": "piracer"}, "name": "car"}]}
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    try:
        ep = next(e for e in engine.ctx.interface.all() if e.name == "ackermann_cmd")
        binding = resolve(ep)
        assert binding.hints["type"] == "ackermann_msgs.msg.AckermannDriveStamped"
        assert binding.hints["topic"] == "drive"
        binding.prepare(resolve_type(binding.hints["type"]))
        msg = AckermannDriveStamped()
        msg.drive.steering_angle, msg.drive.speed = 0.3, 0.0
        ep.write(binding.decode(msg))
        for _ in range(200):
            engine.step()
        drive = next(p for p in engine.plugins if type(p).__name__ == "AckermannDrivePlugin")
        assert drive._steer == pytest.approx(0.3, abs=1e-6)
    finally:
        engine.shutdown()
