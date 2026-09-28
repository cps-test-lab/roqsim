"""The one table from roqsim's neutral types to ROS messages, and how an endpoint binds through it.

Every core type goes through its converters both ways; a type with no row maps onto a named message
by field name, or is refused naming what does not fit; a package adds a row through an entry point;
the QoS is the type's default, the hint's, or the world's; and a real bridge publishes with it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
rclpy = pytest.importorskip("rclpy")

from builtin_interfaces.msg import Time  # noqa: E402
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy  # noqa: E402

from roqsim import endpoint  # noqa: E402
from roqsim import types as T  # noqa: E402
from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.context import SimContext  # noqa: E402
from roqsim.endpoint import QOS_PRESETS, build  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim.plugin import Plugin  # noqa: E402
from roqsim_ros_bridge import typemap  # noqa: E402
from roqsim_ros_bridge.registry import make_tf, resolve_type  # noqa: E402
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge, qos_of  # noqa: E402

STAMP = Time(sec=3, nanosec=250)


def _eq(a, b) -> bool:
    """Dataclass values equal field by field, arrays by value."""
    for name in a.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
                return False
        elif x != y:
            return False
    return True


VALUES = [
    T.Twist(0.1, -0.2, 0.3, 0.4, -0.5, 0.6),
    T.Pose(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5, 0.5]), "map"),
    T.Odometry.planar(1.0, -2.0, 0.7, 0.3, 0.05, -0.4, z=0.2),
    T.JointState(["a", "b"], np.array([0.1, 0.2]), np.array([1.0, -1.0]), np.array([3.0, 4.0])),
    T.JointState(["a"], np.array([0.1]), np.array([1.0])),
    T.JointPositions(["a", "b"], np.array([0.3, -0.3])),
    T.Wrench(np.array([1.0, 2.0, -9.8]), np.array([0.1, 0.0, -0.1])),
    T.Imu(
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.0, 0.0, 9.81]),
        True,
        1e-4,
        2e-4,
        3e-4,
    ),
    T.Imu(orientation_valid=False, angular_velocity=np.array([0.1, 0.0, 0.0])),
    T.LaserScan(np.array([1.0, np.inf, 2.5], dtype=np.float32), -1.0, 1.0, 1.0, 0.1, 10.0),
    T.Image(np.arange(24, dtype=np.uint8).reshape(2, 4, 3), "rgb8"),
    T.Image(np.array([[0.5, 1.5]], dtype=np.float32), "32FC1"),
    T.CameraInfo(640, 480, 500.0, 501.0, 320.0, 240.0, [0.1, 0.0, 0.0, 0.0, 0.0]),
    T.PointCloud(np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)),
    T.Transform("odom", "base_link", np.array([1.0, 2.0, 0.1]), np.array([0.5, 0.5, 0.5, 0.5])),
]


@pytest.mark.parametrize("value", VALUES, ids=lambda v: type(v).__name__)
def test_every_core_type_round_trips_through_every_message_it_travels_as(value):
    rostype = typemap.lookup(type(value))
    wires = [w for w in rostype.wires if w.msg != "sensor_msgs.msg.CompressedImage"]
    assert wires
    for wire in wires:
        msg = resolve_type(wire.msg)()
        wire.fill(msg, value, STAMP, {})
        back = wire.decode(msg)
        if wire.msg == "geometry_msgs.msg.Pose":  # the unstamped message has no frame
            back.frame_id = value.frame_id
        assert _eq(back, value), (wire.msg, back, value)


@pytest.mark.parametrize(
    ("cls", "msg"), [(bool, "Bool"), (float, "Float64"), (int, "Int64"), (str, "String")]
)
def test_scalars_travel_as_std_msgs(cls, msg):
    (wire,) = typemap.lookup(cls).wires
    assert wire.msg == f"std_msgs.msg.{msg}"
    m = resolve_type(wire.msg)()
    wire.fill(m, cls(1), STAMP, {})
    assert wire.decode(m) == cls(1)


def test_a_stamped_message_carries_the_frame_the_hints_name_under_the_bridge_namespace():
    wire = typemap.lookup(T.Wrench).wire("geometry_msgs.msg.WrenchStamped")
    msg = resolve_type(wire.msg)()
    wire.fill(msg, T.Wrench(), STAMP, {"frame_id": "fts", "frame_prefix": "ur5e"})
    assert (msg.header.frame_id, msg.header.stamp) == ("ur5e/fts", STAMP)
    pose = typemap.lookup(T.Pose).wires[0]
    msg = resolve_type(pose.msg)()
    pose.fill(msg, T.Pose(frame_id="odom"), STAMP, {"frame_id": "map", "frame_prefix": "r1"})
    assert msg.header.frame_id == "r1/odom"  # the value's own frame wins over the hint's


def test_odometry_tf_equals_the_planar_tuples():
    planar = (1.0, -2.0, 0.7, 0.3, 0.0, -0.4)
    old = make_tf(planar, STAMP, "odom", "base_link")
    new = make_tf(T.Odometry.planar(*planar[:4], planar[4], planar[5]), STAMP, "odom", "base_link")
    assert old == new


# -- resolving an endpoint ---------------------------------------------------------------------------
@dataclass
class Vec:
    x: float
    y: float
    z: float


@dataclass
class Push:
    linear: Vec
    angular: Vec


@dataclass
class PushStamped:
    accel: Push


@dataclass
class Level:
    data: float


@dataclass
class Odd:
    linear: Vec
    spin: float


class Producer(Plugin):
    @endpoint.out(ros2={"type": "geometry_msgs.msg.Accel"})
    def push(self) -> Push:
        return Push(Vec(1.0, 2.0, 3.0), Vec(0.0, 0.0, 0.5))

    @endpoint.out(ros2={"type": "geometry_msgs.msg.AccelStamped"})
    def push_stamped(self) -> PushStamped:
        return PushStamped(Push(Vec(1.0, 0.0, 0.0), Vec(0.0, 0.0, 0.0)))

    @endpoint.out(ros2={"type": "geometry_msgs.msg.Accel"})
    def odd(self) -> Odd:
        return Odd(Vec(0, 0, 0), 1.0)

    @endpoint.out(ros2={"type": "std_msgs.msg.String"})
    def level(self) -> Level:
        return Level(1.0)

    @endpoint.out
    def unmapped(self) -> Level:
        return Level(1.0)

    @endpoint.out(ros2={"field": "data", "qos": "sensor_data"})
    def level_value(self) -> Level:
        return Level(2.5)

    @endpoint.stream(ros2={"type": "ackermann_msgs.msg.AckermannDrive", "qos": {"depth": 1}})
    def drive(self, steering_angle: T.Angle, speed: T.Speed) -> None: ...

    @endpoint.stream(ros2={"type": "geometry_msgs.msg.Accel"})
    def accel(self, push: Push) -> None: ...

    @endpoint.command
    def zero(self) -> None: ...

    @endpoint.out(ros2=None)
    def private(self) -> float:
        return 0.0


@pytest.fixture(scope="module")
def eps():
    return {e.name: e for e in build(Producer({}, label="p"), SimContext(config={}))}


def test_a_dataclass_without_a_row_maps_by_field_name_both_ways(eps):
    out = typemap.resolve(eps["push"])
    msg_cls = resolve_type(out.hints["type"])
    out.prepare(msg_cls)
    msg = msg_cls()
    out.fill(msg, eps["push"].read(), STAMP, {})
    assert (msg.linear.y, msg.angular.z) == (2.0, 0.5)

    inbound = typemap.resolve(eps["accel"])
    inbound.prepare(msg_cls)
    written = inbound.decode(msg)
    assert written == {"push": Push(Vec(1.0, 2.0, 3.0), Vec(0.0, 0.0, 0.5))}


def test_a_header_the_payload_lacks_is_stamped_with_a_frame_only_where_one_is_stated(eps):
    binding = typemap.resolve(eps["push_stamped"])
    msg_cls = resolve_type(binding.hints["type"])
    binding.prepare(msg_cls)
    bare = msg_cls()
    binding.fill(bare, eps["push_stamped"].read(), STAMP, {"frame_prefix": "r1"})
    assert bare.header.stamp == STAMP and bare.accel.linear.x == 1.0
    assert bare.header.frame_id == "", "no frame stated: not the bare namespace"
    framed = msg_cls()
    binding.fill(
        framed, eps["push_stamped"].read(), STAMP, {"frame_prefix": "r1", "frame_id": "tool"}
    )
    assert framed.header.frame_id == "r1/tool"


def test_plain_parameters_map_onto_a_named_message_by_field_name(eps):
    from ackermann_msgs.msg import AckermannDrive

    binding = typemap.resolve(eps["drive"])
    binding.prepare(AckermannDrive)
    assert binding.decode(AckermannDrive(steering_angle=0.25, speed=1.5)) == {
        "steering_angle": 0.25,
        "speed": 1.5,
    }
    assert binding.hints["qos"] == {**QOS_PRESETS["default"], "depth": 1}


def test_a_field_that_does_not_fit_is_refused_at_bind_naming_it(eps):
    odd = typemap.resolve(eps["odd"])
    with pytest.raises(
        ValueError, match=r"Odd.spin has no field of that name in Accel \(linear, angular\)"
    ):
        odd.prepare(resolve_type("geometry_msgs.msg.Accel"))
    level = typemap.resolve(eps["level"])
    with pytest.raises(ValueError, match=r"Level.data is float but String.data is string"):
        level.prepare(resolve_type("std_msgs.msg.String"))


def test_a_type_with_no_mapping_is_off_ros_and_describe_says_so(eps):
    binding = typemap.resolve(eps["unmapped"])
    assert binding.hints is None and not binding.required
    row = typemap.describe(eps["unmapped"])
    assert row["on_ros"] is False and "Level has no ROS mapping" in row["reason"]
    assert typemap.describe(eps["private"]) == {"on_ros": False, "reason": "its ros2 hint is None"}


def test_a_bridge_refuses_an_endpoint_that_asked_for_ros_it_cannot_have(eps):
    from dataclasses import replace

    wanted = replace(eps["unmapped"], backend={"ros2": {"frame_id": "x"}})
    bridge = Ros2Bridge({})
    with pytest.raises(RuntimeError, match="Level has no ROS mapping"):
        bridge._hints_for(wanted)
    assert Ros2Bridge({})._hints_for(eps["unmapped"]) is None  # asked for nothing: left off, logged


def test_a_field_hint_publishes_that_field_as_its_own_type(eps):
    binding = typemap.resolve(eps["level_value"])
    assert binding.hints["type"] == "std_msgs.msg.Float64"
    assert binding.hints["qos"] == QOS_PRESETS["sensor_data"]
    msg = resolve_type("std_msgs.msg.Float64")()
    binding.fill(msg, eps["level_value"].read(), STAMP, binding.hints)
    assert msg.data == 2.5


def test_a_command_without_parameters_is_a_trigger_service(eps):
    assert typemap.resolve(eps["zero"]).hints == {"service": "std_srvs.srv.Trigger", "name": "zero"}


class Camera(Plugin):
    @endpoint.out(ros2={"topic": "camera/image_raw"})
    def image(self) -> T.Image: ...

    @endpoint.out(ros2={"type": "sensor_msgs.msg.CompressedImage", "topic": "{image}/compressed"})
    def image_compressed(self) -> T.Image: ...

    @endpoint.out(ros2={"topic": "{image}/../camera_info"})
    def camera_info(self) -> T.CameraInfo: ...


def test_a_derived_topic_follows_a_worlds_rename_of_its_sibling():
    def topics(config):
        built = build(Camera(config, label="cam"), SimContext(config={}))
        return {e.name: typemap.resolve(e).hints["topic"] for e in built}

    assert topics({}) == {
        "image": "camera/image_raw",
        "image_compressed": "camera/image_raw/compressed",
        "camera_info": "camera/camera_info",
    }
    assert topics({"topics": {"image": "/drv/rgb/image"}}) == {
        "image": "/drv/rgb/image",
        "image_compressed": "/drv/rgb/image/compressed",
        "camera_info": "/drv/rgb/camera_info",
    }


def test_a_package_maps_its_own_type_through_the_entry_point(monkeypatch, eps):
    def fill(msg, value, stamp, hints):
        msg.data = value.data * 10.0

    rostype = typemap.RosType(
        Level, (typemap.Wire("std_msgs.msg.Float32", fill, lambda m: Level(m.data / 10.0)),)
    )
    entry = SimpleNamespace(name="levels", value="pkg.ros:LEVEL", load=lambda: [rostype])
    monkeypatch.setattr(
        typemap.metadata,
        "entry_points",
        lambda group: [entry] if group == typemap.ENTRY_POINT_GROUP else [],
    )
    typemap._load_entry_points.cache_clear()
    try:
        binding = typemap.resolve(eps["unmapped"])
        assert binding.hints["type"] == "std_msgs.msg.Float32"
        msg = resolve_type("std_msgs.msg.Float32")()
        binding.fill(msg, Level(0.25), STAMP, {})
        assert msg.data == pytest.approx(2.5)
        assert typemap.describe(eps["unmapped"])["on_ros"] is True
    finally:
        typemap.TYPES.pop(Level, None)
        typemap._load_entry_points.cache_clear()


# -- QoS ----------------------------------------------------------------------------------------------
def test_qos_of_turns_a_profile_into_rclpys():
    qos = qos_of(QOS_PRESETS["latched"])
    assert (qos.depth, qos.durability, qos.reliability) == (
        1,
        DurabilityPolicy.TRANSIENT_LOCAL,
        ReliabilityPolicy.RELIABLE,
    )
    assert qos_of(QOS_PRESETS["sensor_data"]).reliability == ReliabilityPolicy.BEST_EFFORT


def test_a_worlds_qos_overrides_the_default_and_a_real_bridge_publishes_with_it():
    # An isolated domain per process, so a parallel run of this suite cannot hear this bridge.
    domain = 150 + os.getpid() % 50
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {"model": "turtlebot3_waffle", "namespace": "tb"},
                "name": "tb",
                "components": [
                    {
                        "diff_drive": {
                            "qos": {
                                "odom": "sensor_data",
                                "cmd_vel": {"durability": "transient_local"},
                            }
                        }
                    }
                ],
            },
            {"ros2_bridge": {"domain_id": domain, "clock_rate_hz": 0}},
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    try:
        (bridge,) = [p for p in engine.plugins if isinstance(p, Ros2Bridge)]
        node = bridge._node
        (odom,) = node.get_publishers_info_by_topic("/tb/odom")
        assert odom.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        (cmd,) = node.get_subscriptions_info_by_topic("/tb/cmd_vel")
        assert (
            cmd.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
        )  # depth is not on the graph
        (joints,) = node.get_publishers_info_by_topic("/tb/joint_states")
        assert (
            joints.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        )  # the default, as before
        (odom_ep,) = [e for e in engine.ctx.interface.all() if e.owner == "tb" and e.name == "odom"]
        assert bridge.bound_name(odom_ep)["qos"] == QOS_PRESETS["sensor_data"]
    finally:
        engine.shutdown()
