"""Authoring an endpoint: the method is the endpoint, options name config keys, the world renames
and tunes, and any dataclass is a payload."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pytest
from numpy.typing import NDArray

from roqsim import endpoint
from roqsim import types as T
from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.endpoint import QOS_PRESETS, Shape, Unit, qos_profile, topic_of, value_type
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.plugins.dummy import DummyPlugin


@dataclass
class Gains:
    """Controller gains.

    Attributes:
        kp: proportional
        kd: derivative
    """

    kp: float
    kd: float = 0.0


@dataclass
class Plan:
    """A custom payload: unit aliases, a nested dataclass, a shaped array.

    Attributes:
        waypoints: (N, 3) positions to pass through
        speed: cruise speed
        gains: tracking gains
    """

    waypoints: Annotated[NDArray[np.float64], Shape(None, 3), Unit("m")]
    speed: T.Speed
    gains: Gains = field(default_factory=lambda: Gains(1.0))


class Base(Plugin):
    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        self.odom_rate_hz = float(self.config.get("odom_rate_hz", 50.0))
        self.publish_joint_states = bool(self.config.get("publish_joint_states", True))
        self.configured = False
        self.twists: list = []
        self.plans: list = []

    def configure(self, ctx: SimContext) -> None:
        self.configured = True

    @endpoint.stream(T.Twist)
    def cmd_vel(self, vx: T.Speed, vy: T.Speed = 0.0, wz: T.AngularSpeed = 0.0) -> None:
        """Body-frame velocity command.

        Args:
            vx: forward speed
            vy: sideways speed
            wz: yaw rate
        """
        self.twists.append((vx, vy, wz))

    @endpoint.out(rate="odom_rate_hz")
    def odom(self) -> T.Odometry:
        """Wheel odometry."""
        return T.Odometry.planar(1.0, 2.0, 0.5, 0.1)

    @endpoint.out(rate="odom_rate_hz", when="publish_joint_states")
    def joint_states(self) -> T.JointState:
        """The wheels."""
        return T.JointState(["l", "r"], np.zeros(2), np.zeros(2))

    @endpoint.out(rate="plan_rate_hz", when="publish_plan")
    def plan(self) -> Plan:
        """The plan being followed."""
        return Plan(np.zeros((2, 3)), 0.5)

    @endpoint.command
    def follow(self, plan: Plan) -> int:
        """Follow a plan.

        Args:
            plan: the whole plan
        """
        self.plans.append(plan)
        return len(plan.waypoints)


def _engine(config=None, **world_config):
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    base = Base({"plan_rate_hz": 5.0, "publish_plan": False, **(config or {})}, entity="bot")
    return Engine(cfg, plugins=[DummyPlugin({}, name="bot"), base], preview=True), base


def _eps(engine) -> dict:
    return {e.name: e for e in engine.ctx.interface.all() if e.owner == "bot"}


# -- the method is the endpoint ------------------------------------------------------------------
def test_the_method_name_is_the_endpoint_name_and_its_first_doc_line_the_doc():
    rows = {s.name: s.describe(Base) for s in endpoint.declared(Base)}
    assert list(rows) == ["cmd_vel", "odom", "joint_states", "plan", "follow"]
    assert rows["cmd_vel"]["doc"] == "Body-frame velocity command."
    assert [p["doc"] for p in rows["cmd_vel"]["params"]] == [
        "forward speed",
        "sideways speed",
        "yaw rate",
    ]
    assert [p["unit"] for p in rows["cmd_vel"]["params"]] == ["m/s", "m/s", "rad/s"]
    assert rows["cmd_vel"]["payload"] == "Twist"


def test_options_name_attributes_or_config_keys_and_describe_says_which():
    engine, _ = _engine({"odom_rate_hz": 20.0})
    with engine:
        eps = _eps(engine)
        assert eps["odom"].rate_hz == 20.0  # the attribute
        assert "plan" not in eps  # `when` read the config key publish_plan: false
        assert "joint_states" in eps
    engine, _ = _engine({"publish_joint_states": False, "publish_plan": True})
    with engine:
        eps = _eps(engine)
        assert "joint_states" not in eps
        assert eps["plan"].rate_hz == 5.0  # the config key, the plugin has no such attribute
    rows = {s.name: s.describe(Base) for s in endpoint.declared(Base)}
    assert rows["odom"]["rate_hz"] == {"from": "odom_rate_hz"}
    assert rows["joint_states"]["when"] == "publish_joint_states"


def test_an_option_naming_nothing_is_refused():
    class Typo(Plugin):
        @endpoint.out(rate="odom_rate")
        def odom(self) -> float:
            return 0.0

    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    engine = Engine(
        cfg, plugins=[DummyPlugin({}, name="bot"), Typo({}, entity="bot")], preview=True
    )
    with pytest.raises(TypeError, match="names 'odom_rate', which is neither an attribute"):
        engine.setup()


def test_an_attribute_hiding_the_endpoint_method_is_refused():
    class Shadowed(Plugin):
        def __init__(self, config=None, **kw):
            super().__init__(config, **kw)
            self.speed = 0.3  # the same name as the endpoint method below

        @endpoint.out
        def speed(self) -> float:
            return self.speed

    ctx = SimContext(config={})
    plugin = Shadowed({}, entity="belt")
    with pytest.raises(TypeError, match="an instance attribute of that name hides the endpoint"):
        plugin.register_endpoints(ctx)


# -- explicit registration --------------------------------------------------------------------------
def test_configure_alone_registers_nothing_and_register_endpoints_adds_them_once():
    ctx = SimContext(config={})
    base = Base({"publish_plan": False}, entity="bot")
    base.configure(ctx)
    assert ctx.interface.all() == []
    added = base.register_endpoints(ctx)
    assert [e.name for e in added] == ["cmd_vel", "odom", "joint_states", "follow"]
    assert base.register_endpoints(ctx) == []  # once per context
    assert len(ctx.interface.all()) == 4


def test_the_engine_registers_after_configure():
    engine, base = _engine()
    with engine:
        assert base.configured and "cmd_vel" in _eps(engine)


# -- payloads ---------------------------------------------------------------------------------------
def test_a_stream_takes_the_fields_of_its_type_it_names():
    engine, base = _engine()
    with engine:
        ep = _eps(engine)["cmd_vel"]
        assert ep.payload_type.cls is T.Twist
        ep.write({"vx": 0.2, "wz": -0.1})
        engine.step()
        assert base.twists == [(0.2, 0.0, -0.1)]


def test_a_parameter_that_is_not_a_field_of_the_type_is_refused():
    class Wrong(Plugin):
        @endpoint.stream(T.Twist)
        def cmd_vel(self, vx: T.Speed, w: T.AngularSpeed = 0.0) -> None: ...

    with pytest.raises(
        TypeError, match=r"its fields, by name \(vx, vy, vz, wx, wy, wz\): 'w' is not a field"
    ):
        endpoint.declared(Wrong)[0].describe(Wrong)

    class Unitless(Plugin):
        @endpoint.stream(T.Twist)
        def cmd_vel(self, vx: Annotated[float, Unit("km/h")]) -> None: ...

    with pytest.raises(TypeError, match=r"'vx' is in km/h but Twist.vx in m/s"):
        endpoint.declared(Unitless)[0].describe(Unitless)


def test_a_custom_dataclass_is_described_and_checked_like_a_core_type():
    rows = {s.name: s.describe(Base) for s in endpoint.declared(Base)}
    plan = rows["plan"]["result"]
    assert (
        plan["type"] == "Plan"
        and plan["doc"] == "A custom payload: unit aliases, a nested dataclass, a shaped array."
    )
    fields = {f["name"]: f for f in plan["fields"]}
    assert fields["waypoints"] == {
        "name": "waypoints",
        "type": "array",
        "unit": "m",
        "doc": "(N, 3) positions to pass through",
        "dtype": "float64",
        "shape": [None, 3],
        "required": True,
    }
    assert fields["speed"]["unit"] == "m/s"
    assert [g["name"] for g in fields["gains"]["fields"]] == ["kp", "kd"]
    assert fields["gains"]["default"] is not None
    assert rows["follow"]["payload"] == "Plan"  # its one dataclass parameter is the whole value

    engine, base = _engine()
    with engine:
        follow = _eps(engine)["follow"]
        refused = follow.write({"plan": {"waypoints": [[0, 0]], "speed": 1.0}})
        with pytest.raises(endpoint.ParameterError, match=r"must have shape \(\*, 3\)"):
            refused.result(timeout=0)
        future = follow.write(
            {"plan": {"waypoints": [[0, 0, 0], [1, 0, 0]], "speed": 1, "gains": {"kp": 2}}}
        )
        engine.step()
        assert future.result(timeout=0) == 2
        (plan,) = base.plans
        assert isinstance(plan.gains, Gains) and plan.gains.kp == 2.0 and plan.speed == 1.0


def test_every_core_type_is_describable():
    for name in (
        "Twist",
        "Pose",
        "Odometry",
        "JointState",
        "JointPositions",
        "Wrench",
        "Imu",
        "LaserScan",
        "Image",
        "CameraInfo",
        "PointCloud",
    ):
        vt = value_type(getattr(T, name))
        assert vt.kind == "struct" and vt.fields and all(f.type.doc for f in vt.fields), name


def test_a_doc_is_given_once_in_the_docstring():
    class Inline(Plugin):
        @endpoint.stream
        def go(self, vx: Annotated[float, Unit("m/s"), "forward"]) -> None: ...

    with pytest.raises(TypeError, match="document a parameter in the docstring's Args"):
        endpoint.declared(Inline)[0].describe(Inline)

    class Stale(Plugin):
        @endpoint.stream
        def go(self, vx: T.Speed) -> None:
            """Go.

            Args:
                speed: what the parameter used to be called
            """

    with pytest.raises(TypeError, match="documents 'speed', which is not among vx"):
        endpoint.declared(Stale)[0].describe(Stale)


# -- the world renames and tunes --------------------------------------------------------------------
def test_qos_presets_and_profiles():
    assert qos_profile("latched") == {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 1,
    }
    assert qos_profile({"reliability": "best_effort", "depth": 3}) == {
        **QOS_PRESETS["default"],
        "reliability": "best_effort",
        "depth": 3,
    }
    with pytest.raises(
        ValueError, match=r"unknown QoS preset 'sensor_dat' \(did you mean 'sensor_data'\?\)"
    ):
        qos_profile("sensor_dat")
    with pytest.raises(
        ValueError, match="depth must be an integer >= 1.*unknown QoS key 'durabilty'"
    ):
        qos_profile({"depth": 0, "durabilty": "volatile"})


def test_a_worlds_qos_and_topics_land_on_the_endpoint():
    engine, _ = _engine({"qos": {"odom": "sensor_data"}, "topics": {"cmd_vel": "/teleop/cmd_vel"}})
    with engine:
        eps = _eps(engine)
        assert eps["odom"].qos == QOS_PRESETS["sensor_data"]
        assert eps["cmd_vel"].qos is None
        assert eps["cmd_vel"].topic == "/teleop/cmd_vel" and eps["odom"].topic is None


class Camera(Plugin):
    """Derived topics and a per-instance ``lazy``."""

    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        self.lazy = bool(self.config.get("lazy", False))

    @endpoint.out(lazy=True, ros2={"topic": "camera/image_raw"})
    def image(self) -> T.Image:
        """The frame."""

    @endpoint.out(
        lazy="lazy",
        ros2={"type": "sensor_msgs.msg.CompressedImage", "topic": "{image}/compressed"},
    )
    def image_compressed(self) -> T.Image:
        """The frame, compressed."""

    @endpoint.out(lazy=lambda self: not self.lazy, ros2={"topic": "{image}/../camera_info"})
    def camera_info(self) -> T.CameraInfo:
        """The intrinsics."""

    @endpoint.command(ros2={"name": "{image}/../reset"})
    def reset(self) -> None:
        """Start over."""

    @endpoint.out(ros2=None)
    def internal(self) -> float:
        """Not on ROS."""


def _camera(config=None) -> dict:
    ctx = SimContext(config={})
    return {e.name: e for e in Camera(config or {}, entity="cam").register_endpoints(ctx)}


def test_a_topic_derived_from_a_sibling_follows_the_worlds_rename_of_it():
    eps = _camera()
    assert topic_of(eps["image_compressed"], "ros2") == "camera/image_raw/compressed"
    assert topic_of(eps["camera_info"], "ros2") == "camera/camera_info"
    assert topic_of(eps["reset"], "ros2") == "camera/reset"  # a service's name
    eps = _camera({"topics": {"image": "/drv/color/image_raw"}})
    assert topic_of(eps["image"], "ros2") == "/drv/color/image_raw"
    assert topic_of(eps["image_compressed"], "ros2") == "/drv/color/image_raw/compressed"
    assert topic_of(eps["camera_info"], "ros2") == "/drv/color/camera_info"
    eps = _camera({"topics": {"image": "rgb", "image_compressed": "/jpeg"}})
    assert topic_of(eps["camera_info"], "ros2") == "camera_info"
    assert topic_of(eps["image_compressed"], "ros2") == "/jpeg"  # its own rename wins
    assert topic_of(eps["internal"], "ros2") is None


def test_a_topic_naming_no_sibling_is_refused():
    class Typo(Plugin):
        @endpoint.out(ros2={"topic": "{imgae}/compressed"})
        def image_compressed(self) -> T.Image:
            """The frame."""

    with pytest.raises(ValueError, match="names 'imgae', which this plugin does not register"):
        Typo({}, entity="cam").register_endpoints(SimContext(config={}))


def test_lazy_is_a_value_a_key_or_a_callable_and_describe_says_which():
    eps = _camera()
    assert (eps["image"].lazy, eps["image_compressed"].lazy, eps["camera_info"].lazy) == (
        True,
        False,
        True,
    )
    eps = _camera({"lazy": True})
    assert (eps["image_compressed"].lazy, eps["camera_info"].lazy) == (True, False)
    rows = {s.name: s.describe(Camera) for s in endpoint.declared(Camera)}
    assert rows["image"]["lazy"] is True
    assert rows["image_compressed"]["lazy"] == {"from": "lazy"}
    assert rows["camera_info"]["lazy"] == "computed"
    assert "lazy" not in rows["internal"]


def test_a_qos_for_an_endpoint_the_plugin_does_not_have_is_refused():
    engine, _ = _engine({"qos": {"odm": "sensor_data"}})
    with pytest.raises(
        ValueError, match=r"qos names 'odm' \(did you mean 'odom'\?\), which it does not register"
    ):
        engine.setup()


def test_a_malformed_world_qos_is_a_config_error():
    errors = Base({}).config_errors({"qos": {"odom": "fast"}})
    assert any("qos['odom']: unknown QoS preset 'fast'" in e for e in errors)
    assert Base({}).config_errors({"qos": ["odom"]}) == [
        "'qos' must be a mapping of endpoint name -> QoS preset or profile"
    ]
