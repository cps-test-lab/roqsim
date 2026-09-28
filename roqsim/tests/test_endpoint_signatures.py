"""An endpoint's schema is its method's signature: checked on write, published by describe."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pytest

from roqsim import endpoint
from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.endpoint import ParameterError
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details
from roqsim.plugin import Plugin
from roqsim.plugins.dummy import DummyPlugin
from roqsim.types import AngularSpeed, Point3, Speed, Torque


@dataclass
class Reading:
    """A reading.

    Attributes:
        speed: along the track
    """

    speed: Speed
    label: str = ""


class Drive(Plugin):
    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        self.speeds: dict[str, float] = {}
        self.applied: list = []
        self.joints = ["left", "right"]

    def configure(self, ctx: SimContext) -> None:
        self.joints = list(self.config.get("joints", self.joints))

    @endpoint.command
    def set_speed(self, vx: Speed, w: AngularSpeed = 0.0) -> bool:
        """Set the target twist.

        Args:
            vx: forward speed
            w: yaw rate
        """
        return vx > 0.0 or w != 0.0

    @endpoint.command
    def tare(self) -> None:
        """Zero it."""

    @endpoint.stream
    def target(self, pos: Point3) -> None:
        self.applied.append(pos)

    @endpoint.out
    def reading(self) -> Reading:
        return Reading(1.5, "ok")

    @endpoint.command(name="joints/{item}/speed", each="joints")
    def joint_speed(self, joint: str, value: AngularSpeed) -> str:
        self.speeds[joint] = value
        return joint

    @endpoint.out(
        each="joints",
        rate=lambda self, joint: 10.0 if joint == "left" else 20.0,
        ros2=lambda self, joint: {"topic": f"{joint}/effort"},
    )
    def effort(self, joint: str) -> Torque:
        return {"left": 1.0, "right": 2.0}[joint]


def _engine(**config):
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    drive = Drive(dict(config), entity="box", label="drive")
    return Engine(cfg, plugins=[DummyPlugin({}, name="box"), drive], preview=True), drive


def _write(engine, name, payload):
    future = engine.ctx.interface.find("box", name).write(payload)
    engine.step()
    return future.result(timeout=0)


def test_named_parameters_reach_the_method_and_its_result_the_future():
    engine, _ = _engine()
    with engine:
        assert _write(engine, "set_speed", {"vx": 0.3}) is True
        assert _write(engine, "set_speed", {"vx": 0, "w": 0}) is False  # ints pass for floats
        assert _write(engine, "tare", None) is None  # no parameters: None or {}
        assert _write(engine, "tare", {}) is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "missing parameter 'vx' (float, m/s)"),
        ({"vx": 0.1, "vxx": 0.2}, "unknown parameter 'vxx' (did you mean 'vx'?)"),
        ({"vx": "fast"}, "parameter 'vx' must be float, got str ('fast')"),
        ({"vx": True}, "parameter 'vx' must be float, got bool"),
        ((0.1, 0.2), "takes named parameters (a mapping), got tuple"),
    ],
)
def test_a_misfit_is_refused_into_the_future_before_anything_is_queued(payload, message):
    engine, _ = _engine()
    with engine:
        future = engine.ctx.interface.find("box", "set_speed").write(payload)
        assert future.done(), "refused on the caller's side, not queued"
        with pytest.raises(ParameterError, match=re.escape(message)):
            future.result(timeout=0)
        assert "It takes vx: float, m/s, w: float, rad/s = 0.0" in str(future._error)


def test_a_refused_command_is_logged_for_a_caller_that_never_reads_the_future(caplog):
    # A topic feeding a command fires and forgets; the refusal must still be visible.
    engine, _ = _engine()
    with engine, caplog.at_level(logging.WARNING):
        engine.ctx.interface.find("box", "set_speed").write({"vxx": 0.2})
    assert "unknown parameter 'vxx' (did you mean 'vx'?)" in caplog.text


def test_every_misfit_is_named_at_once():
    engine, _ = _engine()
    with engine:
        future = engine.ctx.interface.find("box", "set_speed").write({"w": "x", "speed": 1})
        with pytest.raises(ParameterError) as err:
            future.result(timeout=0)
        text = str(err.value)
        assert "unknown parameter 'speed'" in text
        assert "missing parameter 'vx'" in text
        assert "parameter 'w' must be float" in text


def test_a_stream_checks_on_the_callers_side_and_coerces_an_array():
    engine, drive = _engine()
    with engine:
        write = engine.ctx.interface.find("box", "target").write
        with pytest.raises(ParameterError, match=r"must have shape \(3\), got \(2,\)"):
            write({"pos": [1.0, 2.0]})
        write({"pos": [1, 2, 3]})
        engine.step()
        (pos,) = drive.applied
        assert isinstance(pos, np.ndarray) and pos.dtype == np.float64
        assert pos.tolist() == [1.0, 2.0, 3.0]


def test_the_schema_is_data_on_the_endpoint():
    engine, _ = _engine()
    with engine:
        ep = engine.ctx.interface.find("box", "set_speed")
        assert [(p.name, p.type.kind, p.type.unit, p.required) for p in ep.params] == [
            ("vx", "float", "m/s", True),
            ("w", "float", "rad/s", False),
        ]
        assert ep.result.kind == "bool"
        reading = engine.ctx.interface.find("box", "reading")
        assert reading.params is None and reading.result.kind == "struct"
        assert [f.name for f in reading.result.fields] == ["speed", "label"]


def test_describe_shows_types_defaults_units_docs_and_the_result():
    rows = {s.name: s.describe(Drive) for s in endpoint.declared(Drive)}
    assert rows["set_speed"]["params"] == [
        {"name": "vx", "type": "float", "unit": "m/s", "doc": "forward speed", "required": True},
        {
            "name": "w",
            "type": "float",
            "unit": "rad/s",
            "doc": "yaw rate",
            "required": False,
            "default": 0.0,
        },
    ]
    assert rows["set_speed"]["result"] == {"type": "bool"}
    assert rows["tare"]["params"] == [] and rows["tare"]["result"] == {"type": "none"}
    assert rows["target"]["params"][0] == {
        "name": "pos",
        "type": "array",
        "unit": "m",
        "dtype": "float64",
        "shape": [3],
        "required": True,
    }
    assert "result" not in rows["target"]
    assert rows["reading"]["result"]["fields"][0] == {
        "name": "speed",
        "type": "float",
        "unit": "m/s",
        "doc": "along the track",
        "required": True,
    }
    assert rows["joints/{item}/speed"]["family"] == "joints"
    assert rows["joints/{item}/speed"]["params"] == [
        {"name": "value", "type": "float", "unit": "rad/s", "required": True}
    ]


def test_plugins_describe_carries_the_signature_of_a_migrated_plugin():
    (row,) = get_plugin_details("upright_monitor")["endpoints"]
    fields = {f["name"]: f for f in row["result"]["fields"]}
    assert row["result"]["type"] == "UprightReport"
    assert fields["tilt_deg"]["unit"] == "deg" and fields["upright"]["type"] == "bool"


def test_a_family_makes_one_endpoint_per_configured_item():
    engine, drive = _engine(joints=["a", "b", "c"])
    with engine:
        names = sorted(e.name for e in engine.ctx.interface.all() if e.owner == "box")
        assert names == sorted(
            [
                "set_speed",
                "tare",
                "target",
                "reading",
                "joints/a/speed",
                "joints/b/speed",
                "joints/c/speed",
                "effort/a",
                "effort/b",
                "effort/c",
            ]
        )
        assert _write(engine, "joints/b/speed", {"value": 2}) == "b"
        assert drive.speeds == {"b": 2.0}


def test_a_familys_options_receive_the_item():
    engine, _ = _engine()
    with engine:
        left = engine.ctx.interface.find("box", "effort/left")
        right = engine.ctx.interface.find("box", "effort/right")
        assert (left.read(), right.read()) == (1.0, 2.0)
        assert (left.rate_hz, right.rate_hz) == (10.0, 20.0)
        assert right.backend == {"ros2": {"topic": "right/effort"}}
        assert left.result.unit == "N*m"


def test_a_signature_a_bridge_cannot_name_is_refused():
    class Loose(Plugin):
        @endpoint.command
        def go(self, *args) -> None: ...

    with pytest.raises(TypeError, match="parameters are named"):
        endpoint.declared(Loose)[0].describe(Loose)

    class Reads(Plugin):
        @endpoint.out
        def level(self, scale: float) -> float: ...

    with pytest.raises(TypeError, match="an out endpoint takes no parameters"):
        endpoint.declared(Reads)[0].describe(Reads)


def test_an_endpoint_on_another_entity_names_its_owner_and_scope():
    class Carrier(Plugin):
        @endpoint.out(owner=lambda self: "package", namespace="")
        def package_pose(self) -> float:
            return 0.0

        @endpoint.out
        def speed(self) -> float:
            return 1.0

    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    carrier = Carrier({"namespace": "belt"}, entity="box", label="carrier")
    with Engine(cfg, plugins=[DummyPlugin({}, name="box"), carrier], preview=True) as engine:
        package = engine.ctx.interface.find("package", "package_pose")
        speed = engine.ctx.interface.find("box", "speed")
        assert (package.namespace, speed.namespace) == ("", "belt")
