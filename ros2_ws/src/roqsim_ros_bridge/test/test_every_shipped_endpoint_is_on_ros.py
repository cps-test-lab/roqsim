"""Every endpoint a shipped ROS world registers is carried on ROS, or kept off it by its producer.

An endpoint whose payload type has no row in the type table and names no message is not refused by
the bridge: it is logged at info level and left off ROS (:func:`roqsim_ros_bridge.typemap.resolve`),
which is right for a producer that never asked for ROS and silent for one whose type simply has no
mapping yet. Most hints are computed when the plugin configures, so only a built world can say
which one an endpoint is. This builds every world the workspace ships and binds each endpoint as the
bridge would: its message, service or action type resolves, a mapping by field name fits, and a
service or action has a handler.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
pytest.importorskip("rclpy")

from roqsim import types as T  # noqa: E402
from roqsim.config import drop_transport_plugins, load_config  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim_ros_bridge import typemap  # noqa: E402
from roqsim_ros_bridge.actions import get_action_handler  # noqa: E402
from roqsim_ros_bridge.registry import resolve_type  # noqa: E402
from roqsim_ros_bridge.services import get_service_handler  # noqa: E402

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
WORLDS = sorted(WORKSPACE.glob("*/worlds/*.yaml"))


def _unbindable(ep) -> str | None:
    """Why the bridge would not carry *ep* on ROS, or ``None`` when it would (or is told not to)."""
    binding = typemap.resolve(ep)
    if binding is None:  # its producer keeps it off ROS
        return None
    hints = binding.hints
    if hints is None:
        return binding.reason
    try:
        if "service" in hints:
            resolve_type(hints["service"])
            get_service_handler(hints["service"])
        elif "action" in hints:
            resolve_type(hints["action"])
            get_action_handler(hints["action"])
        else:
            binding.prepare(resolve_type(hints["type"]))
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


@pytest.mark.parametrize("world", WORLDS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_endpoint_of_a_shipped_world_binds_on_ros(world):
    cfg = load_config(world)
    drop_transport_plugins(cfg)  # what is bound is checked here, with no node on the graph
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    try:
        endpoints = list(engine.ctx.interface.all())
        unbound = {
            f"{ep.owner}/{ep.name}": why for ep in endpoints if (why := _unbindable(ep)) is not None
        }
    finally:
        engine.shutdown()
    assert endpoints
    assert not unbound, unbound


def test_the_workspace_ships_worlds():
    assert WORLDS


def test_every_neutral_type_has_a_row():
    """A dataclass added to :mod:`roqsim.types` travels on ROS from the start, or this says so."""
    neutral = [
        obj
        for obj in vars(T).values()
        if isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__module__ == T.__name__
    ]
    assert neutral
    assert not [cls.__name__ for cls in neutral if typemap.lookup(cls) is None]
