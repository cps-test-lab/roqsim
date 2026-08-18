"""BridgeBase binding: backend-hint selection, per-endpoint namespaces, optional owner filter."""

from __future__ import annotations

from roqsim.bridge import BridgeBase
from roqsim.context import Endpoint, SimContext


class FakeBridge(BridgeBase):
    """Records what a backend would wire, without any transport."""

    BACKEND = "fake"

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.outputs: list[tuple[str, str]] = []  # (endpoint name, namespace)
        self.inputs: list[tuple[str, str]] = []

    def _make_output(self, ep, hints):
        self.outputs.append((ep.name, ep.namespace))
        return object()

    def _make_input(self, ep, hints, on_payload):
        self.inputs.append((ep.name, ep.namespace))


def _ctx_with_endpoints():
    ctx = SimContext(config={})
    for owner, ns in (("robot1", "robot1"), ("robot2", "robot2")):
        ctx.interface.add(
            Endpoint(
                name="odom",
                direction="out",
                owner=owner,
                namespace=ns,
                read=lambda: (0.0,),
                backend={"fake": {}},
            )
        )
        ctx.interface.add(
            Endpoint(
                name="cmd_vel",
                direction="in",
                owner=owner,
                namespace=ns,
                write=lambda p: None,
                backend={"fake": {}},
            )
        )
    # No hint block for this backend -> must be skipped.
    ctx.interface.add(
        Endpoint(
            name="scan",
            direction="out",
            owner="robot1",
            namespace="robot1",
            read=lambda: (0.0,),
            backend={"other": {}},
        )
    )
    return ctx


def test_one_bridge_serves_all_namespaced_endpoints():
    bridge = FakeBridge({})
    bridge.configure(_ctx_with_endpoints())
    assert bridge.outputs == [("odom", "robot1"), ("odom", "robot2")]
    assert bridge.inputs == [("cmd_vel", "robot1"), ("cmd_vel", "robot2")]


def test_owner_filter_is_optional_and_orthogonal_to_namespace():
    bridge = FakeBridge({"owner": "robot2"})
    bridge.configure(_ctx_with_endpoints())
    assert bridge.outputs == [("odom", "robot2")]
    assert bridge.inputs == [("cmd_vel", "robot2")]


def test_endpoint_namespace_defaults_empty():
    ctx = SimContext(config={})
    ctx.interface.add(
        Endpoint(
            name="speed",
            direction="in",
            owner="conveyor",
            write=lambda p: None,
            backend={"fake": {}},
        )
    )
    bridge = FakeBridge({})
    bridge.configure(ctx)
    assert bridge.inputs == [("speed", "")]
