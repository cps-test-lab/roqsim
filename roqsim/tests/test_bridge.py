"""BridgeBase binding: backend-hint selection, per-endpoint namespaces, optional owner filter."""

from __future__ import annotations

from roqsim.bridge import BridgeBase
from roqsim.context import Endpoint, SimContext


class FakeBridge(BridgeBase):
    """Records what a backend would wire, without any transport."""

    BACKEND = "fake"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
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


class RecordingBridge(FakeBridge):
    """Also records what actually got published, so the lazy gate is observable."""

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.published: list[str] = []
        self._handles: dict[int, str] = {}

    def _make_output(self, ep, hints):
        handle = object()
        self._handles[id(handle)] = ep.name
        super()._make_output(ep, hints)
        return handle

    def _publish(self, handle, payload, stamp):
        self.published.append(self._handles[id(handle)])


def _lazy_ctx(has_subscribers, *, lazy=True):
    """One lazy endpoint and one eager one, both due every step."""
    ctx = SimContext(config={})
    reads: list[str] = []
    ctx.interface.add(
        Endpoint(
            name="image",
            direction="out",
            owner="cam",
            read=lambda: reads.append("image") or (0.0,),
            lazy=lazy,
            has_subscribers=has_subscribers,
            backend={"fake": {}},
        )
    )
    ctx.interface.add(
        Endpoint(
            name="camera_info",
            direction="out",
            owner="cam",
            read=lambda: reads.append("camera_info") or (0.0,),
            has_subscribers=lambda: False,
            backend={"fake": {}},
        )
    )
    return ctx, reads


def test_lazy_endpoint_is_neither_read_nor_published_without_subscribers():
    ctx, reads = _lazy_ctx(lambda: False)
    bridge = RecordingBridge({})
    bridge.configure(ctx)
    bridge.post_step(ctx)
    # The payload is never even produced -- the point is to skip an expensive read, not just the send.
    assert reads == ["camera_info"]
    assert bridge.published == ["camera_info"]


def test_lazy_endpoint_publishes_once_someone_subscribes():
    subscribed = [False]
    ctx, reads = _lazy_ctx(lambda: subscribed[0])
    bridge = RecordingBridge({})
    bridge.configure(ctx)
    bridge.post_step(ctx)
    assert "image" not in bridge.published
    subscribed[0] = True
    bridge.post_step(ctx)
    assert bridge.published.count("image") == 1
    assert reads.count("image") == 1


def test_lazy_endpoint_without_subscriber_introspection_still_publishes():
    """``has_subscribers is None`` means the backend cannot tell -- assume yes, never go silent."""
    ctx, _ = _lazy_ctx(None)
    bridge = RecordingBridge({})
    bridge.configure(ctx)
    bridge.post_step(ctx)
    assert "image" in bridge.published


def test_eager_endpoint_publishes_to_nobody_unchanged():
    """A publish can carry more than its message (TF off an odom payload), so the default stays eager
    even when the transport reports no subscribers."""
    ctx, _ = _lazy_ctx(lambda: False, lazy=False)
    bridge = RecordingBridge({})
    bridge.configure(ctx)
    bridge.post_step(ctx)
    assert bridge.published == ["image", "camera_info"]
