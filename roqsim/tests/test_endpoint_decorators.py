"""Endpoints declared with roqsim.endpoint: registration, marshalling, and the physics thread."""

from __future__ import annotations

import logging
import threading

import pytest

from roqsim import endpoint
from roqsim.bridge import BridgeBase
from roqsim.config import load_config_from_dict
from roqsim.context import CommandFuture, SimContext
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details
from roqsim.plugin import Plugin
from roqsim.plugins.dummy import DummyPlugin


class Probe(Plugin):
    """A plugin with one endpoint of each kind, recording the thread every call runs on."""

    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        self.threads: list[int] = []
        self.applied: list = []
        self.topic = None

    def configure(self, ctx: SimContext) -> None:
        self.topic = "resolved-in-configure"

    @endpoint.out(rate=lambda self: 25.0, ros2=lambda self: {"topic": self.topic})
    def level(self):
        """How full it is."""
        self.threads.append(threading.get_ident())
        return 0.5

    @endpoint.command(ros2={"service": "std_srvs.srv.Trigger"})
    def reset_counter(self, count: int = 0, label: str = "") -> str:
        self.threads.append(threading.get_ident())
        if label == "bad":
            raise ValueError("refused")
        return f"reset {count}"

    @endpoint.stream(name="setpoint", ros2={"type": "std_msgs.msg.Float64"})
    def set_setpoint(self, value: float) -> None:
        self.threads.append(threading.get_ident())
        self.applied.append(value)

    @endpoint.out(when=lambda self: False)
    def hidden(self):
        return None


def _engine(**probe_config):
    cfg = load_config_from_dict({"sim": {}, "components": []})
    probe = Probe(dict(probe_config), entity="box", label="probe")
    engine = Engine(cfg, plugins=[DummyPlugin({}, name="box"), probe], preview=True)
    return engine, probe


def _ep(engine, name):
    return engine.ctx.interface.find("box", name)


def _from_worker(fn):
    out = []
    worker = threading.Thread(target=lambda: out.append(fn()))
    worker.start()
    worker.join()
    return out[0] if out else None


def test_decorated_endpoints_register_with_the_plugins_owner_namespace_and_name():
    engine, _probe = _engine(namespace="tank1")
    with engine:
        eps = {e.name: e for e in engine.ctx.interface.all() if e.owner == "box"}
        assert set(eps) == {"level", "reset_counter", "setpoint"}  # `when` left `hidden` out
        assert {e.owner for e in eps.values()} == {"box"}
        assert {e.namespace for e in eps.values()} == {"tank1"}
        assert (eps["level"].direction, eps["level"].rate_hz) == ("out", 25.0)
        assert eps["reset_counter"].direction == eps["setpoint"].direction == "in"
        # A callable hint is resolved after configure, from the instance.
        assert eps["level"].backend == {"ros2": {"topic": "resolved-in-configure"}}


def test_a_bridge_listed_after_the_producer_binds_its_endpoints():
    class Bound(BridgeBase):
        BACKEND = "ros2"

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.bound = []
            self.inbound = {}

        def _make_output(self, ep, hints):
            self.bound.append(ep.name)
            return object()

        def _make_input(self, ep, hints, on_payload):
            self.bound.append(ep.name)
            self.inbound[ep.name] = on_payload

    cfg = load_config_from_dict({"sim": {}, "components": []})
    probe = Probe({}, entity="box", label="probe")
    bridge = Bound({})
    engine = Engine(cfg, plugins=[DummyPlugin({}, name="box"), probe, bridge], preview=True)
    with engine:
        assert bridge.bound == ["level", "reset_counter", "setpoint"]
        # A marshalled write is handed to the backend as is: it queues itself.
        assert bridge.inbound["reset_counter"] is _ep(engine, "reset_counter").write


def test_a_commands_result_reaches_the_caller():
    engine, _probe = _engine()
    with engine:
        future = _from_worker(lambda: _ep(engine, "reset_counter").write({"count": 7}))
        assert isinstance(future, CommandFuture) and not future.done()
        engine.step()
        assert future.result(timeout=0) == "reset 7"


def test_a_commands_exception_reaches_the_waiting_caller_and_is_not_logged(caplog):
    engine, _probe = _engine()
    with engine:
        future = _from_worker(lambda: _ep(engine, "reset_counter").write({"label": "bad"}))
        raised = []

        def wait():
            try:
                future.result(timeout=5.0)
            except ValueError as exc:
                raised.append(exc)

        waiter = threading.Thread(target=wait)
        waiter.start()
        while future._waiters == 0:  # the caller is blocked on it before the command runs
            threading.Event().wait(0.001)
        with caplog.at_level(logging.ERROR):
            engine.step()
        waiter.join()
        assert [str(e) for e in raised] == ["refused"]
        assert "posted command raised" not in caplog.text


def test_an_unwaited_commands_exception_is_logged(caplog):
    engine, _probe = _engine()
    with engine:
        future = _ep(engine, "reset_counter").write({"label": "bad"})
        with caplog.at_level(logging.ERROR):
            engine.step()
        assert "posted command raised" in caplog.text
        with pytest.raises(ValueError, match="refused"):
            future.result(timeout=0)


def test_a_command_not_yet_run_times_out():
    engine, _probe = _engine()
    with engine:
        future = _ep(engine, "reset_counter").write(None)
        with pytest.raises(TimeoutError):
            future.result(timeout=0.01)


def test_a_stream_applies_only_its_latest_value_once_per_step():
    engine, probe = _engine()
    with engine:
        write = _ep(engine, "setpoint").write
        _from_worker(lambda: [write({"value": v}) for v in (1.0, 2.0, 3.0)])
        assert probe.applied == []
        engine.step()
        assert probe.applied == [3.0]
        engine.step()  # nothing new arrived
        assert probe.applied == [3.0]
        write({"value": 4})  # an int passes for a float, and arrives as one
        engine.step()
        assert probe.applied == [3.0, 4.0]


def test_nothing_runs_off_the_physics_thread():
    engine, probe = _engine()
    with engine:
        physics = threading.get_ident()
        _from_worker(lambda: _ep(engine, "reset_counter").write({"count": 1}))
        _from_worker(lambda: _ep(engine, "setpoint").write({"value": 1.0}))
        engine.step()
        _ep(engine, "level").read()  # a bridge reads in post_step, on this thread
        assert len(probe.threads) == 3
        assert set(probe.threads) == {physics}


def test_commands_stay_fifo_across_post_and_submit():
    ctx = SimContext(config={})
    order = []
    ctx.post(lambda _c: order.append("post"))
    future = ctx.submit(lambda _c: order.append("submit") or "done")
    ctx.post(lambda _c: order.append("after"))
    assert ctx.drain_commands() == 3
    assert order == ["post", "submit", "after"]
    assert future.result(timeout=0) == "done"


def test_declared_endpoints_are_listed_from_the_class_alone():
    specs = endpoint.declared(Probe)
    assert [(s.name, s.kind) for s in specs] == [
        ("level", "out"),
        ("reset_counter", "command"),
        ("setpoint", "stream"),
        ("hidden", "out"),
    ]
    level = specs[0].describe(Probe)
    assert level["backends"] == ["ros2"] and level["doc"] == "How full it is."
    assert level["rate_hz"] == "computed"  # a callable rate is known only per instance


def test_an_undecorated_override_drops_the_declaration():
    class Quiet(Probe):
        def level(self):
            return 0.0

    assert "level" not in [s.name for s in endpoint.declared(Quiet)]


def test_plugins_describe_lists_a_plugins_declared_endpoints():
    details = get_plugin_details("upright_monitor")
    assert [(e["name"], e["kind"], e["direction"]) for e in details["endpoints"]] == [
        ("upright", "out", "out")
    ]
