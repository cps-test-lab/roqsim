"""The ipc bridge: a running simulation's endpoints over ZeroMQ, served only when asked."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from roqsim import control as ctl  # noqa: E402
from roqsim import endpoint, ipc, runner  # noqa: E402
from roqsim.bridge import BridgeBase  # noqa: E402
from roqsim.clock import Pacer  # noqa: E402
from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.context import Endpoint, SimContext  # noqa: E402
from roqsim.control_client import Client, ControlError  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim.ipc.bridge import IpcBridge  # noqa: E402
from roqsim.plugin import Plugin  # noqa: E402
from roqsim.plugins.dummy import DummyPlugin  # noqa: E402
from roqsim.plugins.run_control import RunControlPlugin  # noqa: E402


class Tank(Plugin):
    """One endpoint of each kind, counting every read."""

    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        self.reads = 0
        self.setpoints: list = []
        self.verdict = "none"
        self.pending = False

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx

    @endpoint.out(rate_hz=100.0)
    def level(self):
        """How full it is."""
        self.reads += 1
        return {"litres": 3.5, "profile": np.arange(6, dtype=np.float32).reshape(2, 3)}

    @endpoint.out()
    def report(self):
        return {"verdict": self.verdict}

    @endpoint.command("drain", confirm="report")
    def drain(self, litres=None):
        """Let some out."""
        if litres is not None and litres < 0:
            raise ValueError(f"cannot drain {litres} litres: a drain takes water out")
        self.pending = True
        return {"drained": litres}

    @endpoint.command("stall")
    def stall(self, _payload=None):
        return None

    @endpoint.stream("setpoint")
    def set_setpoint(self, value):
        self.setpoints.append(value)

    def post_step(self, ctx: SimContext) -> None:
        if self.pending:  # the verdict a step records after the command
            self.verdict, self.pending = "landed", False


class Sim:
    """A world driven like `roqsim sim` drives it, on a thread of its own."""

    def __init__(self, uri: str, *extra):
        cfg = load_config_from_dict({"sim": {}, "plugins": []})
        self.tank = Tank({}, entity="box", label="tank")
        self.engine = Engine(
            cfg,
            plugins=[
                RunControlPlugin({}, entity="sim", label="run_control"),
                DummyPlugin({}, name="box"),
                self.tank,
                *extra,
                IpcBridge({"uri": uri, "world": "test"}),
            ],
            preview=True,
        )
        self.uri = uri
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            with self.engine:
                self.engine.reset()
                pacer = Pacer(self.engine.dt, factor=1.0, realtime=True)
                self._ready.set()
                while not self._stop.is_set():
                    runner._tick(self.engine, pacer)
        except BaseException as err:  # noqa: BLE001 - reported by the test
            self.error = err
            self._ready.set()

    def __enter__(self):
        self.thread.start()
        assert self._ready.wait(20)
        if self.error:
            raise self.error
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join(10)


@pytest.fixture
def uri(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return "ipc://" + str(tmp_path / "c.sock")


def test_read_describe_and_array_frames_round_trip(uri):
    with Sim(uri), Client(uri) as sim:
        value = sim.read("box/tank/level")
        assert value["litres"] == 3.5
        assert value["profile"].dtype == np.float32 and value["profile"].shape == (2, 3)
        np.testing.assert_array_equal(value["profile"], np.arange(6).reshape(2, 3))
        assert sim.read("box/tank/level", field="litres") == 3.5
        paths = {e["path"]: e for e in sim.endpoints()}
        assert paths["box/tank/level"]["kind"] == "out"
        assert paths["box/tank/drain"]["kind"] == "command"
        assert paths["box/tank/setpoint"]["kind"] == "stream"
        assert {"sim/run_control/pause", "sim/run_control/step"} <= set(paths)
        full = sim.describe("box/tank/drain")
        assert full["doc"] == "Let some out." and full["confirm"] == "box/tank/report"


def test_a_command_replies_with_its_confirmation_and_a_refusal_with_its_own_text(uri):
    with Sim(uri), Client(uri) as sim:
        reply = sim.call("box/tank/drain", 2)
        assert reply["applied"] and reply["verified"]
        assert reply["result"] == {"drained": 2}
        assert reply["confirmation"] == {"verdict": "landed"}
        with pytest.raises(ControlError, match="cannot drain -1 litres") as err:
            sim.call("box/tank/drain", -1)
        assert err.value.kind == "refused"


def test_a_paused_run_applies_a_command_and_says_it_is_unverified(uri):
    with Sim(uri) as run, Client(uri) as sim:
        sim.pause()
        reply = sim.call("box/tank/drain", 1)
        assert reply["applied"] and reply["verified"] is False
        assert "unverified" in reply["note"]
        assert run.tank.verdict == "none"  # no step ran to record one


def test_a_stream_keeps_the_latest_value(uri):
    with Sim(uri) as run, Client(uri) as sim:
        assert sim.call("box/tank/setpoint", [1.0, 2.0]) == {"queued": True}
        deadline = time.monotonic() + 5
        while not run.tank.setpoints and time.monotonic() < deadline:
            time.sleep(0.01)
        assert run.tank.setpoints[-1] == [1.0, 2.0]


def test_a_command_the_physics_thread_never_runs_is_a_timeout_error(uri):
    with Sim(uri) as run, Client(uri) as sim:
        run._stop.set()  # the loop ends; nothing drains the queue any more
        run.thread.join(5)
        with pytest.raises(ControlError) as err:
            sim.call("box/tank/stall", timeout=0.3)
        assert err.value.kind in ("timeout", "unreachable")


def test_an_unknown_path_names_the_nearest_and_the_siblings(uri):
    with Sim(uri), Client(uri) as sim:
        with pytest.raises(ControlError) as err:
            sim.read("box/tank/levle")
        assert err.value.kind == "unknown_path"
        assert "Did you mean 'box/tank/level'?" in str(err.value)
        assert "box/tank/drain" in str(err.value)
        with pytest.raises(ControlError, match="use call"):
            sim.read("box/tank/drain")


def test_nothing_is_read_per_step_until_someone_subscribes(uri):
    with Sim(uri) as run, Client(uri) as sim:
        time.sleep(0.3)
        assert run.tank.reads == 0, "no client subscribed, yet the endpoint was read"
        with sim.subscribe("box/tank/lev") as sub:
            got = None
            deadline = time.monotonic() + 5
            while got is None and time.monotonic() < deadline:
                got = sub.get(timeout=0.5)
            assert got is not None
            path, t, value = got
            assert path == "box/tank/level" and value["litres"] == 3.5 and t >= 0.0
        reads = run.tank.reads
        time.sleep(0.3)
        assert run.tank.reads - reads <= 1, "still read after the subscriber left"


def test_pause_step_n_and_resume(uri):
    with Sim(uri) as run, Client(uri) as sim:
        paused = sim.pause()
        assert paused["state"] == "paused"
        t0 = sim.state()["sim_time"]
        assert paused["sim_time"] == t0, "the reply is the time the run stopped at"
        reached = sim.step(5)
        dt = run.engine.dt
        assert reached["completed"] and reached["sim_time"] == pytest.approx(t0 + 5 * dt)
        assert sim.state()["sim_time"] == pytest.approx(t0 + 5 * dt)
        assert sim.resume()["state"] == "playing"
        with pytest.raises(ControlError, match="pause it first"):
            sim.step(1)


def test_a_resumed_run_is_not_counted_as_falling_behind():
    cfg = load_config_from_dict({"sim": {}, "plugins": [{"dummy": {}, "name": "d0"}]})
    engine = Engine(cfg, preview=True)
    with engine:
        engine.reset()
        pacer = Pacer(engine.dt, factor=1.0, realtime=True)
        runner._tick(engine, pacer)
        engine.ctx.control.set_state(ctl.PAUSED)
        runner._tick(engine, pacer)
        time.sleep(0.2)  # a pause far longer than a step
        engine.ctx.control.set_state(ctl.PLAYING)
        runner._tick(engine, pacer)
        assert pacer.behind_steps == 0


def test_two_transports_writing_one_stream_warn_once_naming_both(caplog):
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    tank = Tank({}, entity="box", label="tank")
    with Engine(cfg, plugins=[DummyPlugin({}, name="box"), tank], preview=True) as engine:
        ep = engine.ctx.interface.find("box", "setpoint")
        with caplog.at_level(logging.WARNING, logger="roqsim.context"):
            ep.slot.put(1.0, "ros2")
            ep.slot.put(2.0, "ipc")
            ep.slot.put(3.0, "ros2")
        warnings = [r for r in caplog.records if "written by both" in r.getMessage()]
        assert len(warnings) == 1
        assert "ros2" in warnings[0].getMessage() and "ipc" in warnings[0].getMessage()
        engine.ctx.drain_commands()
        assert tank.setpoints == [3.0]


def test_two_endpoints_on_one_path_are_refused_naming_both(uri):
    class Twin(Plugin):
        def configure(self, ctx):
            for _ in range(2):
                ctx.interface.add(Endpoint(name="x", direction="out", read=lambda: 1))

    run = Sim(uri, Twin({}, entity="box", label="twin"))
    with pytest.raises(Exception, match="share the control path 'box/twin/x'"):
        with run:
            pass


def test_an_endpoint_opts_out_with_a_false_hint(uri):
    class Quiet(Plugin):
        @endpoint.out(ipc=False)
        def secret(self):
            return 1

    with Sim(uri, Quiet({}, entity="box", label="quiet")), Client(uri) as sim:
        assert "box/quiet/secret" not in {e["path"] for e in sim.endpoints()}


def test_roqsim_sim_serves_and_registers_the_control_socket(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv(ipc.ENV, raising=False)
    world = tmp_path / "w.yaml"
    world.write_text("sim: {}\ncomponents:\n  - dummy: {}\n    name: d0\n")
    assert runner.main([str(world), "--headless", "--steps", "3", "--pacing", "asap"]) == 0
    assert f"control: ipc://{tmp_path / ipc.SOCKET_NAME}" in capsys.readouterr().out
    assert ipc.running() == []  # unregistered at exit
    assert not (tmp_path / ipc.SOCKET_NAME).exists(), "the socket file outlived the run"


def test_roqsim_sim_control_none_serves_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path))
    world = tmp_path / "w.yaml"
    world.write_text("sim: {}\ncomponents:\n  - dummy: {}\n    name: d0\n")
    seen = []
    real = runner.run
    monkeypatch.setattr(runner, "run", lambda *a, **k: seen.append(k) or real(*a, **k))
    args = [str(world), "--headless", "--steps", "3", "--pacing", "asap", "--control", "none"]
    assert runner.main(args) == 0
    assert seen[0]["control"] is None
    assert "control:" not in capsys.readouterr().out
    assert not (tmp_path / ipc.SOCKET_NAME).exists()
    assert not list(tmp_path.glob("*.json"))


def test_discovery_finds_the_only_running_simulator(uri):
    with Sim(uri):
        assert ipc.discover() == uri
        assert [e["uri"] for e in ipc.running()] == [uri]


def test_describe_names_the_endpoint_on_the_other_transports(uri):
    class Wire(BridgeBase):
        BACKEND = "wire"

        def _make_output(self, ep, hints):
            self._names[id(ep)] = {"topic": "/resolved/" + hints["topic"]}

        def _publish(self, handle, payload, stamp):
            pass

    class Hinted(Plugin):
        @endpoint.out(wire={"topic": "odom"})
        def odom(self):
            return 0.0

    with Sim(uri, Hinted({}, entity="box", label="base"), Wire({})), Client(uri) as sim:
        entry = sim.describe("box/base/odom")
        assert entry["bridges"] == {"wire": {"topic": "/resolved/odom"}}
