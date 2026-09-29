# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The control-socket backend against a running simulation, and the same refusals on both routes.

One world, built twice: stepped in this thread for the in-process backend, and driven the way
``roqsim sim`` drives it -- on a thread of its own, serving its control socket -- for the socket
backend. Every refusal a scenario can meet is run through both and must read the same.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("zmq")
pytest.importorskip("roqsim_nav")

from roqsim import endpoint, runner  # noqa: E402
from roqsim.clock import Pacer  # noqa: E402
from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.context import Entity, SimContext  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim.ipc.bridge import IpcBridge  # noqa: E402
from roqsim.plugin import Plugin  # noqa: E402
from roqsim.plugins.entity_control import EntityControlPlugin  # noqa: E402
from roqsim.plugins.model_override import ModelOverridePlugin  # noqa: E402
from roqsim.plugins.run_control import RunControlPlugin  # noqa: E402
from roqsim_nav.plugins.navigator import NavigatorPlugin  # noqa: E402
from scenario_execution_roqsim.access import AccessError  # noqa: E402
from scenario_execution_roqsim.access.in_process import InProcessAccess  # noqa: E402
from scenario_execution_roqsim.access.ipc import IpcAccess  # noqa: E402

SCENE = """
<mujoco model="ipc_access">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ramp" type="box" size="1 1 0.02" euler="0 20 0" friction="1.0 0.005 0.0001"/>
    <body name="crate" pos="0 0 0.4">
      <freejoint name="crate_free"/>
      <geom name="crate" type="box" size="0.05 0.05 0.05" mass="1"
            priority="1" friction="0.7 0.02 0.001" euler="0 20 0"/>
    </body>
    <body name="fixed_prop" pos="3 3 0.1"><geom name="prop" type="box" size="0.1 0.1 0.1"/></body>
    <body name="cart" pos="4 4 0.1" mocap="true">
      <geom name="cart" type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


#: The sim time at which ``parcel``'s ``tank.full`` report turns true.
FULL_AT = 0.2


class Things(Plugin):
    """The world's entities, a command that refuses, and two reports: one fixed, one that follows
    sim time (``tank.full`` turns true at 0.2 s)."""

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        ctx.entities.add(
            Entity(name="parcel", kind="object", body="crate", meta={"base_joint": "crate_free"})
        )
        ctx.entities.add(Entity(name="prop", kind="object", body="fixed_prop"))
        ctx.entities.add(Entity(name="cart", kind="object", body="cart", meta={"mocap": True}))

    @property
    def endpoint_owner(self) -> str:
        return "parcel"

    @endpoint.command
    def arm(self) -> None:
        raise ValueError("the gripper is not armed")

    @endpoint.command
    def fill(self, litres: float, rate: float) -> None:
        pass

    @endpoint.out
    def level(self) -> dict:
        return {"litres": 3.5, "full": False}

    @endpoint.out
    def tank(self) -> dict:
        t = float(self._ctx.data.time)
        return {"litres": t, "full": t >= FULL_AT, "gauges": [t, 2 * t]}


def _plugins(*extra):
    return [
        Things({}, label="things"),
        NavigatorPlugin({"goals": [], "autostart": False, "output": "mocap"}, entity="cart"),
        ModelOverridePlugin(
            {"overrides": [{"field": "geom_friction", "select": ["crate"], "to": 0.0}]},
            name="grip_fault",
        ),
        *extra,
    ]


def _engine(tmp_path, *extra) -> Engine:
    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)
    cfg = load_config_from_dict({"sim": {"world": str(scene)}, "plugins": []})
    engine = Engine(cfg, plugins=_plugins(*extra), preview=True)
    return engine


class _Sim:
    def __init__(self, ctx):
        self.context = ctx


class Served:
    """The world as `roqsim sim` runs it: its own thread, its control socket, as fast as it goes."""

    def __init__(self, tmp_path, uri):
        self.engine = _engine(
            tmp_path,
            RunControlPlugin({}, entity="sim", label="run_control"),
            EntityControlPlugin({}, entity="sim", label="entities"),
            IpcBridge({"uri": uri}),
        )
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.engine:
            self.engine.reset()
            pacer = Pacer(self.engine.dt, realtime=False)
            self._ready.set()
            while not self._stop.is_set():
                runner._tick(self.engine, pacer)
                time.sleep(0.0005)

    def __enter__(self):
        self.thread.start()
        assert self._ready.wait(20)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join(10)


@pytest.fixture
def routes(tmp_path, monkeypatch):
    """``(in-process access and its engine, socket access)``, over the same world."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    uri = "ipc://" + str(tmp_path / "c.sock")
    (tmp_path / "local").mkdir()
    local = _engine(tmp_path / "local")
    with Served(tmp_path, uri), local:
        local.reset()
        access = IpcAccess(uri)
        deadline = time.monotonic() + 10
        while not access.ready():
            assert time.monotonic() < deadline, access.pending_reason()
            time.sleep(0.01)
        try:
            yield (InProcessAccess(_Sim(local.ctx)), local), access
        finally:
            access.teardown()


def _settle_local(call, engine):
    for _ in range(200):
        outcome = call.poll()
        if outcome is not None:
            return outcome
        engine.step()
    raise AssertionError("no outcome in-process")


def _settle_socket(call):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        outcome = call.poll()
        if outcome is not None:
            return outcome
        time.sleep(0.005)
    raise AssertionError("no outcome over the socket")


def _refusal(route, make) -> str:
    """The text a call is refused with on *route*: an AccessError, or a failed outcome's detail."""
    access, engine = route if isinstance(route, tuple) else (route, None)
    try:
        call = make(access)
        outcome = _settle_local(call, engine) if engine is not None else _settle_socket(call)
    except AccessError as err:
        return f"raised: {err}"
    assert not outcome.ok, outcome
    return f"failed: {outcome.detail}"


REFUSALS = {
    "an unknown command": lambda a: a.call_endpoint("parcel", "arn"),
    "a producer's refusal": lambda a: a.call_endpoint("parcel", "arm"),
    "a bare value for two parameters": lambda a: a.call_endpoint("parcel", "fill", 1.0),
    "welded scenery placed": lambda a: a.set_entity_state("prop", [0, 0, 1], [1, 0, 0, 0]),
    "an unknown entity placed": lambda a: a.set_entity_state("ghost", [0, 0, 1], [1, 0, 0, 0]),
    "a present entity spawned": lambda a: a.set_entity_presence("parcel", True, [0, 0, 1]),
    "a route with nothing configured": lambda a: a.start_route("cart", wait=True),
    "an entity with no navigator": lambda a: a.navigate("parcel", [(1.0, 0.0)], wait=True),
    "an unknown report": lambda a: a.entity_report("parcel", "levle"),
    "an unknown field": lambda a: a.entity_report("parcel", "level", "litre"),
    "a report with no single field": lambda a: a.entity_report("parcel", "level"),
    "a field that is not a single value": lambda a: a.entity_report("parcel", "tank", "gauges"),
}


def test_every_refusal_reads_the_same_on_both_routes(routes):
    local, socket = routes
    for what, make in REFUSALS.items():
        here, there = _refusal(local, make), _refusal(socket, make)
        assert here == there, f"{what}:\n  in-process: {here}\n  socket:     {there}"


def test_a_command_is_confirmed_over_the_socket(routes):
    _local, socket = routes
    time.sleep(0.5)  # the crate settles onto the ramp, so the override has a contact to verify
    outcome = _settle_socket(socket.call_endpoint("grip_fault", "override", True))
    assert outcome.ok and outcome.confirmed
    assert outcome.verified == "landed" and outcome.confirmation["active"] is True


def test_poses_reports_and_placement_over_the_socket(routes):
    _local, socket = routes
    pose = None
    while pose is None:
        pose = socket.entity_pose("parcel")
    assert pose.pos.shape == (3,) and pose.quat.shape == (4,)
    reading = _settle_socket(socket.entity_report("parcel", "level", "full"))
    assert reading.value is False and reading.field == "full"
    placed = _settle_socket(socket.set_entity_state("parcel", [0.0, 0.0, 1.0], [1, 0, 0, 0]))
    assert placed.ok, placed.detail
    gone = _settle_socket(socket.set_entity_presence("parcel", False))
    assert gone.ok, gone.detail
    with pytest.raises(AccessError, match="ABSENT"):
        while socket.entity_pose("parcel") is None:
            pass


def test_a_route_is_followed_to_its_end_over_the_socket(routes):
    _local, socket = routes
    outcome = _settle_socket(socket.navigate("cart", [(4.2, 4.0)], wait=True))
    assert outcome.ok and outcome.detail == "arrived"


# -- entity_monitor: a variable kept current, and a scenario that waits on it -----------------------


def _variable(name, default):
    import py_trees
    from scenario_execution.model.types import VariableReference

    ref = VariableReference(py_trees.blackboard.Client(name=f"test {name}"), f"/{name}")
    ref.set_value(default)
    return ref


class _HostClock:
    def __init__(self):
        self._start = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._start


def test_a_monitor_keeps_its_variable_current_over_the_socket(routes, tmp_path, monkeypatch):
    """Found the way a scenario finds the simulator (``ROQSIM_CONTROL``), and rewritten with each
    reading: the served world's sim time climbs, so every new reading is a new value."""
    pytest.importorskip("scenario_execution")
    from scenario_execution_roqsim.actions.entity_monitor import EntityMonitor

    monkeypatch.setenv("ROQSIM_CONTROL", "ipc://" + str(tmp_path / "c.sock"))
    litres = _variable("litres", -1.0)
    action = EntityMonitor()
    action.setup(clock=_HostClock())
    action.execute(entity="parcel", value="tank.litres", target_variable=litres)
    seen = []
    deadline = time.monotonic() + 10
    while len(seen) < 5:
        assert time.monotonic() < deadline, f"only {seen} over the socket"
        assert action.update().name == "RUNNING", "a monitor never ends on its own"
        if litres.get_value() != -1.0 and (not seen or litres.get_value() != seen[-1]):
            seen.append(litres.get_value())
        time.sleep(0.005)
    action.shutdown()
    assert seen == sorted(seen), "the variable follows the report as it climbs"
    assert "control socket" in action.feedback_message


SCENARIO = """
import osc.helpers
import osc.roqsim

scenario tank_fills:
    timeout(30s)
    var full: bool = false
    var litres: float = 0.0
    do parallel:
        entity_monitor(entity: 'parcel', value: 'tank.full', target_variable: full)
        entity_monitor(entity: 'parcel', value: 'tank.litres', target_variable: litres)
        serial:
            wait full == true and litres >= FULL_AT
            emit end
""".replace("FULL_AT", str(FULL_AT))


def _scenario(tmp_path):
    from scenario_execution.scenario_execution_base import ScenarioExecution

    path = tmp_path / "tank_fills.osc"
    path.write_text(SCENARIO)
    run = ScenarioExecution(
        debug=False,
        log_model=False,
        live_tree=False,
        scenario_file=str(path),
        output_dir=None,
        register_signal=False,
    )
    assert run.parse(), run.results
    return run


def test_a_wait_on_a_monitored_variable_ends_a_stepped_run_when_it_holds(routes, tmp_path):
    """`wait full == true and litres >= 0.2` over two monitors, in a scenario parsed from its
    text: the run ends on the first tick the report says full, and not before."""
    pytest.importorskip("scenario_execution")
    from scenario_execution.simulation import SimulationClock

    (local, engine), _socket = routes
    run = _scenario(tmp_path)
    clock = SimulationClock(engine.dt)
    run.setup(run.tree, simulation=_Sim(engine.ctx), clock=clock)
    try:
        for _ in range(1000):
            if run.shutdown_requested:
                break
            engine.step()
            clock.advance()
            run.behaviour_tree.tick()
    finally:
        run.behaviour_tree.shutdown()
    assert run.process_results(), run.results
    ended = float(engine.ctx.data.time)
    assert FULL_AT <= ended <= FULL_AT + 5 * engine.dt, f"ended at {ended} s"


def test_a_wait_on_a_monitored_variable_ends_a_run_over_the_socket(routes, tmp_path, monkeypatch):
    """The same scenario text, unedited, against the served simulator."""
    pytest.importorskip("scenario_execution")

    monkeypatch.setenv("ROQSIM_CONTROL", "ipc://" + str(tmp_path / "c.sock"))
    run = _scenario(tmp_path)
    run.setup(run.tree, clock=_HostClock())
    deadline = time.monotonic() + 20
    try:
        while not run.shutdown_requested:
            assert time.monotonic() < deadline, "the scenario never ended"
            run.behaviour_tree.tick()
            time.sleep(0.005)
    finally:
        # Each action closes its connection to the simulator.
        run.behaviour_tree.shutdown()
    assert run.process_results(), run.results
