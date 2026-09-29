# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The control-socket backend against a running simulation, and the same refusals on both routes.

One world, built twice: stepped in this thread for the in-process backend, and driven the way
``roqsim sim`` drives it -- on a thread of its own, serving its control socket -- for the socket
backend. Every refusal a scenario can meet is run through both and must read the same.
"""

from __future__ import annotations

import math
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
    "a pose of an unknown entity": lambda a: a.ground_truth_pose("parcle"),
    "a pose of a name no entity has": lambda a: a.ground_truth_pose("ghost"),
}


def test_every_refusal_reads_the_same_on_both_routes(routes):
    local, socket = routes
    for what, make in REFUSALS.items():
        here, there = _refusal(local, make), _refusal(socket, make)
        assert here == there, f"{what}:\n  in-process: {here}\n  socket:     {there}"


def test_an_unknown_entity_is_refused_with_the_closest_names(routes):
    _local, socket = routes
    assert "Did you mean 'parcel'?" in _refusal(socket, REFUSALS["a pose of an unknown entity"])
    assert "Known entities: cart, parcel, prop." in _refusal(
        socket, REFUSALS["a pose of a name no entity has"]
    )


def test_a_bare_value_for_several_parameters_names_them(routes):
    _local, socket = routes
    text = _refusal(socket, REFUSALS["a bare value for two parameters"])
    assert "pass a mapping of litres, rate" in text


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


# -- entity_near, entity_in_region: one verdict, whichever route ----------------------------------
# `cart` stands at (4, 4, 0.1) and the welded `prop` at (3, 3, 0.1) on both routes.

BOX_AT_CART = [{"x": 3.5, "y": 3.5}, {"x": 4.5, "y": 4.5}]
BOX_ELSEWHERE = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
TRIANGLE_AT_CART = [{"x": 3.0, "y": 3.0}, {"x": 5.0, "y": 3.0}, {"x": 4.0, "y": 5.0}]
# An L whose notch is where the cart stands.
L_AROUND_CART = [
    {"x": 3.0, "y": 3.0},
    {"x": 5.0, "y": 3.0},
    {"x": 5.0, "y": 3.5},
    {"x": 3.5, "y": 3.5},
    {"x": 3.5, "y": 5.0},
    {"x": 3.0, "y": 5.0},
]
ABOVE_CART = {"x": 4.0, "y": 4.0, "z": 1.1}

WHERE = {
    "near an entity": ("near", dict(entity="cart", target="prop", distance=1.5), "SUCCESS"),
    "not near enough an entity": (
        "near",
        dict(entity="cart", target="prop", distance=1.3),
        "RUNNING",
    ),
    "near a position": (
        "near",
        dict(entity="cart", position={"x": 4.0, "y": 4.5}, distance=0.6),
        "SUCCESS",
    ),
    "planar ignores height": (
        "near",
        dict(entity="cart", position=ABOVE_CART, distance=0.5),
        "SUCCESS",
    ),
    "spatial counts it": (
        "near",
        dict(entity="cart", position=ABOVE_CART, distance=0.5, mode="spatial"),
        "RUNNING",
    ),
    "inside a box": ("region", dict(entity="cart", region=BOX_AT_CART), "SUCCESS"),
    "not outside a box it is in": (
        "region",
        dict(entity="cart", region=BOX_AT_CART, outside=True),
        "RUNNING",
    ),
    "outside a box": ("region", dict(entity="cart", region=BOX_ELSEWHERE, outside=True), "SUCCESS"),
    "inside a polygon": ("region", dict(entity="cart", region=TRIANGLE_AT_CART), "SUCCESS"),
    "in a polygon's notch": ("region", dict(entity="cart", region=L_AROUND_CART), "RUNNING"),
}


def _where(kind, **args):
    from scenario_execution_roqsim.actions.entity_in_region import EntityInRegion
    from scenario_execution_roqsim.actions.entity_near import EntityNear

    if kind == "near":
        full = {"target": "", "position": None, "mode": "planar"} | args
        return EntityNear(), full
    return EntityInRegion(), {"outside": False} | args


def _judge(routes, route, tmp_path, monkeypatch, kind, args, measured_ticks=10):
    """``(status, message)`` of a where-condition on *route*, after it has measured ten times; the
    message without the transport's name, so the two routes can be compared."""
    (_local, engine), _socket = routes
    action, full = _where(kind, **args)
    if route == "in-process":
        action.setup(simulation=_Sim(engine.ctx), clock=_HostClock())
    else:
        monkeypatch.setenv("ROQSIM_CONTROL", "ipc://" + str(tmp_path / "c.sock"))
        action.setup(clock=_HostClock())
    action.execute(**full)
    deadline = time.monotonic() + 10
    try:
        while True:
            assert time.monotonic() < deadline, action.feedback_message
            status = action.update()
            if status.name != "RUNNING":
                break
            if not action.feedback_message.startswith("waiting for"):
                measured_ticks -= 1
                if measured_ticks == 0:
                    break
            if route == "in-process":
                engine.step()
            else:
                time.sleep(0.005)
    finally:
        action.shutdown()
    return status.name, action.feedback_message.replace(f" ({action.transport})", "")


@pytest.mark.parametrize("case", list(WHERE))
def test_a_where_condition_reads_the_same_on_both_routes(routes, tmp_path, monkeypatch, case):
    pytest.importorskip("scenario_execution")
    kind, args, expected = WHERE[case]
    here = _judge(routes, "in-process", tmp_path, monkeypatch, kind, args)
    there = _judge(routes, "control socket", tmp_path, monkeypatch, kind, args)
    assert here == there
    assert here[0] == expected, here[1]


def test_near_measures_between_reference_points(routes, tmp_path, monkeypatch):
    pytest.importorskip("scenario_execution")
    status, message = _judge(
        routes,
        "control socket",
        tmp_path,
        monkeypatch,
        "near",
        dict(entity="cart", target="prop", distance=1.5),
    )
    assert status == "SUCCESS"
    assert message == "'cart' 1.41 m planar from 'prop' (near: <= 1.5 m)"


def test_an_unknown_target_is_refused_alike_on_both_routes(routes, tmp_path, monkeypatch):
    pytest.importorskip("scenario_execution")
    from scenario_execution.actions.base_action import ActionError

    texts = []
    for route in ("in-process", "control socket"):
        with pytest.raises(ActionError) as err:
            _judge(
                routes,
                route,
                tmp_path,
                monkeypatch,
                "near",
                dict(entity="cart", target="prp", distance=1.0),
            )
        texts.append(str(err.value).replace(f" ({route})", ""))
    assert texts[0] == texts[1]
    assert "no entity called 'prp'" in texts[0] and "Did you mean 'prop'?" in texts[0]


def test_an_absent_entity_is_waited_for_alike_on_both_routes(routes, tmp_path, monkeypatch):
    """Deleted at run time, an entity is nowhere: the condition does not hold, and the run goes on."""
    pytest.importorskip("scenario_execution")
    (local, engine), socket = routes
    assert _settle_local(local.set_entity_presence("parcel", False), engine).ok
    assert _settle_socket(socket.set_entity_presence("parcel", False)).ok
    args = dict(entity="parcel", region=BOX_ELSEWHERE, outside=True)
    here = _judge(routes, "in-process", tmp_path, monkeypatch, "region", args, measured_ticks=3)
    there = _judge(
        routes, "control socket", tmp_path, monkeypatch, "region", args, measured_ticks=3
    )
    assert here == there
    assert here == (
        "RUNNING",
        "entity 'parcel' is absent (deleted at run time): nothing can see or touch it, so it has no "
        "pose until it is spawned again.",
    )


#: The cart drives past the welded prop into a box. The keep-out rule on that box holds only until
#: the cart has come within 1 m of the prop, which it does before it can reach the box.
WHERE_SCENARIO = """
import osc.helpers
import osc.roqsim

scenario cart_arrives:
    timeout(60s)
    event approached
    do parallel:
        entity_navigate(entity: 'cart', goal_poses: [pose_3d(position: position_3d(x: 3.3m, y: 3.3m))])
        serial:
            entity_near(entity: 'cart', target: 'prop', distance: 1.0)
            emit approached
            entity_in_region(entity: 'cart', region: [position_3d(x: 3.0m, y: 3.0m),
                                                      position_3d(x: 3.6m, y: 3.6m)])
            entity_near(entity: 'cart', position: position_3d(x: 3.3m, y: 3.3m, z: 5.0m),
                        distance: 0.3)
            emit end
        serial:
            entity_in_region(entity: 'cart', region: [position_3d(x: 3.0m, y: 3.0m),
                                                      position_3d(x: 3.6m, y: 3.6m)])
            emit fail
        UNTIL
        serial:
            entity_in_region(entity: 'cart', region: [position_3d(x: 5m, y: 5m),
                                                      position_3d(x: 6m, y: 5m),
                                                      position_3d(x: 6m, y: 6m)])
            emit fail
"""

UNTIL = "with:\n            until @approached"


def _where_scenario(tmp_path, until: bool):
    from scenario_execution.scenario_execution_base import ScenarioExecution

    path = tmp_path / "cart_arrives.osc"
    path.write_text(WHERE_SCENARIO.replace("UNTIL", UNTIL if until else ""))
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


def _run_stepped(run, engine):
    from scenario_execution.simulation import SimulationClock

    clock = SimulationClock(engine.dt)
    run.setup(run.tree, simulation=_Sim(engine.ctx), clock=clock)
    try:
        for _ in range(20000):
            if run.shutdown_requested:
                break
            engine.step()
            clock.advance()
            run.behaviour_tree.tick()
    finally:
        run.behaviour_tree.shutdown()
    assert run.shutdown_requested, "the scenario never ended"
    return run.process_results()


def _run_over_the_socket(run, tmp_path, monkeypatch):
    monkeypatch.setenv("ROQSIM_CONTROL", "ipc://" + str(tmp_path / "c.sock"))
    run.setup(run.tree, clock=_HostClock())
    deadline = time.monotonic() + 30
    try:
        while not run.shutdown_requested:
            assert time.monotonic() < deadline, "the scenario never ended"
            run.behaviour_tree.tick()
            time.sleep(0.005)
    finally:
        run.behaviour_tree.shutdown()
    return run.process_results()


def test_a_scenario_of_where_conditions_runs_stepped(routes, tmp_path):
    pytest.importorskip("scenario_execution")
    (_local, engine), _socket = routes
    run = _where_scenario(tmp_path, until=True)
    assert _run_stepped(run, engine), run.results
    x, y, _z = engine.ctx.data.mocap_pos[0]
    assert math.hypot(x - 3.3, y - 3.3) <= 0.3


def test_the_same_scenario_runs_over_the_socket(routes, tmp_path, monkeypatch):
    pytest.importorskip("scenario_execution")
    run = _where_scenario(tmp_path, until=True)
    assert _run_over_the_socket(run, tmp_path, monkeypatch), run.results


def test_without_its_until_the_keep_out_rule_fails_the_run(routes, tmp_path):
    pytest.importorskip("scenario_execution")
    (_local, engine), _socket = routes
    run = _where_scenario(tmp_path, until=False)
    assert not _run_stepped(run, engine)
