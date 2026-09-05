"""The avoidance interface, asserted against a stub rather than against ORCA.

That choice is the test. The interface exists so a local planner nobody here anticipated can be
plugged in, so the thing it is tested with must be one this package knows nothing about -- and it
**must pass with rvo2 uninstalled**. If any of these needed ORCA, the interface would have been
shaped around ORCA, and the next model would discover that the hard way.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim_nav.avoidance import (
    NO_AGENT,
    AvoidanceModel,
    NullAvoidance,
    RegistryError,
    resolve_model,
)

HERE = Path(__file__).parent
STUB = f"{HERE / 'stub_avoidance.py'}:RecordingModel"
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""
ROOM = HERE / "data" / "open_room.xml"  # walls, but nothing to plan around


def _world(tmp_path, *, movers=1, avoidance=None, yields=True, model=STUB, subject=False):
    """``avoidance`` is what each mover names -- there is no world-level entry to declare."""
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    components = []
    for i in range(movers):
        components.append(
            {
                "spawn_model": {
                    "model": str(crate),
                    "prefix": f"m{i}_",
                    "pos": [-2.0, float(i), 0.25],
                    "mocap": True,
                },
                "name": f"mover{i}",
                "components": [
                    {
                        "navigator": {
                            "speed": 0.5,
                            "goals": [[2.0, float(i)]],
                            # With `yields=False` the FIRST mover opts out and the rest still name
                            # the model -- a world where nobody names one has no model at all, which
                            # is a different case and has its own test.
                            "avoidance": {
                                "stop": False,
                                **(
                                    {"steer": "none"}
                                    if (not yields and i == 0) or avoidance is None
                                    else {
                                        "steer": avoidance["model"],
                                        "params": {
                                            k: v for k, v in avoidance.items() if k != "model"
                                        },
                                    }
                                ),
                            },
                        }
                    }
                ],
            }
        )
    if subject:
        # A robot with no navigator: the externally controlled subject.
        components.append(
            {
                "spawn_model": {"model": str(crate), "prefix": "s_", "pos": [0.0, 0.0, 0.25]},
                "name": "subject",
            }
        )
    return load_config_from_dict(
        {"sim": {"pacing": "asap", "world": str(ROOM)}, "components": components}, base_dir=tmp_path
    )


def _stub(engine):
    """The stub model itself, behind the service that owns the once-per-step solve."""
    service = engine.ctx.blackboard.get("nav:avoidance")
    return service.model if service is not None else None


def _run(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


# -- resolution ----------------------------------------------------------------------------------
def test_a_model_resolves_by_file_path():
    assert issubclass(resolve_model(STUB), AvoidanceModel)


def test_a_model_that_is_not_one_is_refused():
    with pytest.raises(RegistryError, match="AvoidanceModel subclass"):
        resolve_model(f"{HERE / 'stub_avoidance.py'}:NotAModel")


def test_an_unknown_model_lists_what_is_available():
    with pytest.raises(RegistryError, match="Available:"):
        resolve_model("magic")


def test_orca_resolves_without_rvo2_and_fails_only_when_used():
    """The optional extra is imported lazily, in ``configure``, not at module import.

    That is deliberate and it is what this pins: ``roqsim plugins`` and a world *listing* orca must
    work on a machine with no compiler, and the failure must arrive when something actually tries to
    run it -- naming the extra to install, rather than as a bare ImportError from inside rvo2.
    """
    from roqsim_nav.avoidance import available

    assert "orca" in available()
    cls = resolve_model("orca")  # resolvable regardless
    if importlib.util.find_spec("rvo2") is None:
        with pytest.raises(ImportError, match="roqsim_nav\\[avoidance\\]"):
            cls().configure(None, {})


# -- the ordering contract -------------------------------------------------------------------------
def test_solve_runs_once_per_step_and_before_any_result(tmp_path):
    """The property that makes plugin order in a world irrelevant."""
    engine = Engine(_world(tmp_path, movers=2, avoidance={"model": STUB}))
    engine.setup()
    engine.reset()
    try:
        stub = _stub(engine)
        stub.calls.clear()
        for _ in range(200):
            engine.step()
        assert stub.solves > 0
        # Between any two solves, no result may precede the solve that serves it.
        chunks, current = [], []
        for call in stub.calls:
            if call == "solve":
                chunks.append(current)
                current = []
            else:
                current.append(call)
        assert all(
            "result" not in chunk[: chunk.index("submit")] if "submit" in chunk else True
            for chunk in chunks
            if chunk
        )
    finally:
        engine.shutdown()


def test_a_mover_that_does_not_yield_still_joins_the_model(tmp_path):
    """Opting out of yielding is not opting out of existing.

    A mover the others cannot see is one they drive into. "One stops and the other goes around it"
    needs the stopping one to be *in* the model as something that must be gone around -- which is
    exactly what broke when a non-yielding mover skipped joining: two robots meeting head-on bumped,
    because the one with avoidance could not see the one without.
    """
    engine = Engine(_world(tmp_path, movers=2, avoidance={"model": STUB}))
    engine.setup()
    engine.reset()
    try:
        navigators = [p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin"]
        engine.step()
        stub = _stub(engine)
        assert len(stub.agents) == len(navigators), "not every mover registered"
        assert all(n._agent != NO_AGENT for n in navigators)
    finally:
        engine.shutdown()


# -- semantics -----------------------------------------------------------------------------------
def test_a_yielding_agent_executes_what_the_model_returned(tmp_path):
    engine = Engine(_world(tmp_path, avoidance={"model": STUB}, yields=True))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 2.0)
        stub = _stub(engine)
        assert stub.agents[0]["yields"] is True
        model = engine.ctx.model
        body = engine.ctx.entities.get("mover0").body
        mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
        # The stub deflects every yielding agent in +y; the mover must have drifted off its line.
        assert abs(float(engine.ctx.data.mocap_pos[mid][1])) > 0.01
    finally:
        engine.shutdown()


def test_a_non_yielding_agent_gets_exactly_what_it_asked_for(tmp_path):
    """`avoidance: none` means this model may never move it -- not "deflected slightly less".

    Two movers, because one that yields has to exist for the model to be created at all: a world
    where nobody asks for avoidance has no model, which is a different case (below).
    """
    engine = Engine(_world(tmp_path, movers=2, avoidance={"model": STUB}, yields=False))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 2.0)
        stub = _stub(engine)
        assert stub.agents[0]["yields"] is False
        model = engine.ctx.model
        body = engine.ctx.entities.get("mover0").body
        mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
        assert abs(float(engine.ctx.data.mocap_pos[mid][1])) < 1e-9, "the model moved it anyway"
    finally:
        engine.shutdown()


def test_the_model_is_given_the_planners_own_wall_polygons(tmp_path):
    """So the global and local layers cannot disagree about where a wall is."""
    engine = Engine(_world(tmp_path, avoidance={"model": STUB}))
    engine.setup()
    try:
        assert _stub(engine).statics, "no static geometry reached the model"
    finally:
        engine.shutdown()


def test_ground_truth_state_is_submitted_not_the_last_command(tmp_path):
    engine = Engine(_world(tmp_path, avoidance={"model": STUB}))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 1.0)
        model = engine.ctx.model
        body = engine.ctx.entities.get("mover0").body
        mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
        submitted = _stub(engine).submissions[0]["pos"]
        assert submitted == pytest.approx(engine.ctx.data.mocap_pos[mid][:2], abs=0.06)
    finally:
        engine.shutdown()


def test_reset_clears_the_models_state(tmp_path):
    engine = Engine(_world(tmp_path, movers=2, avoidance={"model": STUB}))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 1.0)
        before = _stub(engine).resets
        engine.reset()
        assert _stub(engine).resets == before + 1
        # Agents and walls survive an episode; only their velocities do not.
        assert _stub(engine).agents and _stub(engine).statics
    finally:
        engine.shutdown()


# -- configuration ---------------------------------------------------------------------------------
def test_a_param_outside_the_schema_is_refused(tmp_path):
    """A key left behind after switching models must not be silently ignored."""
    with pytest.raises(PluginError, match="does not accept"):
        Engine(_world(tmp_path, avoidance={"model": STUB, "time_horizon": 3.0}))


def test_a_param_in_the_schema_reaches_the_model(tmp_path):
    engine = Engine(_world(tmp_path, avoidance={"model": STUB, "gain": 2.5}))
    engine.setup()
    try:
        assert _stub(engine).configured_with == {"gain": 2.5}
    finally:
        engine.shutdown()


# -- no model at all ---------------------------------------------------------------------------------
def test_a_world_where_nobody_asks_for_avoidance_creates_no_model(tmp_path):
    """No model is built at all, and every mover simply executes what it wanted.

    Worth pinning as its own case: the cost of avoidance should be paid by worlds that use it, and a
    world of movers that only ever stop should not be carrying a solver.
    """
    engine = Engine(_world(tmp_path, avoidance=None, yields=False))
    engine.setup()
    engine.reset()
    try:
        navigator = next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")
        assert engine.ctx.blackboard.get("nav:avoidance") is None
        assert navigator._agent == NO_AGENT
        _run(engine, 6.0)
        model = engine.ctx.model
        body = engine.ctx.entities.get("mover0").body
        mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
        assert float(engine.ctx.data.mocap_pos[mid][0]) > 0.0, "it never set off"
    finally:
        engine.shutdown()


def test_the_null_model_is_the_identity():
    null = NullAvoidance()
    aid = null.add_agent("x", radius=0.3, max_speed=1.0, yields=True, params={})
    null.submit(aid, (0.0, 0.0), (0.0, 0.0), (0.7, -0.2))
    null.solve(0.002)
    assert null.result(aid) == pytest.approx([0.7, -0.2])
