"""The runtime goal interface: sending routes, and telling *your* arrival from someone else's.

The sequence number is what these are really about. Every other property follows from getting it
right, and the failure it prevents is a silent one: a caller that watched a bare "finished" flag
would see the navigator's *previous* route already complete and report an arrival that had not
happened yet.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

DATA = Path(__file__).parent / "data"
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""


def _engine(tmp_path, **nav):
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    navigator = {"speed": 1.0, "avoidance": {"stop": False}}
    navigator.update(nav)
    return Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap"},
                "components": [
                    {
                        "spawn_model": {
                            "model": str(crate),
                            "pos": [0.0, 0.0, 0.25],
                            "mocap": True,
                        },
                        "name": "cart",
                        "components": [{"navigator": navigator}],
                    }
                ],
            },
            base_dir=tmp_path,
        )
    )


def _handle(engine):
    return engine.ctx.blackboard.get("nav:cart:handle")


def _xy(engine):
    model = engine.ctx.model
    body = engine.ctx.entities.get("cart").body
    mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
    return engine.ctx.data.mocap_pos[mid][:2].copy()


def _run(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def _run_until_finished(engine, seq, timeout=40.0):
    handle = _handle(engine)
    deadline = engine.ctx.sim_time + timeout
    while engine.ctx.sim_time < deadline:
        _run(engine, 0.1)
        applied, finished, _, _ = handle.status()
        if applied > seq:
            return "preempted"
        if applied == seq and finished:
            return "finished"
    return "timeout"


@pytest.fixture
def sim(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


def test_the_handle_is_published_for_the_entity(sim):
    handle = _handle(sim)
    assert handle is not None
    assert handle.name == "cart"  # the entity, not the component's address


def test_a_route_is_driven_and_reports_its_own_completion(sim):
    handle = _handle(sim)
    seq = handle.send_goals([(3.0, 0.0)])
    assert seq > 0
    assert _run_until_finished(sim, seq) == "finished"
    assert _xy(sim) == pytest.approx([3.0, 0.0], abs=0.3)


def test_sequence_numbers_rise_and_a_newer_route_preempts_an_older_one(sim):
    handle = _handle(sim)
    first = handle.send_goals([(5.0, 0.0)])
    _run(sim, 0.5)
    second = handle.send_goals([(0.0, 5.0)])
    assert second > first

    # The first caller must learn that it was preempted rather than waiting for an arrival that will
    # never come -- which is exactly what a bare "finished" flag could not tell it.
    assert _run_until_finished(sim, first) == "preempted"
    assert _run_until_finished(sim, second) == "finished"
    assert _xy(sim) == pytest.approx([0.0, 5.0], abs=0.4)


def test_a_stale_completion_is_never_reported_as_a_new_ones(sim):
    """The failure the sequence number exists to prevent.

    After the first route finishes, the navigator is idle and "finished" is true. A caller that
    queued a second route and watched only that flag would conclude it had already arrived.
    """
    handle = _handle(sim)
    first = handle.send_goals([(2.0, 0.0)])
    assert _run_until_finished(sim, first) == "finished"

    second = handle.send_goals([(-2.0, 0.0)])
    applied, finished, _, _ = handle.status()
    assert not (applied == second and finished), (
        "reported arrival before the route was even applied"
    )
    assert _run_until_finished(sim, second) == "finished"
    assert _xy(sim) == pytest.approx([-2.0, 0.0], abs=0.3)


def test_multiple_goals_are_driven_in_order(sim):
    handle = _handle(sim)
    seq = handle.send_goals([(2.0, 0.0), (2.0, 2.0)])
    assert _run_until_finished(sim, seq) == "finished"
    assert _xy(sim) == pytest.approx([2.0, 2.0], abs=0.4)


def test_cancel_stops_it_where_it_stands(sim):
    handle = _handle(sim)
    handle.send_goals([(8.0, 0.0)])
    _run(sim, 1.0)
    moving = _xy(sim)
    handle.cancel()
    _run(sim, 2.0)
    assert _xy(sim) == pytest.approx(moving, abs=0.15), "it kept going after being cancelled"


def test_pose_is_a_latched_snapshot_from_the_last_tick(sim):
    """`NavHandle.pose` is called from other threads, so it must not read ``data`` live.

    It therefore lags the body by at most one nav period -- 50 ms at 20 Hz, so under a mover's
    per-tick travel. The test pins that bound rather than exact equality, because exact equality
    would only hold for an embodiment whose pose is written and would quietly fail for a driven one.
    """
    handle = _handle(sim)
    seq = handle.send_goals([(2.0, 0.0)])
    assert seq > 0
    _run(sim, 1.0)
    x, y, _yaw = handle.pose()
    speed, period = 1.0, 1.0 / 20.0
    assert (x, y) == pytest.approx(_xy(sim), abs=speed * period + 1e-6)


def test_progress_shrinks_as_it_drives(sim):
    handle = _handle(sim)
    handle.send_goals([(6.0, 0.0)])
    _run(sim, 0.5)
    _, _, _, far = handle.status()
    _run(sim, 3.0)
    _, _, _, near = handle.status()
    assert near < far, "distance remaining did not fall while it drove"


def test_an_empty_route_is_refused(sim):
    with pytest.raises(ValueError, match="at least one goal"):
        _handle(sim).send_goals([])


def test_reset_clears_a_commanded_route(tmp_path):
    """Episode N must not inherit the route episode N-1 was given."""
    engine = _engine(tmp_path, goals=[[1.0, 0.0]])
    engine.setup()
    engine.reset()
    try:
        handle = _handle(engine)
        seq = handle.send_goals([(4.0, 0.0)])
        assert _run_until_finished(engine, seq) == "finished"

        engine.reset()
        applied, finished, _, _ = handle.status()
        assert applied == 0 and not finished, "the commanded route survived the reset"
        _run(engine, 6.0)
        # It runs its CONFIGURED route again, not the commanded one.
        assert _xy(engine) == pytest.approx([1.0, 0.0], abs=0.3)
    finally:
        engine.shutdown()
