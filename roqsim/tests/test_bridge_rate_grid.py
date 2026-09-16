"""An output endpoint's publish rate lands on the physics grid, exactly, and says so in proportion.

A rate gate is tested once per physics step, so the rates a world can hold are exactly
``physics_rate / k`` for integer ``k >= 1``. The property that matters is that a run *reports* the
rate it published at: the gate is put on the grid when the endpoint is bound, the move is announced
by how far it went, and both numbers -- requested and realised -- reach the run's record, where a
reader who never opens a log can see them.
"""

from __future__ import annotations

import logging

import mujoco
import pytest
from test_bridge import FakeBridge

from roqsim.capture import SNAP_NOTABLE, SNAP_QUIET, StateRecorder, snap_fps
from roqsim.context import Endpoint, SimContext
from roqsim.recording import open_recording

#: MuJoCo's own default timestep, which stands unless a world sets one: 500 steps per simulated
#: second, so 25 Hz is exact (k=20) and 30 Hz is not (500/17).
_DT = 0.002

_XML = """
<mujoco>
  <option timestep="%s"/>
  <worldbody><geom type="plane" size="1 1 .1"/></worldbody>
</mujoco>
"""


def _bound(rate_hz, *, dt=_DT, config=None, name="scan"):
    """One out endpoint at ``rate_hz``, bound by a bridge against a compiled world."""
    ctx = SimContext(config={})
    ctx.model = mujoco.MjModel.from_xml_string(_XML % dt)
    ctx.data = mujoco.MjData(ctx.model)
    ctx.interface.add(
        Endpoint(
            name=name,
            direction="out",
            owner="robot",
            namespace="robot",
            read=lambda: (0.0,),
            rate_hz=rate_hz,
            backend={"fake": {}},
        )
    )
    bridge = FakeBridge(config or {})
    bridge.configure(ctx)
    return ctx, bridge


def _gate(bridge):
    return bridge._outputs[0].gate


# -- the grid --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("rate", "every"), [(500, 1), (250, 2), (100, 5), (25, 20), (5, 100)])
def test_a_rate_already_on_the_grid_does_not_move(rate, every, caplog):
    """The common case: a rate that is a whole number of steps is served exactly, and silently."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(rate)
    assert _gate(bridge).rate_hz == float(rate)
    assert _gate(bridge).every == every
    assert caplog.records == []


def test_a_rate_off_the_grid_snaps_to_the_nearest_achievable_one():
    """30 Hz needs 16.67 steps, so it is served every 17 -- 500/17, and the gate says so."""
    _, bridge = _bound(30)
    assert _gate(bridge).every == 17
    assert _gate(bridge).rate_hz == pytest.approx(500 / 17)


def test_the_nearest_rate_may_be_the_faster_neighbour():
    """Nearest, not slower: 40 Hz is 12.5 steps and 12 is as close as 13, so the gate takes 12.

    An ungated ``due()`` can only ever reach the slower neighbour, because it fires at the first step
    at or past the period. The snap is what puts the realised rate on whichever side is closer.
    """
    _, bridge = _bound(40)
    assert _gate(bridge).every == 12
    assert _gate(bridge).rate_hz == pytest.approx(500 / 12)
    assert _gate(bridge).rate_hz > 40


def test_a_snapped_gate_fires_exactly_every_k_steps():
    """The whole point: the realised spacing is a constant, and it is the one that was recorded."""
    _, bridge = _bound(30)
    gate = _gate(bridge)
    fired = [i for i in range(500) if gate.due(i * _DT)]
    gaps = {b - a for a, b in zip(fired, fired[1:], strict=False)}
    assert gaps == {gate.every}
    assert len(fired) == pytest.approx(gate.rate_hz, abs=1)  # a simulated second of publications


def test_a_rate_faster_than_the_world_steps_is_served_every_step(caplog):
    """Clamped, not refused: a bound endpoint has no flag to hand back, and the step rate exists."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(800)
    assert _gate(bridge).every == 1
    assert _gate(bridge).rate_hz == pytest.approx(500)
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_an_event_driven_endpoint_stays_ungated(caplog):
    """``rate_hz = 0`` means every step, which is on the grid by construction -- nothing to say."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(0.0)
    assert _gate(bridge).rate_hz == 0.0
    assert _gate(bridge).every is None
    assert _gate(bridge).due(0.0) and _gate(bridge).due(_DT)
    assert caplog.records == []


def test_a_rates_override_is_snapped_like_any_other_rate():
    """The override is a request too, so it lands on the grid rather than beside it."""
    _, bridge = _bound(10, config={"rates": {"scan": 30.0}})
    assert _gate(bridge).every == 17


def test_an_unbound_world_keeps_the_requested_rate():
    """No model, no grid: an embedding driver that binds before compile gets the gate it asked for."""
    ctx = SimContext(config={})
    ctx.interface.add(
        Endpoint(
            name="scan",
            direction="out",
            owner="robot",
            read=lambda: (0.0,),
            rate_hz=30.0,
            backend={"fake": {}},
        )
    )
    bridge = FakeBridge({})
    bridge.configure(ctx)
    assert _gate(bridge).rate_hz == 30.0
    assert _gate(bridge).every is None
    assert ctx.endpoint_rates == []


# -- the report bands ------------------------------------------------------------------------------
#
# The same bands a capture rate is announced on, so one move is as loud whether it hits a recording
# or a topic. Each case pins the deviation first, so a band that moves is visible as the band moving
# rather than as a message that changed.


def test_a_snap_below_the_quiet_band_is_only_a_debug_line(caplog):
    """24.98 -> 25 is 0.08% -- the caller got what they asked for."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(24.98)
    assert abs(_gate(bridge).rate_hz - 24.98) / 24.98 < SNAP_QUIET
    assert [r.levelno for r in caplog.records] == [logging.DEBUG]


def test_a_snap_just_over_the_quiet_band_is_a_note(caplog):
    """24.975 -> 25 is 0.10% -- just over, so it is stated rather than only logged for a debugger."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(24.975)
    assert SNAP_QUIET <= abs(_gate(bridge).rate_hz - 24.975) / 24.975 < SNAP_NOTABLE
    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_a_snap_just_under_the_notable_band_is_still_a_note(caplog):
    """24.8 -> 25 is 0.81%, under 1%: worth stating, not worth alarming about."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(24.8)
    assert SNAP_QUIET <= abs(_gate(bridge).rate_hz - 24.8) / 24.8 < SNAP_NOTABLE
    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_a_snap_at_the_notable_band_warns_and_names_the_neighbours(caplog):
    """24.75 -> 25 is 1.01%, at the band: a warning, with rates that can be asked for instead."""
    with caplog.at_level(logging.DEBUG):
        _, bridge = _bound(24.75)
    assert abs(_gate(bridge).rate_hz - 24.75) / 24.75 >= SNAP_NOTABLE
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert "k=19" in message and "k=21" in message  # both sides, nearest first


def test_a_big_snap_names_the_endpoint_and_the_exact_rational(caplog):
    with caplog.at_level(logging.DEBUG):
        _bound(30)
    message = caplog.records[0].getMessage()
    assert "'scan'" in message
    assert "500/17" in message and "every 17 steps" in message
    assert "30 Hz" in message  # what was asked for, so the two can be compared in one line


# -- the run's record ------------------------------------------------------------------------------


def test_the_record_carries_the_realised_rate_beside_the_requested_one():
    ctx, _ = _bound(30)
    assert ctx.endpoint_rates == [
        {
            "name": "scan",
            "owner": "robot",
            "namespace": "robot",
            "backend": "fake",
            "requested_hz": 30.0,
            "realised_hz": pytest.approx(500 / 17),
            "every_steps": 17,
        }
    ]


def test_an_ungated_endpoint_is_recorded_at_the_world_s_step_rate():
    """An every-step endpoint has a rate too, and a reader needs it beside the others."""
    ctx, _ = _bound(0.0)
    assert ctx.endpoint_rates[0]["requested_hz"] == 0.0
    assert ctx.endpoint_rates[0]["realised_hz"] == pytest.approx(500)
    assert ctx.endpoint_rates[0]["every_steps"] == 1


def test_the_realised_rate_reaches_a_recording(tmp_path):
    """The half that reaches a reader who never opens a container's log."""
    ctx, _ = _bound(30)
    recorder = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, _DT), world="w.yaml")
    for _ in range(50):
        mujoco.mj_step(ctx.model, ctx.data)
        recorder.sample(ctx)
    path = recorder.close()
    rates = open_recording(path).meta["endpoint_rates"]
    assert [(r["name"], r["requested_hz"], r["every_steps"]) for r in rates] == [("scan", 30.0, 17)]
    assert rates[0]["realised_hz"] == pytest.approx(500 / 17)


def test_a_run_with_no_bridge_records_no_rates(tmp_path):
    """A world with no transport has nothing to say here, and says nothing rather than failing."""
    ctx = SimContext(config={})
    ctx.model = mujoco.MjModel.from_xml_string(_XML % _DT)
    ctx.data = mujoco.MjData(ctx.model)
    recorder = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, _DT), world="w.yaml")
    for _ in range(50):
        mujoco.mj_step(ctx.model, ctx.data)
        recorder.sample(ctx)
    assert open_recording(recorder.close()).meta["endpoint_rates"] == []
