"""``StepSimulation`` and ``ResetSimulation`` answer once what they asked for has happened.

``StepSimulation.srv`` returns "only once stepping is complete", and a caller reads the world right
after either call. An answer given when the request was merely queued lets it read the state from
before the steps or the reset, under ``RESULT_OK``. A driver here is the loop ``roqsim sim`` runs:
it consults the run control, steps, and resets the plugins.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from simulation_interfaces.msg import Result

from roqsim import control as ctl
from roqsim_ros_bridge import sim_interfaces
from roqsim_ros_bridge.sim_interfaces import SimInterfacesPlugin

DT = 0.01


class _Driver:
    """The standalone driver's loop over a RunControl, with a step that only advances time."""

    def __init__(self, ctx, plugin, step_wall_s=0.001):
        self.ctx, self.plugin, self.step_wall_s = ctx, plugin, step_wall_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        control = self.ctx.control
        while not self._stop.is_set():
            if control.take_reset():
                self.ctx.sim_time = 0.0
                self.plugin.on_reset(self.ctx)
            if control.should_step():
                time.sleep(self.step_wall_s)
                self.ctx.sim_time = round(self.ctx.sim_time + DT, 9)
            else:
                time.sleep(0.001)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)


def _plugin(state=ctl.PAUSED, sim_time=1.0):
    plugin = SimInterfacesPlugin()
    plugin._ctx = SimpleNamespace(control=ctl.RunControl(state), sim_time=sim_time)
    return plugin


def _resp():
    return SimpleNamespace(result=None)


def test_a_step_answers_once_every_requested_step_has_run():
    plugin = _plugin()
    with _Driver(plugin._ctx, plugin, step_wall_s=0.005):
        resp = plugin._step(SimpleNamespace(steps=20), _resp())
        reached = plugin._ctx.sim_time
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    assert reached == pytest.approx(1.0 + 20 * DT)


def test_zero_steps_is_refused_rather_than_taken_as_one():
    plugin = _plugin()
    resp = plugin._step(SimpleNamespace(steps=0), _resp())
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "at least 1" in resp.result.error_message


def test_a_step_the_simulation_never_takes_fails(monkeypatch):
    monkeypatch.setattr(sim_interfaces, "DEFAULT_TIMEOUT_S", 0.05)
    plugin = _plugin()  # no driver: nothing steps
    resp = plugin._step(SimpleNamespace(steps=3), _resp())
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "took no step" in resp.result.error_message


def test_a_step_cut_short_by_leaving_the_pause_fails():
    plugin = _plugin()
    control = plugin._ctx.control
    threading.Timer(0.05, control.set_state, args=(ctl.STOPPED,)).start()
    resp = plugin._step(SimpleNamespace(steps=1000), _resp())
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "left the paused state" in resp.result.error_message


def test_a_reset_answers_once_the_world_has_reset():
    plugin = _plugin(state=ctl.PLAYING, sim_time=5.0)
    with _Driver(plugin._ctx, plugin):
        resp = plugin._reset(SimpleNamespace(scope=0), _resp())
        # Read before the driver can step again: the reset is what the answer reports.
        assert plugin._ctx.sim_time < 1.0
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message


def test_a_reset_no_driver_performs_fails(monkeypatch):
    monkeypatch.setattr(sim_interfaces, "DEFAULT_TIMEOUT_S", 0.05)
    plugin = _plugin(state=ctl.PLAYING)
    resp = plugin._reset(SimpleNamespace(scope=0), _resp())
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "did not reset" in resp.result.error_message
    assert plugin._reset_waiters == []
