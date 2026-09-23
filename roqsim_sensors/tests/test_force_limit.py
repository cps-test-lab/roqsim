# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Stopping when the contact exceeds what the task allows.

A limit that never trips reads, in a results table, exactly like a trial that stayed inside it. So
the interesting cases here are the ones where it must NOT report safe: a threshold nobody set, a
trip that a later quiet moment erases, and a trip carried into the next repetition.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_sensors.plugins.force_limit import ForceLimitPlugin

from roqsim.config import load_config_from_dict
from roqsim.controllers import ACTIVE, registry_for
from roqsim.engine import Engine


class _Wrench:
    """A sensor under the test's control, so the threshold is what is being tested."""

    def __init__(self, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)):
        self.force, self.torque = np.array(force, float), np.array(torque, float)

    def read(self):
        return self.force, self.torque


class _Ctx:
    def __init__(self, sim_time=1.0):
        self.sim_time = sim_time
        self.stop_requested = False
        self.stop_reason = ""

    def request_stop(self, reason=""):
        self.stop_requested, self.stop_reason = True, reason


def _monitor(**config):
    plugin = ForceLimitPlugin.__new__(ForceLimitPlugin)
    plugin.max_force = float(config.get("max_force", 40.0))
    plugin.max_torque = float(config.get("max_torque", 0.0))
    plugin.settle_s = float(config.get("settle_s", 0.0))
    plugin.latch = bool(config.get("latch", True))
    plugin.stop_run = bool(config.get("stop_run", True))
    plugin.release_controllers = False
    plugin.reports_as = config.get("reports_as", "protective_stop")
    plugin._ft = config.get("ft", _Wrench())
    from roqsim_sensors.plugins.force_limit import LimitReport

    plugin._report = LimitReport()
    return plugin


# -- tripping ------------------------------------------------------------------------------------


def test_a_wrench_inside_the_limit_reports_nothing():
    m = _monitor(ft=_Wrench(force=(0.0, 0.0, -8.0)))
    m.post_step(_Ctx())
    assert m.read_state().tripped is False
    assert m.read_state().at_time == -1.0


def test_the_magnitude_trips_it_not_one_axis():
    """A contact off the tool axis spends its force across all three; watching z alone would miss
    a jam that a real stop catches."""
    m = _monitor(max_force=10.0, ft=_Wrench(force=(7.0, 7.0, 0.0)))  # |f| ~ 9.9
    m.post_step(_Ctx())
    assert m.read_state().tripped is False

    m = _monitor(max_force=10.0, ft=_Wrench(force=(8.0, 8.0, 0.0)))  # |f| ~ 11.3
    m.post_step(_Ctx())
    assert m.read_state().tripped is True


def test_a_trip_records_when_and_why_and_stops_the_run():
    ctx = _Ctx(sim_time=4.25)
    m = _monitor(max_force=40.0, ft=_Wrench(force=(0.0, 0.0, -55.0)))
    m.post_step(ctx)

    report = m.read_state()
    assert report.tripped and report.at_time == 4.25
    assert "55" in report.reason and "40" in report.reason
    assert ctx.stop_requested and report.reason == ctx.stop_reason


def test_the_robots_own_word_for_it_comes_from_config():
    """The capability is generic; the vocabulary is the robot's."""
    m = _monitor(reports_as="safety_stop", ft=_Wrench(force=(0.0, 0.0, -99.0)))
    m.post_step(_Ctx())
    assert m.read_state().reason.startswith("safety_stop:")


def test_a_torque_limit_trips_on_its_own():
    m = _monitor(max_force=0.0, max_torque=2.0, ft=_Wrench(torque=(0.0, 0.0, 5.0)))
    m.post_step(_Ctx())
    assert m.read_state().tripped


# -- what must not be forgotten ------------------------------------------------------------------


def test_a_latched_trip_survives_the_contact_going_away():
    """A trial is failed, not un-failed. The peg that jammed and then slipped free still jammed."""
    sensor = _Wrench(force=(0.0, 0.0, -55.0))
    m = _monitor(ft=sensor)
    m.post_step(_Ctx(sim_time=2.0))
    sensor.force = np.zeros(3)
    m.post_step(_Ctx(sim_time=3.0))

    assert m.read_state().tripped is True
    assert m.read_state().at_time == 2.0, "the FIRST trip is the one that matters"


def test_an_unlatched_monitor_follows_the_contact():
    sensor = _Wrench(force=(0.0, 0.0, -55.0))
    m = _monitor(latch=False, ft=sensor)
    m.post_step(_Ctx())
    assert m.read_state().tripped
    sensor.force = np.zeros(3)
    m.post_step(_Ctx())
    assert m.read_state().tripped is False


def test_the_settle_window_ignores_the_reset_transient():
    """At reset the arm has not yet been commanded, and the wrench that produces is not contact."""
    m = _monitor(settle_s=1.0, ft=_Wrench(force=(0.0, 0.0, -80.0)))
    m.post_step(_Ctx(sim_time=0.5))
    assert m.read_state().tripped is False
    m.post_step(_Ctx(sim_time=1.5))
    assert m.read_state().tripped is True


def test_a_trip_does_not_survive_a_reset():
    """An offset carried into the next episode is a measurement of the previous one."""
    m = _monitor(ft=_Wrench(force=(0.0, 0.0, -99.0)))
    m.post_step(_Ctx())
    assert m.read_state().tripped
    m.on_reset(_Ctx())
    assert m.read_state().tripped is False
    assert m.read_state().at_time == -1.0


def test_a_monitor_with_no_threshold_at_all_is_refused():
    """It would report "nothing exceeded" forever, which in a results table is indistinguishable
    from a trial that stayed within its limits."""
    errors = ForceLimitPlugin({}, entity="arm").validate_config({"max_force": 0.0})
    assert any("watches nothing" in e for e in errors)


# -- in a world ----------------------------------------------------------------------------------


def _world(tmp_path, limit):
    return load_config_from_dict(
        {
            "sim": {"timestep": 0.001},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_", "namespace": "ur5e"},
                    "name": "ur5e",
                    "components": [
                        {"arm_controller": {}},
                        {"force_torque": {"site": "fts_site", "frame": "world"}, "name": "ft"},
                        # The module path rather than the entry-point name: the entry point
                        # resolves only once the package is installed, and this test should not
                        # depend on whether the checkout happens to be.
                        {
                            "roqsim_sensors.plugins.force_limit:ForceLimitPlugin": dict(limit),
                            "name": "safety",
                        },
                    ],
                }
            ],
        },
        base_dir=tmp_path,
    )


def test_it_publishes_an_endpoint_and_a_handle(tmp_path):
    engine = Engine(_world(tmp_path, {"ft": "ft", "max_force": 40.0}))
    engine.setup()
    names = {e.name for e in engine.ctx.interface.all()}
    assert "force_limit" in names
    assert engine.ctx.blackboard.get("force_limit:ur5e.safety") is not None


def test_tripping_releases_whatever_was_driving_the_arm(tmp_path):
    """What a real stop does to the MOTION. The controllers stay loaded, exactly as they do on the
    arm -- this is an effect of the stop, never the mechanism it is reported through.
    """
    engine = Engine(_world(tmp_path, {"ft": "ft", "max_force": 0.001}))
    engine.setup()
    engine.reset()
    registry = registry_for(engine.ctx)
    assert registry.get("arm_controller", "ur5e").state == ACTIVE

    for _ in range(5):
        engine.step()

    assert registry.get("arm_controller", "ur5e").state != ACTIVE
    assert registry.get("arm_controller", "ur5e") is not None, "still loaded, only deactivated"
    assert engine.ctx.stop_requested


def test_a_sensor_that_is_not_there_is_refused_naming_the_key(tmp_path):
    engine = Engine(_world(tmp_path, {"ft": "nonexistent", "max_force": 40.0}))
    with pytest.raises(RuntimeError, match="ft:nonexistent"):
        engine.setup()
