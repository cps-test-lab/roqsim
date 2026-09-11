# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The Cartesian controller runs at the rate it is configured for, whatever the physics clock does.

The simulator's clock is a sum of timesteps in floating point, so a tick due at exactly k periods
can read a hair early. A gate that compares strictly and re-anchors each period on the current time
turns every such hair into a whole missed physics step: at 500 Hz on a 1 ms step it ticked at
472 Hz, while integrating each tick as a full period -- a systematic shortfall in commanded motion
that reported nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from roqsim_manipulation.plugins.cartesian_admittance import CartesianAdmittancePlugin


def _ticks(
    rate_hz: float, timestep: float, steps: int, inactive: tuple[int, int] | None = None
) -> list[int]:
    """The physics steps at which the law runs, with the sim clock advanced as MuJoCo advances it."""
    plugin = CartesianAdmittancePlugin.__new__(CartesianAdmittancePlugin)
    plugin.rate_hz = rate_hz
    plugin.law = "admittance"
    plugin.axes = np.ones(6)
    plugin._active = True
    plugin._next_t = 0.0
    plugin._clamp = lambda t: t
    ticks: list[int] = []
    ctx = SimpleNamespace(
        sim_time=0.0,
        manual_control=False,
        model=SimpleNamespace(opt=SimpleNamespace(timestep=timestep)),
    )
    step = 0
    plugin._admittance_twist = lambda dt: ticks.append(step) or np.zeros(6)
    plugin._apply = lambda ctx, twist, dt: None
    for step in range(steps):
        plugin._active = not (inactive and inactive[0] <= step < inactive[1])
        plugin.pre_step(ctx)
        ctx.sim_time += timestep
    return ticks


def test_a_rate_the_timestep_divides_ticks_on_every_period():
    ticks = _ticks(500.0, 0.001, 1200)
    assert len(ticks) == 600
    assert set(np.diff(ticks)) == {2}, "every tick exactly two physics steps after the last"


@pytest.mark.parametrize("rate_hz", [100.0, 250.0, 500.0])
def test_the_configured_rate_is_the_rate_delivered(rate_hz):
    seconds = 3.0
    ticks = _ticks(rate_hz, 0.001, int(seconds / 0.001))
    assert len(ticks) == pytest.approx(rate_hz * seconds, abs=1)


def test_a_rate_the_timestep_does_not_divide_holds_on_average():
    """300 Hz on a 1 ms step cannot tick evenly; it must still tick 300 times a second."""
    ticks = _ticks(300.0, 0.001, 3000)
    assert len(ticks) == pytest.approx(900, abs=1)
    assert set(np.diff(ticks)) <= {3, 4}


def test_resuming_after_a_pause_does_not_catch_up_in_a_burst():
    """A controller switched off for a while resumes at its rate, not with the ticks it missed."""
    ticks = _ticks(500.0, 0.001, 1000, inactive=(100, 600))
    after = [t for t in ticks if t >= 600]
    assert set(np.diff(after)) == {2}
