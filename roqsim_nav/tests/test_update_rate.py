"""The navigator ticks at its ``update_hz`` on every period the timestep divides."""

from __future__ import annotations

from types import SimpleNamespace

from roqsim_nav.plugins.navigator import NavigatorPlugin


def test_a_rate_the_timestep_divides_ticks_on_every_period():
    """20 Hz on a 5 ms step is every tenth step: the summed timesteps reach the period a hair short
    of it, and the gate must still take that step rather than the next.

    Not started, so a tick ends right after the gate, which is where it resets its accumulator.
    """
    plugin = NavigatorPlugin.__new__(NavigatorPlugin)
    plugin._plan_pending = False
    plugin._started = False
    plugin._accum = 0.0
    plugin._period = 1.0 / 20.0
    ctx = SimpleNamespace(dt=0.005)
    ticks = []
    for step in range(1, 401):
        plugin.pre_step(ctx)
        if plugin._accum == 0.0:
            ticks.append(step)
    gaps = {b - a for a, b in zip(ticks, ticks[1:], strict=False)}
    assert gaps == {10}, f"tick spacing in steps: {sorted(gaps)}"
