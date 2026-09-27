"""The walker controller ticks at its ``update_hz`` on every period the timestep divides."""

from __future__ import annotations

from roqsim_walker.nav.controller import WalkerController


class _Ticked(Exception):
    """Raised by the first read past the gate, so a tick stops there."""


class _Data:
    @property
    def time(self):
        raise _Ticked


def test_a_rate_the_timestep_divides_ticks_on_every_period():
    """20 Hz on a 5 ms step is every tenth step: the summed timesteps reach the period a hair short
    of it, and the gate must still take that step rather than the next."""
    controller = WalkerController.__new__(WalkerController)
    controller._accum = 0.0
    controller._period = 1.0 / 20.0
    controller.data = _Data()
    ticks = []
    for step in range(1, 401):
        try:
            controller.update(0.005)
        except _Ticked:
            ticks.append(step)
    gaps = {b - a for a, b in zip(ticks, ticks[1:], strict=False)}
    assert gaps == {10}, f"tick spacing in steps: {sorted(gaps)}"
