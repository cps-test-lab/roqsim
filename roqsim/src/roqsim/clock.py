"""Wall-clock pacing for the standalone driver.

Three modes, selected by the ``sim.pacing`` config or ``--pacing``:
  * ``realtime``      — one sim second per wall second (factor 1.0).
  * ``{factor: N}``   — N sim seconds per wall second (N>1 faster, N<1 slower).
  * ``asap``          — no sleeping; step as fast as possible.

Under scenario-execution the framework owns timing, so no pacer is used there.
"""

from __future__ import annotations

import time
from typing import Any

#: Report the shortfall only past this share of late steps. A run crosses it when it is
#: genuinely not holding its rate, not when a few steps ran long -- an occasional late step is
#: normal on any loaded machine and saying so every time would train the reader to skip the line.
#: 0.05 is well above what a healthy run produces (measured: 0 late steps in 20) and well below
#: what a failing one does (measured: every step late, 0.34x realtime).
SHORTFALL_REPORT_SHARE = 0.05


class Pacer:
    """Sleeps after each step so wall time tracks ``dt / factor`` (no-op in ``asap`` mode).

    **It also counts the steps it could not pace**, which is the only record that a run failed
    the timing it asked for. Falling behind is absorbed deliberately (see :meth:`wait`), so
    without a counter the run simply produces slower sim time and says nothing -- a campaign
    then has to reconstruct the shortfall from ``run.clock_map.csv`` afterwards, if anyone
    thinks to look. Observed on a lone run that held 0.34x realtime for five minutes and
    reported nothing before its job deadline killed it.
    """

    def __init__(self, dt: float, factor: float = 1.0, realtime: bool = True):
        self.dt = dt
        self.factor = factor
        self.realtime = realtime
        self._next: float | None = None
        #: Steps that were already late when they were due, i.e. where no sleep was possible.
        self.behind_steps = 0
        #: Steps paced normally. ``behind_steps / (behind_steps + paced_steps)`` is the share
        #: of the run that could not hold its requested rate.
        self.paced_steps = 0
        #: Wall seconds owed at the moment each late step was abandoned, summed. Divided by the
        #: run's wall span this is how far short of the requested pacing it fell.
        self.behind_seconds = 0.0

    @classmethod
    def from_config(cls, pacing: Any, dt: float) -> Pacer:
        if pacing == "asap":
            return cls(dt, realtime=False)
        if pacing in (None, "realtime"):
            return cls(dt, factor=1.0, realtime=True)
        if isinstance(pacing, dict) and "factor" in pacing:
            return cls(dt, factor=float(pacing["factor"]), realtime=True)
        raise ValueError(f"invalid pacing spec: {pacing!r}")

    def reset(self) -> None:
        """Drop the schedule. Counters survive: they describe the run, not the current series."""
        self._next = None

    def wait(self) -> None:
        """Block until the next step is due. No-op in ``asap`` mode."""
        if not self.realtime:
            return
        target_period = self.dt / self.factor
        now = time.perf_counter()
        if self._next is None:
            self._next = now + target_period
            return
        sleep = self._next - now
        if sleep > 0:
            time.sleep(sleep)
            self._next += target_period
            self.paced_steps += 1
        else:
            # Fell behind. The deficit is ABANDONED rather than made up: catching up would mean
            # running faster than the requested rate, which for a simulator paced against a
            # live stack is worse than being slow -- it would deliver sim time the stack never
            # asked for. So the schedule restarts from now, and the shortfall is counted
            # instead. Counting is what makes it visible; see the class docstring.
            self.behind_steps += 1
            self.behind_seconds += -sleep
            self._next = now + target_period

    def report_line(self) -> str | None:
        """One line for the run's log, or ``None`` when there is nothing worth saying.

        **Silence is the normal answer.** A run that held its rate says nothing at all, so the
        line's presence is itself the signal -- the same rule the sizing advice follows, and the
        reason it can be read as "something was wrong" rather than as routine output. Emitted
        once at the end of a run, never per step: at 0.01 s that would be a hundred lines a
        second, and the fact being reported is about the whole run anyway.
        """
        s = self.shortfall()
        if not s or s["late_share"] < SHORTFALL_REPORT_SHARE:
            return None
        return (
            f"pacing: held {s['achieved_factor']:.2f}x of the requested {s['requested_factor']:.2f}x"
            f" -- {s['late_steps']} of {s['total_steps']} steps ({s['late_share'] * 100:.0f}%)"
            f" had no time left to sleep, {s['behind_seconds']:.1f} s of pacing never made up."
            " Sim time therefore advanced slower than wall time; a run measured against a wall"
            " clock (a job deadline, a scenario timeout) sees less of the trial than it asked for."
        )

    def shortfall(self) -> dict:
        """What the pacing actually achieved, for a caller that wants to report it.

        ``achieved_factor`` is the rate the run sustained against the one it asked for, so a
        value well below :attr:`factor` means the simulator could not do what it was told --
        the fact a slow run needs and does not otherwise emit. ``{}`` in ``asap`` mode, where
        there is no target to fall short of.
        """
        total = self.paced_steps + self.behind_steps
        if not self.realtime or not total:
            return {}
        late_share = self.behind_steps / total
        # Each late step delivered dt of sim time in (dt/factor + owed) of wall time.
        owed_per_step = self.behind_seconds / total
        achieved = self.dt / (self.dt / self.factor + owed_per_step)
        return {
            "requested_factor": self.factor,
            "achieved_factor": achieved,
            "late_steps": self.behind_steps,
            "total_steps": total,
            "late_share": late_share,
            "behind_seconds": self.behind_seconds,
        }
