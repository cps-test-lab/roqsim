"""Pause, resume, step and reset a running simulation, as endpoints.

``roqsim sim`` adds this plugin itself, as ``sim.run_control``, so every standalone run carries it;
a world does not declare it. What it serves is :class:`roqsim.control.RunControl`, the object the
driver consults each loop, so a transport that wires endpoints reaches play/pause/step without a
control plane of its own::

    sim/run_control/pause     command   stop stepping; commands still run (Engine.idle)
    sim/run_control/resume    command   step again, paced from now
    sim/run_control/step      command   take N steps while paused; replies once they ran
    sim/run_control/reset     command   reset the world; replies once it has
    sim/run_control/state     out       playing/paused/..., sim time and episode

Only the standalone driver honours it. Under scenario-execution the scenario owns stepping, and the
adapter never adds this plugin.
"""

from __future__ import annotations

from .. import control as ctl
from .. import endpoint
from ..context import CommandFuture, SimContext
from ..plugin import Plugin


class RunControlPlugin(Plugin):
    """The driver's run control, served as endpoints."""

    # It builds nothing and holds no simulation state: a consumer that wants the scene drops it.
    transport_only = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._ctx: SimContext | None = None
        self._reset_waiters: list[CommandFuture] = []

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx

    @endpoint.out("state")
    def state(self) -> dict:
        """The run's state (playing, paused, stopped, quitting), its sim time and its episode."""
        ctx = self._ctx
        return {
            "state": ctl.STATE_NAMES.get(ctx.control.state, str(ctx.control.state)),
            "sim_time": float(ctx.sim_time),
            "episode": int(ctx.episode),
        }

    @endpoint.command("pause")
    def pause(self, _payload=None) -> CommandFuture:
        """Stop stepping. Commands sent while paused still run, and time does not advance."""
        self._ctx.control.set_state(ctl.PAUSED)
        return self._after_step_in_flight()

    def _after_step_in_flight(self) -> CommandFuture:
        # A command drains at the start of a step, which still runs: answer after it, with the
        # state and time the run actually stopped at.
        ctx, done = self._ctx, CommandFuture()
        ctx.control.at_next_loop(lambda: ctx.post(lambda _c: done._resolve(self.state())))
        return done

    @endpoint.command("resume")
    def resume(self, _payload=None) -> dict:
        """Step again. The pacing starts afresh, so the pause is not counted as falling behind."""
        self._ctx.control.set_state(ctl.PLAYING)
        return self.state()

    @endpoint.command("step")
    def step(self, n=1) -> CommandFuture:
        """Take N steps (default 1) while paused; the reply comes once they ran, with the sim time."""
        ctx = self._ctx
        if ctx.control.state != ctl.PAUSED:
            raise RuntimeError(
                f"step needs a paused simulation and this one is "
                f"{ctl.STATE_NAMES.get(ctx.control.state, ctx.control.state)}: pause it first"
            )
        count = 1 if n is None else int(n)
        if count < 1:
            raise ValueError(f"step takes a number of steps >= 1, not {n!r}")
        done = CommandFuture()

        def finished(completed: bool) -> None:
            # Runs on the driver's thread just before the last step; posted, so the reply is read
            # at the next drain -- after that step, when the time it reports has been reached.
            ctx.post(
                lambda c: done._resolve(
                    {"steps": count, "completed": completed, "sim_time": float(c.sim_time)}
                )
            )

        ctx.control.request_steps(count, on_done=finished)
        return done

    @endpoint.command("reset")
    def reset(self, _payload=None) -> CommandFuture:
        """Reset the world to its initial state; the reply comes once the reset has run."""
        done = CommandFuture()
        self._reset_waiters.append(done)
        self._ctx.control.request_reset()
        return done

    def on_reset(self, ctx: SimContext) -> None:
        waiters, self._reset_waiters = self._reset_waiters, []
        for done in waiters:
            done._resolve({"sim_time": float(ctx.sim_time), "episode": int(ctx.episode)})

    def validate_config(self, config: dict) -> list[str]:
        return [f"run_control takes no config, got {sorted(config)}"] if config else []
