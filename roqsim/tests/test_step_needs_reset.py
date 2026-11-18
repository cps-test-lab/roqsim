"""``Engine.step()`` refuses an engine that has not been reset since ``setup()``.

``setup()`` poses the data for inspection, but no plugin's ``on_reset`` has run, so a trial stepped
from there starts from a state no driver produces. Every driver resets first; the engine holds a
caller that does not to the same order, naming ``reset()``.
"""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin


def _engine(*components) -> Engine:
    engine = Engine(
        load_config_from_dict(
            {"sim": {}, "components": [{"dummy": {"size": 0.1}, "name": "d0"}, *components]}
        )
    )
    engine.ctx.seed = 1
    return engine


def _counts(engine: Engine) -> dict:
    return engine.ctx.blackboard.get("dummy_counts::d0")


def test_step_without_a_reset_raises_naming_reset():
    engine = _engine()
    with engine:
        with pytest.raises(RuntimeError, match=r"engine\.reset\(\)"):
            engine.step()
        # Refused before anything ran: no hook, no time.
        assert _counts(engine).get("pre_step", 0) == 0
        assert engine.ctx.data.time == 0.0


def test_setup_reset_step_steps():
    engine = _engine()
    with engine:
        engine.reset()
        for _ in range(3):
            engine.step()
        assert _counts(engine)["pre_step"] == 3
        assert engine.ctx.data.time == pytest.approx(3 * engine.dt)


class _Fails(Plugin):
    """Raises from ``on_reset`` while armed, as a plugin whose initial state cannot be placed does."""

    armed = False

    def on_reset(self, ctx: SimContext) -> None:
        if self.armed:
            raise ValueError("no initial state")


def test_a_reset_that_raises_leaves_the_engine_unreset():
    engine = _engine({f"{__name__}:_Fails": {}})
    with engine:
        engine.reset()
        engine.step()
        fails = engine.plugins[-1]
        fails.armed = True
        with pytest.raises(ValueError):
            engine.reset()
        with pytest.raises(RuntimeError, match=r"engine\.reset\(\)"):
            engine.step()
        fails.armed = False
        engine.reset()
        engine.step()


def test_idle_does_not_need_a_reset():
    # idle advances no time and runs no hook: a paused driver may run posted commands before its
    # first reset.
    engine = _engine()
    with engine:
        ran = []
        engine.ctx.post(lambda ctx: ran.append(ctx.data.time))
        assert engine.idle() == 1
        assert ran == [0.0]
