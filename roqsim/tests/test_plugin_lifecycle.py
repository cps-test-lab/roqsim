"""Lifecycle: hook ordering, optional-hook dispatch, and the build->compile->configure sequence."""

from __future__ import annotations

import pytest
from conftest import RecordingPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

REF = "conftest:RecordingPlugin"


def test_full_lifecycle_order():
    cfg = load_config_from_dict(
        {"sim": {}, "plugins": [{REF: {}, "name": "a"}, {REF: {}, "name": "b"}]}
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    engine.step()
    engine.shutdown()

    log = RecordingPlugin.LOG
    # build for all plugins happens before any configure (compile is in between).
    assert log[0] == ("a", "build")
    assert log[1] == ("b", "build")
    assert log[2] == ("a", "configure")
    assert log[3] == ("b", "configure")
    # reset -> on_reset for both.
    assert ("a", "on_reset") in log and ("b", "on_reset") in log
    # a step ticks pre_step for all, then post_step for all.
    pre_idx = log.index(("a", "pre_step"))
    post_idx = log.index(("a", "post_step"))
    assert pre_idx < post_idx
    assert log.index(("b", "pre_step")) < log.index(("a", "post_step"))
    # shutdown runs in reverse order.
    assert log.index(("b", "shutdown")) < log.index(("a", "shutdown"))


class OnlyPreStep(Plugin):
    ticks = 0

    def pre_step(self, ctx):
        type(self).ticks += 1


def test_optional_hooks_are_skipped():
    """A plugin that implements only pre_step must not error on the hooks it omits."""
    OnlyPreStep.ticks = 0
    cfg = load_config_from_dict({"sim": {}, "plugins": [{f"{__name__}:OnlyPreStep": {}}]})
    engine = Engine(cfg, profile=True)
    engine.setup()
    engine.reset()
    engine.step()
    engine.step()
    engine.shutdown()
    assert OnlyPreStep.ticks == 2
    # An un-overridden hook should not appear in the timing report.
    report = engine.timing_report().get("OnlyPreStep", {})
    assert set(report) <= {"pre_step"}


def test_step_before_setup_raises():
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    engine = Engine(cfg)
    with pytest.raises(RuntimeError):
        engine.step()
