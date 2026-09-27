"""Lifecycle: hook ordering, optional-hook dispatch, and the build->compile->configure sequence."""

from __future__ import annotations

import pytest
from recording_plugin import RecordingPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

REF = "recording_plugin:RecordingPlugin"


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


# -- a configure that fails ------------------------------------------------------------------------


class ConfigureRaises(Plugin):
    def configure(self, ctx):
        self.LOG.append((self.name, "configure"))  # opens what it holds, then fails
        raise RuntimeError("configure failed")

    LOG = RecordingPlugin.LOG

    def shutdown(self, ctx):
        self.LOG.append((self.name, "shutdown"))


def test_a_configure_that_fails_shuts_down_what_was_configured_before_it():
    """The plugins configured before the failure hold what configure opened -- a spin thread, a
    file, a node -- and the driver never gets an engine to shut down, so setup() must do it.
    In reverse order, as a shutdown is, and the failing plugin included: it may have opened its
    resources before the line that raised."""
    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {REF: {}, "name": "a"},
                {REF: {}, "name": "b"},
                {"test_plugin_lifecycle:ConfigureRaises": {}, "name": "c"},
                {REF: {}, "name": "d"},
            ],
        }
    )
    engine = Engine(cfg)
    with pytest.raises(RuntimeError, match="configure failed"):
        engine.setup()
    log = RecordingPlugin.LOG
    assert ("d", "configure") not in log, "nothing after the failure is configured"
    shut = [name for name, hook in log if hook == "shutdown"]
    assert shut == ["c", "b", "a"], shut
    engine.shutdown()  # a driver's finally: nothing left to shut down, and nothing shut down twice
    assert [name for name, hook in log if hook == "shutdown"] == ["c", "b", "a"]
