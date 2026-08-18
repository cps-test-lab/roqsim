"""Shared test fixtures and helper plugins for the roqsim plugin-mechanism tests."""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin


class RecordingPlugin(Plugin):
    """Appends ``(name, hook)`` to a shared list on every hook, to assert ordering."""

    #: class-level shared log so ordering across multiple plugin instances is observable
    LOG: list[tuple] = []

    def build(self, spec, ctx):
        self.LOG.append((self.name, "build"))

    def configure(self, ctx):
        self.LOG.append((self.name, "configure"))

    def on_reset(self, ctx):
        self.LOG.append((self.name, "on_reset"))

    def pre_step(self, ctx):
        self.LOG.append((self.name, "pre_step"))

    def post_step(self, ctx):
        self.LOG.append((self.name, "post_step"))

    def shutdown(self, ctx):
        self.LOG.append((self.name, "shutdown"))


@pytest.fixture(autouse=True)
def _clear_recording_log():
    RecordingPlugin.LOG.clear()
    yield
    RecordingPlugin.LOG.clear()


@pytest.fixture(autouse=True)
def _no_gl_reexec(monkeypatch):
    """Neutralise the viewer's libGLEW re-exec guard for the whole test session.

    ``runner.main`` calls ``ensure_gl_preload`` before argparse; on a machine with a ``DISPLAY``
    (a dev box, or CI with an X server) a windowed ``main([...])`` in a CLI test would otherwise
    ``os.execv`` the pytest process itself. The sentinel makes the guard a no-op. The tests that
    exercise the guard directly (``test_viewer``) override this via their own ``monkeypatch``.
    """
    monkeypatch.setenv("ROQSIM_GL_PRELOADED", "1")


@pytest.fixture
def make_engine():
    """Factory: build an Engine from a plugins list (dicts of ref/name/config)."""

    def _factory(plugins):
        cfg = load_config_from_dict({"sim": {}, "plugins": plugins})
        return Engine(cfg)

    return _factory
