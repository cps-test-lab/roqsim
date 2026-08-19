"""Shared test fixtures for the roqsim plugin-mechanism tests.

The ``RecordingPlugin`` helper they clear lives in :mod:`recording_plugin`, not here -- see that
module for why a shared test double must not be addressed as ``conftest:...``.
"""

from __future__ import annotations

import pytest
from recording_plugin import RecordingPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


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
