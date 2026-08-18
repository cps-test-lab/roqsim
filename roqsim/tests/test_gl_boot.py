"""The offscreen GL backend is chosen before ``import mujoco`` binds one.

Every test here runs in a **subprocess**. That is not fussiness: ``MUJOCO_GL`` is read exactly once,
while ``import mujoco`` executes, and both mujoco and roqsim are already imported in the pytest
process by the time any of this runs -- so an in-process test could only ever observe the decision
that was made before it started.

The bug being guarded: roqsim used to select the backend in ``runner.main``, which lives in a module
whose own imports pull in mujoco. The selection therefore ran *after* the binding it was meant to
control, and had no effect. Nothing noticed, because a world with no camera never constructs a
``mujoco.Renderer``, so the wrong backend was never instantiated -- until a camera world reached a
headless cluster and aborted with ``mujoco.FatalError: gladLoadGL error``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Reports the env var and the backend actually bound, which is the pair the bug was hiding between.
_PROBE = (
    "import roqsim, os;"
    "from roqsim.rendering import bound_gl_backend;"
    "print(os.environ.get('MUJOCO_GL'), bound_gl_backend())"
)


#: mujoco imported FIRST, which is the one ordering roqsim's package __init__ cannot fix.
_GUARD_SCRIPT = """
import mujoco, roqsim
from roqsim.rendering import FrameRenderer, GLBackendError

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size=".3 .3 .3"/></worldbody></mujoco>'
)
try:
    FrameRenderer(model, 64, 48)
    print("NO ERROR")
except GLBackendError as err:
    print("RAISED", "MUJOCO_GL" in str(err), "osmesa" in str(err))
"""


def _run(code: str, **env) -> str:
    """Run ``code`` in a clean interpreter with ``env`` applied; ``None`` unsets a variable."""
    child = os.environ.copy()
    # The Makefile exports MUJOCO_GL=egl for the suite, which is precisely the value under test.
    child.pop("MUJOCO_GL", None)
    child.pop("ROQSIM_NO_GL_SELECT", None)
    # PYOPENGL_PLATFORM must go too, and for a reason only a GPU-less machine shows: `import mujoco`
    # in the PARENT sets it from the parent's MUJOCO_GL, and the child inherits it. Where the fresh
    # selection lands on the same backend nothing clashes; on a host with no /dev/dri/renderD128 it
    # picks osmesa, which then refuses to start because the inherited variable still says egl. The
    # point of this helper is a clean interpreter, so it has to be clean of this too.
    child.pop("PYOPENGL_PLATFORM", None)
    for key, value in env.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, env=child, timeout=120
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.mark.parametrize("display", [":0", None])
def test_import_roqsim_binds_an_offscreen_backend(display):
    """Never glfw, and never left unset -- with or without a DISPLAY.

    ``DISPLAY=:0`` is the container case and the one that actually bit: the RoboVAST base image
    exports it unconditionally with no X server behind it, so anything keying off ``DISPLAY`` picks
    the on-screen backend and dies. The answer must not depend on it.
    """
    selected, bound = _run(_PROBE, DISPLAY=display).split()
    assert selected in ("egl", "osmesa"), selected
    assert bound == selected, f"MUJOCO_GL={selected} but mujoco bound {bound}"


def test_an_explicit_backend_is_never_overridden():
    """Including ``glfw``: asking for the on-screen backend is a legitimate thing to want."""
    assert _run(_PROBE, MUJOCO_GL="glfw") == "glfw glfw"


def test_opt_out_leaves_the_decision_to_mujoco():
    """``ROQSIM_NO_GL_SELECT`` is the escape hatch for a caller that wants to own this."""
    selected, _bound = _run(_PROBE, ROQSIM_NO_GL_SELECT="1").split()
    assert selected == "None", selected


def test_selecting_after_mujoco_is_imported_does_not_bind_it():
    """The mechanism itself: proves *when* is what matters, not *whether*.

    If this ever fails, MuJoCo changed to read ``MUJOCO_GL`` lazily and the ordering constraint --
    and the import-time side effect in ``roqsim/__init__.py`` that exists to satisfy it -- can be
    revisited.
    """
    out = _run(
        "import mujoco;"  # binds glfw here, because MUJOCO_GL is unset
        "import os; os.environ['MUJOCO_GL'] = 'egl';"  # too late, by design
        "from mujoco.rendering.classic import gl_context;"
        "print(gl_context.GLContext.__module__.rsplit('.', 1)[-1])"
    )
    assert out == "glfw"


def test_frame_renderer_refuses_an_accidental_glfw_backend():
    """The safety net for the one case the package ``__init__`` cannot reach: mujoco imported first.

    Turns ``mujoco.FatalError: gladLoadGL error`` -- which names neither cause nor fix -- into a
    message that names both.
    """
    out = _run(_GUARD_SCRIPT, ROQSIM_NO_GL_SELECT="1")
    assert out == "RAISED True True", out


def test_an_explicit_glfw_request_is_not_second_guessed():
    """The guard must not fire when glfw is what the caller asked for."""
    out = _run(
        "import roqsim;"
        "from roqsim.rendering import check_gl_backend;"
        "check_gl_backend();"
        "print('allowed')",
        MUJOCO_GL="glfw",
    )
    assert out == "allowed"
