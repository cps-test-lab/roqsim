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
    # Unset it whatever the caller's shell says: the selection under test is what happens when nobody
    # has chosen, and an inherited value would silently make every assertion here vacuous.
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


# -- a GPU handed over without the capability that makes it usable ------------------
#
# The NVIDIA container runtime gates the GL half of the driver behind the `graphics`
# capability and defaults to `compute,utility`. A container given a GPU without asking for
# `graphics` therefore has /dev/nvidia* but no /dev/dri, and this module -- correctly --
# observes no render node and picks software rendering. Nothing errors; the job is just
# many times slower while reporting success. These tests pin the two halves of the answer:
# the *observation* is warned about wherever it is made, and the *refusal* happens where
# something actually renders.


def _fake_devices(monkeypatch, *, nvidiactl, render_node, names=None):
    """Point gl.py's two probes at a scripted answer, without touching /dev.

    *names* is what ``/dev/dri`` contains, so a test can pin WHICH node is present rather
    than only whether one is -- the distinction the ``renderD129`` regression turns on.
    ``render_node=False`` scripts a directory holding no render node at all, which is what
    a GPU handed over without the ``graphics`` capability actually looks like.
    """
    from roqsim import gl

    present = set()
    if nvidiactl:
        present.add(gl._NVIDIA_CONTROL)
    monkeypatch.setattr(gl.os.path, "exists", lambda path: path in present)

    if names is None:
        names = ["by-path", "card0", "renderD128"] if render_node else ["by-path", "card0"]

    def _listdir(path):
        if path == gl._RENDER_NODE_DIR:
            return list(names)
        raise FileNotFoundError(path)

    monkeypatch.setattr(gl.os, "listdir", _listdir)


def test_a_gpu_without_a_render_node_is_recognised(monkeypatch):
    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=True, render_node=False)
    assert gl.gpu_without_render_node() is True


def test_the_ordinary_shapes_are_not_mistaken_for_it(monkeypatch):
    """It must be inert everywhere a GPU was not handed over half-configured: a CPU-only
    node, and any machine whose render node is present. Otherwise it is not safe to refuse
    on."""
    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=False, render_node=False)  # CI, gcp-c4
    assert gl.gpu_without_render_node() is False
    _fake_devices(monkeypatch, nvidiactl=True, render_node=True)  # a working GPU node
    assert gl.gpu_without_render_node() is False
    _fake_devices(monkeypatch, nvidiactl=False, render_node=True)  # laptop, no nvidiactl
    assert gl.gpu_without_render_node() is False


def test_a_discrete_card_numbered_past_128_is_a_render_node(monkeypatch):
    """The regression this detection was written without.

    Render nodes are numbered from 128 in probe order, so a machine with an integrated chip
    and a discrete card puts the discrete one at ``renderD129`` -- and a container handed
    only that card sees no ``renderD128``. Probing the fixed name read a working RTX 4090 as
    a GPU with no render node and REFUSED THE RUN, which is the worst direction for this
    check to fail in: it turns a correct fast configuration into a hard error.
    """
    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=True, render_node=True,
                  names=["by-path", "card1", "renderD129"])
    assert gl.render_node() == "/dev/dri/renderD129"
    assert gl.gpu_without_render_node() is False


def test_a_dri_dir_with_only_card_nodes_is_not_a_render_node(monkeypatch):
    """``card*`` is the privileged primary node, not the one EGL renders through. A
    directory holding only those is the no-graphics-capability case, not a working GPU."""
    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=True, render_node=True, names=["by-path", "card0"])
    assert gl.render_node() is None
    assert gl.gpu_without_render_node() is True


def test_no_dri_directory_at_all_is_the_cpu_only_answer(monkeypatch):
    """A machine with no ``/dev/dri`` must answer "no render node", not raise: that is
    every CPU-only node and every CI runner."""
    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=False, render_node=False, names=None)
    monkeypatch.setattr(gl.os, "listdir", lambda path: (_ for _ in ()).throw(
        FileNotFoundError(path)))
    assert gl.render_node() is None
    assert gl.gpu_without_render_node() is False


def test_selecting_osmesa_next_to_a_gpu_warns(monkeypatch, caplog):
    import logging

    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=True, render_node=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv(gl._OPT_OUT, raising=False)
    with caplog.at_level(logging.WARNING, logger="roqsim.gl"):
        assert gl.select_offscreen_gl() == "osmesa"
    assert "graphics" in caplog.text, "the warning must name the capability to set"


def test_plain_osmesa_selection_is_silent(monkeypatch, caplog):
    """A CPU-only machine is not a misconfiguration and must not be nagged."""
    import logging

    from roqsim import gl

    _fake_devices(monkeypatch, nvidiactl=False, render_node=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv(gl._OPT_OUT, raising=False)
    with caplog.at_level(logging.WARNING, logger="roqsim.gl"):
        assert gl.select_offscreen_gl() == "osmesa"
    assert caplog.text == ""


def test_the_renderer_refuses_a_fallback_next_to_a_gpu(monkeypatch):
    """Where the glfw case aborts loudly inside MjrContext, this one renders correctly and
    slowly -- so it has to be refused rather than left to announce itself."""
    from roqsim import gl, rendering

    monkeypatch.setattr(gl, "_chosen", "osmesa")
    _fake_devices(monkeypatch, nvidiactl=True, render_node=False)
    with pytest.raises(rendering.GLBackendError) as excinfo:
        rendering.check_gl_backend()
    message = str(excinfo.value)
    assert "graphics" in message
    assert "MUJOCO_GL=osmesa" in message, "the message must name the way to opt in deliberately"


def test_an_explicit_osmesa_request_is_honoured(monkeypatch):
    """Asked for by hand, software rendering is a choice -- exactly as an explicit glfw is.
    `_chosen` is None when select_offscreen_gl deferred to a value already set, which is the
    only way the two can be told apart afterwards."""
    from roqsim import gl, rendering

    monkeypatch.setattr(gl, "_chosen", None)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    _fake_devices(monkeypatch, nvidiactl=True, render_node=False)
    rendering.check_gl_backend()  # must not raise


def test_the_gl_device_is_reported_next_to_the_backend():
    """`egl` alone cannot answer "did this use the GPU": it is what any machine with a working
    hardware GL stack reports, and which device it binds is the ICD's decision. So the device
    string is the load-bearing half of the log line."""
    out = _run(
        "import roqsim, mujoco;"
        "from roqsim.rendering import FrameRenderer, bound_gl_backend, bound_gl_device;"
        "m = mujoco.MjModel.from_xml_string("
        '\'<mujoco><worldbody><geom type="box" size=".3 .3 .3"/></worldbody></mujoco>\');'
        # Bound to a name deliberately: an unreferenced FrameRenderer is collected
        # immediately, and its __del__ frees the EGL context -- after which there is no
        # current context left to ask which device it was on.
        "fr = FrameRenderer(m, 32, 32);"
        "print('BACKEND', bound_gl_backend());"
        "print('DEVICE', bound_gl_device())"
    )
    reported = {}
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        if key in ("BACKEND", "DEVICE"):
            reported[key] = value.strip()
    backend, device = reported.get("BACKEND", ""), reported.get("DEVICE", "")
    assert backend in ("egl", "osmesa"), backend
    # A hardware backend must be able to say which device; software rendering names llvmpipe
    # or similar. Either way the string is non-empty, which is what makes the log worth having.
    assert device, "a live context reported no GL_RENDERER"


# -- what actually bound, which is the only authoritative answer --------------------
#
# Everything above reasons from device paths before a context exists. These pin the check
# that reads the bound renderer instead, and with it the case the path heuristics cannot
# see: a render node that belongs to an integrated chip sitting beside the NVIDIA card.


def _bound(monkeypatch, *, device, nvidiactl=True, chosen="egl"):
    """Script "a GPU was handed over and X bound", without a GL context."""
    from roqsim import gl, rendering

    monkeypatch.setattr(gl, "gpu_visible", lambda: nvidiactl)
    monkeypatch.setattr(gl, "chosen_backend", lambda: chosen)
    return rendering


def test_software_rendering_next_to_a_gpu_is_refused(monkeypatch):
    """The missing-'graphics'-capability case, caught by what bound rather than by a path."""
    from roqsim.rendering import GLBackendError

    r = _bound(monkeypatch, device="llvmpipe (LLVM 20.1.2, 256 bits)")
    with pytest.raises(GLBackendError) as err:
        r.check_bound_device("llvmpipe (LLVM 20.1.2, 256 bits)")
    assert "graphics" in str(err.value), "must name the capability to set"
    assert "llvmpipe" in str(err.value), "must name what actually bound"


def test_binding_the_integrated_chip_beside_the_card_is_refused(monkeypatch):
    """The case no device-path test can reach.

    A container given the NVIDIA card AND an iGPU's render node has a render node present, so
    gpu_without_render_node() is False and every path-based check passes -- while EGL binds
    the Intel chip and the discrete card goes unused. Only the bound device string says so.
    """
    from roqsim.rendering import GLBackendError

    r = _bound(monkeypatch, device="Mesa Intel(R) UHD Graphics 770 (ADL-S GT1)")
    with pytest.raises(GLBackendError) as err:
        r.check_bound_device("Mesa Intel(R) UHD Graphics 770 (ADL-S GT1)")
    assert "bound a different device" in str(err.value)
    assert "Intel" in str(err.value)


def test_the_nvidia_card_binding_is_silent(monkeypatch):
    r = _bound(monkeypatch, device="NVIDIA GeForce RTX 4090/PCIe/SSE2")
    r.check_bound_device("NVIDIA GeForce RTX 4090/PCIe/SSE2")  # must not raise


def test_software_rendering_with_no_gpu_is_silent(monkeypatch):
    """A CPU-only node binding llvmpipe is the correct answer for that machine, not a fault."""
    r = _bound(monkeypatch, device="llvmpipe (LLVM 20.1.2, 256 bits)", nvidiactl=False)
    r.check_bound_device("llvmpipe (LLVM 20.1.2, 256 bits)")


def test_an_explicit_backend_is_never_second_guessed(monkeypatch):
    """Same rule the osmesa refusal follows: a request is honoured, only a fallback is judged.

    chosen_backend() is None exactly when MUJOCO_GL was already set, so this is the escape
    hatch for someone who deliberately wants software rendering on a GPU machine.
    """
    r = _bound(monkeypatch, device="llvmpipe (LLVM 20.1.2, 256 bits)", chosen=None)
    r.check_bound_device("llvmpipe (LLVM 20.1.2, 256 bits)")


def test_an_unreadable_device_string_never_fails_a_render(monkeypatch):
    """bound_gl_device() returns "" when it cannot read the renderer. Diagnostics must not be
    the reason a render fails."""
    r = _bound(monkeypatch, device="")
    r.check_bound_device("")


def test_software_renderers_are_recognised():
    from roqsim.rendering import software_rendered

    for name in ("llvmpipe (LLVM 20.1.2, 256 bits)", "softpipe", "SWRAST", "lavapipe"):
        assert software_rendered(name) is True, name
    for name in ("NVIDIA GeForce RTX 4090/PCIe/SSE2", "Mesa Intel(R) UHD Graphics 770", ""):
        assert software_rendered(name) is False, name
