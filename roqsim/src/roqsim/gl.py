"""Which offscreen GL backend MuJoCo renders with, decided before MuJoCo is imported.

**This module must not import mujoco, directly or transitively.** That is its whole reason for
existing separately from :mod:`roqsim.viewer`, which does.

``MUJOCO_GL`` is read exactly once, by ``mujoco/rendering/classic/gl_context.py``, while ``import
mujoco`` runs; it binds ``GLContext`` there and then. Setting the variable afterwards changes
nothing -- and unset is not an error but a *choice*: an empty value falls through to **glfw**, which
on a headless node opens no display and aborts inside the first ``mujoco.Renderer`` with
``mujoco.FatalError: gladLoadGL error``.

That failure is invisible until something renders. A world with no camera never constructs a
``Renderer``, so a process can run to completion against a backend that was already wrong -- which
is how a mis-bound backend survived every campaign this substrate had ever run, and surfaced only
when the first camera world reached a cluster. Hence :func:`select_offscreen_gl` is called from
:mod:`roqsim`'s package ``__init__``, before any ``roqsim`` submodule -- and therefore before any
``import mujoco`` of ours -- can run.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

#: Preferred offscreen backend where a render device exists. ``MUJOCO_GL`` selects only the
#: OFFSCREEN renderer that camera plugins use -- the passive viewer always opens its window via
#: glfw (``mujoco.viewer`` imports and initialises glfw directly, independent of ``MUJOCO_GL``). So
#: ``egl`` is a safe default that leaves the on-screen window untouched while giving cameras their
#: own context: forcing both the window and the cameras onto glfw is what triggers the
#: ``gladLoadGL`` clash on camera worlds. Overridden by setting ``MUJOCO_GL`` yourself.
DEFAULT_MUJOCO_GL = "egl"

#: Where DRI render nodes live -- the things ``egl`` actually needs. The absence of *any* of
#: them is what makes a CPU-only node different from a broken GL install, which is the
#: distinction MuJoCo's own error message cannot draw.
#:
#: Any ``renderD*`` counts, and probing for the specific name ``renderD128`` was a bug: nodes
#: are numbered from 128 in probe order, so on a machine with an integrated chip *and* a
#: discrete card the discrete one is ``renderD129`` -- and a container handed only that card
#: sees no ``renderD128`` at all. That container has a perfectly good GPU, and probing the fixed
#: name reads it as the half-configured case below and refuses to run.
_RENDER_NODE_DIR = "/dev/dri"

#: The prefix a DRI render node's name always has (``renderD128``, ``renderD129``, ...), as
#: opposed to the ``card*`` primary nodes, which need privileges EGL does not use.
_RENDER_NODE_PREFIX = "renderD"

#: Opt out of the selection entirely and let MuJoCo's own default stand. The sibling of
#: ``ROQSIM_NO_GL_PRELOAD`` (see :func:`roqsim.viewer.ensure_gl_preload`), for a caller that wants
#: to own the decision -- including the decision to make no decision.
_OPT_OUT = "ROQSIM_NO_GL_SELECT"

#: An NVIDIA GPU is visible to this process. Paired with *no* render node at all (see
#: :func:`render_node`) it describes one specific mistake and nothing else -- see
#: :func:`gpu_without_render_node`.
_NVIDIA_CONTROL = "/dev/nvidiactl"

#: What :func:`select_offscreen_gl` decided, or ``None`` when it deferred to a value that was
#: already set. Recorded because the two cannot be told apart afterwards: ``MUJOCO_GL=osmesa``
#: reads the same whether a human asked for software rendering or we fell back to it. Only the
#: fallback is ever second-guessed (:func:`roqsim.rendering.check_gl_backend`); a request is
#: honoured, exactly as an explicit ``glfw`` is.
_chosen: str | None = None


def render_node() -> str | None:
    """The first DRI render node this process can see, or ``None``.

    Sorted so the answer is stable rather than dependent on directory order, and any
    ``renderD*`` counts -- see :data:`_RENDER_NODE_DIR` for why pinning ``renderD128``
    was wrong. Which node it returns is diagnostic only: EGL chooses its device through
    the glvnd ICD ordering, not from this path (see :func:`roqsim.rendering.bound_gl_device`).

    An unreadable or missing ``/dev/dri`` is not an error here -- it is the CPU-only answer.
    """
    try:
        names = sorted(os.listdir(_RENDER_NODE_DIR))
    except OSError:
        return None
    for name in names:
        if name.startswith(_RENDER_NODE_PREFIX):
            return os.path.join(_RENDER_NODE_DIR, name)
    return None


def gpu_visible() -> bool:
    """Whether an NVIDIA GPU has been handed to *this process*.

    ``/dev/nvidiactl`` is injected by the container runtime only into a container that
    requested a device, so this is "we were given a GPU", not "the host has one" -- which is
    the question worth asking, because being given a GPU and not rendering on it is the
    mistake (:func:`roqsim.rendering.check_bound_device`).
    """
    return os.path.exists(_NVIDIA_CONTROL)


def gpu_without_render_node() -> bool:
    """True when an NVIDIA GPU is visible but the DRI render node it needs is not.

    This is one mistake with one cause. The NVIDIA container runtime gates the GL/EGL half of
    the driver behind the ``graphics`` capability, and its default is ``compute,utility`` --
    so a container given a GPU *without* asking for ``graphics`` gets ``/dev/nvidia*`` and the
    CUDA stack, but no ``/dev/dri`` and no ``libEGL_nvidia``. Nothing errors: this module then
    correctly observes that there is no render node and picks software rendering, and the only
    symptom is a job that is twenty times slower than it should be while reporting success.

    It cannot fire anywhere a GPU was not deliberately handed over -- a CPU-only node and a
    laptop with a working render node both answer False -- which is what makes it safe to act
    on rather than merely warn about.

    **It is an early warning, not the authoritative test**, and must not be mistaken for one:
    device-node presence is a proxy for "hardware GL works", and it misses the case where a
    render node IS present but belongs to something else -- an integrated chip beside the
    NVIDIA card that was handed over. That process picks ``egl``, binds the wrong device, and
    this function answers False throughout. What the bound context actually is can only be
    read once one exists: :func:`roqsim.rendering.check_bound_device`.
    """
    return gpu_visible() and render_node() is None


def chosen_backend() -> str | None:
    """The backend :func:`select_offscreen_gl` chose, or ``None`` if it deferred (see
    :data:`_chosen`)."""
    return _chosen


def select_offscreen_gl() -> str | None:
    """Choose the **offscreen** GL backend for *this machine*, and apply it if unset.

    ``egl`` needs a render device; without one it fails at import complaining about a
    broken GL install, which is the wrong diagnosis for a node that simply has no GPU.
    So: a render node means ``egl``, and a CPU-only node means ``osmesa``.

    ``DISPLAY`` deliberately does **not** enter into it, though the shell script this
    replaces did check it. That script set ``MUJOCO_GL`` for the whole process, window
    included; this picks only the offscreen renderer, and offscreen never wants ``glfw``
    -- a window is opened by ``mujoco.viewer`` through its own glfw context, independently
    of ``MUJOCO_GL`` (see :data:`DEFAULT_MUJOCO_GL`). Trusting ``DISPLAY`` here is worse
    than useless in a container: a base image may set ``DISPLAY=:0`` unconditionally --
    ours does -- so a headless run would pick glfw and fail against an X server that was
    never started.

    **When** this runs matters as much as what it picks, which is why the call site is
    :mod:`roqsim`'s ``__init__`` and not a driver's ``main``: a driver's module-level
    imports have already pulled in mujoco by the time its ``main`` executes, so a
    selection made there is made too late to have any effect (see the module docstring).

    This also has to happen **where the simulator runs**, not where a run is configured: the
    two are different machines whenever a campaign is dispatched anywhere. It is what
    retires the 22-line shell script three packages had each copied.

    Returns the backend in effect, or ``None`` under :data:`_OPT_OUT`. Never overrides an
    explicit ``MUJOCO_GL``, and is idempotent.
    """
    global _chosen
    if os.environ.get(_OPT_OUT):
        return None
    existing = os.environ.get("MUJOCO_GL")
    if existing:
        return existing
    backend = "egl" if render_node() is not None else "osmesa"
    os.environ["MUJOCO_GL"] = backend
    _chosen = backend
    if backend == "osmesa" and gpu_without_render_node():
        # Warned, not raised: this runs on every import, including for entry points that never
        # build a renderer, and a CUDA-only process has no reason to care. The refusal belongs
        # where something actually renders -- see :func:`roqsim.rendering.check_gl_backend`.
        _logger.warning(
            "An NVIDIA GPU is visible (%s exists) but %s holds no %s* render node, so this "
            "process fell back to software rendering. The container was given the GPU without "
            "the 'graphics' driver capability: set NVIDIA_DRIVER_CAPABILITIES to include "
            "'graphics' (or 'all').",
            _NVIDIA_CONTROL,
            _RENDER_NODE_DIR,
            _RENDER_NODE_PREFIX,
        )
    return backend
