"""Offscreen rendering: the render service every image roqsim draws comes through.

:class:`RenderService` owns the renderers and their contexts. A world's cameras each register a
view with it and are drawn as one batch per tick; a tool with no tick (``roqsim render``, the
thumbnails, the scene review, coverage) draws through :class:`FrameRenderer`, one view of a service
of its own. Behind the service stands a renderer (:mod:`roqsim.render_api`): MuJoCo's own,
:class:`MujocoRenderer`, unless a world names another. A renderer reads a snapshot of the state, never
the live ``MjData``, and never writes the shared ``MjModel``.

MuJoCo's renderer is GL-context- and thread-affine: it is opened, driven and closed on one thread.

**Clip planes.** MuJoCo derives every camera's near and far plane from the model,
``visual/map znear``/``zfar`` times ``stat.extent``, so in a large world the near plane stands tens
of centimetres out. A :class:`FrameRenderer` given ``near``/``far`` renders with its own planes
instead: it sets them on the GL cameras of its own ``MjvScene`` after each ``update_scene``, and
linearises its depth reads with them. The planes live in that scene, never in the shared model,
which a viewer and every other renderer read meanwhile.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import logging
import math
import os
import sys
import threading
from collections.abc import Collection, Mapping, Sequence

import mujoco
import numpy as np

from . import raycast
from .exit_status import NO_GL
from .plugin import PluginError
from .presence import ABSENT_GEOM_GROUP, PARKED_BELOW_Z, subtree_body_ids
from .render_api import (
    CAMERA_MODELS,
    DEFAULT_RENDERER,
    IMAGES,
    RENDER_MODES,
    CameraRequest,
    CameraSpec,
    Frame,
    Renderer,
    RendererCapabilities,
    RenderError,
    SceneDescription,
    Setting,
    StateSnapshot,
    image_mismatch,
)

_logger = logging.getLogger(__name__)

#: Keys that walk a free camera, as lowercase names (the tkinter ``keysym`` / GLFW letter).
WALK_KEYS = frozenset("wasdqe")

#: Whether this MuJoCo takes the ``MjvScene`` argument in ``mjv_moveCamera`` (dropped in 3.11);
#: ``None`` until the first call probes it.
_MOVE_CAMERA_TAKES_SCENE: bool | None = None


def view_forward(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit view direction (eye -> lookat) of a free camera at these orbit angles."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def walk_delta(azimuth_deg: float, keys, step: float, elevation_deg: float = 0.0) -> np.ndarray:
    """The world-frame translation for the currently held walk keys.

    ``W``/``S`` fly along the direction the camera is *looking* -- look down and W descends, like a
    spectator camera -- while ``A``/``D`` strafe horizontally (a banking strafe would roll the
    horizon) and ``E``/``Q`` are world up/down. Diagonals are normalised to ``step`` metres, otherwise
    two keys would move faster than one. Unknown keys and an empty set give a zero vector.
    """
    az = math.radians(azimuth_deg)
    fwd = view_forward(azimuth_deg, elevation_deg)
    right = np.array([math.sin(az), -math.cos(az), 0.0])  # forward x up, with up = +z
    up = np.array([0.0, 0.0, 1.0])
    axes = {"w": fwd, "s": -fwd, "d": right, "a": -right, "e": up, "q": -up}
    vec = np.zeros(3)
    for key in keys:
        vec += axes.get(str(key).lower(), 0.0)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3)
    return vec / norm * step


def default_free_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """A free camera framed on the whole model (MuJoCo's model-derived default)."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    return cam


def bounding_sphere(model: mujoco.MjModel, data: mujoco.MjData, body_ids):
    """World-space enclosing sphere ``(center, radius)`` of ``body_ids`` and everything below them.

    Returns the sphere over those bodies' geoms alone (and their descendants -- a robot's links, not
    just its root), ignoring the rest of the model, or ``None`` when they carry no framable geometry
    (only planes, ``rbound`` 0, or nothing). Each geom is taken as its own world sphere -- centre
    ``geom_xpos``, radius ``geom_rbound`` -- which is the placement-correct bound even for a mesh
    MuJoCo recentred on its inertia frame (a box built from ``geom_aabb`` is not: it is stated in that
    rotated local frame). A parked geom (below :data:`~roqsim.presence.PARKED_BELOW_Z`) is skipped.
    ``data`` must be forward-kinematics-current (call ``mj_forward`` first).
    """
    roots = {int(b) for b in body_ids if int(b) >= 0}
    keep = set(roots)
    for root in roots - {0}:  # pull in descendants so a robot's links count, not just its root
        keep.update(subtree_body_ids(model, root))

    spheres = [
        (data.geom_xpos[g], float(model.geom_rbound[g]))
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in keep
        and float(model.geom_rbound[g]) > 0.0
        and float(data.geom_xpos[g][2]) >= PARKED_BELOW_Z
    ]
    if not spheres:  # e.g. the bodies carry only planes (rbound 0) or no geometry at all
        return None
    # Enclosing sphere: centre on the AABB of the per-geom spheres, then grow the radius to cover the
    # farthest one. Tight enough for framing without a full minimal-sphere solve.
    lo = np.min([c - r for c, r in spheres], axis=0)
    hi = np.max([c + r for c, r in spheres], axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.linalg.norm(c - center)) + r for c, r in spheres)  # > 0: every r > 0
    return center, radius


def _frame_distance(model: mujoco.MjModel, radius: float, aspect: float, margin: float) -> float:
    """Orbit distance that makes a sphere of ``radius`` fill the model's field of view.

    ``aspect`` (width/height of the target frame) lets the narrower of the two fields govern so
    nothing clips on a tall window.
    """
    half_fovy = math.radians(float(model.vis.global_.fovy)) / 2.0
    half_fov = min(half_fovy, math.atan(math.tan(half_fovy) * max(float(aspect), 1e-6)))
    return margin * radius / math.sin(half_fov)


def scene_corners(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    """The eight corners of the world-space box around every geom with a bound, or ``None``.

    Each geom's own ``geom_aabb`` (stated in its rotated frame) is carried into world space corner by
    corner, so a wall stays a wall and not the sphere around it -- which is what makes this tight for
    the rooms recordings are made in. Planes have no bound and are skipped; the geoms that stand on
    them say where the scene is, and parked geoms (below :data:`~roqsim.presence.PARKED_BELOW_Z`)
    are no part of it. ``data`` must be forward-kinematics-current.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for g in range(model.ngeom):
        if float(model.geom_rbound[g]) <= 0.0 or float(data.geom_xpos[g][2]) < PARKED_BELOW_Z:
            continue
        center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        xmat = data.geom_xmat[g].reshape(3, 3)
        corners = data.geom_xpos[g] + (center + signs * half) @ xmat.T
        lo = np.minimum(lo, corners.min(axis=0))
        hi = np.maximum(hi, corners.max(axis=0))
    if not np.all(np.isfinite(lo)):
        return None
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def scene_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    aspect: float = 1.0,
    azimuth: float = 90.0,
    elevation: float = -45.0,
    margin: float = 1.05,
) -> mujoco.MjvCamera:
    """The whole scene from above: every geom in the frame, as close as the field of view allows.

    Looks at the centre of the scene's box from ``azimuth``/``elevation`` (MuJoCo's default orbit:
    angled, not top-down, so walls and heights read) and backs off exactly far enough that all eight
    corners of that box fall inside the view, in both the vertical field and the one the frame's
    ``aspect`` gives it horizontally. Falls back to :func:`default_free_camera` when nothing in the
    model has a bound.
    """
    cam = default_free_camera(model)
    corners = scene_corners(model, data)
    if corners is None:
        return cam
    cam.azimuth, cam.elevation = float(azimuth), float(elevation)
    cam.lookat[:] = (corners.min(axis=0) + corners.max(axis=0)) / 2.0
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(right, forward)
    half_fovy = math.radians(float(model.vis.global_.fovy)) / 2.0
    tan_v = math.tan(half_fovy)
    tan_h = tan_v * max(float(aspect), 1e-6)
    needed = 0.0
    for corner in corners:
        rel = corner - cam.lookat
        depth = float(rel @ forward)  # positive: beyond the lookat, away from the eye
        # The eye must be far enough behind the lookat that the corner's lateral offset fits the
        # field of view at the corner's own depth: (distance + depth) * tan >= |offset|.
        needed = max(
            needed,
            abs(float(rel @ right)) / tan_h - depth,
            abs(float(rel @ up)) / tan_v - depth,
        )
    cam.distance = max(margin * needed, 1e-3)
    return cam


def autoframe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids,
    *,
    aspect: float = 1.0,
    margin: float = 1.1,
) -> mujoco.MjvCamera:
    """A free camera zoomed to fill the view with ``body_ids`` and everything attached below them.

    Frames on the world-space bounding sphere of those bodies' geoms alone (see
    :func:`bounding_sphere`), ignoring the rest of the model -- so a single prop or robot previewed
    inside the 10 m ``empty_room`` fills the window at a sensible size instead of being lost in the
    box. Keeps MuJoCo's default orbit angle and recomputes only ``lookat`` and ``distance`` from the
    model's field of view. A sphere is the natural fit for a free camera that orbits, since it never
    clips at any angle. Falls back to :func:`default_free_camera` when the bodies carry no framable
    geometry. ``data`` must be forward-kinematics-current (call ``mj_forward`` first).
    """
    cam = default_free_camera(model)
    sphere = bounding_sphere(model, data, body_ids)
    if sphere is None:
        return cam
    center, radius = sphere
    cam.lookat[:] = center
    cam.distance = _frame_distance(model, radius, aspect, margin)
    return cam


def preview_camera(
    model: mujoco.MjModel, data: mujoco.MjData, entities, *, aspect: float = 1.0
) -> mujoco.MjvCamera:
    """The camera for showing ``entities`` (a prop or robot) by themselves -- the one call the windowed
    viewers make for a single-model preview.

    Brings forward kinematics current, resolves each entity's body, and :func:`autoframe`\\ s on them.
    ``entities`` is any iterable of objects with a ``body`` name (e.g. ``ctx.entities.all()``); those
    with no body or an unresolved name are skipped.
    """
    mujoco.mj_forward(model, data)  # geom world poses must be current before measuring
    ids = []
    for e in entities:
        name = getattr(e, "body", None)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
        if bid >= 0:
            ids.append(bid)
    return autoframe(model, data, ids, aspect=aspect)


def eye_position(cam: mujoco.MjvCamera) -> np.ndarray:
    """World-space eye position of a free camera from its orbit params.

    MuJoCo's free camera points into the scene along ``forward`` (eye -> lookat) at
    ``[cos(el)cos(az), cos(el)sin(az), sin(el)]`` and sits behind ``lookat`` at ``distance``. With the
    default negative elevation (looking down) this puts the eye above the floor.
    """
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.asarray(cam.lookat) - cam.distance * forward


def orbit_from_eye(eye, target) -> tuple[list[float], float, float, float]:
    """The inverse of :func:`eye_position`: ``(lookat, distance, azimuth, elevation)`` in metres and
    degrees for a camera standing at ``eye`` and looking at ``target``, so a pose stated in world
    coordinates can be written onto a free camera.
    """
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    distance = float(np.linalg.norm(forward))
    if distance < 1e-9:
        raise ValueError("eye and target coincide; the camera has nowhere to look")
    fx, fy, fz = forward / distance
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, fz))))
    azimuth = math.degrees(math.atan2(fy, fx))
    return [float(v) for v in target], distance, azimuth, elevation


def look_in_place(cam: mujoco.MjvCamera, dx: float, dy: float, sensitivity: float = 180.0) -> None:
    """Aim a free camera **about the eye** -- first-person mouse-look, not an orbit.

    MuJoCo's own rotate keeps ``lookat`` fixed and swings the eye around it, which reads as circling a
    pivot floating in front of you. Here the eye stays put and ``lookat`` is re-derived from the new
    angles, so the view turns like a head. ``dx``/``dy`` are mouse motions as a fraction of window
    height, and ``sensitivity`` degrees-per-unit matches ``mjv_moveCamera``'s 180.
    """
    eye = eye_position(cam)
    cam.azimuth -= sensitivity * dx
    # Straight up/down is a singularity for an azimuth/elevation camera (the horizon spins), so stop
    # just short of the poles the way every first-person camera does.
    cam.elevation = float(np.clip(cam.elevation - sensitivity * dy, -89.9, 89.9))
    cam.lookat[:] = eye + cam.distance * view_forward(cam.azimuth, cam.elevation)


def set_orbit_radius(cam: mujoco.MjvCamera, radius: float) -> None:
    """Re-parameterise a free camera to orbit ``radius`` metres in front of the eye. **Same picture.**

    Only the eye and the view angles reach the image -- ``lookat`` and ``distance`` are just how a free
    camera spells the eye out (``eye = lookat - distance * forward``). Pulling ``lookat`` in to
    ``radius`` and shrinking ``distance`` to match therefore renders identically, while changing the
    one thing that is *not* drawn: the point the mouse orbits. A metre-scale radius turns MuJoCo's own
    orbit-drag into something close to a first-person head turn, with no per-frame fight against its
    render loop -- which is what makes this work in a window we do not draw.

    The cost is that MuJoCo scales pan and zoom by ``distance``, so both get slower as the radius
    shrinks; that is the trade being made when picking one.
    """
    if abs(float(cam.distance) - radius) < 1e-9:
        return
    eye = eye_position(cam)
    cam.distance = radius
    cam.lookat[:] = eye + radius * view_forward(cam.azimuth, cam.elevation)


def dolly(cam: mujoco.MjvCamera, step: float) -> None:
    """Move a free camera ``step`` metres along its view direction (forward > 0), eye and all.

    A wheel that shrinks ``distance`` instead would pull the eye toward a fixed point and stall there;
    translating keeps the orbit radius -- and therefore the mouse-look pivot -- constant.
    """
    cam.lookat[:] = np.asarray(cam.lookat) + step * view_forward(cam.azimuth, cam.elevation)


def _line_of_sight_clear(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eye: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    eps: float = 1e-3,
) -> bool:
    """True if nothing occludes the straight path from ``eye`` to the target sphere's near surface.

    Casts a single ray ``eye -> center`` (the same predicate the coverage engine uses): the path is
    clear iff nothing is hit (``dist < 0``) or the first hit is at/behind the sphere's near surface
    (``dist >= dist - radius - eps``) -- i.e. the first thing the ray meets is the object itself, not
    a wall in front of it.

    Through :func:`roqsim.raycast.cast`, so an *absent* entity does not occlude -- which is what you
    want when framing a camera on something: a body nothing can see should not push the view to
    another angle. Before the seam existed this passed ``geomgroup=None`` and an absent obstacle
    steered the framing.
    """
    delta = np.asarray(center, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist <= eps:
        return True
    hits = raycast.cast(
        model,
        data,
        eye,
        delta / dist,
        cutoff=dist + 1.0,  # culling hint
        flg_static=True,  # walls/floor are static, must occlude
    )
    rd = float(hits.dist[0])
    return rd < 0.0 or rd >= dist - radius - eps


# Candidate viewing angles, tried in order: the default orbit angle first (unoccluded scenes keep
# today's feel), then azimuth swept around the object nearest-first, then the same sweep at steeper
# elevations to look over walls in roofless floorplan worlds.
_FOCUS_AZIMUTH_STEPS = tuple(
    range(0, 360, 30)
)  # offsets from the default azimuth, nearest-first below
_FOCUS_ELEVATIONS = (None, -55.0, -75.0)  # None = keep the default elevation


def focus_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids,
    *,
    aspect: float = 1.0,
    margin: float = 1.1,
) -> mujoco.MjvCamera:
    """A free camera framed on ``body_ids`` *and* rotated to a viewpoint with a clear line of sight.

    Like :func:`autoframe` it zooms to fill the view with the target's bounding sphere, but instead
    of blindly keeping the default orbit angle -- which in a populated indoor world often points
    straight through a wall -- it searches candidate ``(azimuth, elevation)`` angles and picks the
    first from which nothing occludes the object (one ``mj_multiRay`` per candidate; no GL, so this
    runs headless). Falls back to :func:`default_free_camera` when the bodies carry no framable
    geometry, and to the plain :func:`autoframe` framing when no angle is unoccluded (a fully
    enclosed object) -- best-effort, never an exception. ``data`` must be forward-kinematics-current.
    """
    cam = default_free_camera(model)
    sphere = bounding_sphere(model, data, body_ids)
    if sphere is None:
        return cam
    center, radius = sphere
    cam.lookat[:] = center
    cam.distance = _frame_distance(model, radius, aspect, margin)

    default_az, default_el = float(cam.azimuth), float(cam.elevation)
    # Nearest-first azimuth offsets: 0, +30, -30, +60, -60, ... so the chosen angle stays as close to
    # the default as occlusion allows.
    offsets = sorted(_FOCUS_AZIMUTH_STEPS, key=lambda a: min(a % 360, 360 - (a % 360)))
    signed = [0.0]
    for a in offsets:
        if a == 0:
            continue
        signed += [float(a), float(-a)]
    for elevation in _FOCUS_ELEVATIONS:
        cam.elevation = default_el if elevation is None else elevation
        for off in signed:
            cam.azimuth = default_az + off
            eye = eye_position(cam)
            if _line_of_sight_clear(model, data, eye, center, radius):
                return cam
    # No unoccluded angle found: restore the default framing and let the caller decide.
    cam.azimuth, cam.elevation = default_az, default_el
    return cam


class GLBackendError(RuntimeError):
    """MuJoCo bound a GL backend that cannot render here (see :func:`check_gl_backend`)."""

    exit_status = NO_GL


def bound_gl_backend() -> str:
    """The backend ``import mujoco`` actually bound: ``glfw``, ``egl``, ``osmesa`` or ``cgl``.

    Read from the bound ``GLContext`` rather than from ``MUJOCO_GL``, because the whole class of bug
    this guards is the two disagreeing: the variable is read once during ``import mujoco`` and
    changing it afterwards moves the variable and nothing else.
    """
    from mujoco.rendering.classic import gl_context

    ctx = getattr(gl_context, "GLContext", None)
    return "" if ctx is None else getattr(ctx, "__module__", "").rsplit(".", 1)[-1]


def bound_gl_device() -> str:
    """The GL renderer string of the *current* context, e.g. ``NVIDIA RTX A2000 12GB``.

    The backend name alone cannot answer "did this use the GPU". ``egl`` is what a machine with
    any working hardware GL stack reports, and which device it then binds is decided by the
    glvnd ICD ordering rather than by anything visible here -- on a hybrid laptop whose only
    ``renderD128`` is the integrated chip, EGL still binds the discrete NVIDIA card. So the
    device string is the answer and the backend is not.

    Requires a current context, so it is only meaningful once a renderer exists. Returns ``""``
    if it cannot be read: this feeds a log line and must never be the reason a render fails.
    """
    try:
        from OpenGL import GL

        renderer = GL.glGetString(GL.GL_RENDERER)
        if not renderer:
            return ""
        return bytes(renderer).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - diagnostics must not break rendering
        return ""


def check_gl_backend() -> None:
    """Refuse to build a renderer on a backend that cannot render what was asked for.

    MuJoCo picks glfw when ``MUJOCO_GL`` was unset *at import time* -- an empty value is not an
    error there, it is a choice -- and on a headless machine the first ``mujoco.Renderer`` then
    dies inside ``MjrContext`` with ``mujoco.FatalError: gladLoadGL error``, a message that names
    neither the cause nor the fix. Say both instead.

    Deliberately no ``DISPLAY`` test: whether a window could open is a different question, and
    trusting it is how this failed in the first place (a container image sets ``DISPLAY=:0`` with
    no X server behind it). An explicit ``MUJOCO_GL=glfw`` is honoured -- someone who asked for the
    on-screen backend gets it, and gets MuJoCo's own error if it does not work.

    Also refuses the reverse mistake: software rendering *fallen back to* on a machine that was
    handed a GPU (see :func:`roqsim.gl.gpu_without_render_node`). Unlike the glfw case that one
    does not crash -- it produces correct frames, twenty times slower, and reports success --
    which is why it has to be refused rather than left to announce itself.
    """
    from .gl import chosen_backend, gpu_without_render_node

    # Only a fallback is second-guessed. `MUJOCO_GL=osmesa` asked for by hand means someone wants
    # software rendering on this machine, and that is as legitimate as an explicit glfw.
    if chosen_backend() == "osmesa" and gpu_without_render_node():
        raise GLBackendError(
            "This process was given an NVIDIA GPU but no DRI render node, so it fell back to "
            "software rendering (osmesa) and would produce correct frames many times slower.\n"
            "  /dev/nvidiactl exists but /dev/dri holds no renderD* node -- the signature of "
            "a container started without the 'graphics' driver capability.\n"
            "  - Set NVIDIA_DRIVER_CAPABILITIES to include 'graphics' (or use 'all'), or\n"
            "  - set MUJOCO_GL=osmesa explicitly if software rendering is what you want here."
        )
    if bound_gl_backend() != "glfw" or os.environ.get("MUJOCO_GL", "").lower().strip() == "glfw":
        return
    raise GLBackendError(
        "MuJoCo bound the glfw (on-screen) GL backend, so there is no offscreen renderer.\n"
        "  MUJOCO_GL is read once, while `import mujoco` runs; setting it later has no effect.\n"
        f"  It is now {os.environ.get('MUJOCO_GL') or 'unset'}, which means mujoco was imported "
        "before it was set.\n"
        "  - Import roqsim before mujoco (roqsim selects a backend for this machine on import), or\n"
        "  - set MUJOCO_GL=egl (a render device) or MUJOCO_GL=osmesa (CPU only) in the environment."
    )


#: Set once the GL line has been logged. Per process, not per renderer: a camera world builds
#: one renderer per camera and the answer cannot differ between them.
_gl_logged = False

#: Substrings identifying a SOFTWARE renderer in a GL device string. Mesa's CPU rasterisers
#: (``llvmpipe``, ``softpipe``, ``lavapipe``) and the ancient ``swrast`` are the ones reachable
#: here. Matched case-insensitively as substrings because the full string carries a version and
#: a word size -- ``llvmpipe (LLVM 20.1.2, 256 bits)`` -- that must not have to be predicted.
_SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "lavapipe", "swrast")


def software_rendered(device: str) -> bool:
    """Whether *device* (a :func:`bound_gl_device` string) is a CPU rasteriser."""
    lowered = device.lower()
    return any(marker in lowered for marker in _SOFTWARE_RENDERERS)


def check_bound_device(device: str) -> None:
    """Refuse a GPU that was handed to this process and then not rendered on.

    **This is the authoritative check, and the only one that can be.** Every other test in
    this module reasons from device paths *before* a context exists, and a path is a proxy:
    it can say a render node is present without saying whether the GL stack behind it is the
    card we were given. Two ways that goes wrong, and the second is invisible to
    :func:`roqsim.gl.gpu_without_render_node` by construction:

    * the ``graphics`` driver capability was not requested, so ``libEGL_nvidia`` was never
      mounted and EGL falls through to Mesa's CPU rasteriser;
    * a render node IS present but belongs to an **integrated chip beside** the NVIDIA card,
      so EGL binds the iGPU. Nothing is missing, nothing errors, and the discrete card sits
      idle.

    Reading the renderer string collapses both, and every future variant, into one question
    that needs no device paths at all: *we were given an NVIDIA GPU -- is that what bound?*
    A run that answers no produces correct frames many times slower while reporting success,
    which is why this raises rather than warns (measured: 26-32x more CPU for the same frames).

    Honoured escape hatches, matching the rest of this module: an explicit ``MUJOCO_GL``
    is never second-guessed (:func:`roqsim.gl.chosen_backend` returns ``None`` then), and
    ``ROQSIM_NO_GL_SELECT`` opts out of the selection entirely. Someone who asked for software
    rendering on a GPU machine gets it.

    Silent where no GPU was handed over: a CPU-only node binding ``llvmpipe`` is not a
    misconfiguration, it is the correct answer for that machine.
    """
    from .gl import chosen_backend, gpu_visible

    # Only a backend WE chose is second-guessed -- the same rule check_gl_backend applies to
    # osmesa. An explicit request is a decision, not a fallback.
    if chosen_backend() is None or not gpu_visible():
        return
    if not device:
        return  # the renderer string could not be read; diagnostics must not fail a render
    if "nvidia" in device.lower():
        return
    how = (
        "fell back to software rendering"
        if software_rendered(device)
        else "bound a different device"
    )
    raise GLBackendError(
        f"This process was given an NVIDIA GPU but {how}, so it would produce correct frames "
        f"many times slower while reporting success.\n"
        f"  /dev/nvidiactl exists, but the GL context bound {device!r}.\n"
        "  - Set NVIDIA_DRIVER_CAPABILITIES to include 'graphics' (or use 'all'), which is "
        "what mounts the NVIDIA EGL driver, or\n"
        "  - set MUJOCO_GL explicitly if rendering off the NVIDIA card is what you want here."
    )


def _log_gl_once() -> None:
    """Log the backend *and* the device it bound, once, then verify the device.

    This is the line that answers "did this run use the GPU", and it is worth a log entry
    because the alternative -- inferring it from wall-clock afterwards -- lets a mis-bound
    backend go unnoticed across any number of campaigns.

    The verification lives here because this is the first point in the process where a GL
    context exists, and :func:`check_bound_device` cannot answer anything before one does.
    It runs on every renderer while the log line is emitted once: the check is a single
    ``glGetString`` and a caller that catches the refusal and builds another renderer must
    not be met with silence the second time.
    """
    global _gl_logged
    device = bound_gl_device()
    if not _gl_logged:
        _gl_logged = True
        _logger.info(
            "offscreen GL: %s%s",
            bound_gl_backend() or "unknown",
            f" -- {device}" if device else "",
        )
    check_bound_device(device)


@contextlib.contextmanager
def hold_gc():
    """Keep the cyclic garbage collector out of a GL critical section.

    A ``mujoco.Renderer`` is torn down in its ``__del__``, so an unreferenced one -- an engine a
    test dropped, a camera whose plugin is gone -- is destroyed whenever the collector happens to
    run. Its teardown makes its own context current and destroys it, leaving no context current;
    a render interrupted by that (the collector can run at any allocation, and MuJoCo allocates
    the output array between making its context current and drawing) draws into nothing and reads
    back uninitialised memory, which surfaces as impossible pixel values. Holding the collector for
    the few milliseconds of a render is what keeps the two apart.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def set_clip_planes(scene: mujoco.MjvScene, near: float | None, far: float | None) -> None:
    """Move the clip planes of *scene*'s GL cameras to *near*/*far* metres, keeping the view.

    ``mjv_updateScene`` fills each ``MjvGLCamera`` from the model's planes; this overrides them in
    the scene alone. A perspective frustum's sides are given at the near plane, so they scale with
    it, which keeps the field of view; an orthographic one's are absolute. ``None`` keeps a plane.
    """
    for cam in scene.camera:
        if near is not None:
            old = float(cam.frustum_near)
            if not cam.orthographic:
                scale = near / old
                cam.frustum_bottom *= scale
                cam.frustum_top *= scale
                cam.frustum_center *= scale
                cam.frustum_width *= scale
            cam.frustum_near = near
        if far is not None:
            cam.frustum_far = far


def linear_depth(
    buffer: np.ndarray, near: float, far: float, orthographic: bool = False
) -> np.ndarray:
    """Metres along the view axis from a depth buffer *buffer* rendered with planes *near*/*far*.

    The formula ``mujoco.Renderer.render`` applies (reversed Z, the projection's terms in float32
    as ``glFrustum`` computes them, the inversion in float64), but with the planes of this render
    rather than the model's.
    """
    zfar, znear = np.float32(far), np.float32(near)
    z = buffer.astype(np.float64)
    if orthographic:
        out = zfar - z * (zfar - znear)
    else:
        c_coef = -(zfar + znear) / (zfar - znear)
        d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
        d_coef = np.float32(-0.5) * d_coef
        out = d_coef / (z + c_coef)
    return out.astype(np.float32)


def has_clear_geoms(model: mujoco.MjModel) -> bool:
    """Whether any geom of ``model`` is drawn see-through: its own or its material's alpha below 1."""
    alpha = np.array(model.geom_rgba[:, 3])
    mat = np.asarray(model.geom_matid)
    textured = mat >= 0
    alpha[textured] = np.asarray(model.mat_rgba)[mat[textured], 3]
    return bool(np.any((alpha > 0.0) & (alpha < 1.0)))


def clear_geoms_cast_no_shadow(scene: mujoco.MjvScene) -> None:
    """Take every see-through geom of a populated scene out of the shadow pass.

    MuJoCo's shadow map is binary: every geom it draws into it blocks the light completely, so a
    pane of glass or a shrink film at 10 % opacity throws the same full shadow as a steel plate, and
    a part wrapped in film renders as if it stood in the dark. MuJoCo leaves a geom out of the shadow
    pass when its category is ``mjCAT_DECOR`` and uses the category for nothing else once the scene
    is built, so a transparent geom is moved there. It is still drawn and still blended; it only no
    longer darkens what is behind it.
    """
    decor = int(mujoco.mjtCatBit.mjCAT_DECOR)
    for i in range(scene.ngeom):
        geom = scene.geoms[i]
        # By its resolved alpha: MuJoCo sets `transparent` itself, but only inside mjr_render.
        if geom.rgba[3] < 1.0:
            geom.category = decor


#: The model arrays that change at run time and that drawing reads: :mod:`roqsim.presence` hides an
#: absent entity by moving its geoms and its flexes to :data:`~roqsim.presence.ABSENT_GEOM_GROUP`
#: and zeroing their alpha. A snapshot carries them, so a renderer off the physics thread or in
#: another process draws the presence of the tick it renders and never reads the shared model while
#: the physics thread writes it.
RUNTIME_MODEL_FIELDS = ("geom_rgba", "geom_group", "flex_rgba", "flex_group")

#: The geoms a scene holds unless a camera asks for more (``mujoco.Renderer``'s default).
DEFAULT_MAX_GEOM = 10000


def _fixed_camera(camera_id: int) -> mujoco.MjvCamera:
    """The ``MjvCamera`` ``mujoco.Renderer.update_scene`` makes of a camera id (``-1``: free)."""
    cam = mujoco.MjvCamera()
    cam.fixedcamid = camera_id
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED if camera_id >= 0 else mujoco.mjtCamera.mjCAMERA_FREE
    return cam


def resolve_camera(model: mujoco.MjModel, camera) -> mujoco.MjvCamera | int:
    """A camera argument as a renderer takes it: an ``MjvCamera`` as given, a name or id as an id."""
    if isinstance(camera, mujoco.MjvCamera):
        return camera
    camera_id = camera
    if isinstance(camera_id, str):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if camera_id == -1:
            raise ValueError(f'The camera "{camera}" does not exist.')
    camera_id = int(camera_id)
    if camera_id < -1 or camera_id >= model.ncam:
        raise ValueError(f"The camera id {camera_id} is out of range [-1, {model.ncam}).")
    return camera_id


def camera_model_of(model: mujoco.MjModel, camera) -> str:
    """Which :data:`roqsim.render_api.CAMERA_MODELS` entry *camera* projects as."""
    if isinstance(camera, mujoco.MjvCamera):
        if camera.type == mujoco.mjtCamera.mjCAMERA_FIXED and camera.fixedcamid >= 0:
            camera = int(camera.fixedcamid)
        else:
            return "free"
    if camera < 0:
        return "free"
    ortho = int(model.cam_projection[camera]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
    return "orthographic" if ortho else "pinhole"


def moving_bodies(model: mujoco.MjModel) -> np.ndarray:
    """Per body, whether it moves: it is not welded to the world, or it is a mocap body."""
    return (np.asarray(model.body_weldid) != 0) | (np.asarray(model.body_mocapid) >= 0)


def _vopt(geomgroup) -> mujoco.MjvOption:
    """The scene options every roqsim render draws with; see :class:`FrameRenderer`."""
    vopt = mujoco.MjvOption()
    # An absent entity is not in the picture either. Its geoms already have zero alpha, but
    # excluding the group is what makes absence one decision rather than several that could
    # drift -- and it is the same group every raycast mask excludes.
    vopt.geomgroup[ABSENT_GEOM_GROUP] = 0
    if geomgroup is not None:
        # Draw only the named groups. A model separates what it SHOWS from what it COLLIDES --
        # group 2 for visual meshes, group 3 for the primitives physics uses -- and a picture of
        # the visual half cannot answer a question about the other one. Absence stays excluded
        # regardless: it is a property of the entity, not a drawing preference, and letting a
        # caller re-enable it here would make absence two decisions instead of one.
        for group in range(len(vopt.geomgroup)):
            vopt.geomgroup[group] = 1 if group in geomgroup else 0
        vopt.geomgroup[ABSENT_GEOM_GROUP] = 0
    return vopt


class _Atlas:
    """One single-sampled colour framebuffer every view's colour image is copied into, read back once.

    MuJoCo draws each view at the origin of its own offscreen framebuffer, as a lone
    ``mujoco.Renderer`` does, so a view's pixels do not depend on where its tile is; the view is
    then copied into its tile on the GPU (``glBlitFramebuffer``, an exact copy between buffers of
    one format). Read through MuJoCo's resolve buffer first when MuJoCo multisamples, which is what
    ``mjr_readPixels`` does too. Tiles are shelf-packed per batch, so a view nobody wanted takes no
    tile; the buffer grows to the largest batch it has held and never shrinks.
    """

    def __init__(self) -> None:
        from OpenGL import GL  # noqa: PLC0415 -- MuJoCo's egl and osmesa backends need it anyway

        self.GL = GL
        self.fbo = None
        self.rbo = None
        self.width = 0
        self.height = 0
        self.max_size = int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE))

    def _clear_errors(self) -> None:
        """Drop GL errors MuJoCo's own calls left behind, which it does not check.

        PyOpenGL checks ``glGetError`` after every call it makes, so an error MuJoCo raised and
        ignored -- a light past the eight it draws is a ``GL_INVALID_ENUM`` in every frame -- would
        be reported as this atlas's.
        """
        GL = self.GL
        for _ in range(64):
            if GL.glGetError() == GL.GL_NO_ERROR:
                return

    def pack(self, sizes: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Tile origins for *sizes*, in order; grows the buffer to hold them."""
        row_limit = max(
            max(w for w, _ in sizes),
            int(math.ceil(math.sqrt(len(sizes))) * max(w for w, _ in sizes)),
        )
        row_limit = min(row_limit, self.max_size)
        x = y = shelf = 0
        origins = []
        width = 0
        for w, h in sizes:
            if x + w > row_limit:
                x, y, shelf = 0, y + shelf, 0
            origins.append((x, y))
            x += w
            shelf = max(shelf, h)
            width = max(width, x)
        height = y + shelf
        if width > self.max_size or height > self.max_size:
            raise RenderError(
                f"the colour images of one batch ({len(sizes)} views) need a {width}x{height} "
                f"buffer, more than this GL allows ({self.max_size}); render fewer views per batch"
            )
        self._ensure(width, height)
        return origins

    def _ensure(self, width: int, height: int) -> None:
        GL = self.GL
        if width <= self.width and height <= self.height:
            return
        self._clear_errors()
        width, height = max(width, self.width), max(height, self.height)
        self.free()
        self.fbo = GL.glGenFramebuffers(1)
        self.rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, width, height)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.rbo
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RenderError(f"the colour atlas framebuffer is incomplete (GL status {status})")
        self.width, self.height = width, height

    def copy_in(self, con: mujoco.MjrContext, w: int, h: int, origin: tuple[int, int]) -> None:
        """Copy the ``w x h`` image MuJoCo just drew at its origin into the tile at *origin*."""
        GL = self.GL
        self._clear_errors()
        src = con.offFBO
        if con.offSamples:
            # Resolve exactly as mjr_readPixels does, then copy the resolved pixels.
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, con.offFBO)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, con.offFBO_r)
            GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            src = con.offFBO_r
        x, y = origin
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, src)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glBlitFramebuffer(0, 0, w, h, x, y, x + w, y + h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, con)

    def read(self, width: int, height: int, con: mujoco.MjrContext) -> np.ndarray:
        """The atlas' bottom-left ``width x height``, as GL stores it (bottom row first)."""
        GL = self.GL
        self._clear_errors()
        out = np.empty((height, width, 3), dtype=np.uint8)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, out)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, con)
        return out

    def free(self) -> None:
        GL = self.GL
        if self.fbo is not None:
            GL.glDeleteFramebuffers(1, [self.fbo])
            GL.glDeleteRenderbuffers(1, [self.rbo])
        self.fbo = self.rbo = None
        self.width = self.height = 0


class MujocoRenderer(Renderer):
    """MuJoCo's own OpenGL renderer: one context and one upload for every camera of a world.

    Per batch it builds the scene once from the snapshot; each camera then only sets its view, its
    lights (the headlight follows the camera) and its clip planes, is drawn at the origin of the
    offscreen buffer and copied into its tile of one colour atlas, which is read back once. Depth
    and segmentation keep a pass of their own per camera (they draw with flat ID colours). A camera
    with its own geom groups or decorations gets a scene built for it alone. The pictures are
    ``mujoco.Renderer``'s: see the module docstring for the planes, and :class:`FrameRenderer` for
    the see-through geoms that cast no shadow.
    """

    capabilities = RendererCapabilities(
        images=frozenset(IMAGES),
        camera_models=frozenset(CAMERA_MODELS),
        composes_moving_bodies=True,
        draws_static=True,
        batched=True,
        deterministic=True,
        in_process=True,
        decorations=True,
        geom_groups=True,
    )
    SETTINGS = {
        "atlas": Setting(
            True,
            "copy every camera's colour image into one buffer and read it back once per batch; "
            "false reads each camera back by itself (the same pixels, more read-backs)",
        ),
    }

    @classmethod
    def validate_config(cls, settings) -> list[str]:
        errors = super().validate_config(settings)
        if isinstance(settings, Mapping) and not isinstance(settings.get("atlas", True), bool):
            errors.append(f"'atlas' must be true or false, got {settings['atlas']!r}")
        return errors

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self.model: mujoco.MjModel | None = None
        self._private = False
        self._gl = None
        self._con: mujoco.MjrContext | None = None
        self._scene: mujoco.MjvScene | None = None
        self._atlas: _Atlas | None = None
        self._clear_geoms = False
        self._vopts: dict = {}
        #: Scene builds and draws since open (what a test counts).
        self.stats = {"scene_builds": 0, "draws": 0, "readbacks": 0, "batches": 0}

    # -- lifecycle -------------------------------------------------------------------------------

    def open(self, scene: SceneDescription) -> None:
        # Before anything touches GL: a wrong backend aborts the process inside MjrContext with a
        # message that explains nothing, and every renderer in the tree goes through here, so one
        # check covers cameras, the thumbnailer and `roqsim render` alike.
        check_gl_backend()
        from mujoco.rendering.classic import gl_context  # noqa: PLC0415 -- bound by import mujoco

        model = scene.model
        self.model = model
        self._private = bool(scene.private_model)
        width = max([int(model.vis.global_.offwidth)] + [c.width for c in scene.cameras])
        height = max([int(model.vis.global_.offheight)] + [c.height for c in scene.cameras])
        if gl_context.GLContext is not None:
            self._gl = gl_context.GLContext(width, height)
            self._gl.make_current()
        self._con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # The framebuffer must hold the largest camera. It is sized here, in this renderer's own
        # context, so the shared model's declared size is never written.
        if width > self._con.offWidth or height > self._con.offHeight:
            mujoco.mjr_resizeOffscreen(width, height, self._con)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._con)
        self._con.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        max_geom = max([DEFAULT_MAX_GEOM] + [c.max_geom or 0 for c in scene.cameras])
        self._scene = mujoco.MjvScene(model=model, maxgeom=max_geom)
        self._clear_geoms = has_clear_geoms(model)
        _log_gl_once()

    def close(self) -> None:
        """Free the context with ITS OWN GL context current, so its deletes hit its own objects.

        ``MjrContext.free`` is a run of ``glDelete*`` calls that land in whatever context is current
        at that moment. With more than one renderer alive that is the other renderer's context, and
        GL object names are per context and start from one -- so the framebuffer and textures of a
        live renderer would be deleted in place of the dead one's, and it would render garbage from
        then on. Idempotent.
        """
        con, self._con = self._con, None
        gl, self._gl = self._gl, None
        if gl is not None:
            gl.make_current()
        if self._atlas is not None and gl is not None:
            with contextlib.suppress(Exception):
                self._atlas.free()
        self._atlas = None
        if con is not None:
            con.free()
        if gl is not None:
            gl.free()
        self._scene = None

    @property
    def scene(self) -> mujoco.MjvScene | None:
        """The ``MjvScene`` the last view was drawn with."""
        return self._scene

    # -- drawing ---------------------------------------------------------------------------------

    def render(self, snapshot: StateSnapshot, requests: Sequence[CameraRequest]) -> list[Frame]:
        if self._con is None:
            raise RenderError("MuJoCo's renderer was used before open() or after close()")
        model, data = self.model, snapshot.data
        if self._private:
            # This renderer's own copy of the model, off the physics thread: the values presence
            # changes at run time come with the snapshot.
            for name, values in snapshot.model_values.items():
                np.copyto(getattr(model, name), values)
        self.stats["batches"] += 1
        frames = [Frame(r.id, snapshot.sim_time, snapshot.step) for r in requests]
        use_atlas = bool(self.settings.get("atlas", True))
        colour = [i for i, r in enumerate(requests) if "rgb" in r.images]
        # One GL critical section for the whole batch: the collector must not run a dead renderer's
        # teardown (which makes its own context current) between this context's calls; see hold_gc.
        with hold_gc():
            if self._gl is not None:
                self._gl.make_current()
            origins: dict[int, tuple[int, int]] = {}
            if use_atlas and colour:
                if self._atlas is None:
                    self._atlas = _Atlas()
                sizes = [(requests[i].camera.width, requests[i].camera.height) for i in colour]
                origins = dict(zip(colour, self._atlas.pack(sizes), strict=True))
                # Growing the atlas binds its framebuffer; MuJoCo draws into its own.
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._con)
            for group in self._groups(requests):
                self._build(model, data, requests[group[0]])
                for i in group:
                    self._draw(model, data, requests[i], frames[i], origins.get(i))
            if origins:
                self._read_atlas(requests, frames, origins)
        return frames

    @staticmethod
    def _scene_key(request: CameraRequest):
        spec = request.camera
        decorate = None if request.decorate is None else id(request)
        return (spec.geomgroup, decorate)

    def _groups(self, requests: Sequence[CameraRequest]) -> list[list[int]]:
        """Requests that share one scene build: the same geom groups and no decorations of their own."""
        groups: dict = {}
        for i, r in enumerate(requests):
            groups.setdefault(self._scene_key(r), []).append(i)
        return list(groups.values())

    def _vopt_for(self, geomgroup) -> mujoco.MjvOption:
        if geomgroup not in self._vopts:
            self._vopts[geomgroup] = _vopt(geomgroup)
        return self._vopts[geomgroup]

    def _mjv_camera(self, request: CameraRequest) -> mujoco.MjvCamera:
        cam = request.camera.mujoco_camera
        if isinstance(cam, mujoco.MjvCamera):
            return cam
        if cam is None:
            raise RenderError(
                f"camera {request.id!r} names no MuJoCo camera; MuJoCo's renderer draws through one"
            )
        return _fixed_camera(int(cam))

    def _build(self, model, data, first: CameraRequest) -> None:
        """Populate the scene from *data*: every geom, once, for all the requests that share it."""
        scene = self._scene
        mujoco.mjv_updateScene(
            model,
            data,
            self._vopt_for(first.camera.geomgroup),
            None,
            self._mjv_camera(first),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )
        self.stats["scene_builds"] += 1
        if self._clear_geoms:
            clear_geoms_cast_no_shadow(scene)

    def _set_view(self, model, data, request: CameraRequest) -> tuple[float, float, bool]:
        """Point the built scene at *request*'s camera: its view, its lights and its planes."""
        scene = self._scene
        mujoco.mjv_updateCamera(model, data, self._mjv_camera(request), scene)
        # The headlight follows the camera, and mjv_makeLights reads the scene's camera.
        mujoco.mjv_makeLights(model, data, scene)
        spec = request.camera
        set_clip_planes(scene, spec.near, spec.far)
        for cam in scene.camera:
            if cam.frustum_far <= cam.frustum_near:
                raise ValueError(
                    f"far clip plane {cam.frustum_far:g} m is not beyond the near one "
                    f"{cam.frustum_near:g} m"
                )
        cam = scene.camera[0]
        return float(cam.frustum_near), float(cam.frustum_far), bool(cam.orthographic)

    def _draw(self, model, data, request, frame: Frame, origin) -> None:
        spec = request.camera
        w, h = spec.width, spec.height
        rect = mujoco.MjrRect(0, 0, w, h)
        scene, con = self._scene, self._con
        near, far, ortho = self._set_view(model, data, request)
        if request.decorate is not None:
            # A decorated request has its scene to itself (see _groups); its markers go in once
            # the scene and its planes are this camera's.
            request.decorate(scene)
        flip = self._gl is not None  # GL reads the bottom row first (EGL, OSMesa and GLFW alike)
        if "rgb" in request.images:
            mujoco.mjr_render(rect, scene, con)
            self.stats["draws"] += 1
            if origin is not None:
                self._atlas.copy_in(con, w, h, origin)
            else:
                rgb = np.empty((h, w, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, rect, con)
                self.stats["readbacks"] += 1
                frame.rgb = np.flipud(rgb) if flip else rgb
        if "depth" in request.images:
            frame.depth = self._depth(rect, near, far, ortho, flip)
        if "segmentation" in request.images:
            frame.segmentation = self._segmentation(rect, flip)

    def _flat_pass(self, rect, rgb=None, depth=None) -> None:
        """Draw with flat ID colours, as ``mujoco.Renderer`` does for depth and segmentation."""
        scene = self._scene
        flags = scene.flags.copy()
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        try:
            mujoco.mjr_render(rect, scene, self._con)
            mujoco.mjr_readPixels(rgb, depth, rect, self._con)
        finally:
            np.copyto(scene.flags, flags)
        self.stats["draws"] += 1
        self.stats["readbacks"] += 1

    def _depth(self, rect, near: float, far: float, ortho: bool, flip: bool) -> np.ndarray:
        """Depth in metres, linearised with the planes the scene was drawn with; ``inf`` for none.

        ``mujoco.Renderer``'s own depth mode linearises with the model's planes, so this reads the
        depth buffer and linearises it with this camera's. A pixel nothing was drawn at (cleared to
        0 under reversed Z) reads ``inf``, not the far plane.
        """
        buffer = np.empty((rect.height, rect.width), dtype=np.float32)
        self._flat_pass(rect, depth=buffer)
        depth = linear_depth(buffer, near, far, ortho)
        depth[buffer == 0.0] = np.inf
        return np.ascontiguousarray(np.flipud(depth)) if flip else depth

    def _segmentation(self, rect, flip: bool) -> np.ndarray:
        """``(H, W, 2)`` int32 object id and type, ``(-1, -1)`` for background."""
        out = np.empty((rect.height, rect.width, 3), dtype=np.uint8)
        self._flat_pass(rect, rgb=out)
        image3 = out.astype(np.uint32)
        segimage = image3[:, :, 0] + image3[:, :, 1] * (2**8) + image3[:, :, 2] * (2**16)
        scene = self._scene
        ngeoms = scene.ngeom
        segid2output = np.full((ngeoms + 1, 2), fill_value=-1, dtype=np.int32)
        visible = [g for g in scene.geoms[:ngeoms] if g.segid != -1]
        segids = np.array([g.segid + 1 for g in visible], np.int32)
        segid2output[segids, 0] = np.array([g.objid for g in visible], np.int32)
        segid2output[segids, 1] = np.array([g.objtype for g in visible], np.int32)
        seg = segid2output[segimage]
        return np.ascontiguousarray(np.flipud(seg)) if flip else seg

    def _read_atlas(self, requests, frames, origins) -> None:
        width = max(x + requests[i].camera.width for i, (x, _) in origins.items())
        height = max(y + requests[i].camera.height for i, (_, y) in origins.items())
        atlas = self._atlas.read(width, height, self._con)
        self.stats["readbacks"] += 1
        # One flip of the whole atlas (a view), and each frame a view of its tile: the atlas is
        # fresh per batch, so a frame shares it with no other batch, and a consumer that needs its
        # own contiguous copy makes the one copy it needs.
        flip = self._gl is not None
        if flip:
            atlas = np.flipud(atlas)
        for i, (x, y) in origins.items():
            w, h = requests[i].camera.width, requests[i].camera.height
            top = height - (y + h) if flip else y
            frames[i].rgb = atlas[top : top + h, x : x + w]


def draws_with_gl(renderer: str, base_dir=None) -> bool:
    """Whether *renderer* is MuJoCo's GL renderer (or built on it), so a GL failure is its own."""
    from .registry import resolve_renderer  # noqa: PLC0415 -- avoids an import cycle

    try:
        cls = resolve_renderer(renderer, base_dir)
    except PluginError:
        return False
    return issubclass(cls, MujocoRenderer)


class RenderView:
    """One camera registered with a :class:`RenderService`: what it is, and its newest frame.

    ``wants`` is the camera's own gate, asked every tick: it returns the images it wants drawn this
    tick, or nothing, so a camera nobody consumes is in no batch and costs nothing.
    """

    def __init__(self, service: RenderService, spec: CameraSpec, renderer: str, wants) -> None:
        self.service = service
        self.spec = spec
        self.renderer = renderer
        self.wants = wants
        #: Which instance of its renderer draws it (:attr:`RenderService.workers`).
        self.shard = 0
        self._frame: Frame | None = None
        self._fresh = False

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def frame(self) -> Frame | None:
        """The newest frame delivered to this view, taken or not."""
        return self._frame

    def take(self) -> Frame | None:
        """The newest frame delivered since the last ``take``, else ``None``.

        Raises the failure of a render worker that died, rather than answering "no frame yet".
        """
        self.service.raise_worker_error()
        with self.service._lock:
            if not self._fresh:
                return None
            self._fresh = False
            return self._frame

    def _deliver(self, frame: Frame) -> None:
        self._frame = frame
        self._fresh = True

    def clear(self) -> None:
        """Forget the frames: a reset restarts sim time, and an old frame would carry a future stamp."""
        with self.service._lock:
            self._frame = None
            self._fresh = False


class _Job:
    """One batch for a worker: the snapshot it is drawn from, and the views and images asked."""

    def __init__(self, snapshot: StateSnapshot, batch, requests, epoch: int, buffer) -> None:
        self.snapshot = snapshot
        self.batch = batch
        self.requests = requests
        self.epoch = epoch
        self.buffer = buffer
        self.frames: list[Frame] | None = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class _Worker:
    """A thread that owns one renderer and its context, and renders the newest batch it is handed.

    A slot of one: a batch posted while another waits replaces it, and the replaced one is dropped
    (counted in the service's ``dropped``), so a renderer slower than the ticks falls behind by at
    most one batch and nothing queues. The renderer is opened, used and closed on this thread only,
    so a thread-affine GL context is safe, and it draws its own copy of the model, updated from each
    snapshot's runtime values, so it never reads the shared model while the physics thread writes it.
    A renderer that raises ends the worker; the error is raised on the physics thread at the next
    tick or read.
    """

    #: Snapshot buffers: one being drawn, one waiting, one being filled.
    BUFFERS = 3

    def __init__(self, service: RenderService, name: str, shard: int = 0) -> None:
        self.service = service
        self.name = name
        self.shard = shard
        self.cv = threading.Condition()
        self.pending: _Job | None = None
        self.stopping = False
        #: Set once the thread has ended; a batch posted after it would never be drawn.
        self.ended = False
        self.error: BaseException | None = None
        self.renderer: Renderer | None = None
        self.opened = threading.Event()
        # On the physics thread, before the worker starts: the copy and the buffers read the model.
        self.scene = service.describe(name, private=True, shard=shard)
        self.free = [mujoco.MjData(service.model) for _ in range(self.BUFFERS)]
        self.thread = threading.Thread(
            target=self._run, name=f"roqsim-render-{name}-{shard}", daemon=True
        )
        self.thread.start()
        self.opened.wait()
        if self.error is not None:
            raise RenderError(f"renderer {name!r} failed to open: {self.error}") from self.error

    def buffer(self) -> mujoco.MjData:
        with self.cv:
            return self.free.pop()

    def post(self, job: _Job) -> None:
        with self.cv:
            if self.ended:
                self.free.append(job.buffer)
                raise RenderError(
                    f"the render worker {self.shard} of {self.name!r} has ended"
                    + (f": {self.error}" if self.error is not None else "")
                ) from self.error
            dropped, self.pending = self.pending, job
            if dropped is not None:
                self.free.append(dropped.buffer)
                self.service.stats["dropped"] += 1
                dropped.done.set()
            self.cv.notify()

    def _run(self) -> None:
        try:
            self.renderer = self.service._construct(self.name)
            self.renderer.open(self.scene)
        except BaseException as err:  # noqa: BLE001 -- handed to the physics thread
            self.error = err
            # What it took before it failed -- a context, a server process -- goes with it.
            renderer, self.renderer = self.renderer, None
            if renderer is not None:
                with contextlib.suppress(Exception):
                    renderer.close()
            with self.cv:
                self.ended = True
            self.opened.set()
            return
        self.opened.set()
        try:
            while True:
                with self.cv:
                    while self.pending is None and not self.stopping:
                        self.cv.wait()
                    if self.pending is None:
                        return
                    job, self.pending = self.pending, None
                try:
                    job.frames = self.service._draw(self.name, self.renderer, job)
                except BaseException as err:  # noqa: BLE001 -- handed to the physics thread
                    job.error = self.error = err
                    return
                finally:
                    with self.cv:
                        self.free.append(job.buffer)
                    job.done.set()
        finally:
            with self.cv:
                # A batch posted while the last one failed is answered, not left waiting.
                self.ended = True
                pending, self.pending = self.pending, None
                if pending is not None:
                    pending.error = self.error
                    self.free.append(pending.buffer)
                    pending.done.set()
            renderer, self.renderer = self.renderer, None
            with contextlib.suppress(Exception):
                renderer.close()

    def stop(self) -> None:
        with self.cv:
            self.stopping = True
            pending, self.pending = self.pending, None
            if pending is not None:
                pending.done.set()
            self.cv.notify()
        self.thread.join(timeout=30)


class RenderService:
    """The one place images come from: every camera's renderer, its GL context and its batches.

    A world's cameras register a :class:`RenderView` each (:meth:`view`). Each tick the engine calls
    :meth:`tick` between the physics step and the plugins' ``post_step``: every view is asked whether
    it is due, one :class:`~roqsim.render_api.StateSnapshot` is taken, and the due views are handed
    to their renderers as one batch per renderer; a view's frame is then read in ``post_step``
    (:meth:`RenderView.take`). :meth:`render_now` draws given views at once, for a caller with no
    tick (``roqsim render``, a thumbnail, :class:`FrameRenderer`).

    A renderer is resolved and checked against its views when the world loads (:meth:`check`), and
    constructed and opened only when its first batch comes, so a world whose cameras nobody reads
    opens no context.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        renderer: str = DEFAULT_RENDERER,
        renderers: Mapping[str, Mapping] | None = None,
        mode: str = "sync",
        workers: int = 1,
        base_dir=None,
        logger: logging.Logger | None = None,
    ) -> None:
        if mode not in RENDER_MODES:
            raise ValueError(f"render mode {mode!r}; one of {', '.join(RENDER_MODES)}")
        if int(workers) < 1:
            raise ValueError(f"render workers {workers!r}: at least 1")
        #: How many instances of each renderer share its cameras (``sim.render.workers``): one
        #: context each, and in ``async`` mode one thread each. Cameras are dealt to them in turn.
        self.workers = int(workers)
        self.model = model
        self.default_renderer = renderer
        self.settings = {str(k): dict(v or {}) for k, v in (renderers or {}).items()}
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger or _logger
        self.views: list[RenderView] = []
        self._classes: dict[str, type] = {}
        self._renderers: dict[tuple[str, int], Renderer] = {}
        self._snapshot_data: mujoco.MjData | None = None
        self._closed = False
        #: Guards the views' frames, which a worker delivers while the physics thread reads them.
        self._lock = threading.Lock()
        self._workers: dict[tuple[str, int], _Worker] = {}
        #: Bumped by :meth:`clear`: a batch drawn from before a reset is not delivered after it.
        self._epoch = 0
        #: Batches and frames since construction; ``dropped`` counts stale batches an ``async``
        #: worker never drew because a newer one replaced them.
        self.stats = {"batches": 0, "frames": 0, "ticks": 0, "dropped": 0}

    @classmethod
    def for_config(cls, model: mujoco.MjModel, config, logger=None) -> RenderService:
        """The service a world's ``sim`` block describes (a :class:`~roqsim.config.SimConfig`)."""
        return cls(
            model,
            renderer=config.renderer,
            renderers=config.renderers,
            mode=config.render["mode"],
            workers=config.render["workers"],
            base_dir=getattr(config, "base_dir", None),
            logger=logger,
        )

    @classmethod
    def of(cls, ctx) -> RenderService:
        """*ctx*'s render service, made from its world's ``sim`` block if the engine set none.

        The engine sets one at setup; a harness that configures plugins on a bare
        :class:`~roqsim.context.SimContext` gets one here, on first use.
        """
        if ctx.render is None:
            sim = (ctx.config or {}).get("sim") or {}
            ctx.render = cls(
                ctx.model,
                renderer=sim.get("renderer") or DEFAULT_RENDERER,
                renderers=sim.get("renderers") or {},
                mode=(sim.get("render") or {}).get("mode", "sync"),
                workers=(sim.get("render") or {}).get("workers", 1),
                logger=ctx.logger,
            )
        return ctx.render

    # -- registration ----------------------------------------------------------------------------

    def view(
        self,
        id: str,
        *,
        camera,
        width: int,
        height: int,
        near: float | None = None,
        far: float | None = None,
        images: Collection[str] = ("rgb",),
        renderer: str | None = None,
        wants=None,
        geomgroup: Sequence[int] | None = None,
        decorated: bool = False,
        max_geom: int | None = None,
    ) -> RenderView:
        """Register a camera; nothing renders until it is due (``wants``) or asked (:meth:`render_now`)."""
        camera = resolve_camera(self.model, camera)
        unknown = sorted(set(images) - set(IMAGES))
        if unknown:
            raise ValueError(f"view {id!r}: unknown image(s) {unknown}; one of {', '.join(IMAGES)}")
        spec = CameraSpec(
            id=id,
            width=int(width),
            height=int(height),
            model=camera_model_of(self.model, camera),
            fovy=self._fovy(camera),
            near=near,
            far=far,
            images=frozenset(images),
            mujoco_camera=camera,
            geomgroup=None if geomgroup is None else tuple(int(g) for g in geomgroup),
            decorated=decorated,
            max_geom=max_geom,
        )
        view = RenderView(self, spec, renderer or self.default_renderer, wants)
        # Dealt to the renderer's instances in turn: the shard is the service's choice, never a
        # camera's, so a second context, GPU or process changes no camera plugin.
        view.shard = sum(1 for v in self.views if v.renderer == view.renderer) % self.workers
        if any(name == view.renderer for name, _ in [*self._renderers, *self._workers]):
            raise RenderError(
                f"view {id!r} registered after renderer {view.renderer!r} opened; register every "
                "camera in configure"
            )
        self.views.append(view)
        return view

    def _fovy(self, camera) -> float:
        if isinstance(camera, mujoco.MjvCamera):
            if camera.type == mujoco.mjtCamera.mjCAMERA_FIXED and camera.fixedcamid >= 0:
                return float(self.model.cam_fovy[camera.fixedcamid])
            return float(self.model.vis.global_.fovy)
        if camera < 0:
            return float(self.model.vis.global_.fovy)
        return float(self.model.cam_fovy[camera])

    # -- the renderers ---------------------------------------------------------------------------

    def renderer_class(self, name: str) -> type:
        if name not in self._classes:
            from .registry import resolve_renderer  # noqa: PLC0415 -- avoids an import cycle

            self._classes[name] = resolve_renderer(name, self.base_dir)
        return self._classes[name]

    def check(self) -> None:
        """Refuse, naming the camera, a view its renderer cannot give; resolves every renderer.

        Called when the world loads, and raises the world-load error (:class:`PluginError`). A
        renderer that cannot give what a camera asks for is never stood in for by another.
        """
        problems = []
        for view in self.views:
            cls = self.renderer_class(view.renderer)
            caps = cls.capabilities_for(self.settings.get(view.renderer, {}))
            missing = caps.missing(
                sorted(view.spec.images),
                view.spec.model,
                view.spec.decorated,
                view.spec.geomgroup is not None,
            )
            if missing:
                problems.append(
                    f"camera {view.id!r} asks renderer {view.renderer!r} for {', '.join(missing)}; "
                    f"it renders images {', '.join(sorted(caps.images))} through camera models "
                    f"{', '.join(sorted(caps.camera_models))}"
                    + ("" if caps.decorations else ", draws no decorations")
                    + ("" if caps.geom_groups else ", draws every geom group")
                )
        if problems:
            raise PluginError("; ".join(problems))

    def _renderer(self, name: str, shard: int = 0) -> Renderer:
        """The opened instance *shard* of renderer *name*, constructed and opened on first use."""
        if self._closed:
            raise RenderError("the render service is closed")
        renderer = self._renderers.get((name, shard))
        if renderer is None:
            renderer = self._construct(name)
            try:
                renderer.open(self.describe(name, shard=shard))
            except BaseException:
                # What it took before it failed -- a context, a server process -- goes with it.
                with contextlib.suppress(Exception):
                    renderer.close()
                raise
            self._renderers[(name, shard)] = renderer
        return renderer

    def _construct(self, name: str) -> Renderer:
        cls = self.renderer_class(name)
        settings = self.settings.get(name, {})
        errors = cls.validate_config(settings)
        if errors:
            raise RenderError("; ".join(f"sim.renderers.{name}: {e}" for e in errors))
        return cls(settings)

    def describe(
        self, name: str, private: bool = False, shard: int | None = None
    ) -> SceneDescription:
        """The scene renderer *name* is opened with: the model and every camera it will draw.

        *private* hands it a copy of the model of its own (a renderer off the physics thread);
        *shard* keeps the cameras dealt to that instance.
        """
        return SceneDescription(
            model=copy.copy(self.model) if private else self.model,
            private_model=private,
            moving=moving_bodies(self.model),
            cameras=[
                v.spec
                for v in self.views
                if v.renderer == name and (shard is None or v.shard == shard)
            ],
            settings=dict(self.settings.get(name, {})),
            base_dir=self.base_dir,
            body_names=[self.model.body(i).name for i in range(self.model.nbody)],
        )

    def renderer(self, name: str | None = None, shard: int = 0) -> Renderer | None:
        """The opened renderer *name* (the default one), or ``None`` before its first batch.

        *shard* picks one of its instances when ``sim.render.workers`` deals its cameras to more.
        """
        name = name or self.default_renderer
        worker = self._workers.get((name, shard))
        if worker is not None:
            return worker.renderer
        return self._renderers.get((name, shard))

    def raise_worker_error(self) -> None:
        """Raise, on the caller's thread, the failure of a render worker that died."""
        for (name, shard), worker in self._workers.items():
            if worker.error is not None:
                raise RenderError(
                    f"the render worker {shard} of {name!r} died: {worker.error}"
                ) from worker.error

    # -- per tick --------------------------------------------------------------------------------

    def snapshot(self, data: mujoco.MjData, buffer: mujoco.MjData | None = None) -> StateSnapshot:
        """Copy what a render reads off *data*; the renderer never sees the live ``MjData``."""
        if buffer is None:
            if self._snapshot_data is None:
                self._snapshot_data = mujoco.MjData(self.model)
            buffer = self._snapshot_data
        mujoco.mj_copyData(buffer, self.model, data)
        timestep = float(self.model.opt.timestep)
        return StateSnapshot(
            sim_time=float(data.time),
            step=int(round(float(data.time) / timestep)) if timestep > 0 else 0,
            xpos=np.array(data.xpos),
            xquat=np.array(data.xquat),
            model_values={f: np.array(getattr(self.model, f)) for f in RUNTIME_MODEL_FIELDS},
            data=buffer,
        )

    def request(self, view: RenderView, images, snapshot: StateSnapshot, decorate=None):
        """The :class:`~roqsim.render_api.CameraRequest` for *view* at *snapshot*."""
        pos, quat = self._camera_pose(view.spec.mujoco_camera, snapshot.data)
        return CameraRequest(
            camera=view.spec,
            images=frozenset(images),
            pos=pos,
            quat=quat,
            fovy=view.spec.fovy,
            decorate=decorate,
        )

    def _camera_pose(self, camera, data) -> tuple[np.ndarray, np.ndarray]:
        """A camera's world pose (``w, x, y, z``; MuJoCo's camera frame) in the state *data* holds."""
        quat = np.zeros(4)
        if isinstance(camera, mujoco.MjvCamera):
            if camera.type == mujoco.mjtCamera.mjCAMERA_FIXED and camera.fixedcamid >= 0:
                camera = int(camera.fixedcamid)
            else:
                # A free or tracking camera: its eye and frame as MuJoCo derives them.
                forward, up = np.zeros(3), np.zeros(3)
                pos = np.zeros(3)
                lookat = np.array(camera.lookat, dtype=float)
                az, el = math.radians(camera.azimuth), math.radians(camera.elevation)
                forward[:] = [
                    math.cos(el) * math.cos(az),
                    math.cos(el) * math.sin(az),
                    math.sin(el),
                ]
                pos[:] = lookat - float(camera.distance) * forward
                up[:] = [-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)]
                right = np.cross(forward, up)
                mat = np.column_stack((right, up, -forward)).reshape(-1)
                mujoco.mju_mat2Quat(quat, mat)
                return pos, quat
        if camera < 0:
            return self._camera_pose(default_free_camera(self.model), data)
        mujoco.mju_mat2Quat(quat, np.asarray(data.cam_xmat[camera], dtype=float))
        return np.array(data.cam_xpos[camera]), quat

    def tick(self, data: mujoco.MjData) -> int:
        """Render every view due now; the engine calls it after the physics step. Returns how many."""
        self.stats["ticks"] += 1
        due = []
        for view in self.views:
            if view.wants is None:
                continue
            images = view.wants()
            if images:
                due.append((view, images))
        if not due:
            return 0
        self._render(due, data, wait=False)
        return len(due)

    def render_now(self, views: Sequence[tuple[RenderView, Collection[str]]], data, decorate=None):
        """Draw *views* (each with the images asked for) from the state *data* holds, now."""
        return self._render(list(views), data, decorate)

    def _render(self, due, data, decorate=None, wait: bool = True) -> list[Frame]:
        """Draw *due* from *data*: here (``sync``), or by each renderer's worker (``async``).

        In ``async`` mode a tick (``wait`` false) hands the batches over and returns; the frames
        reach the views when drawn. :meth:`render_now` waits for them in either mode.
        """
        self.raise_worker_error()
        by_renderer: dict[tuple[str, int], list] = {}
        for view, images in due:
            by_renderer.setdefault((view.renderer, view.shard), []).append((view, images))
        if self.mode == "sync":
            snapshot = self.snapshot(data)
            frames: list[Frame] = []
            for (name, shard), batch in by_renderer.items():
                requests = [self.request(v, imgs, snapshot, decorate) for v, imgs in batch]
                job = _Job(snapshot, batch, requests, self._epoch, None)
                frames += self._draw(name, self._renderer(name, shard), job)
            return frames
        jobs = []
        for (name, shard), batch in by_renderer.items():
            worker = self._worker(name, shard)
            buffer = worker.buffer()
            snapshot = self.snapshot(data, buffer)
            requests = [self.request(v, imgs, snapshot, decorate) for v, imgs in batch]
            job = _Job(snapshot, batch, requests, self._epoch, buffer)
            worker.post(job)
            jobs.append(job)
        if not wait:
            return []
        frames = []
        for job in jobs:
            job.done.wait()
            if job.error is not None:
                raise RenderError(f"renderer failed: {job.error}") from job.error
            if job.frames is None:
                # Replaced by a tick's newer batch before it was drawn: ask again.
                return self._render(due, data, decorate, wait=True)
            frames += job.frames
        return frames

    def _worker(self, name: str, shard: int = 0) -> _Worker:
        if self._closed:
            raise RenderError("the render service is closed")
        worker = self._workers.get((name, shard))
        if worker is None:
            worker = self._workers[(name, shard)] = _Worker(self, name, shard)
        return worker

    def _draw(self, name: str, renderer: Renderer, job: _Job) -> list[Frame]:
        """Render one job and deliver its frames to its views (on whichever thread renders)."""
        try:
            out = renderer.render(job.snapshot, job.requests)
        except RenderError:
            raise
        except Exception as err:
            raise RenderError(f"renderer {name!r} failed: {err}") from err
        if len(out) != len(job.requests):
            raise RenderError(
                f"renderer {name!r} returned {len(out)} frames for {len(job.requests)} requests"
            )
        for (view, images), frame in zip(job.batch, out, strict=True):
            self._check_frame(name, view, images, frame, job.snapshot)
        with self._lock:
            if job.epoch == self._epoch:
                for (view, _), frame in zip(job.batch, out, strict=True):
                    view._deliver(frame)
            self.stats["batches"] += 1
            self.stats["frames"] += len(out)
        return list(out)

    @staticmethod
    def _check_frame(name: str, view: RenderView, images, frame: Frame, snapshot) -> None:
        """Refuse a frame that is not the one asked for: its camera, its stamp, its images."""
        problems = []
        if frame.id != view.id:
            problems.append(f"it is camera {frame.id!r}'s")
        if frame.sim_time != snapshot.sim_time or frame.step != snapshot.step:
            problems.append(
                f"it is stamped t={frame.sim_time!r} step {frame.step!r}, the snapshot "
                f"t={snapshot.sim_time!r} step {snapshot.step!r}"
            )
        for kind in images:
            image = frame.image(kind)
            if image is None:
                problems.append(f"it has no {kind}")
                continue
            mismatch = image_mismatch(kind, image, view.spec.width, view.spec.height)
            if mismatch:
                problems.append(f"its {mismatch}")
        if problems:
            raise RenderError(
                f"renderer {name!r} returned a frame for camera {view.id!r} that is not the one "
                f"asked for: {'; '.join(problems)}"
            )

    def clear(self) -> None:
        """Forget every view's frames (an engine reset); a batch still being drawn is not delivered."""
        with self._lock:
            self._epoch += 1
        for view in self.views:
            view.clear()

    def close(self) -> None:
        """Close every renderer it opened. Idempotent; the service renders nothing afterwards."""
        workers, self._workers = self._workers, {}
        for worker in workers.values():
            worker.stop()  # the worker closes its renderer on its own thread
        renderers, self._renderers = self._renderers, {}
        self._closed = True
        for name, renderer in renderers.items():
            try:
                renderer.close()
            except Exception:  # noqa: BLE001 -- one renderer's failure must not keep another open
                # At interpreter exit the GL display may already be gone; nothing is left to free.
                if not sys.is_finalizing():
                    self.logger.exception("renderer %s failed to close", name)

    def __del__(self) -> None:
        # Attribute access can fail during interpreter shutdown; there is nothing to do then.
        with contextlib.suppress(Exception):
            self.close()


class FrameRenderer:
    """One view drawn through a render service of its own: what a tool with no world tick uses.

    ``camera`` may be a :class:`mujoco.MjvCamera` (a free/orbit camera; the default is
    :func:`default_free_camera`) or an ``int`` camera id / camera name for a fixed MJCF ``<camera>``.
    :meth:`render` returns a fresh ``(height, width, 3)`` uint8 array each call; :meth:`render_depth`
    and :meth:`render_segmentation` are the other passes. ``renderer`` names the renderer it draws
    through (MuJoCo's by default) and ``settings`` its settings block, as a world's ``sim.renderer``
    and ``sim.renderers.<name>`` do. ``max_geom`` raises the geoms its scene holds, for a caller that
    decorates it with many markers.

    For interactive navigation of a free camera, :meth:`move_camera` forwards mouse deltas to
    MuJoCo's own :func:`mujoco.mjv_moveCamera` -- the exact routine the native viewer uses -- so the
    orbit/pan/zoom feel matches, and :meth:`select` ray-picks MuJoCo's scene.

    ``near``/``far`` are this renderer's clip planes in metres (see the module docstring); ``None``
    keeps the model's. Both are settable between renders. A see-through geom throws no shadow:
    every renderer here moves a geom whose resolved alpha is below 1 out of MuJoCo's shadow pass
    (:func:`clear_geoms_cast_no_shadow`).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int,
        height: int,
        camera: mujoco.MjvCamera | int | str | None = None,
        geomgroup: Sequence[int] | None = None,
        *,
        near: float | None = None,
        far: float | None = None,
        renderer: str = DEFAULT_RENDERER,
        settings: Mapping | None = None,
        max_geom: int | None = None,
    ) -> None:
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.camera: mujoco.MjvCamera | int | str = (
            camera if camera is not None else default_free_camera(model)
        )
        self._geomgroup = geomgroup
        self._max_geom = max_geom
        self._service = RenderService(
            model, renderer=renderer, renderers={renderer: dict(settings or {})}
        )
        self._view: RenderView | None = None
        self._pick_scene: mujoco.MjvScene | None = None
        self._vopt = _vopt(geomgroup)
        self._near: float | None = None
        self._far: float | None = None
        self.near = near
        self.far = far
        # Open the renderer now, as a renderer of a single view always has: a GL that cannot render
        # here fails at construction, not at the first frame.
        self._view_for(self.camera)
        self._service._renderer(renderer)

    @property
    def near(self) -> float | None:
        """The near clip plane in metres; ``None`` is the model's own."""
        return self._near

    @near.setter
    def near(self, value: float | None) -> None:
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"near clip plane must be a finite distance > 0 m, got {value!r}")
        self._near = None if value is None else float(value)

    @property
    def far(self) -> float | None:
        """The far clip plane in metres; ``None`` is the model's own."""
        return self._far

    @far.setter
    def far(self, value: float | None) -> None:
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"far clip plane must be a finite distance > 0 m, got {value!r}")
        self._far = None if value is None else float(value)

    @property
    def service(self) -> RenderService:
        """The render service this renderer draws through."""
        return self._service

    def _view_for(self, camera) -> RenderView:
        """This renderer's one view, pointed at *camera*, with its model and planes as they are now."""
        resolved = resolve_camera(self.model, camera)
        if self._view is None:
            self._view = self._service.view(
                "frame_renderer",
                camera=resolved,
                width=self.width,
                height=self.height,
                images=IMAGES,
                geomgroup=self._geomgroup,
                max_geom=self._max_geom,
            )
        spec = self._view.spec
        spec.mujoco_camera = resolved
        spec.model = camera_model_of(self.model, resolved)
        spec.fovy = self._service._fovy(resolved)
        spec.near, spec.far = self._near, self._far
        return self._view

    def _draw(self, data: mujoco.MjData, image: str, decorate=None) -> Frame:
        if self._service is None:
            raise AttributeError("this FrameRenderer is closed")
        view = self._view_for(self.camera)
        caps = self._service.renderer_class(view.renderer).capabilities_for(
            self._service.settings.get(view.renderer, {})
        )
        missing = caps.missing(
            [image], view.spec.model, decorate is not None, view.spec.geomgroup is not None
        )
        if missing:
            raise RenderError(
                f"renderer {view.renderer!r} cannot give {', '.join(missing)} for this view"
            )
        (frame,) = self._service.render_now([(view, (image,))], data, decorate)
        return frame

    @property
    def scene(self) -> mujoco.MjvScene:
        """The ``MjvScene`` the last frame was drawn from (needed by :func:`mujoco.mjv_moveCamera`).

        MuJoCo's renderer's own scene; for another renderer, one built on the CPU for picking.
        """
        renderer = self._service.renderer() if self._service is not None else None
        scene = getattr(renderer, "scene", None)
        if scene is not None:
            return scene
        if self._pick_scene is None:
            self._pick_scene = mujoco.MjvScene(self.model, maxgeom=DEFAULT_MAX_GEOM)
        return self._pick_scene

    def render(self, data: mujoco.MjData, decorate=None) -> np.ndarray:
        """Update the scene from ``data`` through :attr:`camera` and return an RGB frame.

        ``decorate`` is an optional ``callable(scene)`` invoked after the scene is populated and
        before rasterisation, so a caller can append transient marker geoms (e.g. via
        :func:`mujoco.mjv_initGeom`) that render in 3D and track the camera.
        """
        return np.array(self._draw(data, "rgb", decorate).rgb, order="C")

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Render ``data`` through :attr:`camera`; depth in metres, ``(height, width)`` float32.

        Linearised with the planes the scene was drawn with; a pixel nothing was drawn at reads
        ``inf``, not the far plane.
        """
        return self._draw(data, "depth").depth

    def render_segmentation(
        self, data: mujoco.MjData, scene_option: mujoco.MjvOption | None = None
    ) -> np.ndarray:
        """The segmentation pass through :attr:`camera`, with this renderer's planes.

        ``(height, width, 2)`` int32 object id and type, ``(-1, -1)`` for background.
        *scene_option* is accepted for the options of :class:`FrameRenderer`'s own (absence
        excluded); one that draws other geom groups is refused.
        """
        if scene_option is not None and list(scene_option.geomgroup) != list(self._vopt.geomgroup):
            raise ValueError(
                "render_segmentation: a scene_option drawing other geom groups than this "
                "renderer's; construct the FrameRenderer with that geomgroup instead"
            )
        return self._draw(data, "segmentation").segmentation

    def move_camera(self, action: int, dx: float, dy: float) -> None:
        """Apply a mouse move to a free camera via MuJoCo's own handler.

        ``action`` is a :class:`mujoco.mjtMouse` value (rotate/move/zoom); ``dx``/``dy`` are motions
        as a fraction of window height. No-op unless :attr:`camera` is an ``MjvCamera``.
        """
        if not isinstance(self.camera, mujoco.MjvCamera):
            return
        # MuJoCo 3.11 dropped the MjvScene argument from mjv_moveCamera; roqsim supports >= 3.0, so try
        # the current signature and fall back to the legacy one (probed once, not per drag event).
        global _MOVE_CAMERA_TAKES_SCENE
        if _MOVE_CAMERA_TAKES_SCENE is None:
            try:
                mujoco.mjv_moveCamera(self.model, action, dx, dy, self.camera)
            except TypeError:
                _MOVE_CAMERA_TAKES_SCENE = True
            else:
                _MOVE_CAMERA_TAKES_SCENE = False
                return
        if _MOVE_CAMERA_TAKES_SCENE:
            mujoco.mjv_moveCamera(self.model, action, dx, dy, self.scene, self.camera)
        else:
            mujoco.mjv_moveCamera(self.model, action, dx, dy, self.camera)

    def look_camera(self, dx: float, dy: float) -> None:
        """First-person mouse-look for a free camera (see :func:`look_in_place`)."""
        if isinstance(self.camera, mujoco.MjvCamera):
            look_in_place(self.camera, dx, dy)

    def dolly_camera(self, step: float) -> None:
        """Move a free camera along its view direction (see :func:`dolly`)."""
        if isinstance(self.camera, mujoco.MjvCamera):
            dolly(self.camera, step)

    def walk_camera(self, keys, step: float) -> None:
        """Translate a free camera for the held WASD/QE keys (see :func:`walk_delta`).

        Shifting the orbit point carries the eye with it -- the free camera sits a fixed distance
        behind ``lookat`` -- which is what makes this a walk rather than an orbit. No-op unless
        :attr:`camera` is an ``MjvCamera``.
        """
        if isinstance(self.camera, mujoco.MjvCamera):
            cam = self.camera
            cam.lookat[:] = np.array(cam.lookat) + walk_delta(
                cam.azimuth, keys, step, cam.elevation
            )

    def select(self, data: mujoco.MjData, relx: float, rely: float) -> tuple[int, int, np.ndarray]:
        """Ray-pick the last-rendered scene at a click; return ``(geom_id, body_id, world_point)``.

        ``relx``/``rely`` are normalized in ``[0, 1]``: ``relx`` from the left, ``rely`` from the
        **bottom** (OpenGL convention). ``geom_id`` and ``body_id`` are ``-1`` when the click misses
        all geometry; ``world_point`` is the 3D hit point (meaningful only on a hit). Call after a
        :meth:`render`, whose scene this casts against.
        """
        selpnt = np.zeros(3)
        geomid = np.zeros(1, np.int32)
        flexid = np.zeros(1, np.int32)
        skinid = np.zeros(1, np.int32)
        body_id = mujoco.mjv_select(
            self.model,
            data,
            self._vopt,
            self.width / self.height,
            relx,
            rely,
            self.scene,
            selpnt,
            geomid,
            flexid,
            skinid,
        )
        return int(geomid[0]), int(body_id), selpnt.copy()

    def close(self) -> None:
        """Close the renderer and its context. Idempotent, and also what ``__del__`` runs."""
        service, self._service = self._service, None
        if service is not None:
            service.close()

    def __del__(self) -> None:
        # Attribute access can fail during interpreter shutdown; there is nothing to do then.
        with contextlib.suppress(Exception):
            self.close()
