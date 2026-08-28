"""Offscreen frame rendering — one owner of the ``mujoco.Renderer`` lifecycle.

Rendering an RGB frame from a model + data is the same three calls everywhere (create a
``mujoco.Renderer``, ``update_scene(data, camera)``, ``render()``), so this collects them behind
:class:`FrameRenderer`: the thumbnail tool, the RGB(-D) camera sensor, and the interactive
scene-review tool all share it instead of re-deriving the pattern.

Rendering is GL-context- and thread-affine (see ``docs/future_work.rst``): create and drive a
``FrameRenderer`` on one thread only, and honour the single-writer rule — the ``data`` you render
from must not be mutated by another thread meanwhile.
"""

from __future__ import annotations

import logging
import math
import os

import mujoco
import numpy as np

from . import raycast
from .presence import ABSENT_GEOM_GROUP

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
    rotated local frame). ``data`` must be forward-kinematics-current (call ``mj_forward`` first).
    """
    roots = {int(b) for b in body_ids if int(b) >= 0}
    keep = set(roots)
    for b in range(model.nbody):  # pull in descendants so a robot's links count, not just its root
        p = b
        while p > 0:
            if p in roots:
                keep.add(b)
                break
            p = int(model.body_parentid[p])

    spheres = [
        (data.geom_xpos[g], float(model.geom_rbound[g]))
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in keep and float(model.geom_rbound[g]) > 0.0
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
            "  /dev/nvidiactl exists, /dev/dri/renderD128 does not -- the signature of a "
            "container started without the 'graphics' driver capability.\n"
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


def _log_gl_once() -> None:
    """Log the backend *and* the device it bound, once, at INFO.

    This is the line that answers "did this run use the GPU", and it is worth a log entry
    because the alternative -- inferring it from wall-clock afterwards -- is how a mis-bound
    backend went unnoticed across every campaign this substrate had run.
    """
    global _gl_logged
    if _gl_logged:
        return
    _gl_logged = True
    device = bound_gl_device()
    _logger.info(
        "offscreen GL: %s%s", bound_gl_backend() or "unknown", f" -- {device}" if device else ""
    )


class FrameRenderer:
    """Owns a ``mujoco.Renderer`` and the camera it renders through.

    ``camera`` may be a :class:`mujoco.MjvCamera` (a free/orbit camera; the default is
    :func:`default_free_camera`) or an ``int`` camera id / camera name for a fixed MJCF ``<camera>``.
    :meth:`render` returns a fresh ``(height, width, 3)`` uint8 array each call.

    For interactive navigation of a free camera, :meth:`move_camera` forwards mouse deltas to
    MuJoCo's own :func:`mujoco.mjv_moveCamera` — the exact routine the native viewer uses — so the
    orbit/pan/zoom feel matches. The underlying renderer is exposed as :attr:`raw` for callers that
    need extra passes (e.g. depth).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int,
        height: int,
        camera: mujoco.MjvCamera | int | str | None = None,
    ) -> None:
        # Before anything touches GL: a wrong backend aborts the process inside MjrContext with a
        # message that explains nothing, and this is the funnel every renderer in the tree goes
        # through, so one check here covers cameras, the thumbnailer and `roqsim render` alike.
        check_gl_backend()
        # The offscreen framebuffer must be at least as large as the requested frame, or MuJoCo
        # clips the render; bump the model's declared size before constructing the renderer.
        model.vis.global_.offwidth = max(int(width), int(model.vis.global_.offwidth))
        model.vis.global_.offheight = max(int(height), int(model.vis.global_.offheight))
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._renderer = mujoco.Renderer(model, self.height, self.width)
        _log_gl_once()
        self._vopt = mujoco.MjvOption()
        # An absent entity is not in the picture either. Its geoms already have zero alpha, but
        # excluding the group is what makes absence one decision rather than several that could
        # drift -- and it is the same group every raycast mask excludes.
        self._vopt.geomgroup[ABSENT_GEOM_GROUP] = 0
        self.camera: mujoco.MjvCamera | int | str = (
            camera if camera is not None else default_free_camera(model)
        )

    @property
    def raw(self) -> mujoco.Renderer:
        """The underlying ``mujoco.Renderer`` (for extra passes such as depth)."""
        return self._renderer

    @property
    def scene(self) -> mujoco.MjvScene:
        """The renderer's ``MjvScene`` (needed by :func:`mujoco.mjv_moveCamera`)."""
        return self._renderer.scene

    def render(self, data: mujoco.MjData, decorate=None) -> np.ndarray:
        """Update the scene from ``data`` through :attr:`camera` and return an RGB frame.

        ``decorate`` is an optional ``callable(scene)`` invoked after the scene is populated and
        before rasterisation, so a caller can append transient marker geoms (e.g. via
        :func:`mujoco.mjv_initGeom`) that render in 3D and track the camera.
        """
        self._renderer.update_scene(data, self.camera)
        if decorate is not None:
            decorate(self._renderer.scene)
        return self._renderer.render()

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
            mujoco.mjv_moveCamera(self.model, action, dx, dy, self._renderer.scene, self.camera)
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
        :meth:`render`, whose ``update_scene`` populates the scene this casts against.
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
            self._renderer.scene,
            selpnt,
            geomid,
            flexid,
            skinid,
        )
        return int(geomid[0]), int(body_id), selpnt.copy()

    def close(self) -> None:
        self._renderer.close()
