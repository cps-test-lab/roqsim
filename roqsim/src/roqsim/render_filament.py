# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo's Filament renderer (``mjrf``) behind the render service, headless through Vulkan.

Filament is MuJoCo's physically based renderer: per-pixel lighting from any number of lights,
soft shadows (``filament.shadows.type``, a light's ``bulbradius``), image-based light
(``light type="image"``), metallic-roughness materials and material layers, screen-space ambient
occlusion, tone mapping and colour grading (``filament.*`` custom numerics and texts). It reads
all of that from the model itself (:func:`mjrf.Scene.configure_from_model`, ``ModelResources``,
``ModelLights``), so a world turns it on in its MJCF, and a world that sets none of it is drawn by
Filament's own fallback for models authored for the classic renderer: a default environment light,
the headlight, the model's lights sharing a fixed intensity (MuJoCo's rendering documentation,
"Lighting").

What this module adds is roqsim's contract (:mod:`roqsim.render_api`):

* **Headless, Vulkan only.** Filament's OpenGL backend opens an X display (GLX, not EGL) and ends
  the process when there is none; its Vulkan backend needs no display. A Vulkan context on a
  machine without a Vulkan driver also ends the process, inside Filament, so before the first
  context in a process a child process opens one (:func:`probe`): a failure is then a
  :class:`~roqsim.render_api.RenderError` with Filament's own words, and the device Filament
  selects is known. A process given an NVIDIA GPU (:func:`roqsim.gl.gpu_visible`) that would
  render on a CPU rasteriser is refused unless ``software: true`` asks for that.
* **Contexts are borrowed, never destroyed.** Filament's engine aborts the process when it is
  destroyed off the thread that made it, and destroying a Vulkan context on an NVIDIA driver
  terminates the EGL display MuJoCo's classic renderer shares, so every classic renderer of the
  process fails from then on. And one context holds the resources of one textured model at a
  time: a second model's renders fail while the first is loaded (MuJoCo 4.0). So each renderer
  borrows a context of its thread's pool (:func:`borrow`), made when none is free, frees only its
  world's meshes, textures and renderables on close, on its own thread, and gives the context
  back; contexts are kept until the process ends. A thread holds as many as it had renderers
  open at once; a thread that ends leaves its own behind (an ``async`` render worker's).
* **Depth and labels as every renderer gives them.** Depth is read in Filament's reversed-Z and
  linearised to metres along the view axis, ``inf`` where nothing was drawn; segmentation is read
  as Filament's ids (an object's renderable's ``segmentation_id``, 0 for the background) and turned
  into ``(object id, mjtObj type)`` from the renderables themselves.
* **Colour textures as the classic renderer shows them.** A material's colour texture whose
  colorspace the model leaves at ``auto`` is read as sRGB, as its ``rgba`` colours are, on the
  renderer's own copy of the model (``colour_textures``; :func:`colour_textures_left_auto`).
* **The camera MuJoCo would draw**, from ``mjv_camera2GLCamera``, with the camera's own clip planes.

What it does not do: MuJoCo decorations (``mjvGeom`` markers), the moving bodies alone. A camera
that asks for either is refused when the world loads (:attr:`FilamentRenderer.capabilities`).
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import logging
import re
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from .render_api import (
    CAMERA_MODELS,
    IMAGES,
    CameraRequest,
    Frame,
    Renderer,
    RendererCapabilities,
    RenderError,
    SceneDescription,
    Setting,
    StateSnapshot,
)
from .rendering import _vopt, set_camera_planes, software_rendered

_logger = logging.getLogger(__name__)

#: The headlight's intensity in lux when the model lights the classic way (no light with a positive
#: ``intensity``) and sets no ``filament.fallback.head_light_intensity``: MuJoCo's documented default.
FALLBACK_HEADLIGHT_LUX = 40000.0

#: The objects whose renderables carry a segmentation id, in the order ``ModelRenderables`` makes
#: them, with the model's count of each.
_SEGMENTED = (
    (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
    (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    (mujoco.mjtObj.mjOBJ_SKIN, "nskin"),
    (mujoco.mjtObj.mjOBJ_FLEX, "nflex"),
    (mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
)

_DEVICE_LINE = re.compile(r"Selected physical device '([^']*)'")

#: The child that opens a Vulkan context and reports what Filament selected.
_PROBE = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # Filament aborts where it cannot open: no core file
import mujoco.studio.window  # registers the filament: assets (materials) Filament loads
from mujoco import mjrf
cfg = mjrf.ContextConfig()
cfg.graphics_api = mjrf.GraphicsApi.GRAPHICS_API_VULKAN
ctx = mjrf.Context(cfg)
print("ok", flush=True)
"""

_probe_lock = threading.Lock()
_probe_result: tuple[str, str] | None = None

#: Objects held for the rest of the process: contexts, and a world's objects closed off its thread.
_kept: list = []

#: This thread's free Filament contexts (:func:`borrow`).
_local = threading.local()


def probe(timeout_s: float = 120.0) -> tuple[str, str]:
    """Open a Filament Vulkan context in a child process: ``(device, log)``.

    Once per process; the answer is kept. Raises :class:`RenderError` with the child's output if it
    cannot, where the same call in this process would have ended it. ``device`` is the physical
    device Filament selected (``""`` if its log did not name one).
    """
    global _probe_result
    with _probe_lock:
        if _probe_result is not None:
            return _probe_result
        try:
            run = subprocess.run(
                [sys.executable, "-c", _PROBE],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as err:
            raise RenderError(
                f"filament: opening a Vulkan context did not finish within {timeout_s:g} s"
            ) from err
        log = (run.stdout or "") + (run.stderr or "")
        if run.returncode != 0 or "ok" not in (run.stdout or "").split():
            raise RenderError(
                "filament: Filament could not open a headless Vulkan context here (exit status "
                f"{run.returncode}); it needs a Vulkan driver for the GPU (for NVIDIA in a "
                "container, the 'graphics' driver capability, which mounts the Vulkan ICD). "
                "Filament said:\n  " + "\n  ".join(log.strip().splitlines()[-12:])
            )
        match = _DEVICE_LINE.search(log)
        _probe_result = (match.group(1) if match else "", log)
        return _probe_result


def check_device(device: str, allow_software: bool) -> None:
    """Refuse a CPU rasteriser in a process that was given an NVIDIA GPU, unless asked for."""
    from .gl import gpu_visible  # noqa: PLC0415 -- the device checks live with the GL ones

    if allow_software or not device or not software_rendered(device) or not gpu_visible():
        return
    raise RenderError(
        f"filament: this process was given an NVIDIA GPU but Vulkan selected {device!r}, a CPU "
        "rasteriser, which draws correct frames many times slower. Mount the NVIDIA Vulkan driver "
        "(the container's 'graphics' driver capability), or set "
        "sim.renderers.filament.software: true if rendering on the CPU is what you want."
    )


#: The roles in which a material's texture is a colour, read as one (``mjtTextureRole``).
_COLOUR_ROLES = (
    mujoco.mjtTextureRole.mjTEXROLE_RGB,
    mujoco.mjtTextureRole.mjTEXROLE_RGBA,
    mujoco.mjtTextureRole.mjTEXROLE_EMISSIVE,
)


def colour_textures_left_auto(model: mujoco.MjModel) -> np.ndarray:
    """The ids of the textures some material uses as a colour whose colorspace is ``auto``.

    MuJoCo reads such a texture as linear unless its file says otherwise, and the classic renderer
    shows its bytes as display values whatever it says; a world authored under it has display
    (sRGB) colours in them. Filament honours the colourspace, so read as linear they are drawn
    lighter and paler than the same colour as an ``rgba``, which it reads as sRGB.
    """
    if not model.nmat or not model.ntex:
        return np.zeros(0, dtype=int)
    ids = np.asarray(model.mat_texid)[:, [int(r) for r in _COLOUR_ROLES]].ravel()
    ids = np.unique(ids[ids >= 0])
    auto = np.asarray(model.tex_colorspace)[ids] == int(mujoco.mjtColorSpace.mjCOLORSPACE_AUTO)
    return ids[auto]


def _custom_numeric(model: mujoco.MjModel, name: str) -> np.ndarray | None:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if i < 0:
        return None
    adr, size = int(model.numeric_adr[i]), int(model.numeric_size[i])
    return np.array(model.numeric_data[adr : adr + size], dtype=float)


def photometric(model: mujoco.MjModel) -> bool:
    """Whether Filament lights *model* photometrically: some light has a positive ``intensity``."""
    return bool(model.nlight and np.any(np.asarray(model.light_intensity) > 0))


def headlight(model: mujoco.MjModel) -> tuple[bool, tuple[float, float, float], float]:
    """``(on, colour, lux)`` of the headlight Filament draws *model* with.

    MuJoCo's rule (rendering documentation, "Lighting"): a model with a photometric light has no
    headlight; one lit the classic way has it, at ``filament.fallback.head_light_intensity`` lux
    (40 000 when unset), coloured by ``visual/headlight/diffuse`` as MuJoCo Studio draws it. The
    headlight's ``active`` is on for 1 and for 2 (MuJoCo 4.0's ``auto``) and off for 0.
    """
    colour = tuple(float(c) for c in model.vis.headlight.diffuse)
    if photometric(model) or int(model.vis.headlight.active) == 0:
        return False, colour, 0.0
    lux = _custom_numeric(model, "filament.fallback.head_light_intensity")
    return True, colour, float(lux[0]) if lux is not None and lux.size else FALLBACK_HEADLIGHT_LUX


def linear_depth(
    buffer: np.ndarray, near: float, far: float, orthographic: bool = False
) -> np.ndarray:
    """Metres along the view axis from Filament's reversed-Z depth; ``inf`` where it is 0.

    Perspective: ``d = n (f - z) / (z (f - n))``, so ``z = n f / (d (f - n) + n)``; orthographic:
    ``d = (f - z) / (f - n)``.
    """
    d = buffer.astype(np.float64)
    n, f = float(near), float(far)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = f - d * (f - n) if orthographic else n * f / (d * (f - n) + n)
    out = z.astype(np.float32)
    out[buffer <= 0.0] = np.inf
    return out


def borrow():
    """A Filament Vulkan context of this thread no renderer holds: a free one, else a new one.

    Never destroyed (the module docstring says why); :func:`give_back` returns it to the pool.
    """
    free = getattr(_local, "free", None)
    if free is None:
        free = _local.free = []
    if free:
        return free.pop()
    from mujoco import mjrf  # noqa: PLC0415

    cfg = mjrf.ContextConfig()
    cfg.graphics_api = mjrf.GraphicsApi.GRAPHICS_API_VULKAN
    ctx = mjrf.Context(cfg)
    _keep_forever([ctx])
    return ctx


def give_back(ctx) -> None:
    """Return *ctx*, made on this thread, to its pool once its world's resources are freed."""
    ctx.flush()
    getattr(_local, "free", []).append(ctx)


def _keep_forever(objects: list) -> None:
    """Hold *objects* until the process ends, past interpreter teardown: never destroyed."""
    for obj in objects:
        if obj is None:
            continue
        _kept.append(obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(obj))


class FilamentRenderer(Renderer):
    """MuJoCo's Filament renderer, headless through Vulkan: one context and one scene per world.

    Per batch it updates the scene's lights and renderables from the snapshot once per set of geom
    groups, then submits every camera's images of that set -- colour in the default draw mode,
    depth and segmentation in Filament's own draw modes -- one ``mjrf`` render call per image (it
    reads back one target per call), and waits once they are all submitted.
    """

    capabilities = RendererCapabilities(
        images=frozenset(IMAGES),
        camera_models=frozenset(CAMERA_MODELS),
        composes_moving_bodies=True,
        draws_static=True,
        batched=True,
        deterministic=False,
        in_process=True,
        decorations=False,
        geom_groups=True,
        principal_point=True,
        non_square_pixels=True,
        moving_only=False,
    )
    SETTINGS = {
        "shadows": Setting(True, "draw shadows (their kind is the model's filament.shadows.type)"),
        "post_processing": Setting(
            True,
            "tone mapping, colour grading, ambient occlusion, anti-aliasing and bloom as the "
            "model's filament.* flags set them; false draws without them, and only then is a "
            "frame a function of the state alone (Filament's dithering and ambient occlusion "
            "change from frame to frame)",
        ),
        "reflections": Setting(True, "draw the reflections of materials with a reflectance"),
        "colour_textures": Setting(
            "srgb",
            "how a colour texture (a material's rgb, rgba or emissive) whose colorspace the model "
            "leaves at auto is read: srgb, as the classic renderer shows it and as rgba colours "
            "are read; model, as MuJoCo reads it (linear unless the image file says otherwise), "
            "which draws it lighter and paler than the same colour as rgba. An explicit "
            "colorspace is always honoured",
        ),
        "software": Setting(
            False,
            "allow a CPU rasteriser (lavapipe) in a process that was given an NVIDIA GPU; "
            "refused otherwise, as it draws correct frames many times slower",
        ),
    }

    @classmethod
    def capabilities_for(cls, settings: Mapping) -> RendererCapabilities:
        """Deterministic only without post-processing: Filament's dithering and ambient occlusion
        draw a noise that changes from frame to frame, so with it a frame depends on the frames
        drawn before it, not on the snapshot alone."""
        post = bool(dict(settings or {}).get("post_processing", True))
        return dataclasses.replace(cls.capabilities, deterministic=not post)

    @classmethod
    def validate_config(cls, settings, base_dir=None) -> list[str]:
        errors = super().validate_config(settings, base_dir)
        if isinstance(settings, Mapping):
            if settings.get("colour_textures", "srgb") not in ("srgb", "model"):
                errors.append(
                    f"'colour_textures' must be srgb or model, got {settings['colour_textures']!r}"
                )
            for key in cls.SETTINGS:
                if key == "colour_textures":
                    continue
                if key in settings and not isinstance(settings[key], bool):
                    errors.append(f"{key!r} must be true or false, got {settings[key]!r}")
        return errors

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self.model: mujoco.MjModel | None = None
        self.device = ""
        self._private = False
        self._thread: int | None = None
        self._ctx = None
        self._scene = None
        self._resources = None
        self._lights = None
        self._renderables = None
        self._targets: dict = {}
        self._segtable: np.ndarray | None = None
        self._headlight = (False, (1.0, 1.0, 1.0), 0.0)
        self._vopts: dict = {}
        self._options = None
        #: Per object type, the groups its renderables were made with (Filament's own visibility).
        self._built_groups: dict = {}
        #: Renderables taken out of the scene: hidden at run time (:meth:`_apply_visibility`).
        self._removed: dict = {}
        #: Render calls and images read since open (what a test counts).
        self.stats = {"batches": 0, "submits": 0, "reads": 0}

    # -- lifecycle -------------------------------------------------------------------------------

    def open(self, scene: SceneDescription) -> None:
        self.device, _log = probe()
        check_device(self.device, bool(self.settings.get("software", False)))
        try:
            import mujoco.studio.window  # noqa: F401, PLC0415 -- registers Filament's materials
            from mujoco import mjrf  # noqa: PLC0415
        except ImportError as err:
            raise RenderError(
                f"filament: MuJoCo's Filament bindings are not installed ({err}); they come with "
                "the full 'mujoco' package (mujoco-studio)"
            ) from err
        self._mjrf = mjrf
        model = scene.model
        self._private = bool(scene.private_model)
        colour = colour_textures_left_auto(model)
        if len(colour) and self.settings.get("colour_textures", "srgb") == "srgb":
            if not self._private:
                # A copy of its own, which this renderer may write; the runtime values come with
                # each snapshot, as for a renderer off the physics thread.
                model = copy.copy(model)
                self._private = True
            model.tex_colorspace[colour] = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
        self.model = model
        self._thread = threading.get_ident()
        self._ctx = borrow()
        clear = _custom_numeric(model, "filament.clear_color")
        self._ctx.set_clear_color(list(clear[:3]) if clear is not None else [0.0, 0.0, 0.0])
        self._scene = self._ctx.create_scene(mjrf.SceneParams())
        self._scene.configure_from_model(model)
        self._resources = mjrf.ModelResources(self._ctx, model)
        self._lights = mjrf.ModelLights(self._scene, self._resources)
        self._renderables = mjrf.ModelRenderables(self._scene, self._resources)
        self._segtable = self._segmentation_table(model)
        self._built_groups = {
            "geom": np.array(model.geom_group, copy=True),
            "flex": np.array(model.flex_group, copy=True),
        }
        self._removed = {}
        self._headlight = headlight(model)
        _logger.info("filament: Vulkan on %s", self.device or "an unnamed device")

    def _segmentation_table(self, model) -> np.ndarray:
        """Row ``i``: the ``(object id, mjtObj type)`` Filament labels ``i``; row 0 the background."""
        pairs: dict[int, tuple[int, int]] = {}
        for objtype, count in _SEGMENTED:
            for objid in range(int(getattr(model, count))):
                try:
                    r = self._renderables.get_renderable(objtype, objid, -1)
                except Exception:  # noqa: BLE001 -- an object Filament makes nothing for
                    continue
                if r is None:
                    continue
                segid = int(r.get_material().segmentation_id)
                if segid > 0:
                    pairs[segid] = (objid, int(objtype))
        table = np.full((max(pairs, default=0) + 1, 2), -1, dtype=np.int32)
        for segid, pair in pairs.items():
            table[segid] = pair
        return table

    def close(self) -> None:
        """Free the world's meshes, textures and renderables and give the context back, on the
        thread that opened it; from any other, keep them all. Idempotent."""
        # Freed in this order: what uses the resources before the resources.
        objects = [self._renderables, self._lights, self._resources, self._targets]
        self._renderables = self._lights = self._resources = self._scene = None
        self._targets = {}
        self._removed = {}
        ctx, self._ctx = self._ctx, None
        if self._thread is not None and threading.get_ident() != self._thread:
            # Filament aborts the process when its objects are destroyed off their thread.
            _keep_forever(objects)
            _logger.warning(
                "filament: closed off the thread that opened it; its world is kept, not freed"
            )
        elif ctx is not None:
            while objects:
                objects.pop(0)
            give_back(ctx)
        self._thread = None

    def __del__(self) -> None:
        # Collected without close(): the collector may run on any thread, so keep, never destroy.
        if getattr(self, "_ctx", None) is not None:
            _keep_forever([self._renderables, self._lights, self._resources, self._targets])

    # -- drawing ---------------------------------------------------------------------------------

    def render(self, snapshot: StateSnapshot, requests: Sequence[CameraRequest]) -> list[Frame]:
        if self._ctx is None:
            raise RenderError("filament: render() before open() or after close()")
        model, data = self.model, snapshot.data
        if data is None:
            raise RenderError("filament: the snapshot carries no MjData; it draws in-process")
        if self._private:
            for name, values in snapshot.model_values.items():
                np.copyto(getattr(model, name), values)
        self.stats["batches"] += 1
        frames = [Frame(r.id, snapshot.sim_time, snapshot.step) for r in requests]
        self._lights.update(data)
        groups: dict = {}
        for i, r in enumerate(requests):
            groups.setdefault(r.camera.geomgroup, []).append(i)
        for geomgroup, members in groups.items():
            self._set_options(geomgroup)
            self._renderables.update(data)
            self._apply_visibility(model, self._vopts[geomgroup])
            self._submit(model, data, [requests[i] for i in members], [frames[i] for i in members])
        return frames

    def _set_options(self, geomgroup) -> None:
        if geomgroup not in self._vopts:
            self._vopts[geomgroup] = _vopt(geomgroup)
        if self._options is not geomgroup or self._options is None:
            self._renderables.set_options(self._vopts[geomgroup])
            self._options = geomgroup

    def _apply_visibility(self, model, vopt) -> None:
        """Hide what roqsim hides at run time, which Filament does not follow.

        ``ModelRenderables`` takes a geom's or a flex's group when it is made, and draws a geom of
        zero alpha into the labels; :mod:`roqsim.presence` hides an absent entity by moving it to
        its own group at run time and zeroing its alpha. An object whose group is hidden now, or
        whose alpha is 0, is taken out of the scene, and put back once it is drawn again.
        """
        shown = np.asarray(vopt.geomgroup, dtype=bool)
        last = len(shown) - 1
        for kind, objtype, groups, rgba in (
            ("geom", mujoco.mjtObj.mjOBJ_GEOM, model.geom_group, model.geom_rgba),
            ("flex", mujoco.mjtObj.mjOBJ_FLEX, model.flex_group, model.flex_rgba),
        ):
            if not len(groups):
                continue
            now = shown[np.clip(np.asarray(groups), 0, last)] & (np.asarray(rgba)[:, 3] > 0)
            built = shown[np.clip(self._built_groups[kind], 0, last)]
            hide = set(np.flatnonzero(built & ~now).tolist())
            removed = self._removed.setdefault(kind, set())
            for i in hide - removed:
                r = self._renderables.get_renderable(objtype, int(i), -1)
                if r is not None:
                    self._scene.remove_renderable(r)
            for i in removed - hide:
                r = self._renderables.get_renderable(objtype, int(i), -1)
                if r is not None:
                    self._scene.add_renderable(r)
            self._removed[kind] = hide

    def _target(self, key, width: int, height: int, kind: str):
        mjrf = self._mjrf
        fmt, shape, dtype = {
            "rgb": (mjrf.PixelFormat.PIXEL_FORMAT_RGB8, (height, width, 3), np.uint8),
            "depth": (mjrf.PixelFormat.PIXEL_FORMAT_R32F, (height, width), np.float32),
            "segmentation": (mjrf.PixelFormat.PIXEL_FORMAT_RGBA8, (height, width, 4), np.uint8),
        }[kind]
        entry = self._targets.get(key)
        if entry is None:
            cfg = mjrf.RenderTargetConfig()
            cfg.width, cfg.height, cfg.color_format = width, height, fmt
            target = self._ctx.create_render_target(cfg)
            entry = self._targets[key] = [target, None, None]
        target = entry[0]
        buffer = np.empty(shape, dtype)
        read = mjrf.ReadRequest(target=target)
        read.set_buffer(buffer)
        entry[1], entry[2] = read, buffer  # the read and its buffer live until the next submit
        return target, read, buffer

    def _camera(self, model, data, request: CameraRequest):
        """The ``mjrf.Camera`` of *request* and its planes ``(near, far, orthographic)``."""
        cam = request.camera.mujoco_camera
        if isinstance(cam, mujoco.MjvCamera):
            vcam = cam
        elif cam is not None:
            vcam = mujoco.MjvCamera()
            vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            vcam.fixedcamid = int(cam)
        else:
            raise RenderError(
                f"filament: camera {request.id!r} names no MuJoCo camera; it draws through one"
            )
        gl = mujoco.mjv_camera2GLCamera(model, data, vcam)
        set_camera_planes(gl, request.camera.near, request.camera.far)
        if gl.frustum_far <= gl.frustum_near:
            raise ValueError(
                f"far clip plane {gl.frustum_far:g} m is not beyond the near one "
                f"{gl.frustum_near:g} m"
            )
        out = self._mjrf.Camera(
            pos=np.asarray(gl.pos, np.float32),
            forward=np.asarray(gl.forward, np.float32),
            up=np.asarray(gl.up, np.float32),
            frustum_center=float(gl.frustum_center),
            frustum_width=float(gl.frustum_width),
            frustum_bottom=float(gl.frustum_bottom),
            frustum_top=float(gl.frustum_top),
            frustum_near=float(gl.frustum_near),
            frustum_far=float(gl.frustum_far),
            orthographic=bool(gl.orthographic),
        )
        return out, (float(gl.frustum_near), float(gl.frustum_far), bool(gl.orthographic))

    def _submit(self, model, data, requests, frames) -> None:
        mjrf = self._mjrf
        modes = {
            "rgb": mjrf.DrawMode.DRAW_MODE_DEFAULT,
            "depth": mjrf.DrawMode.DRAW_MODE_DEPTH,
            "segmentation": mjrf.DrawMode.DRAW_MODE_SEGMENTATION_BY_ID,
        }
        on, colour, lux = self._headlight
        renders, reads, outputs = [], [], []
        for request, frame in zip(requests, frames, strict=True):
            camera, planes = self._camera(model, data, request)
            w, h = request.camera.width, request.camera.height
            for kind in IMAGES:
                if kind not in request.images:
                    continue
                target, read, buffer = self._target((request.id, kind, w, h), w, h, kind)
                q = mjrf.RenderRequest()
                q.scene, q.target, q.camera = self._scene, target, camera
                q.viewport.left, q.viewport.bottom = 0, 0
                q.viewport.width, q.viewport.height = w, h
                q.draw_mode = modes[kind]
                colour_pass = kind == "rgb"
                q.shadows = colour_pass and bool(self.settings["shadows"])
                q.post_processing = colour_pass and bool(self.settings["post_processing"])
                q.reflections = colour_pass and bool(self.settings["reflections"])
                q.skybox = colour_pass
                q.haze = colour_pass
                q.headlight = colour_pass and on
                q.headlight_color = colour
                q.headlight_intensity = lux if colour_pass else 0.0
                renders.append(q)
                reads.append(read)
                outputs.append((frame, kind, buffer, planes))
        if not renders:
            return
        # MuJoCo 4.0's Filament reads back one target per render call ("Only one read request is
        # supported for now"; with more it aborts or skips the rest), so each image is a call of
        # its own. All are submitted before the first wait; frames finish in order.
        handles = [self._ctx.render([q], [read]) for q, read in zip(renders, reads, strict=True)]
        for handle in handles:
            self._ctx.wait_for_frame(handle)
        self.stats["submits"] += len(handles)
        self.stats["reads"] += len(reads)
        for frame, kind, buffer, (near, far, ortho) in outputs:
            if kind == "rgb":
                frame.rgb = buffer
            elif kind == "depth":
                frame.depth = linear_depth(buffer, near, far, ortho)
            else:
                ids = (
                    buffer[..., 0].astype(np.int64)
                    | (buffer[..., 1].astype(np.int64) << 8)
                    | (buffer[..., 2].astype(np.int64) << 16)
                )
                table = self._segtable
                unknown = ids >= len(table)
                if unknown.any():
                    raise RenderError(
                        f"filament: segmentation id {int(ids[unknown].max())} labels no object "
                        f"of the model (it labels {len(table) - 1})"
                    )
                frame.segmentation = table[ids]
