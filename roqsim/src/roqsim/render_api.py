# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""The renderer contract: what every renderer behind roqsim's render service is given and returns.

A renderer is an extension. It registers under the ``roqsim.renderers`` entry-point group (or is
named ``module:Class`` / ``file.py:Class``, as a plugin is), and a world chooses it with
``sim.renderer``; MuJoCo's own renderer is the one registered as ``mujoco`` and the default. The
render service (:class:`roqsim.rendering.RenderService`) is the only caller:

1. :meth:`Renderer.open` once, with a :class:`SceneDescription` -- the compiled model, which bodies
   move, the cameras it will be asked for and its own settings;
2. :meth:`Renderer.render` per tick, with a :class:`StateSnapshot` (sim time, body poses, the model
   values that change at run time) and a *batch* of :class:`CameraRequest` -- every camera due that
   tick at once, so a renderer can share work across them; it returns one :class:`Frame` per
   request, stamped with the snapshot's sim time;
3. :meth:`Renderer.close` once.

What a renderer can give is declared, not discovered: :attr:`Renderer.capabilities` (a
:class:`RendererCapabilities`). The service checks every camera against it when the world loads, so
a world that asks a renderer for what it cannot give fails then, saying so. Nothing falls back to
another renderer.

Nothing here assumes a GL context or the renderer's process: a request carries the camera's pose and
intrinsics in the world frame, and a frame carries plain arrays, so a renderer in another process
(:mod:`roqsim.render_socket`) stands behind this unchanged. This module imports no GL.

The golden rules hold for every renderer: it reads the snapshot it is handed and never the live
``MjData``, it never writes the shared ``MjModel`` (the windowed viewer draws from it on its own
thread), and a frame is a function of the snapshot and the seed when it declares ``deterministic``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

#: The images a camera may ask for. ``rgb`` is ``(H, W, 3)`` uint8; ``depth`` is ``(H, W)`` float32
#: metres along the view axis, ``inf`` where nothing was drawn; ``segmentation`` is ``(H, W, 2)``
#: int32 object id and ``mjtObj`` type, ``(-1, -1)`` for background.
IMAGES = ("rgb", "depth", "segmentation")

#: Each image's channels after ``(H, W)`` and its dtype, as :data:`IMAGES` states them.
IMAGE_FORMATS = {
    "rgb": ((3,), np.dtype(np.uint8)),
    "depth": ((), np.dtype(np.float32)),
    "segmentation": ((2,), np.dtype(np.int32)),
}


def image_mismatch(kind: str, image: np.ndarray, width: int, height: int) -> str | None:
    """How *image* differs from the ``kind`` image of a ``width x height`` camera, or ``None``."""
    channels, dtype = IMAGE_FORMATS[kind]
    want = (int(height), int(width), *channels)
    got = tuple(np.shape(image))
    if got != want or np.asarray(image).dtype != dtype:
        return f"{kind} {got} {np.asarray(image).dtype}, expected {want} {dtype}"
    return None


#: How a camera projects. ``pinhole`` and ``orthographic`` are a model camera's; ``free`` is a
#: viewer-style orbit camera (``lookat``/``distance``/``azimuth``/``elevation``), what ``roqsim
#: render`` and the scene review draw through.
CAMERA_MODELS = ("pinhole", "orthographic", "free")

#: Where a renderer's settings live in a world: ``sim.renderers.<name>``.
SETTINGS_BLOCK = "sim.renderers"

#: The renderer a world draws through when it names none: MuJoCo's own, registered as ``mujoco``.
DEFAULT_RENDERER = "mujoco"

#: Which bodies a camera draws. ``all`` is the world as it is; ``moving`` is the moving bodies alone
#: (a robot, a prop a plugin moves; :attr:`SceneDescription.moving`), with no static geometry, no
#: floor and nothing static to cast a shadow on them -- what a renderer that brings its own static
#: world (a captured splat) composites into it by depth.
BODIES = ("all", "moving")

#: ``sim.render.mode``: ``sync`` waits each tick for the frames due in it; ``async`` renders them on a
#: worker thread from a snapshot while physics steps on.
RENDER_MODES = ("sync", "async")


class RenderError(RuntimeError):
    """A render that could not be given: a renderer that failed, a worker that died, a missing GPU.

    Raised to the caller of the tick that needed the frame, never answered with a stale or a
    substitute frame.
    """


@dataclass(frozen=True)
class RendererCapabilities:
    """What a renderer can give, declared on its class and checked against a world at load."""

    #: Which of :data:`IMAGES` it renders.
    images: frozenset[str] = frozenset({"rgb"})
    #: Which of :data:`CAMERA_MODELS` it renders through.
    camera_models: frozenset[str] = frozenset({"pinhole"})
    #: Whether it draws MuJoCo's moving bodies (robots, props a plugin moves).
    composes_moving_bodies: bool = True
    #: Whether it draws the model's static geometry too; a renderer that brings its own static world
    #: (a captured splat) does not.
    draws_static: bool = True
    #: Whether one :meth:`Renderer.render` call renders the batch; the service sends the batch
    #: either way, so this informs a reader, it does not change the call.
    batched: bool = False
    #: Whether a frame is a function of the snapshot and the seed alone.
    deterministic: bool = False
    #: Whether it runs in roqsim's process (and may therefore read the ``MjModel`` it is given).
    in_process: bool = True
    #: Whether it draws a caller's MuJoCo ``mjvGeom`` decorations (:attr:`CameraRequest.decorate`).
    decorations: bool = False
    #: Whether it draws only the geom groups a camera names (:attr:`CameraSpec.geomgroup`).
    geom_groups: bool = False
    #: Whether it projects a camera's principal point off the image centre
    #: (:attr:`CameraSpec.intrinsics` ``cx, cy``), as a model camera with ``principal`` has.
    principal_point: bool = False
    #: Whether it projects non-square pixels (``fx != fy``).
    non_square_pixels: bool = False
    #: Whether it draws the moving bodies alone for a camera that asks (:attr:`CameraSpec.bodies`).
    moving_only: bool = False

    def missing(
        self,
        images: Sequence[str] = (),
        model: str | None = None,
        decorate: bool = False,
        geomgroup: bool = False,
        off_centre: bool = False,
        non_square: bool = False,
        moving_only: bool = False,
    ) -> list[str]:
        """What of *images*, *model*, decorations, geom groups, an *off_centre* principal point,
        *non_square* pixels and the moving bodies alone (*moving_only*) this renderer does not give."""
        out = [f"image {i!r}" for i in images if i not in self.images]
        if model is not None and model not in self.camera_models:
            out.append(f"camera model {model!r}")
        if decorate and not self.decorations:
            out.append("decorations")
        if geomgroup and not self.geom_groups:
            out.append("only some geom groups")
        if off_centre and not self.principal_point:
            out.append("an off-centre principal point")
        if non_square and not self.non_square_pixels:
            out.append("non-square pixels (fx != fy)")
        if moving_only and not self.moving_only:
            out.append("the moving bodies alone")
        return out

    def as_dict(self) -> dict:
        return {
            "images": sorted(self.images),
            "camera_models": sorted(self.camera_models),
            "composes_moving_bodies": self.composes_moving_bodies,
            "draws_static": self.draws_static,
            "batched": self.batched,
            "deterministic": self.deterministic,
            "in_process": self.in_process,
            "decorations": self.decorations,
            "geom_groups": self.geom_groups,
            "principal_point": self.principal_point,
            "non_square_pixels": self.non_square_pixels,
            "moving_only": self.moving_only,
        }


@dataclass(frozen=True)
class Setting:
    """One key of a renderer's settings block: its default and what it means."""

    default: Any
    meaning: str


@dataclass
class CameraSpec:
    """A camera as it is registered with the service: everything about it that does not move.

    ``mujoco_camera`` is the MuJoCo camera it renders through -- a camera id, or for the ``free``
    model the ``MjvCamera`` an in-process caller steers -- so MuJoCo's renderer reproduces exactly the
    picture MuJoCo would draw; every other renderer works from the pose and intrinsics the service
    computes per tick (:class:`CameraRequest`).
    """

    id: str
    width: int
    height: int
    model: str = "pinhole"
    #: Vertical field of view in degrees (pinhole), or the vertical extent in metres (orthographic).
    fovy: float = 45.0
    near: float | None = None
    far: float | None = None
    #: The images this camera may ask for; a tick asks for a subset.
    images: frozenset[str] = frozenset({"rgb"})
    mujoco_camera: Any = None
    #: The geom groups drawn (``None``: MuJoCo's default groups).
    geomgroup: tuple[int, ...] | None = None
    #: Which bodies it draws, one of :data:`BODIES`: ``moving`` asks for the moving bodies alone.
    bodies: str = "all"
    #: Whether the caller decorates the scene (MuJoCo renderers only).
    decorated: bool = False
    #: The most geoms its scene may hold (MuJoCo renderers; ``None`` is MuJoCo's default).
    max_geom: int | None = None
    #: ``(fx, fy, cx, cy)`` in pixels, the projection MuJoCo draws this camera with at this size:
    #: from the camera's ``sensorsize`` and ``focal``/``principal`` where the model states them,
    #: else the symmetric pinhole of ``fovy``. ``cx, cy`` count pixel centres from the top-left
    #: pixel's, as a ``sensor_msgs/CameraInfo`` does. ``None`` (orthographic, or a ``fovy`` of 0): use ``fovy``.
    intrinsics: tuple[float, float, float, float] | None = None

    @property
    def fx(self) -> float:
        """The focal length in pixels a pinhole camera of this size and ``fovy`` has."""
        return 0.5 * self.height / float(np.tan(np.radians(self.fovy) / 2.0))

    def missing_from(self, caps: RendererCapabilities, images=None, decorate=None) -> list[str]:
        """What of this camera *caps* does not give: its *images* (default: all it may ask for), its
        camera model, its decorations (default: :attr:`decorated`), geom groups, principal point,
        pixels and bodies."""
        return caps.missing(
            sorted(self.images if images is None else images),
            self.model,
            self.decorated if decorate is None else decorate,
            self.geomgroup is not None,
            self.off_centre,
            self.non_square,
            self.bodies == "moving",
        )

    @property
    def off_centre(self) -> bool:
        """Whether its principal point is off the image centre (by more than a thousandth of a pixel)."""
        if self.intrinsics is None:
            return False
        _fx, _fy, cx, cy = self.intrinsics
        return abs(cx - (self.width - 1) / 2.0) > 1e-3 or abs(cy - (self.height - 1) / 2.0) > 1e-3

    @property
    def non_square(self) -> bool:
        """Whether its pixels are not square: ``fx != fy``."""
        if self.intrinsics is None:
            return False
        fx, fy, _cx, _cy = self.intrinsics
        return abs(fx - fy) > 1e-6 * max(abs(fx), 1.0)


@dataclass
class SceneDescription:
    """What a renderer is given once, before the first frame.

    ``model`` is the compiled ``MjModel``, read-only. ``moving`` marks each body that moves (it is
    not welded to the world, or it is a mocap body); the rest is static. ``cameras`` are every camera
    the renderer will be asked for. ``settings`` is the renderer's own block of the world
    (``sim.renderers.<name>``), already validated by :meth:`Renderer.validate_config`, and
    ``base_dir`` the directory a relative path in it is resolved against.
    """

    model: Any
    moving: np.ndarray
    cameras: list[CameraSpec]
    settings: dict
    base_dir: Any = None
    #: Every body's name, by id: what a renderer elsewhere matches poses to.
    body_names: list[str] = field(default_factory=list)
    #: Whether ``model`` is this renderer's own copy (a renderer off the physics thread gets one),
    #: which it updates from each snapshot's ``model_values``; the shared model it never writes.
    private_model: bool = False


@dataclass
class StateSnapshot:
    """The state one tick renders, copied off the physics thread; never the live ``MjData``.

    ``xpos``/``xquat`` are every body's world pose (``nbody`` rows; ``xquat`` is ``w, x, y, z``).
    ``model_values`` are the model arrays that change at run time and that drawing reads
    (:data:`roqsim.rendering.RUNTIME_MODEL_FIELDS`, set by :mod:`roqsim.presence`). ``data`` is a
    copy of the whole ``MjData`` that an in-process MuJoCo renderer builds its scene from; a renderer
    elsewhere needs only the poses.
    """

    sim_time: float
    step: int
    xpos: np.ndarray
    xquat: np.ndarray
    model_values: dict[str, np.ndarray]
    data: Any = None


@dataclass
class CameraRequest:
    """One camera's part of a tick's batch.

    ``pos``/``quat`` are the camera's world pose (``quat`` ``w, x, y, z``, MuJoCo's camera frame:
    looking down its -z, +y up), computed from the snapshot. ``images`` is what this tick asks for.
    ``decorate`` is a ``callable(scene)`` an in-process caller adds MuJoCo markers with, for a
    renderer that declares ``decorations``.
    """

    camera: CameraSpec
    images: frozenset[str]
    pos: np.ndarray
    quat: np.ndarray
    #: Vertical field of view in degrees for this tick (a free camera's follows the model).
    fovy: float
    decorate: Callable | None = None

    @property
    def id(self) -> str:
        return self.camera.id


@dataclass
class Frame:
    """One camera's images for one snapshot, stamped with that snapshot's sim time."""

    id: str
    sim_time: float
    step: int
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None

    def image(self, kind: str) -> np.ndarray | None:
        return getattr(self, kind)


def unknown_keys(block: Mapping, valid: Collection[str], where: str = "") -> list[str]:
    """An error per key of *block* outside *valid*, naming the valid keys and the nearest one.

    *where* is the dotted path of a block nested in a renderer's settings (``splat.pose``), so a
    mistyped key deep in it is named with the block it sits in; the service prefixes every error
    with ``sim.renderers.<name>``.
    """
    from difflib import get_close_matches  # noqa: PLC0415 -- only a mistake pays

    names = sorted(str(k) for k in valid)
    errors = []
    for key in block:
        if key in valid:
            continue
        near = get_close_matches(str(key), names, n=1, cutoff=0.8)
        errors.append(
            f"unknown key {key!r}{f' in {where}' if where else ''}; "
            f"valid keys: {', '.join(names) if names else '(none)'}"
            + (f"; did you mean {near[0]!r}?" if near else "")
        )
    return errors


def validate_renderer_config(cls, settings: Mapping, base_dir=None) -> list[str]:
    """``cls.validate_config(settings, base_dir)``, or ``(settings)`` for an override that takes no
    *base_dir*."""
    try:
        params = inspect.signature(cls.validate_config).parameters
    except (TypeError, ValueError):
        params = {}
    takes = "base_dir" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
    return (
        cls.validate_config(settings, base_dir=base_dir) if takes else cls.validate_config(settings)
    )


class Renderer:
    """Base of every renderer. Subclass it, declare :attr:`capabilities` and :attr:`SETTINGS`, and
    implement :meth:`open`, :meth:`render` and :meth:`close`.

    The service constructs a renderer with its validated settings, calls :meth:`open` on the thread
    it will render on, and calls :meth:`render` on that thread only; a renderer may therefore hold a
    thread-affine context.
    """

    capabilities: ClassVar[RendererCapabilities] = RendererCapabilities()
    #: Its settings block, key by key: ``{key: Setting(default, meaning)}``.
    SETTINGS: ClassVar[Mapping[str, Setting]] = {}

    def __init__(self, settings: Mapping | None = None) -> None:
        self.settings = {k: s.default for k, s in self.SETTINGS.items()}
        self.settings.update(dict(settings or {}))

    @classmethod
    def capabilities_for(cls, settings: Mapping) -> RendererCapabilities:
        """The capabilities a renderer configured with *settings* has; the class's by default."""
        return cls.capabilities

    @classmethod
    def open_timeout_s(cls, settings: Mapping) -> float | None:
        """How long :meth:`open` may take with *settings*, in s, where that is longer than a tick.

        A renderer that starts a process of its own and is slow to get ready states it; the
        service then waits for the longer of this and ``sim.render.timeout_s`` before it names the
        renderer as hung. ``None``: ``sim.render.timeout_s`` alone.
        """
        return None

    @classmethod
    def validate_config(cls, settings: Mapping, base_dir=None) -> list[str]:
        """Errors in *settings*, each naming the key; an unknown key names the valid ones.

        *base_dir* is the directory of the world that states them (``None``: none, a tool's own), what
        a relative path in them resolves against, so a renderer can name a missing or wrong file at
        load rather than when it opens. An override that takes only *settings* is called so, and is
        not given it (:func:`validate_renderer_config`).
        """
        if not isinstance(settings, Mapping):
            return [f"settings must be a mapping, got {type(settings).__name__}"]
        return unknown_keys(settings, cls.SETTINGS)

    def open(self, scene: SceneDescription) -> None:
        """Take the scene: upload what is static, remember what moves. Called once."""

    def render(self, snapshot: StateSnapshot, requests: Sequence[CameraRequest]) -> list[Frame]:
        """Draw every request of the batch from *snapshot*; one :class:`Frame` per request, in order."""
        raise NotImplementedError

    def close(self) -> None:
        """Free what :meth:`open` took. Idempotent."""
