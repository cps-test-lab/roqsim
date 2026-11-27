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

"""A list of camera views, each a pose and pinhole intrinsics, drawn through any renderer.

A *views file* (JSON or YAML) names its convention and lists its views::

    convention: colmap        # a preset, or {axes: ..., rotation: ..., pixel_origin: ...}
    near: 0.05                # optional, m (default 0.01)
    far: 30.0                 # optional, m (default: the model's own far plane)
    views:
      - {name: "2_1276", width: 353, height: 628, fx: 623.8, fy: 626.5, cx: 176.5, cy: 314.0,
         R: [[...], [...], [...]], position: [x, y, z]}

Three keys state a convention (:data:`KEYS`); a preset names a set of them after where its data comes
from (:data:`PRESETS`): ``colmap``, ``opencv``, ``ros`` and ``mujoco``. A file that states neither is
refused, never read with a default. ``roqsim views describe [PRESET]`` prints the conventions and a
ready example file.

Every view is read into one form (:class:`View`): MuJoCo's camera frame (looking down -z, +y up) as
``pos`` and ``quat`` (``w, x, y, z``, camera to world) and ``(fx, fy, cx, cy)`` with ``cx, cy``
counted from the top-left pixel's centre -- what a :class:`~roqsim.render_api.CameraRequest` carries.
:func:`render_views` draws them through the render service's posed cameras
(:meth:`~roqsim.rendering.RenderService.view_at`), all of a file's views on one renderer;
:func:`write_views` writes what ``roqsim render --views`` writes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path

import numpy as np

#: The three keys that state a convention, each value with what it means.
KEYS: dict[str, dict[str, str]] = {
    "axes": {
        "opencv": "the camera frame is x right, y down, z forward into the scene "
        "(also ROS's camera optical frame)",
        "mujoco": "the camera frame is x right, y up, looking down -z (MuJoCo, OpenGL)",
    },
    "rotation": {
        "world_to_camera": "R maps the world to the camera: x_cam = R (x_world - position); "
        "t may be given for position: x_cam = R x_world + t",
        "camera_to_world": "R maps the camera to the world: x_world = R x_cam + position; "
        "its columns are the camera's axes in the world",
    },
    "pixel_origin": {
        "corner": "cx, cy count from the top-left corner of the image: the top-left pixel's "
        "centre is (0.5, 0.5)",
        "centre": "cx, cy count from the top-left pixel's centre: that centre is (0, 0)",
    },
}


@dataclass(frozen=True)
class Convention:
    """How a views file writes a camera: its :data:`KEYS`, and the preset it was named by."""

    axes: str
    rotation: str
    pixel_origin: str
    #: The preset this convention is, by name or by its keys; ``None`` for one of its own.
    preset: str | None = None
    #: Where a preset's data comes from, in a few words.
    source: str = ""

    def keys(self) -> dict[str, str]:
        return {"axes": self.axes, "rotation": self.rotation, "pixel_origin": self.pixel_origin}

    def as_record(self) -> dict:
        out = self.keys()
        if self.preset:
            out["preset"] = self.preset
        return out

    def summary(self) -> str:
        """One line: the rotation, the axes and the pixel origin, in plain words."""
        axes = (
            "x right y down z forward" if self.axes == "opencv" else "x right y up looking down -z"
        )
        rot = "R,t world->camera" if self.rotation == "world_to_camera" else "camera in the world"
        origin = "the pixel corner" if self.pixel_origin == "corner" else "the pixel centre"
        return f"{rot}, {axes}, cx,cy from {origin}"


#: The presets, each named after where its data comes from: the one declaration the help of
#: ``roqsim render --views``, ``roqsim views describe``, a refusal and the docs are made from.
PRESETS: dict[str, Convention] = {
    "colmap": Convention(
        "opencv",
        "world_to_camera",
        "corner",
        "colmap",
        "SfM reconstruction (COLMAP images.txt/cameras.txt)",
    ),
    "opencv": Convention(
        "opencv",
        "world_to_camera",
        "centre",
        "opencv",
        "camera calibration (OpenCV rvec/tvec: R = cv2.Rodrigues(rvec), t = tvec)",
    ),
    "ros": Convention(
        "opencv",
        "camera_to_world",
        "centre",
        "ros",
        "robot data (the camera optical frame in the world, with its sensor_msgs/CameraInfo)",
    ),
    "mujoco": Convention(
        "mujoco",
        "camera_to_world",
        "centre",
        "mujoco",
        "MuJoCo and roqsim (a model camera, a CameraRequest)",
    ),
}

#: The command that prints the conventions and an example file; every refusal names it.
DESCRIBE = "roqsim views describe <preset>"

#: The near clip plane a view draws with unless it states one: a sensor camera's ``render_near``.
DEFAULT_NEAR = 0.01

#: How far a written ``R`` may be from a rotation (``R Rᵀ`` from ``I``, ``det R`` from 1): its entries
#: rounded to four decimals stay inside, a scaled or mirrored matrix does not. The rotation nearest
#: it is what is drawn.
ROTATION_TOLERANCE = 1e-3

#: What a view's name may be: it becomes a file stem.
_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

#: The keys of a file, and of one view.
FILE_KEYS = ("convention", "near", "far", "views")
VIEW_KEYS = ("name", "width", "height", "fx", "fy", "cx", "cy", "R", "quat", "position", "t")
VIEW_KEYS = (*VIEW_KEYS, "near", "far")


class ViewsError(ValueError):
    """A views file that cannot be read as it is written: each problem named with its view and key."""


def preset_lines(indent: str = "") -> list[str]:
    """One line per preset: its name, where its data comes from and its convention."""
    width = max(len(n) for n in PRESETS)
    return [
        f"{indent}{name:<{width}}  {c.source.split(' (')[0]}: {c.summary()}"
        for name, c in PRESETS.items()
    ]


def _refuse_convention(value) -> str:
    near = ""
    if isinstance(value, str):
        close = get_close_matches(value, list(PRESETS), n=1, cutoff=0.6)
        near = f" (did you mean {close[0]!r}?)" if close else ""
    return (
        f"unknown convention {value!r}{near}; presets: {', '.join(PRESETS)}, or the keys "
        f"{', '.join(KEYS)}; '{DESCRIBE}' prints them with an example views file"
    )


def read_convention(value) -> Convention:
    """The :class:`Convention` a file's ``convention`` states: a preset's name, or all three keys."""
    if isinstance(value, str):
        if value in PRESETS:
            return PRESETS[value]
        raise ViewsError(_refuse_convention(value))
    if isinstance(value, Mapping):
        missing = [k for k in KEYS if k not in value]
        unknown = [k for k in value if k not in KEYS]
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if unknown:
                parts.append(f"unknown {', '.join(map(str, unknown))}")
            raise ViewsError(
                f"convention {dict(value)!r}: {'; '.join(parts)}; it is a preset "
                f"({', '.join(PRESETS)}) or all three keys {', '.join(KEYS)}; '{DESCRIBE}' "
                "prints them"
            )
        for key, allowed in KEYS.items():
            if value[key] not in allowed:
                raise ViewsError(
                    f"convention {key}: {value[key]!r}; one of {', '.join(allowed)} "
                    f"('{DESCRIBE}' prints what each means)"
                )
        keys = {k: str(value[k]) for k in KEYS}
        for name, preset in PRESETS.items():
            if preset.keys() == keys:
                return Convention(**keys, preset=name, source=preset.source)
        return Convention(**keys)
    raise ViewsError(_refuse_convention(value))


@dataclass(frozen=True)
class View:
    """One camera of a views file, in roqsim's form.

    ``pos`` (m, world) and ``quat`` (``w, x, y, z``, MuJoCo's camera frame, camera to world) are the
    pose; ``intrinsics`` ``(fx, fy, cx, cy)`` in pixels, ``cx, cy`` from the top-left pixel's centre;
    ``near``/``far`` its clip planes in metres (``far`` ``None``: the model's own).
    """

    name: str
    width: int
    height: int
    intrinsics: tuple[float, float, float, float]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    near: float = DEFAULT_NEAR
    far: float | None = None
    #: The convention it was written in.
    convention: Convention | None = field(default=None, compare=False)

    @classmethod
    def from_pose(cls, convention: str | Mapping | Convention, **entry) -> View:
        """A view from the keys of a views file's entry (``name``, ``width``, ... ``R``/``quat``,
        ``position``/``t``) written in *convention*; refuses as :func:`load_views` does."""
        conv = convention if isinstance(convention, Convention) else read_convention(convention)
        errors: list[str] = []
        view = _view(entry, conv, {}, errors)
        if errors:
            raise ViewsError("; ".join(errors))
        return view

    @property
    def rotation(self) -> np.ndarray:
        """The camera's axes in the world as columns (MuJoCo's camera frame)."""
        return _quat_to_mat(np.asarray(self.quat, float))

    def describe(self) -> dict:
        """How it was read, in the world frame: where it stands, which way it looks, what is up in
        its picture, and where its principal point lies; with ``text``, the same as one line, and
        ``warnings`` for what a wrong convention typically produces."""
        mat = self.rotation
        fwd, up = -mat[:, 2], mat[:, 1]
        zw = np.array([0.0, 0.0, 1.0])
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, float(fwd @ zw)))))
        flat = fwd - (fwd @ zw) * zw
        if np.linalg.norm(flat) < 1e-6:
            toward = "+z" if fwd[2] > 0 else "-z"
        else:
            toward = _nearest_axis(flat, ("x", "y"))
        zp = zw - (zw @ fwd) * fwd
        if np.linalg.norm(zp) < 1e-6:
            roll = None
        else:
            zp = zp / np.linalg.norm(zp)
            right0 = np.cross(fwd, zp)
            roll = math.degrees(math.atan2(float(up @ right0), float(up @ zp)))
        fx, fy, cx, cy = self.intrinsics
        dx, dy = cx - (self.width - 1) / 2.0, cy - (self.height - 1) / 2.0
        warnings = []
        if float(up @ zw) < 0:
            warnings.append(
                "upside down: the picture's up points down the world's z "
                "(a y-down frame read as y-up, or the rotation inverted?)"
            )
        if not (-0.5 <= cx <= self.width - 0.5 and -0.5 <= cy <= self.height - 0.5):
            warnings.append("principal point outside the image")
        elif abs(dx) > self.width / 4 or abs(dy) > self.height / 4:
            warnings.append("principal point more than a quarter of the image from its centre")
        conv = self.convention
        label = conv.preset if conv and conv.preset else "keys" if conv else "roqsim"
        if abs(elevation) < 0.5:
            look = f"{toward}, level"
        else:
            look = f"{toward}, {abs(elevation):.0f}° {'up' if elevation > 0 else 'down'}"
        if roll is None:
            straight = "up" if fwd[2] > 0 else "down"
            up_text = f"up is {_vec(up)}, roll not defined (looking straight {straight})"
        else:
            side = "right" if roll > 0 else "left"
            up_text = (
                f"up is {_vec(up)}, toward {_nearest_axis(up, ('x', 'y', 'z'))}, "
                f"rolled {abs(roll):.0f}°{'' if abs(roll) < 0.5 else ' ' + side}"
            )
        pp = (
            f"principal point {abs(dx):.1f} px {'right' if dx >= 0 else 'left'} of and "
            f"{abs(dy):.1f} px {'below' if dy > 0 else 'above'} the image centre "
            f"({self.width}x{self.height})"
        )
        text = (
            f"{self.name} ({label}): stands at x {self.pos[0]:.3f}, y {self.pos[1]:.3f}, "
            f"z {self.pos[2]:.3f} m; looks along {_vec(fwd)}, toward {look}; {up_text}; {pp}"
        )
        if warnings:
            text += "; " + "; ".join(f"WARNING {w}" for w in warnings)
        return {
            "name": self.name,
            "convention": label,
            "position": [round(float(v), 6) for v in self.pos],
            "looks_along": [round(float(v), 6) for v in fwd],
            "toward": toward,
            "elevation_deg": round(elevation, 3),
            "up": [round(float(v), 6) for v in up],
            "roll_deg": None if roll is None else round(roll, 3),
            "principal_offset_px": {"right": round(dx, 3), "below": round(dy, 3)},
            "warnings": warnings,
            "text": text,
        }

    def as_record(self) -> dict:
        """The view as roqsim drew it (MuJoCo frame, pixel centres), for a manifest."""
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "intrinsics": list(self.intrinsics),
            "pos": list(self.pos),
            "quat": list(self.quat),
            "near": self.near,
            "far": self.far,
        }


def _vec(v) -> str:
    return " ".join(f"{round(float(c), 2) + 0.0:+.2f}" for c in v)


def _nearest_axis(v, names: Sequence[str]) -> str:
    idx = {"x": 0, "y": 1, "z": 2}
    best = max(names, key=lambda n: abs(float(v[idx[n]])))
    return ("+" if v[idx[best]] >= 0 else "-") + best


@dataclass
class ViewsFile:
    """A read views file: its views in roqsim's form, its convention, and where it came from."""

    views: list[View]
    convention: Convention
    path: str | None = None
    sha256: str | None = None

    def __iter__(self) -> Iterator[View]:
        return iter(self.views)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, i):
        return self.views[i]


def load_views(path: str | Path) -> ViewsFile:
    """Read a views file (``.json``, ``.yaml``, ``.yml``); refuses, naming each view and key, what
    cannot be read as written."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise ViewsError(
            f"views file {path}: a .json, .yaml or .yml file, got {suffix or '(none)'}"
        )
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise ViewsError(f"views file {path}: {err.strerror or err}") from None
    try:
        if suffix == ".json":
            doc = json.loads(raw)
        else:
            import yaml  # noqa: PLC0415 -- only a YAML file pays

            doc = yaml.safe_load(raw)
    except Exception as err:  # noqa: BLE001 -- a parser's own error, named
        raise ViewsError(f"views file {path}: not {suffix[1:].upper()}: {err}") from None
    out = parse_views(doc, where=f"views file {path}")
    out.path = str(path)
    out.sha256 = hashlib.sha256(raw).hexdigest()
    return out


def parse_views(doc, where: str = "views") -> ViewsFile:
    """A views file's document (as parsed) read into :class:`View` s; see :func:`load_views`."""
    if not isinstance(doc, Mapping):
        raise ViewsError(f"{where}: a mapping of {', '.join(FILE_KEYS)}, got {type(doc).__name__}")
    errors = [f"{where}: {e}" for e in _unknown(doc, FILE_KEYS)]
    if "convention" not in doc:
        raise ViewsError(
            f"{where}: no convention; a views file names one: a preset "
            f"({', '.join(PRESETS)}) or the keys {', '.join(KEYS)}; '{DESCRIBE}' prints them "
            "with an example views file"
        )
    try:
        conv = read_convention(doc["convention"])
    except ViewsError as err:
        raise ViewsError(f"{where}: {err}") from None
    defaults = {}
    for key in ("near", "far"):
        if key in doc and doc[key] is not None:
            defaults[key] = doc[key]
    entries = doc.get("views")
    if not isinstance(entries, list) or not entries:
        raise ViewsError(f"{where}: 'views' is a list of at least one view")
    views, names = [], set()
    for i, entry in enumerate(entries):
        label = entry.get("name") if isinstance(entry, Mapping) else None
        label = f"view {label!r}" if isinstance(label, str) else f"view #{i}"
        problems: list[str] = []
        if not isinstance(entry, Mapping):
            errors.append(f"{where}: {label}: a mapping, got {type(entry).__name__}")
            continue
        view = _view(entry, conv, defaults, problems)
        errors += [f"{where}: {label}: {p}" for p in problems]
        if view is not None:
            if view.name in names:
                errors.append(f"{where}: {label}: the name is used twice")
            names.add(view.name)
            views.append(view)
    if errors:
        raise ViewsError("\n".join(errors))
    return ViewsFile(views, conv)


def _unknown(block: Mapping, valid: Sequence[str]) -> list[str]:
    out = []
    for key in block:
        if key in valid:
            continue
        hint = "position" if key in ("pos", "centre", "center") else None
        if hint is None:
            close = get_close_matches(str(key), list(valid), n=1, cutoff=0.7)
            hint = close[0] if close else None
        out.append(
            f"unknown key {key!r}"
            + (f" (did you mean {hint!r}?)" if hint else "")
            + f"; keys: {', '.join(valid)}"
        )
    return out


def _number(entry, key, errors, *, positive=False, integer=False):
    value = entry.get(key)
    if value is None:
        errors.append(f"{key!r} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        errors.append(f"{key!r} is a number, got {value!r}")
        return None
    if integer and int(value) != value:
        errors.append(f"{key!r} is a whole number of pixels, got {value!r}")
        return None
    if positive and not value > 0:
        errors.append(f"{key!r} must be > 0, got {value!r}")
        return None
    return int(value) if integer else float(value)


def _vector(value, n: int, key: str, errors, names=("x", "y", "z")):
    """A list of *n* numbers, or a mapping of *names*."""
    if isinstance(value, Mapping):
        if set(value) != set(names):
            errors.append(
                f"{key!r} as a mapping has the keys {', '.join(names)}, got {list(value)}"
            )
            return None
        value = [value[k] for k in names]
    if not isinstance(value, list | tuple) or len(value) != n:
        errors.append(f"{key!r} is {n} numbers, got {value!r}")
        return None
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value):
        errors.append(f"{key!r} is {n} numbers, got {value!r}")
        return None
    out = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(out)):
        errors.append(f"{key!r} is {n} finite numbers, got {value!r}")
        return None
    return out


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    import mujoco  # noqa: PLC0415 -- after roqsim selected the GL backend

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(m, dtype=float).reshape(-1))
    return q


def _rotation(entry, conv: Convention, errors) -> np.ndarray | None:
    has_r, has_q = "R" in entry, "quat" in entry
    if has_r == has_q:
        errors.append("give the rotation as exactly one of 'R' (3x3) or 'quat'")
        return None
    if has_r:
        r = entry["R"]
        try:
            mat = np.asarray(r, dtype=float)
        except (TypeError, ValueError):
            mat = None
        if mat is None or mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
            errors.append(f"'R' is 3 rows of 3 numbers, got {r!r}")
            return None
        err = float(np.abs(mat @ mat.T - np.eye(3)).max())
        det = float(np.linalg.det(mat))
        if err > ROTATION_TOLERANCE or abs(det - 1.0) > ROTATION_TOLERANCE:
            errors.append(
                f"'R' is not a rotation: R Rᵀ differs from I by {err:.2g}, det {det:.6g} "
                f"(orthonormal, det +1, to {ROTATION_TOLERANCE:g})"
            )
            return None
        # The rotation nearest the matrix as written, to the digits it was written with.
        u, _, vt = np.linalg.svd(mat)
        return u @ vt
    q = entry["quat"]
    if isinstance(q, Mapping):
        quat = _vector(q, 4, "quat", errors, names=("x", "y", "z", "w"))
        if quat is None:
            return None
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        if conv.preset == "ros":
            errors.append(
                "'quat' under ros is the mapping {x, y, z, w}, as a ROS message spells it; a "
                "list would be read as w, x, y, z, which a ROS list is not"
            )
            return None
        quat = _vector(q, 4, "quat", errors, names=("w", "x", "y", "z"))
        if quat is None:
            return None
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        errors.append("'quat' has no length")
        return None
    return _quat_to_mat(quat / norm)


def _view(entry: Mapping, conv: Convention, defaults: Mapping, errors: list[str]) -> View | None:
    n0 = len(errors)
    errors += _unknown(entry, VIEW_KEYS)
    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        errors.append(f"'name' is a file stem of letters, digits, '.', '_', '-', got {name!r}")
    width = _number(entry, "width", errors, positive=True, integer=True)
    height = _number(entry, "height", errors, positive=True, integer=True)
    fx = _number(entry, "fx", errors, positive=True)
    fy = _number(entry, "fy", errors, positive=True)
    cx = _number(entry, "cx", errors)
    cy = _number(entry, "cy", errors)
    rot = _rotation(entry, conv, errors)
    has_pos, has_t = "position" in entry, "t" in entry
    position = None
    if has_t and conv.rotation != "world_to_camera":
        errors.append("'t' is for a world_to_camera rotation; give 'position' (the camera centre)")
    elif has_pos == has_t:
        errors.append(
            "give the camera's place as exactly one of 'position' (its centre in the world)"
            + (" or 't'" if conv.rotation == "world_to_camera" else "")
        )
    elif has_pos:
        position = _vector(entry["position"], 3, "position", errors)
    else:
        t = _vector(entry["t"], 3, "t", errors)
        if t is not None and rot is not None:
            position = -rot.T @ t
    near = entry.get("near", defaults.get("near", DEFAULT_NEAR))
    far = entry.get("far", defaults.get("far"))
    if isinstance(near, bool) or not isinstance(near, int | float) or not near > 0:
        errors.append(f"'near' is a distance > 0 m, got {near!r}")
    elif far is not None and (
        isinstance(far, bool) or not isinstance(far, int | float) or not far > near
    ):
        errors.append(f"far clip plane {far!r} m is not beyond the near one {near!r} m")
    if len(errors) > n0 or rot is None or position is None:
        return None
    c2w = rot.T if conv.rotation == "world_to_camera" else rot
    if conv.axes == "opencv":
        c2w = c2w @ np.diag([1.0, -1.0, -1.0])
    if conv.pixel_origin == "corner":
        cx, cy = cx - 0.5, cy - 0.5
    return View(
        name=name,
        width=width,
        height=height,
        intrinsics=(fx, fy, cx, cy),
        pos=tuple(float(v) for v in position),
        quat=tuple(float(v) for v in _mat_to_quat(c2w)),
        near=float(near),
        far=None if far is None else float(far),
        convention=conv,
    )


# -- drawing -------------------------------------------------------------------------------------

#: A batch holds at most this many views, and at most this many pixels: MuJoCo's colour atlas grows
#: to the largest batch it is given, and a splat pass to the GPU memory of its cameras.
BATCH_VIEWS = 64
BATCH_PIXELS = 64_000_000


class RenderedViews:
    """The frames of a views file, drawn as they are iterated: ``(View, Frame)`` per view, in order.

    One render service and one renderer serve every view (each registered as a posed camera before
    the renderer opens), drawn in batches from one state. ``model``, ``renderer``, ``settings``,
    ``capabilities`` and ``images`` say what drew them. Close it (or exhaust it) to free the
    renderer, and the world it built for a target.
    """

    def __init__(
        self,
        model,
        data,
        views: Sequence[View],
        *,
        images: Sequence[str],
        renderer: str,
        settings: Mapping,
        base_dir=None,
        geomgroup=None,
        ctx=None,
        on_close=None,
    ) -> None:
        from .plugin import PluginError  # noqa: PLC0415 -- after the GL backend is chosen
        from .render_api import IMAGES, RenderError  # noqa: PLC0415
        from .rendering import RenderService  # noqa: PLC0415

        self.model, self.data, self.ctx = model, data, ctx
        self.views = list(views)
        self.images = tuple(images)
        self.renderer = renderer
        self.settings = dict(settings or {})
        self._on_close = on_close
        unknown = sorted(set(self.images) - set(IMAGES))
        if unknown or not self.images:
            raise ValueError(f"images {list(self.images)}: some of {', '.join(IMAGES)}")
        self.service = RenderService(
            model, renderer=renderer, renderers={renderer: self.settings}, base_dir=base_dir
        )
        cls = self.service.renderer_class(renderer)
        self.capabilities = cls.capabilities_for(self.settings)
        lacking = [i for i in self.images if i not in self.capabilities.images]
        if lacking:
            self.close()
            raise RenderError(
                f"renderer {renderer!r} does not give image {', '.join(map(repr, lacking))}; it "
                f"gives {', '.join(sorted(self.capabilities.images))}"
            )
        self._views = [
            self.service.view_at(
                v.name,
                width=v.width,
                height=v.height,
                intrinsics=v.intrinsics,
                pos=v.pos,
                quat=v.quat,
                near=v.near,
                far=v.far,
                images=self.images,
                geomgroup=geomgroup,
            )
            for v in self.views
        ]
        try:
            self.service.check()
        except PluginError as err:
            self.close()
            raise RenderError(str(err)) from None

    def batches(self) -> list[list[int]]:
        """The views' indices in the batches they are drawn in."""
        out, cur, pixels = [], [], 0
        for i, v in enumerate(self.views):
            size = v.width * v.height
            if cur and (len(cur) >= BATCH_VIEWS or pixels + size > BATCH_PIXELS):
                out.append(cur)
                cur, pixels = [], 0
            cur.append(i)
            pixels += size
        if cur:
            out.append(cur)
        return out

    def __iter__(self):
        try:
            for batch in self.batches():
                frames = self.service.render_now(
                    [(self._views[i], self.images) for i in batch], self.data
                )
                for i, frame in zip(batch, frames, strict=True):
                    yield self.views[i], frame
        finally:
            self.close()

    def close(self) -> None:
        """Free the renderer, and the world built for a target. Idempotent."""
        service, self.service = getattr(self, "service", None), None
        if service is not None:
            service.close()
        on_close, self._on_close = self._on_close, None
        if on_close is not None:
            on_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def render_views(
    source,
    views: Sequence[View] | ViewsFile,
    *,
    images: Sequence[str] = ("rgb",),
    renderer: str | None = None,
    settings: Mapping | None = None,
    geomgroup: Sequence[int] | None = None,
    overrides: Mapping | None = None,
    base_dir=None,
    no_ceiling: bool = False,
) -> RenderedViews:
    """Draw *views* of *source*: ``(View, Frame)`` per view, as the result is iterated.

    *source* is a target as ``roqsim render`` takes it (a world YAML, an MJCF, a model reference, a
    mesh), built as ``roqsim render`` builds it with *overrides*, or a compiled ``(model, data)``
    pair, drawn as it is (*no_ceiling*: a target's ceiling left out, as ``--no-ceiling``). *images* are drawn for every view (colour by default); *renderer* (the
    world's ``sim.renderer`` by default, MuJoCo's for a bare model) and its *settings* lay over the
    world's. A renderer that does not give an image or a posed camera is refused, with the reason,
    before anything is drawn.
    """
    on_close, ctx = None, None
    if isinstance(source, str | Path):
        from .render import build_target, renderer_choice  # noqa: PLC0415 -- avoids a cycle

        model, data, ctx, _view, _cam = build_target(
            str(source), dict(overrides or {}) or None, no_ceiling=no_ceiling
        )
        if ctx is not None:
            on_close = ctx.engine.shutdown
        asked = {k: dict(v) if isinstance(v, Mapping) else v for k, v in (overrides or {}).items()}
        if renderer:
            asked.setdefault("sim", {})["renderer"] = renderer
        choice = renderer_choice(ctx, asked)
    else:
        model, data = source
        choice = {}
    name = renderer or choice.get("renderer") or "mujoco"
    merged = dict(choice.get("settings") or {})
    merged.update(dict(settings or {}))
    try:
        return RenderedViews(
            model,
            data,
            list(views),
            images=images,
            renderer=name,
            settings=merged,
            base_dir=base_dir if base_dir is not None else choice.get("base_dir"),
            geomgroup=geomgroup,
            ctx=ctx,
            on_close=on_close,
        )
    except BaseException:
        if on_close is not None:
            on_close()
        raise


# -- writing -------------------------------------------------------------------------------------

#: How depth is written, as the manifest states it.
DEPTH_CONVENTION = (
    "metres along the view axis (z-depth, not the distance along the pixel's ray), float32, "
    "+inf where nothing was drawn"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Legend:
    """Segmentation values: index 0 is background, each other an (object type, id) of the model."""

    def __init__(self, model, ctx=None) -> None:
        self.model, self.ctx = model, ctx
        self.index: dict[tuple[int, int], int] = {}
        self.entries: list[dict] = [{"index": 0, "type": None, "name": "background"}]

    def _entry(self, objtype: int, objid: int) -> dict:
        import mujoco  # noqa: PLC0415

        kind = mujoco.mjtObj(objtype).name.removeprefix("mjOBJ_").lower()
        name = mujoco.mj_id2name(self.model, objtype, objid) or ""
        body = None
        if objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
            body = self.model.body(int(self.model.geom_bodyid[objid])).name
        elif objtype == int(mujoco.mjtObj.mjOBJ_SITE):
            body = self.model.body(int(self.model.site_bodyid[objid])).name
        entry = {"type": kind, "id": objid, "name": name, "body": body}
        entity = self._entity(body)
        if entity:
            entry["entity"] = entity
        return entry

    def _entity(self, body: str | None) -> str | None:
        if not body or self.ctx is None or getattr(self.ctx, "entities", None) is None:
            return None
        import mujoco  # noqa: PLC0415

        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        from .presence import entity_bodies  # noqa: PLC0415

        for ent in self.ctx.entities.all():
            try:
                if bid in set(entity_bodies(self.model, ent)):
                    return ent.name
            except Exception:  # noqa: BLE001,S112 -- an entity that cannot be resolved names none
                continue
        return None

    def encode(self, seg: np.ndarray) -> np.ndarray:
        """``(H, W, 2)`` object id and type as a ``(H, W)`` uint16 of legend indices."""
        ids, types = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
        out = np.zeros(ids.shape, dtype=np.int64)
        drawn = ids >= 0
        if drawn.any():
            pairs, inverse = np.unique(
                np.stack([types[drawn], ids[drawn]], axis=1), axis=0, return_inverse=True
            )
            lut = np.empty(len(pairs), dtype=np.int64)
            for j, (t, i) in enumerate(pairs):
                key = (int(t), int(i))
                if key not in self.index:
                    if len(self.entries) > 65535:
                        raise ValueError(
                            "more than 65535 labelled objects: the segmentation PNG holds 16 bits"
                        )
                    self.index[key] = len(self.entries)
                    self.entries.append({"index": len(self.entries), **self._entry(*key)})
                lut[j] = self.index[key]
            out[drawn] = lut[inverse.reshape(-1)]
        return out.astype(np.uint16)


#: The file name each image of a view is written under, after the view's name.
FILE_SUFFIX = {"rgb": ".png", "depth": ".depth.npy", "segmentation": ".seg.png"}


def check_files(views: Sequence[View], images: Sequence[str]) -> None:
    """Refuse *views* two of which would write *images* into one file (a view ``x.seg`` writes its
    colour where ``x`` writes its segmentation), naming both, before anything is drawn."""
    owner: dict[str, str] = {}
    for view in views:
        for image in images:
            name = view.name + FILE_SUFFIX[image]
            if name in owner:
                raise ViewsError(
                    f"view {view.name!r} would write {name}, which view {owner[name]!r} writes; "
                    "rename one of them"
                )
            owner[name] = view.name


def write_views(out_dir: str | Path, rendered: RenderedViews, extra: Mapping | None = None) -> dict:
    """Draw *rendered* into *out_dir* and write its ``manifest.json``; returns the manifest.

    Per view, each image asked for: ``<name>.png`` (8-bit RGB as the renderer gave it),
    ``<name>.depth.npy`` (float32, :data:`DEPTH_CONVENTION`), ``<name>.seg.png`` (a 16-bit
    greyscale PNG of indices into the manifest's ``legend``, 0 for background;
    :func:`read_segmentation` reads it).
    """
    from PIL import Image  # noqa: PLC0415

    from .conditions import of_model as conditions_of_model  # noqa: PLC0415

    try:
        check_files(rendered.views, rendered.images)
    except ViewsError:
        rendered.close()
        raise
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    legend = _Legend(rendered.model, rendered.ctx)
    per_view = []
    for view, frame in rendered:
        files = {}
        if frame.rgb is not None:
            path = out / (view.name + FILE_SUFFIX["rgb"])
            Image.fromarray(np.ascontiguousarray(frame.rgb)).save(path)
            files["rgb"] = path.name
        if frame.depth is not None:
            path = out / (view.name + FILE_SUFFIX["depth"])
            np.save(path, np.asarray(frame.depth, dtype=np.float32))
            files["depth"] = path.name
        if frame.segmentation is not None:
            path = out / (view.name + FILE_SUFFIX["segmentation"])
            Image.fromarray(legend.encode(frame.segmentation)).save(path)
            files["segmentation"] = path.name
        record = view.as_record()
        record["files"] = {k: {"path": f, "sha256": _sha256(out / f)} for k, f in files.items()}
        per_view.append(record)
    try:
        from importlib.metadata import version  # noqa: PLC0415

        roqsim_version = version("roqsim")
    except Exception:  # noqa: BLE001 -- a source tree without metadata
        roqsim_version = None
    manifest = {
        "roqsim": roqsim_version,
        **dict(extra or {}),
        "renderer": rendered.renderer,
        "settings": rendered.settings,
        "capabilities": rendered.capabilities.as_dict(),
        "conditions": conditions_of_model(rendered.model),
        "images": list(rendered.images),
        "depth": DEPTH_CONVENTION if "depth" in rendered.images else None,
        "legend": legend.entries if "segmentation" in rendered.images else None,
        "views": per_view,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


# -- describing the conventions ------------------------------------------------------------------

#: The units every views file is written in, whatever its convention.
UNITS = (
    "positions in metres, in the world frame of the target as built (after its components spawn)",
    "fx, fy, cx, cy in pixels of the image written, at its width and height",
    "x counts right from the image's left edge, y down from its top edge, under every convention",
    "a quaternion list is w, x, y, z; a mapping {x, y, z, w} names its parts (ros: mapping only)",
    "near and far are clip planes in metres; near defaults to 0.01, far to the model's own",
)

#: The example view: a 640x480 camera 2 m from the world's origin, 1 m up, looking along +x at it.
_EXAMPLE = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 319.5, "cy": 239.5}
_EXAMPLE_POSITION = [-2.0, 0.0, 1.0]
#: Its axes in the world as columns, MuJoCo's camera frame: right -y, up +z, back -x.
_EXAMPLE_C2W_MJ = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def read_segmentation(path: str | Path) -> np.ndarray:
    """A written ``<name>.seg.png`` as its ``(H, W)`` uint16 legend indices.

    Pillow opens a 16-bit greyscale PNG as ``I;16`` or as 32-bit ``I``, depending on its version, so
    an array made from the opened image is uint16 with one and int32 with another. The indices are
    the same; this returns them in the type they were written in.
    """
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        if image.mode not in ("I;16", "I"):
            raise ViewsError(
                f"{path}: not a segmentation image: a 16-bit greyscale PNG is expected, "
                f"this one opens as mode {image.mode!r}"
            )
        return np.asarray(image).astype(np.uint16)


def _clean(values) -> list:
    return [0.0 if abs(v) < 1e-12 else round(float(v), 6) for v in values]


def example_file(preset: str) -> tuple[dict, str]:
    """A ready views file of one view in *preset*'s own form: as data, and as commented YAML."""
    if preset not in PRESETS:
        raise ViewsError(_refuse_convention(preset))
    conv = PRESETS[preset]
    c2w = _EXAMPLE_C2W_MJ @ (np.diag([1.0, -1.0, -1.0]) if conv.axes == "opencv" else np.eye(3))
    rot = c2w.T if conv.rotation == "world_to_camera" else c2w
    shift = 0.5 if conv.pixel_origin == "corner" else 0.0
    view: dict = {"name": "view_0001", **_EXAMPLE}
    view["cx"], view["cy"] = _EXAMPLE["cx"] + shift, _EXAMPLE["cy"] + shift
    axes_text = (
        "x right, y down, z forward" if conv.axes == "opencv" else "x right, y up, looking down -z"
    )
    origin_text = (
        "from the image's top-left corner (top-left pixel's centre: 0.5, 0.5)"
        if conv.pixel_origin == "corner"
        else "from the top-left pixel's centre (that centre: 0, 0)"
    )
    comments = {
        "name": "unique; the file stem of its images: letters, digits, . _ -",
        "width": "pixels",
        "height": "pixels",
        "fx": "focal length, pixels",
        "fy": "focal length, pixels",
        "cx": f"principal point, pixels, {origin_text}",
        "cy": "principal point, pixels, counted as cx",
    }
    if preset == "ros":
        q = _clean(_mat_to_quat(rot))
        view["quat"] = {"x": q[1], "y": q[2], "z": q[3], "w": q[0]}
        view["position"] = dict(zip("xyz", _EXAMPLE_POSITION, strict=True))
        comments["quat"] = f"the camera's orientation in the world ({axes_text}), as ROS spells it"
        comments["position"] = "the camera centre in the world, m"
    elif preset == "mujoco":
        view["quat"] = _clean(_mat_to_quat(rot))
        view["position"] = list(_EXAMPLE_POSITION)
        comments["quat"] = f"w, x, y, z: camera to world ({axes_text}); or R, its axes as columns"
        comments["position"] = "the camera centre in the world, m"
    else:
        view["R"] = [_clean(row) for row in rot]
        view["t"] = _clean(-rot @ np.asarray(_EXAMPLE_POSITION))
        comments["R"] = f"world to camera, rows ({axes_text}); or quat: [w, x, y, z]"
        comments["t"] = "x_cam = R x_world + t, m; or position: the camera centre in the world"
    doc = {"convention": preset, "near": 0.05, "far": 50.0, "views": [view]}

    def flow(value) -> str:
        return json.dumps(value).replace('"', "")

    head = [
        f"# A views file for `roqsim render <target> --views FILE --out DIR`, convention {preset}:",
        f"#   {conv.source}",
        f"#   {conv.summary()}",
        f"#   = axes: {conv.axes}, rotation: {conv.rotation}, pixel_origin: {conv.pixel_origin}",
        "# The example view stands 2 m from the world's origin, 1 m up, looking along +x at it.",
        "# Units: " + UNITS[0] + ".",
    ]
    rows = [
        (f"convention: {preset}", f"a preset ({', '.join(PRESETS)}) or the three keys"),
        ("near: 0.05", "optional: the near clip plane, m (default 0.01)"),
        ("far: 50.0", "optional: the far clip plane, m (default: the model's own)"),
        ("views:", "one entry per camera; add as many as you have"),
    ]
    first = True
    for key, value in view.items():
        text = f"{key}: {value}" if key == "name" else f"{key}: {flow(value)}"
        rows.append((("  - " if first else "    ") + text, comments.get(key, "")))
        first = False
    width = max(len(r) for r, _ in rows) + 2
    lines = head + [f"{r:<{width}}# {c}" if c else r for r, c in rows]
    return doc, "\n".join(lines) + "\n"


def describe_conventions(preset: str | None = None) -> dict:
    """What ``roqsim views describe`` prints as JSON: the keys, the presets and the units, and with
    a *preset* its example file as data and as text."""
    out = {
        "keys": KEYS,
        "presets": {
            name: {**c.keys(), "source": c.source, "summary": c.summary()}
            for name, c in PRESETS.items()
        },
        "units": list(UNITS),
        "describe": DESCRIBE,
    }
    if preset is not None:
        doc, text = example_file(preset)
        out["preset"] = preset
        out["example"] = doc
        out["example_text"] = text
    return out


def conventions_text() -> str:
    """The keys, the units and the table of presets, as YAML comments."""
    lines = ["# A views file names its convention: a preset, or the three keys.", "#"]
    for key, values in KEYS.items():
        lines.append(f"# {key}:")
        lines += [f"#   {v:<16} {meaning}" for v, meaning in values.items()]
    lines += ["#", "# presets (axes / rotation / pixel_origin):"]
    width = max(len(n) for n in PRESETS)
    for name, c in PRESETS.items():
        lines.append(
            f"#   {name:<{width}}  {c.axes} / {c.rotation} / {c.pixel_origin}  -- {c.source}"
        )
    lines += ["#", "# units:"] + [f"#   {u}" for u in UNITS]
    lines.append(f"#\n# `{DESCRIBE}` prints a ready example file for that preset.")
    return "\n".join(lines) + "\n"


def presets_rst() -> list[str]:
    """The table of presets as reStructuredText lines: what the docs' ``roqsim-view-presets``
    directive inserts, so the docs show :data:`PRESETS` as installed."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - preset",
        "     - axes",
        "     - rotation",
        "     - pixel_origin",
        "     - where the data comes from",
    ]
    for name, c in PRESETS.items():
        lines += [
            f"   * - ``{name}``",
            f"     - ``{c.axes}``",
            f"     - ``{c.rotation}``",
            f"     - ``{c.pixel_origin}``",
            f"     - {c.source}",
        ]
    return [*lines, ""]
