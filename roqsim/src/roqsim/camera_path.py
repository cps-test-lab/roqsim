"""A camera path: keyframes over simulated time, interpolated per frame of a video.

A recording is drawn one sample per frame, and each frame asks where the camera is. The base camera
comes from ``--view``/``--focus``, the recorded session camera or the world's own ``sim.view``; a path
moves it. A path is a small document, written by hand, by an agent or by a person flying the replay
window::

    ease: smoothstep            # linear | smoothstep
    keyframes:                  # t: seconds, `onset`, `onset+2.5` -- or a moment a caller names
      - {t: onset,    distance: 2.5, azimuth: 180, elevation: -20}
      - {t: onset+10, azimuth: 540, distance: 4.0}
      - {t: 30,       eye: [3, -2, 2], target: [0, 0, 0.3]}

**Each key is its own track.** ``lookat``, ``distance``, ``azimuth`` and ``elevation`` interpolate
between the nearest keyframes that *state* that key and hold beyond the first and last of them; a key
no keyframe states is never written, so the base camera keeps it. That is what lets a path of
``azimuth`` alone orbit a tracking camera while MuJoCo keeps driving its ``lookat``.

``eye``/``target`` is the world-metres form, for when "stand here, look there" is easier than orbit
angles: it converts to the four orbit keys at load (:func:`roqsim.rendering.orbit_from_eye`), so the
rest of the code sees one vocabulary.

Azimuth takes the shortest arc between keyframes (350 -> 10 passes through 0), which is what a person
means nine times in ten. A full orbit needs ``wrap: false`` and an angle past 180, e.g. 180 -> 540.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: The four keys a free camera has, and the only ones a keyframe interpolates.
ORBIT_KEYS = ("lookat", "distance", "azimuth", "elevation")

#: What a keyframe may state: the orbit keys, or the eye/target pair that converts to them.
KEYFRAME_KEYS = frozenset({"t", *ORBIT_KEYS, "eye", "target"})

#: What a path document may state beside its keyframes.
DOCUMENT_KEYS = frozenset({"keyframes", "ease", "wrap"})

EASES = ("linear", "smoothstep")

#: ``onset`` and ``onset+2.5``: a moment by name with an optional offset in seconds.
_MOMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:([+-])\s*([0-9]*\.?[0-9]+))?\s*$")


class CameraPathError(ValueError):
    """A path document says something a render cannot do (see the message)."""


@dataclass(frozen=True)
class Moment:
    """A time named rather than numbered: ``onset``, ``onset+2.5``, ``t_goal-1``.

    Resolved late, against the recording, because the number is the recording's to give: where the
    run first moved, or a moment it marked.
    """

    name: str
    offset: float = 0.0

    def resolve(self, moments: dict) -> float:
        if self.name not in moments:
            have = ", ".join(sorted(moments)) or "none"
            raise CameraPathError(
                f"{self}: {self.name!r} is not a moment this render knows. It knows: {have}."
            )
        return float(moments[self.name]) + self.offset

    def __str__(self) -> str:
        if not self.offset:
            return self.name
        return f"{self.name}{self.offset:+g}"


def parse_moment(value) -> float | Moment:
    """A time as ``--at``/``--from``/``--to`` and a keyframe take it: seconds, or a named moment."""
    if isinstance(value, bool):
        raise CameraPathError(f"{value!r} is not a time")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    try:
        return float(text)
    except ValueError:
        pass
    if (m := _MOMENT_RE.match(text)) is None:
        raise CameraPathError(
            f"{text!r} is neither a time in seconds nor a moment such as 'onset' or 'onset+2.5'"
        )
    name, sign, amount = m.groups()
    offset = float(amount) * (-1.0 if sign == "-" else 1.0) if amount else 0.0
    return Moment(name, offset)


@dataclass(frozen=True)
class Keyframe:
    t: float | Moment
    values: dict = field(default_factory=dict)  # a subset of ORBIT_KEYS

    @property
    def time(self) -> float:
        if isinstance(self.t, Moment):
            raise CameraPathError(f"keyframe at {self.t} is not anchored yet")
        return self.t


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


_EASE_FN = {"linear": lambda u: u, "smoothstep": _smoothstep}


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class CameraPath:
    """Keyframes, and the camera they give at any time between them."""

    def __init__(self, keyframes: list[Keyframe], *, ease: str = "linear", wrap: bool = True):
        if not keyframes:
            raise CameraPathError("a camera path needs at least one keyframe")
        if ease not in EASES:
            raise CameraPathError(f"ease {ease!r} is not one of {', '.join(EASES)}")
        self.keyframes = list(keyframes)
        self.ease = ease
        self.wrap = bool(wrap)
        self._tracks: dict[str, list[tuple[float, object]]] | None = None
        if self.anchored:
            self._index()

    # -- loading -------------------------------------------------------------------------------

    @classmethod
    def from_doc(cls, doc) -> CameraPath:
        """A path from its document (a mapping with ``keyframes``, or a bare list of keyframes)."""
        if isinstance(doc, list):
            doc = {"keyframes": doc}
        if not isinstance(doc, dict):
            raise CameraPathError(
                "a camera path is a mapping with 'keyframes' (or a list of them), not "
                f"{type(doc).__name__}"
            )
        if unknown := set(doc) - DOCUMENT_KEYS:
            raise CameraPathError(
                f"camera path: unknown key(s) {', '.join(sorted(unknown))}; "
                f"it takes {', '.join(sorted(DOCUMENT_KEYS))}"
            )
        raw = doc.get("keyframes")
        if not isinstance(raw, list) or not raw:
            raise CameraPathError("camera path: 'keyframes' must be a non-empty list")
        frames = [_keyframe(entry, i) for i, entry in enumerate(raw)]
        return cls(frames, ease=str(doc.get("ease", "linear")), wrap=bool(doc.get("wrap", True)))

    @classmethod
    def from_arg(cls, text: str) -> CameraPath:
        """``--camera-path``: a YAML/JSON file, or the document inline as JSON (``{``/``[`` first)."""
        import yaml

        stripped = text.strip()
        if stripped[:1] in "{[":
            try:
                return cls.from_doc(json.loads(stripped))
            except json.JSONDecodeError as err:
                raise CameraPathError(
                    f"--camera-path: inline document is not JSON: {err}"
                ) from None
        path = Path(text)
        if not path.is_file():
            raise CameraPathError(
                f"--camera-path {text!r}: no such file. Give a YAML/JSON file, or the document "
                "inline as JSON."
            )
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        try:
            return cls.from_doc(doc)
        except CameraPathError as err:
            raise CameraPathError(f"{path}: {err}") from None

    # -- anchoring ---------------------------------------------------------------------------

    @property
    def anchored(self) -> bool:
        return all(not isinstance(k.t, Moment) for k in self.keyframes)

    @property
    def moments(self) -> frozenset[str]:
        """The moment names the keyframes use, which a render must resolve before drawing."""
        return frozenset(k.t.name for k in self.keyframes if isinstance(k.t, Moment))

    def anchor(self, moments: dict) -> CameraPath:
        """The same path with every named moment resolved to seconds."""
        frames = [
            Keyframe(k.t.resolve(moments) if isinstance(k.t, Moment) else k.t, k.values)
            for k in self.keyframes
        ]
        return CameraPath(frames, ease=self.ease, wrap=self.wrap)

    def _index(self) -> None:
        times = [k.time for k in self.keyframes]
        for a, b in zip(times, times[1:], strict=False):
            if b < a:
                raise CameraPathError(
                    f"keyframes must be in time order; {b:g} s comes after {a:g} s"
                )
        tracks: dict[str, list] = {key: [] for key in ORBIT_KEYS}
        for k in self.keyframes:
            for key, value in k.values.items():
                tracks[key].append((k.time, value))
        self._tracks = {key: pts for key, pts in tracks.items() if pts}

    # -- evaluation --------------------------------------------------------------------------

    @property
    def keys(self) -> frozenset[str]:
        """Which camera keys this path writes."""
        return frozenset(key for k in self.keyframes for key in k.values)

    @property
    def span(self) -> tuple[float, float]:
        return self.keyframes[0].time, self.keyframes[-1].time

    def at(self, t: float) -> dict:
        """The camera keys at ``t``: each interpolated on its own track, held outside it."""
        if self._tracks is None:
            raise CameraPathError(
                f"camera path uses {', '.join(sorted(self.moments))}, which must be resolved first"
            )
        out = {}
        ease = _EASE_FN[self.ease]
        for key, pts in self._tracks.items():
            if t <= pts[0][0]:
                out[key] = pts[0][1]
                continue
            if t >= pts[-1][0]:
                out[key] = pts[-1][1]
                continue
            for (t0, v0), (t1, v1) in zip(pts, pts[1:], strict=False):
                if t0 <= t <= t1:
                    u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
                    out[key] = self._lerp(key, v0, v1, u)
                    break
        return out

    def _lerp(self, key: str, v0, v1, u: float):
        if key == "lookat":
            return [float(a + (b - a) * u) for a, b in zip(v0, v1, strict=True)]
        if key == "azimuth" and self.wrap:
            return float(v0 + _wrap180(v1 - v0) * u)
        return float(v0 + (v1 - v0) * u)

    def apply(self, cam, t: float, *, tracking=None) -> None:
        """Write the path's keys at ``t`` onto ``cam`` (a ``mujoco.MjvCamera``).

        Under a tracking camera MuJoCo owns ``lookat``, so that key is skipped; and under
        ``follow_heading`` the azimuth is an *offset behind the robot*, which is what a path then
        animates -- through :attr:`roqsim.viewer.TrackingCamera.azimuth_offset`, never the raw
        ``cam.azimuth`` (a change there is folded in as if the mouse had dragged it, and drifts).
        """
        import mujoco

        values = self.at(t)
        tracked = int(cam.type) == int(mujoco.mjtCamera.mjCAMERA_TRACKING)
        if "lookat" in values and not tracked:
            cam.lookat[:] = values["lookat"]
        if "distance" in values:
            cam.distance = values["distance"]
        if "elevation" in values:
            cam.elevation = values["elevation"]
        if "azimuth" in values:
            if tracking is not None and tracking.follow_heading:
                tracking.azimuth_offset = values["azimuth"]
            else:
                cam.azimuth = values["azimuth"]

    def to_doc(self) -> dict:
        """The document :meth:`from_doc` reads back, numbers rounded to what a camera can mean."""
        keyframes = []
        for k in self.keyframes:
            entry = {"t": str(k.t) if isinstance(k.t, Moment) else round(float(k.t), 6)}
            entry.update(_rounded(k.values))
            keyframes.append(entry)
        return {"ease": self.ease, "wrap": self.wrap, "keyframes": keyframes}

    def write(self, path: str | Path) -> Path:
        """Write the document to a YAML file, one keyframe per line, and return its path."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.to_doc()
        lines = [
            f"ease: {doc['ease']}",
            f"wrap: {'true' if doc['wrap'] else 'false'}",
            "keyframes:",
        ]
        for entry in doc["keyframes"]:
            lines.append(
                "  - " + yaml.safe_dump(entry, default_flow_style=True, width=10_000).strip()
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def describe(self) -> dict:
        """The path as the render record reports it: what it animates, over which times."""
        out = {
            "keyframes": len(self.keyframes),
            "keys": sorted(self.keys),
            "ease": self.ease,
            "wrap": self.wrap,
        }
        if self.anchored:
            out["span"] = [round(v, 6) for v in self.span]
            out["resolved"] = [
                {"t": round(k.time, 6), **_rounded(k.values)} for k in self.keyframes
            ]
        else:
            out["moments"] = sorted(self.moments)
        return out


def _keyframe(entry, index: int) -> Keyframe:
    where = f"keyframe {index}"
    if not isinstance(entry, dict):
        raise CameraPathError(f"{where}: a keyframe is a mapping such as {{t: 0, azimuth: 90}}")
    if unknown := set(entry) - KEYFRAME_KEYS:
        raise CameraPathError(
            f"{where}: unknown key(s) {', '.join(sorted(unknown))}; a keyframe takes t, "
            f"{', '.join(ORBIT_KEYS)}, or eye + target"
        )
    if "t" not in entry:
        raise CameraPathError(f"{where}: no 't'")
    try:
        t = parse_moment(entry["t"])
    except CameraPathError as err:
        raise CameraPathError(f"{where}: t: {err}") from None

    values: dict = {}
    if ("eye" in entry) != ("target" in entry):
        raise CameraPathError(
            f"{where}: eye and target go together (where to stand, where to look)"
        )
    if "eye" in entry:
        if stated := [k for k in ORBIT_KEYS if k in entry]:
            raise CameraPathError(
                f"{where}: eye/target already say where the camera is; drop {', '.join(stated)}"
            )
        from .rendering import orbit_from_eye

        eye = _vector(entry["eye"], f"{where}: eye")
        target = _vector(entry["target"], f"{where}: target")
        try:
            lookat, distance, azimuth, elevation = orbit_from_eye(eye, target)
        except ValueError as err:
            raise CameraPathError(f"{where}: {err}") from None
        values = {
            "lookat": lookat,
            "distance": distance,
            "azimuth": azimuth,
            "elevation": elevation,
        }
    for key in ORBIT_KEYS:
        if key not in entry:
            continue
        if key == "lookat":
            values[key] = _vector(entry[key], f"{where}: lookat")
        else:
            values[key] = _number(entry[key], f"{where}: {key}")
    if not values:
        raise CameraPathError(f"{where}: states no camera key (only t)")
    if "distance" in values and values["distance"] <= 0:
        raise CameraPathError(f"{where}: distance must be positive, got {values['distance']:g}")
    return Keyframe(t, values)


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraPathError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _vector(value, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CameraPathError(f"{where}: expected three numbers, e.g. [1, 2, 0.5], got {value!r}")
    return [_number(v, where) for v in value]


def _rounded(values: dict) -> dict:
    out = {}
    for key, value in values.items():
        if isinstance(value, list):
            out[key] = [round(float(v), 4) for v in value]
        else:
            out[key] = round(float(value), 4)
    return out


__all__ = ["CameraPath", "CameraPathError", "Keyframe", "Moment", "parse_moment", "ORBIT_KEYS"]
