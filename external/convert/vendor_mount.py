"""Shared by the device builders that seat a device model on the frame its vendor macro places.

A device model whose ``mount`` body is the frame its vendor macro attaches to the macro's ``parent``
is placed by the ``origin`` a robot description gives that macro. This module holds what every such
builder needs and nothing device-specific: rotations in the URDF convention, a reader for one xacro
macro's properties and fixed joints, number formatting, and the splice that rewrites a manifest's
generated block. Builders: ``build_realsense_devices.py``, ``build_oakd_pro.py``.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

#: Both spellings of the xacro namespace, since vendor files use both.
XACRO_NS = ("{http://ros.org/wiki/xacro}", "{http://www.ros.org/wiki/xacro}")


# -- rotations -------------------------------------------------------------------------------------


def rpy_matrix(rpy) -> np.ndarray:
    """Fixed-axis XYZ (URDF) roll/pitch/yaw as a rotation matrix, ``Rz @ Ry @ Rx``."""
    r, p, y = (float(v) for v in rpy)
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    """A unit quaternion (w, x, y, z) with w >= 0 for a rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = ((m[k, j] - m[j, k]) / s, *v)
    q = np.asarray(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))


# -- the vendor description ----------------------------------------------------------------------


class Xacro:
    """One macro's properties and fixed joints, with its ``${...}`` expressions evaluated."""

    def __init__(self, path: Path, name: str, values: dict[str, float] | None = None):
        self.path = path
        self.root = ET.parse(path).getroot()
        #: ``pi``, plus what the macro reads from files it includes (``cm2m``, say), which a
        #: caller names rather than this class resolving the include.
        self.values: dict[str, float | str] = {"pi": math.pi, "name": name, **(values or {})}
        for prop in self._iter("property"):
            self.values[prop.get("name")] = self.eval(prop.get("value"))
        self.name = name

    def _iter(self, tag: str, under=None):
        for el in (under if under is not None else self.root).iter():
            if el.tag in {ns + tag for ns in XACRO_NS}:
                yield el

    def eval(self, text: str):
        def one(expr: str) -> str:
            value = eval(expr, {"__builtins__": {}}, dict(self.values))  # noqa: S307
            return value if isinstance(value, str) else repr(value)

        out = re.sub(r"\$\{([^}]*)\}", lambda m: one(m.group(1)), text.strip())
        try:
            return float(out)
        except ValueError:
            return out

    def vector(self, text: str | None) -> tuple[float, float, float]:
        if not text:
            return (0.0, 0.0, 0.0)
        parts = re.findall(r"\$\{[^}]*\}|[^\s]+", text.strip())
        return tuple(float(self.eval(p)) for p in parts)

    def joint(self, suffix: str) -> tuple[tuple[float, ...], tuple[float, ...], str]:
        """``(xyz, rpy, parent)`` of the fixed joint ``${name}_<suffix>``."""
        for joint in self.root.iter("joint"):
            if joint.get("name") == f"${{name}}_{suffix}":
                origin = joint.find("origin")
                parent = joint.find("parent").get("link").replace("${name}", self.name)
                return self.vector(origin.get("xyz")), self.vector(origin.get("rpy")), parent
        raise RuntimeError(f"{self.path.name}: no joint ${{name}}_{suffix}")

    def mesh_origin(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The ``${name}_link`` visual origin that goes with its mesh (the ``use_mesh`` branch).

        The origin is the sibling of the ``<geometry>`` holding the ``<mesh>``, whether the visual
        wraps the pair in an ``xacro:if`` or not.
        """
        for link in self.root.iter("link"):
            if link.get("name") != "${name}_link":
                continue
            for parent in link.iter():
                geometry = parent.find("geometry")
                origin = parent.find("origin")
                if (
                    geometry is not None
                    and geometry.find("mesh") is not None
                    and origin is not None
                ):
                    return self.vector(origin.get("xyz")), self.vector(origin.get("rpy"))
        raise RuntimeError(f"{self.path.name}: no mesh visual origin on ${{name}}_link")


# -- output ----------------------------------------------------------------------------------------


def fmt_num(v: float) -> str:
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fmt_vec(values) -> str:
    return " ".join(fmt_num(v) for v in values)


def frame_pose(pos=None, rpy=None) -> str:
    """``, pose: {...}`` for a ``frames:`` entry, zero components left out; ``""`` for the identity."""
    parts = []
    position = [f"{a}: {fmt_num(v)}" for a, v in zip("xyz", pos or (), strict=False) if v != 0]
    if position:
        parts.append("position: {" + ", ".join(position) + "}")
    angles = zip(("roll", "pitch", "yaw"), rpy or (), strict=False)
    orientation = [f"{a}: {fmt_num(v)}" for a, v in angles if v != 0]
    if orientation:
        parts.append("orientation: {" + ", ".join(orientation) + "}")
    return f", pose: {{{', '.join(parts)}}}" if parts else ""


def splice(manifest: Path, block: str, begin: str, end: str) -> None:
    """Replace the lines from *begin* to *end* in *manifest* with *block*, which carries both.

    The rest of the file is authored by hand and stays as written; a manifest without the two
    marker lines is refused rather than appended to.
    """
    text = manifest.read_text()
    if begin not in text or end not in text:
        raise RuntimeError(f"{manifest}: no generated block between {begin!r} and {end!r}")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    manifest.write_text(head + block + (tail[1:] if tail.startswith("\n") else tail))
