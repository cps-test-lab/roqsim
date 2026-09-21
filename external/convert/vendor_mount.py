"""Shared by the device builders that re-seat a device model on its vendor link.

A device model whose ``mount`` body is the link its vendor macro builds everything from is placed by
the vendor joint origin; one whose ``mount`` was pre-rotated into a display convention was not. When
a builder moves a model from the second to the first it renames it (the old name is refused at load,
see ``roqsim.models.RetiredModel``) and every existing mount must be re-expressed: a mount of the old
model at ``T_old`` becomes ``T_old * D``, for the ``D`` its builder derives.

This module holds what every such builder needs and nothing device-specific: rotations in the URDF
convention, a reader for one xacro macro's properties and fixed joints, the splice that rewrites a
manifest's generated block, and :func:`rewrite_mounts`, which applies a ``D`` to the world files it
is given. Builders: ``build_realsense_devices.py``, ``build_oakd_pro.py``.
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


def matrix_rpy(m: np.ndarray) -> tuple[float, float, float]:
    """The inverse of :func:`rpy_matrix`, with pitch in [-pi/2, pi/2]."""
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(pitch)) < 1e-9:  # gimbal lock: fold the roll into the yaw
        return 0.0, pitch, math.atan2(-m[0, 1], m[1, 1])
    return math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


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


def rewrite_pose(pos, rpy, delta) -> tuple[list[float], list[float]]:
    r_d, t_d = delta
    r_old = rpy_matrix(rpy)
    p_old = np.asarray(list(pos) + [0.0] * (3 - len(pos)), dtype=float)
    return (p_old + r_old @ t_d).tolist(), list(matrix_rpy(r_old @ r_d))


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


def fmt_list(values) -> str:
    return "[" + ", ".join(fmt_num(v) for v in values) + "]"


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


# -- re-expressing existing mounts -----------------------------------------------------------------

_MODEL = re.compile(
    r"^(?P<indent>\s*)model:\s*(?P<quote>['\"]?)(?P<name>[\w:]+)(?P=quote)\s*(#.*)?$"
)


def _block(lines: list[str], i: int, width: int) -> tuple[int, int]:
    """The ``[start, end)`` lines of the mapping whose keys sit at *width*, around line *i*."""

    def inside(line: str) -> bool:
        body = line.strip()
        return not body or body.startswith("#") or len(line) - len(line.lstrip(" ")) >= width

    start = i
    while start > 0 and inside(lines[start - 1]):
        start -= 1
    end = i + 1
    while end < len(lines) and inside(lines[end]):
        end += 1
    return start, end


_FLOW = re.compile(r"spawn_sensor:\s*(\{.*\})")


def _flow_mount(line: str, deltas: dict) -> str | None:
    """*line* re-expressed if it is a one-line flow ``spawn_sensor: {model: <retired>, ...}``."""
    import yaml

    fm = _FLOW.search(line)
    if not fm:
        return None
    body = fm.group(1)
    try:
        cfg = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    name = str(cfg.get("model", "")).split(":")[-1] if isinstance(cfg, dict) else ""
    if name not in deltas:
        return None
    new_model, delta = deltas[name]
    new_pos, new_rpy = rewrite_pose(cfg.get("pos", [0, 0, 0]), cfg.get("rpy", [0, 0, 0]), delta)
    out = re.sub(rf"(model:\s*['\"]?(?:\w+:)?){name}\b", rf"\g<1>{new_model}", body, count=1)
    for key, value in (("pos", new_pos), ("rpy", new_rpy)):
        if re.search(rf"\b{key}:\s*\[", out):
            out = re.sub(rf"\b{key}:\s*\[[^\]]*\]", f"{key}: {fmt_list(value)}", out, count=1)
        else:
            out = out[:-1].rstrip() + f", {key}: {fmt_list(value)}}}"
    if yaml.safe_load(out).get("model") is None:
        raise RuntimeError(f"rewriting {line.strip()!r} lost its model")
    return line[: fm.start(1)] + out + line[fm.end(1) :]


def rewrite_mounts(path: Path, deltas: dict) -> int:
    """Rename and re-pose every ``spawn_sensor`` of a retired model in *path*.

    *deltas* maps each retired model name to ``(new name, (R, t))``: a mount of the old model at
    ``T_old`` is written as ``T_old * (R, t)`` (:func:`rewrite_pose`).

    Line-based, so comments and layout survive. A block-style mount is named by its ``model:`` line;
    its sibling ``pos:``/``rpy:`` lines (same indentation, flow lists) are rewritten in place, and a
    missing one is added after ``pos:``. A one-line flow mount (``spawn_sensor: {model: ..., pos:
    [...]}``) is rewritten within its line. Anything else naming a retired model -- a pose written as
    a block list, a mount split over lines some other way -- is refused, since a pose left
    unconverted would load at the wrong place silently.
    """
    import yaml

    names = "|".join(re.escape(n) for n in sorted(deltas))
    retired = re.compile(rf"model:\s*['\"]?(\w+:)?({names})\b")
    lines = path.read_text().splitlines(keepends=True)
    changed = 0
    i = 0
    while i < len(lines):
        flow = _flow_mount(lines[i], deltas)
        if flow is not None:
            lines[i] = flow
            changed += 1
            i += 1
            continue
        m = _MODEL.match(lines[i].rstrip("\n"))
        name = m and m.group("name").split(":")[-1]
        if not m or name not in deltas:
            if retired.search(lines[i]):
                raise RuntimeError(f"{path}:{i + 1}: a retired mount this rewrite cannot parse")
            i += 1
            continue
        indent = m.group("indent")
        start, end = _block(lines, i, len(indent))
        keys = {}
        for j in range(start, end):
            km = re.match(
                rf"^{re.escape(indent)}(pos|rpy):\s*(\[.*\])\s*(#.*)?$", lines[j].rstrip("\n")
            )
            if km:
                keys[km.group(1)] = (j, yaml.safe_load(km.group(2)), km.group(3) or "")
            elif re.match(rf"^{re.escape(indent)}(pos|rpy):", lines[j]):
                raise RuntimeError(f"{path}:{j + 1}: pos/rpy must be a flow list to be rewritten")
        pos = keys.get("pos", (None, [0.0, 0.0, 0.0], ""))[1]
        rpy = keys.get("rpy", (None, [0.0, 0.0, 0.0], ""))[1]
        new_pos, new_rpy = rewrite_pose(pos, rpy, deltas[name][1])
        new_model = deltas[name][0]
        lines[i] = lines[i].replace(m.group("name"), m.group("name").replace(name, new_model), 1)
        pos_line = f"{indent}pos: {fmt_list(new_pos)}\n"
        rpy_line = f"{indent}rpy: {fmt_list(new_rpy)}\n"
        if "pos" in keys:
            lines[keys["pos"][0]] = pos_line
        if "rpy" in keys:
            lines[keys["rpy"][0]] = rpy_line
        insert = []
        if "pos" not in keys:
            insert.append(pos_line)
        if "rpy" not in keys:
            insert.append(rpy_line)
        at = (keys["pos"][0] + 1) if "pos" in keys else i + 1
        lines[at:at] = insert
        changed += 1
        i += 1 + len(insert)
    path.write_text("".join(lines))
    return changed
