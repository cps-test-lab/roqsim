"""Named fixed frames: the links a vendor description has and a compiled MJCF flattens away.

A URDF chains fixed links (``base_link -> shell_link -> rplidar_link``) that an MJCF port merges
into one body, and a consumer that builds its TF tree from the vendor description still expects
every one of them. A model states them in a ``frames:`` block -- in its manifest, or in a spawn's
config -- as the vendor writes them::

    frames:
      - {name: shell_link, parent: base_link, pos: [0, 0, 0.0945], rpy: [0, 0, 0]}
      - {name: rplidar_link, parent: shell_link, pos: [-0.04, 0, 0.0987], rpy: [0, 0, 1.5708]}

``parent`` is a body of the model or a frame declared before this one; ``pos``/``rpy`` are the
fixed joint's origin relative to it (metres, radians; both default to zero). Each frame becomes a
site of the model at build time, on the body its chain ends at, so the pose lives in the compiled
model and a mount can name the frame as where it hangs. At configure the chain is published as
static transforms read back from that compiled model, never recomputed from the numbers above.

ROS-free: a transform is plain numbers, and the bridge turns it into a message.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import mujoco
import numpy as np

from .context import Endpoint
from .plugin import PluginError
from .pose import rpy_to_quat

_FRAME_KEYS = frozenset({"name", "parent", "pos", "rpy"})

#: Geom/site group the frame sites live in. Not a rendered one: a frame is a coordinate system, and
#: a marker drawn at every flattened link would clutter every render of the robot.
FRAME_SITE_GROUP = 5


@dataclass(frozen=True)
class FrameDecl:
    name: str
    parent: str
    pos: tuple[float, float, float]
    rpy: tuple[float, float, float]


def substitute(value, values: dict[str, str], where: str):
    """*value* with ``{placeholder}`` fields in every string replaced from *values*, recursively.

    Only the names in *values* exist. Anything else in braces -- a misspelt placeholder, a format
    spec, a positional ``{}`` -- is refused rather than passed through, because a frame named
    ``{frame_idd}`` reaches a TF tree as a literal and nothing downstream says why it is orphaned.
    ``{{``/``}}`` spell a literal brace.
    """
    if isinstance(value, dict):
        return {k: substitute(v, values, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, values, f"{where}[{i}]") for i, v in enumerate(value)]
    if not isinstance(value, str):
        return value
    out = []
    try:
        parsed = list(string.Formatter().parse(value))
    except ValueError as exc:
        raise PluginError(f"{where}: {value!r} is not a valid template: {exc}") from None
    for literal, field, spec, conversion in parsed:
        out.append(literal)
        if field is None:
            continue
        if spec or conversion:
            raise PluginError(
                f"{where}: {value!r} formats {{{field}}}; a placeholder is replaced by its value "
                f"as it stands, with no conversion or format spec."
            )
        if field not in values:
            known = ", ".join("{" + k + "}" for k in sorted(values))
            raise PluginError(
                f"{where}: {value!r} uses the placeholder {{{field}}}, which is not one this "
                f"document fills in. Known: {known}. Write a literal brace as '{{{{' or '}}}}'."
            )
        out.append(str(values[field]))
    return "".join(out)


def _triple(value, key: str, where: str) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise PluginError(f"{where}: '{key}' must be [x, y, z] / [roll, pitch, yaw], got {value!r}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        raise PluginError(f"{where}: '{key}' must be three numbers, got {value!r}") from None


def parse_frames(entries, where: str) -> list[FrameDecl]:
    """Validate a ``frames:`` block into declarations, in order. ``None`` is no frames."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise PluginError(f"{where}: 'frames' must be a list of {{name, parent, pos, rpy}} entries")
    out: list[FrameDecl] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        at = f"{where}.frames[{i}]"
        if not isinstance(entry, dict):
            raise PluginError(f"{at}: must be a mapping of name, parent, pos, rpy")
        unknown = sorted(set(entry) - _FRAME_KEYS)
        if unknown:
            raise PluginError(f"{at}: unknown key(s) {unknown}; a frame has {sorted(_FRAME_KEYS)}")
        name, parent = entry.get("name"), entry.get("parent")
        for key, val in (("name", name), ("parent", parent)):
            if not isinstance(val, str) or not val:
                raise PluginError(f"{at}: '{key}' is required and must be a non-empty string")
        if name in seen:
            raise PluginError(f"{at}: frame {name!r} is declared twice")
        if parent == name:
            raise PluginError(f"{at}: frame {name!r} cannot be its own parent")
        seen.add(name)
        out.append(
            FrameDecl(
                name,
                parent,
                _triple(entry.get("pos"), "pos", at),
                _triple(entry.get("rpy"), "rpy", at),
            )
        )
    return out


def _rotate(quat, vec) -> np.ndarray:
    res = np.zeros(3)
    mujoco.mju_rotVecQuat(
        res, np.asarray(vec, dtype=np.float64), np.asarray(quat, dtype=np.float64)
    )
    return res


def _compose(quat_a, quat_b) -> np.ndarray:
    res = np.zeros(4)
    mujoco.mju_mulQuat(
        res, np.asarray(quat_a, dtype=np.float64), np.asarray(quat_b, dtype=np.float64)
    )
    return res


def add_frame_sites(spec: mujoco.MjSpec, frames: list[FrameDecl], where: str) -> None:
    """Add one site per frame to *spec* (a model, before it is attached), named as the frame.

    Each site sits on the body its parent chain ends at, at the pose composed along that chain, so
    a frame whose parent is another frame costs no body. A frame name that is already a body or a
    site of the model is refused: the two would be one name for two poses, and a mount or a TF
    consumer asking for it would get whichever MuJoCo resolved first.
    """
    bodies = {b.name for b in spec.bodies if b.name and b is not spec.worldbody}
    sites = {s.name for s in spec.sites}
    placed: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}
    for frame in frames:
        if frame.name in bodies or frame.name in sites:
            kind = "body" if frame.name in bodies else "site"
            raise PluginError(
                f"{where}: frame {frame.name!r} is already a {kind} of this model. A frame names a "
                f"pose the model does not have yet; to publish an existing {kind}, name it as a "
                f"parent instead."
            )
        if frame.parent in placed:
            body, ppos, pquat = placed[frame.parent]
        elif frame.parent in bodies:
            body, ppos, pquat = spec.body(frame.parent), np.zeros(3), np.array([1.0, 0, 0, 0])
        else:
            raise PluginError(
                f"{where}: frame {frame.name!r} hangs from {frame.parent!r}, which is neither a body "
                f"of this model nor a frame declared before it."
            )
        quat = _compose(pquat, rpy_to_quat(*frame.rpy))
        pos = ppos + _rotate(pquat, frame.pos)
        site = body.add_site(name=frame.name, pos=pos.tolist(), quat=quat.tolist())
        site.group = FRAME_SITE_GROUP
        placed[frame.name] = (body, pos, quat)


def _pose(model, data, name: str, where: str) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation matrix of a compiled site, else body, called *name*."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
    raise PluginError(f"{where}: {name!r} is neither a site nor a body of the compiled model")


def static_transforms(model, links: list[tuple[str, str, str, str]], where: str) -> list[dict]:
    """``parent -> child`` transforms read off the compiled model at its reference pose.

    *links* are ``(parent, parent_in_model, child, child_in_model)``: the bare frame names a TF
    consumer sees, and the (prefixed) site or body names they are measured between. The transform
    between two frames welded to one chain is rigid, so the reference pose is as good as any.
    Rotation is ``(w, x, y, z)``, as every other static transform a producer hands a bridge.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = []
    for parent, parent_full, child, child_full in links:
        ppos, pmat = _pose(model, data, parent_full, where)
        cpos, cmat = _pose(model, data, child_full, where)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(pmat.T @ cmat).reshape(-1))
        out.append(
            {
                "parent": parent,
                "child": child,
                "translation": [float(v) for v in pmat.T @ (cpos - ppos)],
                "rotation": [float(v) for v in quat],
            }
        )
    return out


def static_tf_endpoint(name: str, owner: str, namespace: str, transforms: list[dict]) -> Endpoint:
    """An output endpoint that carries only static transforms, published once by a bridge.

    The shape ``spawn_model`` uses for a welded prop's frame, generalised to a chain: ``read`` has
    nothing to stream, and ``static_tf`` is a list of ``{parent, child, translation, rotation}``.
    Frame names are bare; the bridge scopes them by ``namespace``.
    """
    return Endpoint(
        name=name,
        direction="out",
        owner=owner,
        namespace=namespace,
        read=lambda: None,
        backend={
            "ros2": {
                "type": "tf2_msgs.msg.TFMessage",
                "topic": "tf",
                "frame_id": transforms[0]["parent"] if transforms else "",
                "static_tf": transforms,
            }
        },
    )
