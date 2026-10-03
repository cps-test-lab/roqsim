"""Named fixed frames: the links a vendor description has and a compiled MJCF flattens away.

A URDF chains fixed links (``base_link -> shell_link -> rplidar_link``) that an MJCF port merges
into one body, and a consumer that builds its TF tree from the vendor description still expects
every one of them. A model states them in a ``frames:`` block -- in its manifest, or in a spawn's
config -- as the vendor writes them::

    frames:
      - {name: shell_link, parent: base_link, pose: {position: {z: 0.0945}}}
      - name: rplidar_link
        parent: shell_link
        pose: {position: {x: -0.04, z: 0.0987}, orientation: {yaw: 1.5708}}

``parent`` is a body of the model or a frame declared before this one; ``pose`` is the fixed joint's
origin relative to it, a ``geometry_msgs/Pose`` read by :func:`roqsim.pose.parse_pose` with
``relative=True``: every omitted component is zero, so an absent ``pose`` is the parent itself.
Each frame becomes a site of the model at build time, on the body its chain ends at, so the pose
lives in the compiled model and a device can name the frame as where it hangs (``parent_frame``).
At configure the chain is published as static transforms read back from that compiled model, never
recomputed from the numbers above.

Every frame of a world is also named by a path (:mod:`roqsim.paths`): :func:`resolve_frame` finds
an entity's root, bodies, sites, cameras and declared or device frames by it, and a body, site or
camera of the world's own MJCF that no entity owns by its MuJoCo name; :func:`frame_pose` reads
one's pose, in the world or relative to another, from the core pose data. A camera's frame is
MuJoCo's: it looks along its -z axis with +y up, where an optical frame looks along +z with +y down.

ROS-free: a transform is plain numbers, and the bridge turns it into a message.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import mujoco
import numpy as np

from .context import Endpoint
from .endpoint import value_type
from .plugin import PluginError
from .pose import PoseError, parse_pose, refuse_pos_rpy
from .types import Transform, Transforms

_FRAME_KEYS = frozenset({"name", "parent", "pose"})

#: Geom/site group the frame sites live in. Not a rendered one: a frame is a coordinate system, and
#: a marker drawn at every flattened link would clutter every render of the robot.
FRAME_SITE_GROUP = 5


@dataclass(frozen=True)
class FrameDecl:
    name: str
    parent: str
    pos: tuple[float, float, float]
    #: ``(w, x, y, z)``, MuJoCo's order.
    quat: tuple[float, float, float, float]


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


def parse_frames(entries, where: str) -> list[FrameDecl]:
    """Validate a ``frames:`` block into declarations, in order. ``None`` is no frames."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise PluginError(f"{where}: 'frames' must be a list of {{name, parent, pose}} entries")
    out: list[FrameDecl] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        at = f"{where}.frames[{i}]"
        if not isinstance(entry, dict):
            raise PluginError(f"{at}: must be a mapping of name, parent, pose")
        if refusal := refuse_pos_rpy(entry, at):
            raise PluginError(refusal)
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
        try:
            pos, quat = parse_pose(entry.get("pose") or {}, relative=True)
        except PoseError as exc:
            raise PluginError(f"{at}: {exc}") from None
        seen.add(name)
        out.append(FrameDecl(name, parent, tuple(pos), tuple(quat)))
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
        quat = _compose(pquat, frame.quat)
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


def static_transforms_of(links: list[dict]) -> Transforms:
    """*links* -- ``{parent, child, translation, rotation}`` each -- as the value of an endpoint that
    publishes them once as static transforms (``ros2={"static": True}``)::

        @endpoint.out(ros2={"static": True})
        def frames(self) -> Transforms:
            return static_transforms_of(self.links)
    """
    return Transforms(
        [
            Transform(
                link["parent"],
                link["child"],
                np.asarray(link["translation"], dtype=float),
                np.asarray(link["rotation"], dtype=float),
            )
            for link in links
        ]
    )


_TRANSFORMS = value_type(Transforms)


def static_tf_endpoint(name: str, owner: str, namespace: str, transforms: list[dict]) -> Endpoint:
    """An output endpoint that carries only static transforms, sent once by a bridge.

    The endpoint a plugin that builds its endpoints by hand adds for a chain of fixed frames:
    :func:`static_transforms_of` of *transforms*, with the ``static`` hint. Frame names are bare; the
    bridge scopes them by ``namespace``.
    """
    value = static_transforms_of(transforms)
    return Endpoint(
        name=name,
        direction="out",
        owner=owner,
        namespace=namespace,
        read=lambda: value,
        backend={"ros2": {"static": True}},
        result=_TRANSFORMS,
        payload_type=_TRANSFORMS,
        transport=True,
    )


# -- frames by path ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """A frame a path names (:func:`resolve_frame`).

    Attributes:
        path: its path (``robot/oakd/oakd_link``, ``gantry``)
        name: its name as TF shows it: the body's, site's or camera's without the model prefix, the
            entity's own name for its root, the MuJoCo name for an unowned one
        kind: ``root``, ``body``, ``site``, ``frame`` (a declared or device frame) or ``camera``
        entity: the entity it belongs to; empty for a body, site or camera of the world that no
            entity owns
        index: the body id (a root's is its entity's body), the site id or the camera id
    """

    path: str
    name: str
    kind: str
    entity: str
    index: int

    @property
    def is_camera(self) -> bool:
        """Whether it is a MuJoCo camera, whose intrinsics a consumer can read by :attr:`index`."""
        return self.kind == "camera"


def _owners(ctx) -> dict[int, list]:
    """Body id -> the entities it belongs to: those of the nearest body up its chain that is an
    entity's root. MuJoCo numbers a parent before its children, so one pass in id order suffices.
    The world body (id 0) and a body no entity's root is above map to no entity."""
    model = ctx.model
    roots: dict[int, list] = {}
    for entity in ctx.entities.all():
        if not entity.body:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if bid > 0:
            roots.setdefault(bid, []).append(entity)
    owners: dict[int, list] = {0: []}
    for bid in range(1, model.nbody):
        owners[bid] = roots.get(bid) or owners[int(model.body_parentid[bid])]
    return owners


def frame_offers(ctx) -> list:
    """Every frame the world offers, as :class:`roqsim.paths.Offer` of kind ``frame``.

    An entity offers its root (its own path), and every body under it and every site and camera on
    those bodies -- down to where a nested entity's root takes over -- named without the entity's
    model prefix. A site of the frame group is a frame a ``frames:`` block declared or a device's
    frame chain, and is offered as a ``frame``. A named body, site or camera no entity owns -- one
    of the world's own MJCF -- is offered by its MuJoCo name alone (``gantry``); its sort reads
    ``unowned body``, so a clash with an entity's path of the same spelling is refused naming both.
    """
    from .paths import Offer, address_path

    model = ctx.model
    owners = _owners(ctx)

    def offer(entity, name: str, what: str, index: int) -> Offer:
        if entity is None:
            frame = Frame(name, name, what, "", index)
            return Offer("", name, "frame", f"unowned {what}", target=frame)
        prefix = entity.meta.get("prefix", "")
        bare = name[len(prefix) :] if prefix and name.startswith(prefix) else name
        component = address_path(entity.name)
        frame = Frame(f"{component}/{bare}", bare, what, entity.name, index)
        return Offer(component, bare, "frame", what, target=frame)

    def owned_by(bid: int) -> list:
        return owners[bid] or [None]

    out = []
    for entity in ctx.entities.all():
        if not entity.body:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if bid <= 0:
            continue
        component, _, last = address_path(entity.name).rpartition("/")
        root = Frame(address_path(entity.name), entity.name, "root", entity.name, bid)
        out.append(Offer(component, last, "frame", "root", target=root))
    for bid in range(1, model.nbody):
        name = model.body(bid).name
        for entity in owned_by(bid) if name else ():
            out.append(offer(entity, name, "body", bid))
    for sid in range(model.nsite):
        name = model.site(sid).name
        what = "frame" if int(model.site_group[sid]) == FRAME_SITE_GROUP else "site"
        for entity in owned_by(int(model.site_bodyid[sid])) if name else ():
            out.append(offer(entity, name, what, sid))
    for cid in range(model.ncam):
        name = model.camera(cid).name
        for entity in owned_by(int(model.cam_bodyid[cid])) if name else ():
            out.append(offer(entity, name, "camera", cid))
    return out


def resolve_frame(ctx, path: str, *, within: str | None = None) -> Frame:
    """The frame *path* names (:mod:`roqsim.paths`), or :class:`roqsim.paths.PathError`.

    With *within* (an entity's name), *path* is relative to that entity: ``.`` is its root and
    ``mouse`` is ``<within>/mouse``; a leading ``/`` makes it absolute again.
    """
    from .paths import address_path, resolve

    if within is not None and not path.startswith("/"):
        base = address_path(within)
        path = base if path.strip() in ("", ".") else f"{base}/{path}"
    return resolve(frame_offers(ctx), path, "frame").target


def entity_body(ctx, entity: str | None, path: str = "", *, who: str) -> Frame:
    """The body a component of *entity* measures or acts on, as a :class:`Frame` of kind ``root``
    or ``body`` whose ``index`` is the body id.

    Without *path* it is the root body the entity registered. With one, *path* is read as
    :func:`resolve_frame` reads it within the entity (``base_link`` is ``<entity>/base_link``, a
    leading ``/`` makes it absolute) and must name a body. *who* names the caller in the errors.

    Raises :class:`RuntimeError` for an entity that is not registered, or that registered no body
    while *path* is relative to it, and for a path that names no body: a component resolved to a
    body chosen by naming convention measures whichever body happens to carry that name.
    """
    from .paths import PathError

    absolute = path.startswith("/")
    if not absolute:
        registered = ctx.entities.get(entity) if entity else None
        if registered is None:
            raise RuntimeError(
                f"{who}: no entity {entity!r} is registered, so there is no body to resolve. Nest "
                f"this entry under the entry that spawns the entity; that entry registers it."
            )
        if not registered.body:
            raise RuntimeError(
                f"{who}: entity {entity!r} registered no body. The entry that creates it must "
                f"register the entity with its root body (Entity.body) for {who} to resolve one."
            )
        if mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, registered.body) <= 0:
            raise RuntimeError(
                f"{who}: base body {registered.body!r} not found; entity {entity!r} registered "
                f"it, and the compiled model has no body of that name."
            )
    try:
        frame = resolve_frame(ctx, path, within=entity)
    except PathError as err:
        raise RuntimeError(f"{who}: {err}") from err
    if frame.kind not in ("root", "body"):
        raise RuntimeError(f"{who}: {frame.path!r} names a {frame.kind}, not a body.")
    return frame


def _world_pose(ctx, frame: Frame):
    """``(position, quaternion, rotation matrix)`` of *frame* in the world, or ``None`` while its
    entity is absent. A root's comes from its entity's core pose endpoint, a body's, a site's and a
    camera's from the physics state that endpoint reads. An unowned frame has no entity to be
    absent, so its pose is always there."""
    if frame.entity:
        entity = ctx.entities.get(frame.entity)
        if entity is None or not entity.present:
            return None
    d = ctx.data
    if frame.kind == "root":
        from . import entity_pose

        ep = ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name(frame.entity))
        pose = ep.read() if ep is not None else None
        if pose is None:
            return None
        return pose.position, pose.orientation, d.xmat[frame.index].reshape(3, 3)
    if frame.kind == "body":
        i = frame.index
        return d.xpos[i].copy(), d.xquat[i].copy(), d.xmat[i].reshape(3, 3)
    if frame.kind == "camera":
        pos, mat = d.cam_xpos[frame.index], d.cam_xmat[frame.index]
    else:
        pos, mat = d.site_xpos[frame.index], d.site_xmat[frame.index]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, mat)
    return pos.copy(), quat, mat.reshape(3, 3).copy()


def frame_pose(ctx, path: str | Frame, relative_to: str | Frame | None = None) -> Transform | None:
    """The pose of the frame *path* names, in the world or relative to *relative_to*.

    A :class:`~roqsim.types.Transform` from *relative_to*'s name (``world`` for the world) to the
    frame's, rotation ``(w, x, y, z)``; ``None`` while either frame's entity is absent. Paths are
    resolved with :func:`resolve_frame`; a caller that reads every step resolves once and passes the
    :class:`Frame`.
    """
    frame = path if isinstance(path, Frame) else resolve_frame(ctx, path)
    world = _world_pose(ctx, frame)
    if world is None:
        return None
    pos, quat, _ = world
    if relative_to is None:
        return Transform("world", frame.name, pos, quat)
    ref = relative_to if isinstance(relative_to, Frame) else resolve_frame(ctx, relative_to)
    base = _world_pose(ctx, ref)
    if base is None:
        return None
    ref_pos, ref_quat, ref_mat = base
    # pos_rel = R_ref^T (pos - ref_pos); quat_rel = ref_quat^-1 * quat.
    inv = np.empty(4)
    mujoco.mju_negQuat(inv, ref_quat)
    rel_quat = np.empty(4)
    mujoco.mju_mulQuat(rel_quat, inv, quat)
    return Transform(ref.name, frame.name, ref_mat.T @ (pos - ref_pos), rel_quat)
