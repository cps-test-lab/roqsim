"""A compiled flex as a skin: which bodies move its vertices, and with what weights.

A flex has no mesh of its own that a viewer could animate. MuJoCo computes each vertex position
(``flexvert_xpos``) from bodies every step, and those bodies are all a run capture records. This
module states that dependency in the one form a viewer already animates -- linear-blend skinning
over named bodies, at most four per vertex -- so :mod:`roqsim.export_web` can ship a flex as an
ordinary skin, and a replay deforms it from the bodies' pose tracks alone.

Where a vertex's position comes from depends on how the flex was declared:

* **Per-vertex bodies** (``dof="full"``, ``"radial"``, a ``<flex>`` naming bodies): a vertex sits on
  ``flex_vertbodyid`` at the offset ``flex_vert`` in that body's frame. That is one bone with weight
  1, exactly. A **pinned** vertex is the same case: its body is the flex's parent, and the offset is
  where on it the vertex is pinned.
* **Interpolated** (``flex_interp > 0``: ``dof="trilinear"`` or ``"quadratic"``): a vertex is a
  weighted sum of the positions of the node bodies (``flex_nodebodyid``). The weights are
  **measured**, not re-derived: :func:`rig` moves each node body in turn and reads how every vertex
  follows through MuJoCo's own ``mj_flex``, so a change in MuJoCo's basis cannot drift from this.

A skin blends each bone's *rigid* motion since the bind pose, which reproduces an interpolated flex
exactly when the weights sum to one and reproduce the vertex's own position from the bones' --
provided the node bodies share one frame orientation, as MuJoCo's node bodies (siblings on slide
joints) do. Trilinear gives a surface vertex at most four non-zero weights, so it is exact.
Quadratic gives one on a face nine, past the four a viewer's skinning accepts; :func:`rig` then
keeps the four largest and corrects them minimally so they still reproduce the vertex's rest position
and sum to one. That keeps the rest pose, every rigid motion and every affine deformation exact, and
leaves only curvature between the nodes approximate; the vertices it touched are reported.

:func:`owned_bodies` answers the other half: which bodies exist only to carry a flex. They are one per
vertex or node, so a table of named bodies (``sim_poses.csv``) leaves them out; a run capture keeps
them, because they are what a skin's bones are.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: Bones per vertex a viewer's skinning takes (three.js ``SkinnedMesh``, and the skin export's cap).
MAX_INFLUENCES = 4

#: A measured weight below this is a zero: the finite difference is exact for a linear map, so the
#: only non-zero noise is floating-point round-off, many orders below this.
_WEIGHT_EPS = 1e-9

#: Displacement used to measure an interpolated vertex's dependence on a node body. The map is
#: linear, so the step size only has to be large against round-off.
_STEP = 1e-3


@dataclass(frozen=True)
class FlexRig:
    """How one flex's vertices follow bodies, at the bind state :func:`rig` was given.

    Vertex ``i`` of the flex is ``sum_k weight[i, k] * T(bone[index[i, k]]) * inv(B(...)) * vert[i]``,
    where ``T`` is a bone body's current world transform and ``B`` its transform at the bind state.
    """

    #: Body ids of the bones, in the order ``index`` refers to.
    bones: list[int]
    #: Each bone's world position and orientation (wxyz) at the bind state, ``(n_bones, 3|4)``.
    bind_pos: np.ndarray
    bind_quat: np.ndarray
    #: The flex's vertex positions at the bind state, world frame, ``(n_vert, 3)``.
    vert: np.ndarray
    #: Per vertex, up to four bone slots and their weights (unused slots weigh 0), ``(n_vert, 4)``.
    index: np.ndarray
    weight: np.ndarray
    #: Vertices that depend on more than four bodies and were reduced to four (see the module doc).
    reduced: np.ndarray


def flex_name(model: mujoco.MjModel, flex: int) -> str:
    """Flex ``flex``'s name, or ``#<id>`` for an unnamed one."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex) or f"#{flex}"


def anchor_bodies(model: mujoco.MjModel, flex: int) -> np.ndarray:
    """The bodies flex ``flex``'s positions are defined on: one per node if interpolated, else per vertex."""
    if int(model.flex_interp[flex]):
        adr, num = int(model.flex_nodeadr[flex]), int(model.flex_nodenum[flex])
        return model.flex_nodebodyid[adr : adr + num]
    adr, num = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
    return model.flex_vertbodyid[adr : adr + num]


def owned_bodies(model: mujoco.MjModel) -> dict[int, list[int]]:
    """Per flex id, the bodies that exist only to carry that flex's vertices or nodes.

    Those are its anchor bodies less the one it is declared in: a pinned vertex sits on the flex's
    parent body -- an end effector, a table -- which is a body in its own right, and it is recognised
    as the parent of the flex's other anchors. A flex whose vertices all sit on one body (a rigid or
    fully pinned flex) owns none. Flexes that own nothing are left out.
    """
    out: dict[int, list[int]] = {}
    for f in range(model.nflex):
        anchors = list(dict.fromkeys(int(b) for b in anchor_bodies(model, f) if b >= 0))
        if len(anchors) <= 1:
            continue
        parents = {int(model.body_parentid[b]) for b in anchors}
        owned = [b for b in anchors if b not in parents]
        if owned:
            out[f] = owned
    return out


def surface(model: mujoco.MjModel, flex: int) -> np.ndarray:
    """What a viewer draws of flex ``flex``, in flex-local vertex ids.

    ``(n, 3)`` triangles for a solid (its boundary, ``flex_shell``, wound counter-clockwise seen from
    outside) and for a sheet (its elements, whose winding says nothing about a side); ``(n, 2)`` edges
    for a line flex, which has no surface until the caller gives it a radius.
    """
    dim = int(model.flex_dim[flex])
    if dim == 3:
        adr, num = int(model.flex_shelldataadr[flex]), int(model.flex_shellnum[flex])
        return model.flex_shell[adr : adr + num * 3].reshape(-1, 3)
    adr, num = int(model.flex_elemdataadr[flex]), int(model.flex_elemnum[flex])
    return model.flex_elem[adr : adr + num * (dim + 1)].reshape(-1, dim + 1)


def rig(model: mujoco.MjModel, data: mujoco.MjData, flex: int) -> FlexRig:
    """Measure how flex ``flex``'s vertices follow bodies, binding at the state in ``data``.

    ``data`` must have had ``mj_kinematics`` run; it is used as scratch (``xpos`` perturbed and
    restored, ``flexvert_xpos`` recomputed) and left as it was found.
    """
    mujoco.mj_flex(model, data)
    vadr, nvert = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
    vert = data.flexvert_xpos[vadr : vadr + nvert].copy()
    anchors = anchor_bodies(model, flex)
    bones = list(dict.fromkeys(int(b) for b in anchors))
    if not int(model.flex_interp[flex]):
        slot = {b: i for i, b in enumerate(bones)}
        dense = np.zeros((nvert, len(bones)))
        dense[np.arange(nvert), [slot[int(b)] for b in anchors]] = 1.0
    else:
        dense = _measure(model, data, flex, bones, vert)
    bind_pos = data.xpos[bones].copy()
    index, weight, reduced = _cap(dense, bind_pos, vert)
    return FlexRig(
        bones=bones,
        bind_pos=bind_pos,
        bind_quat=data.xquat[bones].copy(),
        vert=vert,
        index=index,
        weight=weight,
        reduced=reduced,
    )


def _measure(model, data, flex, bones, vert) -> np.ndarray:
    """``(n_vert, n_bones)``: how far each vertex moves per unit move of each node body, per MuJoCo."""
    vadr, nvert = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
    dense = np.zeros((nvert, len(bones)))
    for k, body in enumerate(bones):
        rest = data.xpos[body].copy()
        gain = np.zeros((nvert, 3))
        for axis in range(3):
            moved = []
            for sign in (1.0, -1.0):
                data.xpos[body] = rest
                data.xpos[body, axis] += sign * _STEP
                mujoco.mj_flex(model, data)
                moved.append(data.flexvert_xpos[vadr : vadr + nvert, axis].copy())
            gain[:, axis] = (moved[0] - moved[1]) / (2 * _STEP)
        data.xpos[body] = rest
        dense[:, k] = gain.mean(axis=1)
    mujoco.mj_flex(model, data)
    dense[np.abs(dense) < _WEIGHT_EPS] = 0.0
    return dense


def _cap(dense: np.ndarray, bone_pos: np.ndarray, vert: np.ndarray):
    """Pack dense weights into four slots per vertex, reducing a vertex that has more (module doc)."""
    nvert = dense.shape[0]
    index = np.zeros((nvert, MAX_INFLUENCES), np.uint16)
    weight = np.zeros((nvert, MAX_INFLUENCES), np.float32)
    reduced = np.zeros(nvert, bool)
    for v in range(nvert):
        row = dense[v]
        nonzero = np.flatnonzero(row)
        if len(nonzero) <= MAX_INFLUENCES:
            keep, w = nonzero, row[nonzero]
        else:
            reduced[v] = True
            keep = np.argsort(-row, kind="stable")[:MAX_INFLUENCES]
            w = row[keep] / row[keep].sum()
            # The smallest change to those weights that makes them sum to one and reproduce the
            # vertex's own position from the bones' (affine precision).
            a = np.vstack([bone_pos[keep].T, np.ones(len(keep))])
            b = np.append(vert[v], 1.0)
            w = w + np.linalg.pinv(a) @ (b - a @ w)
        index[v, : len(keep)] = keep
        weight[v, : len(keep)] = w
    return index, weight, reduced
