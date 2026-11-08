"""Skin a CARLA (or other) character mesh onto the humanoid skeleton as MuJoCo ``<skin>``s.

The walker's body pose is FK over a flat set of mocap bodies (one per skeleton joint, see
:mod:`roqsim_walker.humanoid`). Instead of capsule geoms, this binds a real character mesh to
those bodies with deformable skins that follow the bones every step (render-only -- the per-limb
collision capsules are untouched).

:func:`add_skin` loads the mesh per **material group** (a CARLA character has Skin/Jacket/Pants/
Shoes/Hair/... groups, each with its own UVs and texture), rigs each group's vertices to the bones,
and adds one ``spec.add_skin`` per group -- textured from the material's BaseColor image, or a flat
colour where there is none. CARLA meshes are exported in a **T-pose**, so the bind is computed with
the arms posed horizontally out to match; our arms-down animation then deforms it correctly.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from roqsim_walker.humanoid import (
    CHILDREN,
    DEFAULT_SKELETON,
    JOINT_NAMES,
    SKELETON,
    forward_kinematics,
)

_DEFAULT_RGBA = (0.80, 0.66, 0.52, 1.0)
# Outward stub direction for leaf joints (no child bone), so head/hand/foot vertices bind to their
# own body rather than the parent's.
_LEAF_DIR = {
    "head": (0.0, 0.0, 0.16),
    "wrist_l": (0.0, 0.0, -0.12),
    "wrist_r": (0.0, 0.0, -0.12),
    "toe_l": (0.06, 0.0, -0.02),
    "toe_r": (0.06, 0.0, -0.02),
}


def _minrot(a, b) -> np.ndarray:
    """Shortest quaternion (w,x,y,z) rotating unit vector ``a`` onto ``b``."""
    a = np.asarray(a, float) / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float) / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -0.999999:
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    ax = np.cross(a, b)
    q = np.array([1.0 + d, *ax])
    return q / np.linalg.norm(q)


def _bind_bones(tpose=False, skeleton=None) -> dict:
    """``name -> (pos, quat)`` for every joint body at the bind pose. ``tpose`` rotates the shoulders
    so the arms point horizontally out (matching a CARLA mesh's T-pose); otherwise arms-down rest.
    This is the frame the mesh is rigged against; the runtime (animation) pose deforms relative to
    it."""
    skel = skeleton or DEFAULT_SKELETON
    jr = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    if tpose:
        jr["shoulder_l"] = _minrot((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))  # arm -> +Y
        jr["shoulder_r"] = _minrot((0.0, 0.0, -1.0), (0.0, -1.0, 0.0))  # arm -> -Y
    return forward_kinematics([0.0, 0.0, skel.root_height], 0.0, jr, skeleton=skel)


def _segments(poses) -> list:
    """``(jointA, jointB, pA, pB)`` line segments covering every bone. A bone's weight is split
    between its two endpoint joints by where the vertex projects along it, so deformation blends
    smoothly across each joint. Leaves get a stub."""
    segs = []
    for j in SKELETON:
        for child in CHILDREN[j.name]:
            segs.append((j.name, child, poses[j.name][0], poses[child][0]))
        if not CHILDREN[j.name]:
            d = np.array(_LEAF_DIR.get(j.name, (0.0, 0.0, 0.1)))
            segs.append((j.name, j.name, poses[j.name][0], poses[j.name][0] + d))
    return segs


def _weights(verts, segs, k=3, power=6.0) -> np.ndarray:
    """(nvert, njoint) bone weights: nearest ``k`` segments, inverse-distance, split between each
    segment's endpoint joints by the projection t; normalised. The auto-rig fallback used when no
    authored ``*.weights.npz`` sidecar is present."""
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    n, s = len(verts), len(segs)
    dist = np.zeros((n, s))
    tval = np.zeros((n, s))
    ja = np.array([idx[a] for a, _, _, _ in segs])
    jb = np.array([idx[b] for _, b, _, _ in segs])
    for si, (_, _, a, b) in enumerate(segs):
        ab = b - a
        t = np.clip((verts - a) @ ab / (ab @ ab + 1e-9), 0.0, 1.0)
        dist[:, si] = np.linalg.norm(verts - (a + t[:, None] * ab), axis=1)
        tval[:, si] = t
    nearest = np.argsort(dist, axis=1)[:, :k]
    w = np.zeros((n, len(JOINT_NAMES)))
    rows = np.arange(n)
    for c in range(nearest.shape[1]):
        si = nearest[:, c]
        contrib = 1.0 / (dist[rows, si] + 1e-4) ** power
        t = tval[rows, si]
        np.add.at(w, (rows, ja[si]), contrib * (1.0 - t))
        np.add.at(w, (rows, jb[si]), contrib * t)
    return w / (w.sum(axis=1, keepdims=True) + 1e-9)


def _aligner(all_verts, flip=False):
    """Place the character mesh on its **per-walker** skeleton: the mesh and the skeleton come from
    the same source rig at the same native scale, so just drop the soles to the floor (z=0) -- **no
    rescale** (rescaling to a fixed height is what makes a child walker adult-sized). ``flip`` turns
    the mesh 180deg about Z for a -X facing (CARLA); rigs that already face +X (the Open-RMF actors)
    pass ``flip=False``."""
    zmin = float(all_verts[:, 2].min())

    def apply(v):
        u = v.copy()
        u[:, 2] -= zmin  # soles -> floor (mesh already ~0)
        if flip:
            u[:, :2] *= -1.0  # 180deg about Z: CARLA mesh faces -X, ours +X
        return u

    return apply


def _load_weights(mesh_path):
    """Return ``lookup(verts) -> (n, J) weights`` from the converter's ``<obj>.weights.npz`` (CARLA's
    authored skin weights, aggregated onto our joints), position-matched to the OBJ verts; ``None``
    if no sidecar. Using the real weights avoids the auto-rig pinching the waist when the arms
    drop."""
    path = os.path.splitext(os.path.abspath(mesh_path))[0] + ".weights.npz"
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    pos, wmat = d["positions"], d["weights"]
    if "joints" in d:  # remap columns to current JOINT_NAMES
        src = [str(j) for j in d["joints"]]
        col = {n: i for i, n in enumerate(src)}
        remap = np.zeros((wmat.shape[0], len(JOINT_NAMES)))
        for k, n in enumerate(JOINT_NAMES):
            if n in col:
                remap[:, k] = wmat[:, col[n]]
        wmat = remap
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pos)
        return lambda verts: wmat[tree.query(np.asarray(verts, float))[1]]
    except Exception:  # pragma: no cover - scipy fallback

        def lookup(verts):
            verts = np.asarray(verts, float)
            idx = [int(np.argmin(((pos - p) ** 2).sum(1))) for p in verts]
            return wmat[idx]

        return lookup


def _load_parts(mesh_path) -> list:
    """``[(material_name, verts, faces, uv), ...]`` -- one entry per OBJ material group. trimesh
    splits a *multi*-material OBJ into a Scene (keyed by material name); a *single*-material OBJ
    loads as one mesh, whose material name we recover from ``visual.material`` so a one-texture
    character (e.g. the Open-RMF MaleVisitor) still matches its walker.json instead of binding flat."""
    import trimesh

    loaded = trimesh.load(mesh_path, process=False)
    if hasattr(loaded, "geometry"):
        items = list(loaded.geometry.items())
    else:
        mat = getattr(getattr(loaded, "visual", None), "material", None)
        items = [(getattr(mat, "name", None), loaded)]
    parts = []
    for key, g in items:
        uv = getattr(getattr(g, "visual", None), "uv", None)
        uv = np.asarray(uv, float) if uv is not None and len(uv) else None
        parts.append(
            (key, *_merge_split(np.asarray(g.vertices, float), np.asarray(g.faces, np.int32), uv))
        )
    return parts


def _merge_split(verts, faces, uv):
    """One vertex per distinct (position, texture coordinate): ``(verts, faces, uv)``.

    trimesh splits an OBJ's vertex for every distinct ``v/vt`` index pair, and an OBJ may write a
    ``vt`` per face corner with the same coordinates repeated (the Open-RMF actors do). MuJoCo sums a
    skin vertex's normal from that vertex's own faces, so every such split is drawn as an edge: a
    vertex shared by one face or two is shaded as a flat facet. Merging the copies that agree on
    position and texture coordinate leaves a split only where the UV map really has a seam."""
    keyed = verts if uv is None else np.hstack([verts, uv])
    _, first, inverse = np.unique(
        np.round(keyed, 7), axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    keep = np.sort(first)  # keep the file's vertex order
    renumber = np.empty(len(first), np.int64)
    renumber[np.argsort(first)] = np.arange(len(first))
    new_index = renumber[inverse]
    merged_uv = None if uv is None else uv[keep]
    return verts[keep], new_index[faces].astype(np.int32), merged_uv


def add_skin(
    spec,
    name,
    mesh_path,
    materials=None,
    rgba=_DEFAULT_RGBA,
    tpose=False,
    skeleton=None,
    flip=None,
    hands=None,
) -> None:
    """Bind the character mesh at ``mesh_path`` to the ``{name}/{joint}`` humanoid bodies (call after
    :func:`roqsim_walker.humanoid.build_humanoid`, before compile).

    ``materials`` maps a material name -> ``{"texture": <png next to the obj>|None, "normal": ...,
    "color": rgba}`` (from the converter's ``*.walker.json``). Each material group becomes a textured
    (or flat) ``<skin>``. ``tpose`` binds for a T-pose mesh; ``skeleton`` is the walker's per-rig
    bone table. ``flip`` turns a -X-facing mesh to our +X; it defaults to ``tpose`` because CARLA
    exports are both T-posed *and* -X-facing, but the two are independent -- the Open-RMF actors are
    T-posed and already face +X, and set ``flip: false``.

    ``hands`` (a walker with ``hands: true``) maps each side to its
    :class:`~roqsim_motion.hands.HandLayout`: the fingers' bones join the skin. Where the blueprint
    has finger weights (``<obj>.fingers.npz``), each hand vertex's share on the source's thumb and
    finger chains moves from the wrist to those bones, the four-finger chain split between the four
    fingers by the bone nearest the vertex; where it has none, the hand stays bound to the wrist as
    one rigid part (:func:`hand_weights`)."""
    poses = _bind_bones(tpose=tpose, skeleton=skeleton)
    bones = list(JOINT_NAMES)
    if hands:
        poses = {**poses, **_bind_fingers(poses, hands)}
        bones += [b for side in ("l", "r") for b in _finger_bones(side)]
    segs = _segments(poses)
    objdir = os.path.dirname(os.path.abspath(mesh_path))
    materials = materials or {}
    parts = _load_parts(mesh_path)
    apply = _aligner(
        np.vstack([v for _, v, _, _ in parts]), flip=tpose if flip is None else bool(flip)
    )
    lookup = _load_weights(mesh_path)  # real CARLA skin weights, if exported
    fingers = _load_fingers(mesh_path) if hands else None
    for i, (matname, verts, faces, uv) in enumerate(parts):
        w = lookup(verts) if lookup is not None else None  # match raw (pre-align) verts
        shares = fingers(verts) if fingers is not None else None
        verts = apply(verts)
        if w is None:
            w = _weights(verts, segs)  # fallback: auto-rig
        if hands:
            w = hand_weights(w, verts, shares, poses)
        spec_mat = materials.get(matname, {})
        tex = spec_mat.get("texture")
        nrm = spec_mat.get("normal")
        texfile = os.path.join(objdir, tex) if tex else None
        nrmfile = os.path.join(objdir, nrm) if nrm else None
        color = spec_mat.get("color", rgba)
        _emit_part(spec, name, i, verts, faces, uv, w, poses, texfile, color, nrmfile, bones)


def _finger_bones(side: str) -> list[str]:
    from roqsim_motion.hands import bone_names

    return bone_names(side)


def _bind_fingers(poses: dict, hands: dict) -> dict:
    """The finger bones' bind poses: each hand's fingers open on its wrist at the bind pose."""
    from roqsim_motion.hands import POSES, finger_fk

    out = {}
    for side, lay in hands.items():
        wp, wq = poses[f"wrist_{side}"]
        pos, rot = finger_fk(lay, wp, wq, np.array(POSES["flat"], float))
        for k, bone in enumerate(_finger_bones(side)):
            out[bone] = (pos[k], rot[k])
    return out


def _load_fingers(mesh_path):
    """``lookup(verts) -> (n, 12)`` shares on the source's finger chains (``<obj>.fingers.npz``,
    written by ``roqsim walker import-actor --fingers-only``), or ``None`` without one."""
    path = os.path.splitext(os.path.abspath(mesh_path))[0] + ".fingers.npz"
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    from scipy.spatial import cKDTree

    tree = cKDTree(d["positions"])
    shares = d["weights"]
    return lambda verts: shares[tree.query(np.asarray(verts, float))[1]]


def hand_weights(w17: np.ndarray, verts: np.ndarray, shares, poses: dict) -> np.ndarray:
    """(n, 47) weights for a skin with finger bones, from the 17 joints' (n, 17).

    With ``shares`` (n, 12; :data:`roqsim_walker.cli.import_actor.FINGER_CHAINS` order) a vertex's
    share on a thumb bone moves from its wrist to that thumb bone, and its share on a bone of the
    four-finger chain to the same bone of the finger nearest the vertex. Without them the hand stays
    on the wrist: a rigid part, not a hand deformed from weights the mesh does not have."""
    n = w17.shape[0]
    out = np.zeros((n, len(JOINT_NAMES) + 30))
    out[:, : len(JOINT_NAMES)] = w17
    if shares is None:
        return out
    col = {name: i for i, name in enumerate(JOINT_NAMES)}
    for s_i, side in enumerate(("l", "r")):
        bones = _finger_bones(side)
        base = len(JOINT_NAMES) + 15 * s_i
        wrist = col[f"wrist_{side}"]
        for k in range(3):
            thumb = shares[:, 6 * s_i + k]
            chain = shares[:, 6 * s_i + 3 + k]
            # thumb bone k: index k in the bone list (thumb first, three bones a finger)
            out[:, base + k] += thumb
            # the four fingers' bone k: the nearest at bind
            centres = np.array([poses[bones[3 * f + k]][0] for f in range(1, 5)])
            nearest = np.argmin(np.linalg.norm(verts[:, None, :] - centres[None], axis=2), axis=1)
            for f in range(4):
                sel = nearest == f
                out[sel, base + 3 * (f + 1) + k] += chain[sel]
            out[:, wrist] -= thumb + chain
    out[:, : len(JOINT_NAMES)] = np.clip(out[:, : len(JOINT_NAMES)], 0.0, None)
    return out / (out.sum(axis=1, keepdims=True) + 1e-9)


def _emit_part(spec, name, i, verts, faces, uv, w, poses, texfile, color, nrmfile=None, bones=None) -> None:
    bones = bones or JOINT_NAMES
    used = [bi for bi in range(len(bones)) if np.any(w[:, bi] > 1e-3)]
    if not used:
        return
    sk = spec.add_skin()
    sk.name = f"{name}_sk{i}"
    sk.vert = verts.flatten()
    sk.face = faces.flatten()
    sk.bodyname = [f"{name}/{bones[bi]}" for bi in used]
    sk.bindpos = np.array([poses[bones[bi]][0] for bi in used]).flatten()
    sk.bindquat = np.array([poses[bones[bi]][1] for bi in used]).flatten()
    vertid, vertweight = [], []
    for bi in used:
        ids = np.where(w[:, bi] > 1e-3)[0].astype(np.int32)
        vertid.append(ids)
        vertweight.append(w[ids, bi].astype(float))
    sk.vertid = vertid
    sk.vertweight = vertweight
    if texfile and uv is not None and os.path.isfile(texfile):
        t = spec.add_texture()
        t.name = f"{name}_t{i}"
        t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.file = texfile
        m = spec.add_material()
        m.name = f"{name}_m{i}"
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_t{i}"
        if nrmfile and os.path.isfile(nrmfile):  # CARLA normal map -> surface detail
            tn = spec.add_texture()
            tn.name = f"{name}_n{i}"
            tn.type = mujoco.mjtTexture.mjTEXTURE_2D
            tn.file = nrmfile
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = f"{name}_n{i}"
        uvc = uv.copy()
        uvc[:, 1] = 1.0 - uvc[:, 1]  # MuJoCo texcoord V is flipped
        sk.texcoord = uvc.flatten()
        sk.material = f"{name}_m{i}"
    else:
        sk.rgba = list(color)
