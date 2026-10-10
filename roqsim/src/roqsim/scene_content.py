"""What a compiled world's scene contains: the one walk every scene export serialises.

``roqsim export web`` and ``roqsim export gltf`` write the same scene in two formats. What the scene
*contains* -- which bodies, where each sits, which geoms are drawn, the meshes with their texture
coordinates, the skins and the flexes drawn as skins, the materials and the textures they use -- is
decided here, once, so the two files cannot disagree about it. A writer decides only how to spell
it.

:func:`walk` reads a compiled :class:`mujoco.MjModel` and the :class:`mujoco.MjData` holding the
state to export, and returns a :class:`SceneContent`:

* **bodies**, in MuJoCo's order, each with its parent and the local pose a format with joints
  starts from: a free body's from ``qpos``, a mocap body's from ``xpos``/``xquat``, every other
  body's from ``body_pos``/``body_quat``, its joints at rest (:func:`body_poses`);
* **joints**, with the configured value of each named hinge and slide, which a format with joints
  applies to those poses;
* **geoms** that are drawn (:func:`geom_drawn`), with the dense index of the mesh each mesh geom
  uses;
* **meshes**, one per mesh a drawn geom uses, with per-vertex texture coordinates where the mesh has
  them (:func:`mesh_geometry`);
* **skins**: every drawn MuJoCo ``<skin>``, then every drawn flex as a skin whose bones are the
  bodies its vertices follow (:mod:`roqsim.flex_skin`);
* **materials** and the RGB-role **textures** they reference, in first-use order;
* every body's **world pose** at the exported state (:func:`exported_state`), for a format that
  places bodies rather than joints.

A geom, skin or flex in the collision group (:data:`COLLISION_GROUP`) is not drawn. That is the
whole drawn-or-not rule, and :func:`drawn_group` is the only place it is spelled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import flex_skin

#: The repo convention: group-3 geometry is collision-only and never drawn.
COLLISION_GROUP = 3

# MuJoCo joint types (mjtJoint) -> the name a scene description uses.
JOINT_TYPE = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}

# MuJoCo geom types (mjtGeom) a scene draws -> their name. A type not listed (a heightfield, an SDF)
# is not drawn.
GEOM_TYPE = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}

TEXROLE_RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)

#: Sides of the tube a line flex (``dim=1``) is drawn as.
TUBE_SIDES = 8

_SCALAR_JOINT_TYPES = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


@dataclass
class Body:
    name: str
    parent: int
    pos: list  # local, parent-relative
    quat: list  # local, wxyz


@dataclass
class Joint:
    name: str
    body: int
    type: str
    axis: list
    pos: list
    qposadr: int


@dataclass
class Geom:
    id: int  # the MuJoCo geom id
    name: str
    body: int
    type: str
    pos: list
    quat: list  # wxyz
    size: list
    matid: int
    rgba: list
    mesh: int | None  # index into SceneContent.meshes for a mesh geom


@dataclass
class Mesh:
    vert: np.ndarray  # (n, 3)
    index: np.ndarray  # (m, 3), into vert
    uv: np.ndarray | None  # (n, 2), MuJoCo's convention (v from the image's top row)
    source: int | None = None  # the MuJoCo mesh id, for a geom mesh


@dataclass
class Skin:
    """A deformable surface: a MuJoCo skin, or a flex drawn as one. Vertices are world-frame."""

    mesh: Mesh
    bones: list[str]  # bone body names
    bone_ids: list[int]
    bindpos: list  # per bone, world frame
    bindquat: list  # per bone, wxyz
    skin_index: np.ndarray  # (n, 4) uint16, into bones
    skin_weight: np.ndarray  # (n, 4) float32, rows sum to 1
    matid: int
    rgba: list
    flex: str | None = None  # the flex this skin draws, if it draws one


@dataclass
class Material:
    id: int  # the MuJoCo material id
    rgba: list
    texture: int  # index into SceneContent.textures, -1 for none
    texrepeat: list
    texuniform: bool


@dataclass
class Texture:
    id: int  # the MuJoCo texture id
    width: int
    height: int
    channels: int
    path: str | None  # the path MuJoCo recorded for it, None for a procedural texture

    def pixels(self, model: mujoco.MjModel) -> np.ndarray:
        """The compiled pixels, ``(height, width, channels)`` uint8, rows top first."""
        adr = int(model.tex_adr[self.id])
        n = self.height * self.width * self.channels
        return np.asarray(model.tex_data[adr : adr + n], dtype=np.uint8).reshape(
            self.height, self.width, self.channels
        )

    def source_file(self) -> Path | None:
        """The image file it came from, when the recorded path resolves on disk."""
        if self.path and Path(self.path).is_file():
            return Path(self.path)
        return None


@dataclass
class SceneContent:
    bodies: list[Body] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    initial_joints: dict[str, float] = field(default_factory=dict)
    geoms: list[Geom] = field(default_factory=list)
    meshes: list[Mesh] = field(default_factory=list)
    skins: list[Skin] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    textures: list[Texture] = field(default_factory=list)
    #: Every body's world (pos, wxyz quat) at :func:`exported_state`.
    world_poses: list[tuple[list, list]] = field(default_factory=list)

    @property
    def flex_count(self) -> int:
        return sum(1 for s in self.skins if s.flex is not None)


# -- what is drawn ---------------------------------------------------------------------------------


def drawn_group(group: int) -> bool:
    """Whether geometry in ``group`` is drawn: everything but the collision group."""
    return int(group) != COLLISION_GROUP


def geom_drawn(model: mujoco.MjModel, g: int) -> bool:
    """Whether geom ``g`` is part of the drawn scene."""
    if not drawn_group(model.geom_group[g]):
        return False
    gtype = int(model.geom_type[g])
    if gtype not in GEOM_TYPE:
        return False
    return not (gtype == int(mujoco.mjtGeom.mjGEOM_MESH) and int(model.geom_dataid[g]) < 0)


#: A geom's ``rgba`` when the MJCF gives none. MuJoCo draws a geom in its material's colour unless
#: its own differs from this.
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


def resolved_rgba(model: mujoco.MjModel, matid: int, rgba) -> list[float]:
    """The colour MuJoCo draws a geom in (``setMaterial`` in its visualiser).

    The material's ``rgba``, unless the geom's own differs from :data:`DEFAULT_RGBA` -- then the
    geom's, which is how a geom tints or hides (alpha 0) one instance of a shared material. With no
    material, the geom's own.
    """
    own = [float(c) for c in rgba]
    if matid < 0 or any(abs(a - b) > 1e-6 for a, b in zip(own, DEFAULT_RGBA, strict=True)):
        return own
    return [float(c) for c in model.mat_rgba[matid]]


# -- strings ---------------------------------------------------------------------------------------


def model_path(model: mujoco.MjModel, adr: int) -> str | None:
    """Read a NUL-terminated string at ``adr`` in ``model.paths`` (the packed texture file paths)."""
    if adr < 0:
        return None
    paths = model.paths
    end = paths.find(b"\x00", adr)
    return paths[adr:end].decode() if end >= 0 else paths[adr:].decode()


# -- bodies and joints -----------------------------------------------------------------------------


def _free_body_poses(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, tuple[list, list]]:
    """Map each free-jointed body -> its initial (pos, quat) from ``data.qpos``.

    A free body's ``body_pos``/``body_quat`` is just a static offset (usually identity); its real
    initial placement lives in ``qpos`` (3 pos + 4 quat). This reads the *configured* state
    (``data.qpos`` after ``setup()``, which applies each spawn plugin's initial pose), not
    ``model.qpos0``, which is the bare model default (often all zeros).
    """
    out: dict[int, tuple[list, list]] = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            adr = int(model.jnt_qposadr[j])
            pos = data.qpos[adr : adr + 3].tolist()
            quat = data.qpos[adr + 3 : adr + 7].tolist()  # wxyz
            out[int(model.jnt_bodyid[j])] = (pos, quat)
    return out


def _mocap_body_poses(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, tuple[list, list]]:
    """Map each mocap body -> its world (pos, quat) from ``data.xpos``/``data.xquat``.

    A mocap body's ``body_pos``/``body_quat`` is a static placeholder (the walker parks its bones at
    z=-50 until the first pose is written); its real placement is written each step to
    ``data.mocap_pos``/``mocap_quat`` and propagated into ``data.xpos``/``xquat`` by ``mj_forward``.
    Reading the world transform seats the exported pose at the configured stance instead of the park
    pose. Mocap bodies are world children, so the world transform is also the local one.
    """
    out: dict[int, tuple[list, list]] = {}
    for i in range(model.nbody):
        if int(model.body_mocapid[i]) >= 0:
            out[i] = (data.xpos[i].tolist(), data.xquat[i].tolist())
    return out


def body_poses(model: mujoco.MjModel, data: mujoco.MjData) -> list[Body]:
    """Every body with its parent and its local pose at the exported state."""
    free = _free_body_poses(model, data)
    mocap = _mocap_body_poses(model, data)
    bodies = []
    for i in range(model.nbody):
        if i in free:
            pos, quat = free[i]
        elif i in mocap:
            pos, quat = mocap[i]
        else:
            pos, quat = model.body_pos[i].tolist(), model.body_quat[i].tolist()
        bodies.append(Body(model.body(i).name, int(model.body_parentid[i]), pos, quat))
    return bodies


def joints(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[Joint], dict[str, float]]:
    """Return (every joint, {name: configured value}) for the named hinge and slide joints.

    The values come from the configured ``data.qpos`` (set by the spawn plugins during ``setup()``),
    not ``model.qpos0``.
    """
    out = []
    initial: dict[str, float] = {}
    for i in range(model.njnt):
        jtype = JOINT_TYPE.get(int(model.jnt_type[i]), "unknown")
        name = model.joint(i).name
        qadr = int(model.jnt_qposadr[i])
        out.append(
            Joint(
                name,
                int(model.jnt_bodyid[i]),
                jtype,
                model.jnt_axis[i].tolist(),
                model.jnt_pos[i].tolist(),
                qadr,
            )
        )
        # An unnamed joint has no key a viewer could look its value up by: every unnamed joint would
        # share the key "", and the last one's value would seat them all. Left out, it rests at its
        # reference position (:func:`exported_state` relies on that).
        if jtype in ("hinge", "slide") and name:
            initial[name] = float(data.qpos[qadr])
    return out, initial


# -- meshes ----------------------------------------------------------------------------------------


def reindex(faces: np.ndarray, *corner_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One GPU index over several OBJ-style attribute indices.

    ``faces`` and each array in ``corner_indices`` are ``(m, 3)`` indices into their own attribute.
    Returns ``(unique, index)``: ``unique[k]`` is the tuple of attribute indices of new vertex ``k``
    (column 0 into the vertices, then one column per extra index), and ``index`` the ``(m, 3)``
    faces over the new vertices. Only the combinations the faces use become vertices.
    """
    pairs = np.stack([faces.ravel(), *(c.ravel() for c in corner_indices)], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique, inverse.reshape(-1, 3)


def mesh_geometry(model: mujoco.MjModel, mid: int, logger: logging.Logger) -> Mesh:
    """Indexed geometry (positions + faces + UVs) of mesh ``mid``.

    Face indices are mesh-local, so slicing ``mesh_vert``/``mesh_face`` yields a self-contained
    indexed mesh. Texcoords must travel: a texture atlas -- what every ``roqsim_assets`` prop and
    every baked scene mesh uses -- cannot be reconstructed from geometry. MuJoCo indexes them
    OBJ-style (``mesh_facetexcoord``), which for some meshes is already per-vertex
    (``mesh_texcoordnum == mesh_vertnum`` with identical face indices) and for others is not (a
    mesh with more texcoords than vertices). One index is what a GPU buffer takes, so the second
    case is re-indexed over the unique (vertex, texcoord) pairs the faces use (:func:`reindex`).

    UVs are in MuJoCo's convention (v measured from the image's top row).
    """
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = model.mesh_vert[va : va + vn]
    faces = model.mesh_face[fa : fa + fn]
    tca = int(model.mesh_texcoordadr[mid])
    uv = None
    if tca >= 0:
        tcn = int(model.mesh_texcoordnum[mid])
        texcoord = model.mesh_texcoord[tca : tca + tcn]
        ftex = model.mesh_facetexcoord[fa : fa + fn]
        if tcn == vn and np.array_equal(ftex, faces):
            uv = texcoord
        else:
            unique, faces = reindex(faces, ftex)
            verts = verts[unique[:, 0]]
            uv = texcoord[unique[:, 1]]
            logger.debug(
                "mesh %r: split %d vertices into %d to carry its %d texcoords",
                model.mesh(mid).name,
                vn,
                len(unique),
                tcn,
            )
    return Mesh(verts, faces, uv, mid)


# -- skins and flexes ------------------------------------------------------------------------------


def _skins(model: mujoco.MjModel, logger: logging.Logger) -> list[Skin]:
    """Every drawn MuJoCo ``<skin>``: bind-pose geometry plus a per-vertex bone block.

    The bind block transposes MuJoCo's per-bone vertex lists (``skin_bonevert*``) into per-vertex
    bone indices and weights, at most four per vertex (the cap of a GPU skinning attribute), and
    renormalised; it carries the bone body names and each bone's world bind pose.
    """
    skins: list[Skin] = []
    over_cap = 0
    for i in range(model.nskin):
        if not drawn_group(model.skin_group[i]):
            continue
        va, vn = int(model.skin_vertadr[i]), int(model.skin_vertnum[i])
        fa, fn = int(model.skin_faceadr[i]), int(model.skin_facenum[i])
        tca = int(model.skin_texcoordadr[i])
        uv = model.skin_texcoord[tca : tca + vn] if tca >= 0 else None  # one (u, v) per vertex
        mesh = Mesh(model.skin_vert[va : va + vn], model.skin_face[fa : fa + fn], uv)

        ba, bn = int(model.skin_boneadr[i]), int(model.skin_bonenum[i])
        bones, bone_ids, bindpos, bindquat = [], [], [], []
        skin_index = np.zeros((vn, 4), np.uint16)
        skin_weight = np.zeros((vn, 4), np.float32)
        slot = np.zeros(vn, np.int32)  # next free (index, weight) slot per vertex
        for j in range(bn):
            bid = int(model.skin_bonebodyid[ba + j])
            bones.append(model.body(bid).name)
            bone_ids.append(bid)
            bindpos.append(model.skin_bonebindpos[ba + j].tolist())  # world bind pose
            bindquat.append(model.skin_bonebindquat[ba + j].tolist())  # wxyz
            bva = int(model.skin_bonevertadr[ba + j])
            bvn = int(model.skin_bonevertnum[ba + j])
            vids = model.skin_bonevertid[bva : bva + bvn]
            wts = model.skin_bonevertweight[bva : bva + bvn]
            for vid, w in zip(vids.tolist(), wts.tolist(), strict=True):
                s = int(slot[vid])
                if s < 4:
                    skin_index[vid, s] = j
                    skin_weight[vid, s] = w
                    slot[vid] = s + 1
                else:
                    over_cap += 1
        # Renormalise (MuJoCo weights already sum to ~1; re-normalising guards the >4-bone drop).
        wsum = skin_weight.sum(axis=1, keepdims=True)
        wsum[wsum == 0] = 1.0
        skin_weight /= wsum
        skins.append(
            Skin(
                mesh,
                bones,
                bone_ids,
                bindpos,
                bindquat,
                skin_index,
                skin_weight,
                int(model.skin_matid[i]),
                model.skin_rgba[i].tolist(),
            )
        )
    if over_cap:
        logger.warning(
            "skin export: %d bone-weight entries past the 4-bones/vertex cap were dropped "
            "(weights renormalised)",
            over_cap,
        )
    return skins


def exported_state(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjData:
    """The state a scene shows before any track: ``data`` with every unnamed joint at rest.

    That is ``data`` -- free bodies, mocap bodies and named joints, as :func:`body_poses` and
    :func:`joints` read them -- with every unnamed hinge and slide joint at its reference position,
    where a viewer of ``roqsim.web_scene`` shows it, having no value for it. A flex binds here (its
    vertex bodies move on unnamed slide joints, so its bind shape is the flex at rest on its posed
    parent, whatever it had settled into in ``data``), and a format with no joints, such as glTF,
    places each body where this state puts it.
    """
    bind = mujoco.MjData(model)
    bind.qpos[:] = data.qpos
    bind.mocap_pos[:] = data.mocap_pos
    bind.mocap_quat[:] = data.mocap_quat
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in _SCALAR_JOINT_TYPES and not model.joint(j).name:
            adr = int(model.jnt_qposadr[j])
            bind.qpos[adr] = model.qpos0[adr]
    mujoco.mj_kinematics(model, bind)
    return bind


def _tube(rig: flex_skin.FlexRig, edges: np.ndarray, radius: float):
    """A line flex's edges as open tubes: vertices, triangles, and the flex vertex each one follows.

    Each edge gets its own ring of :data:`TUBE_SIDES` points at either end, placed around the edge's
    rest direction and bound to the bones of the vertex at that end -- so a ring follows its vertex,
    and keeps its rest orientation rather than turning with a bend.
    """
    theta = np.linspace(0.0, 2 * np.pi, TUBE_SIDES, endpoint=False)
    verts, tris, rows = [], [], []
    for a, b in edges:
        axis = rig.vert[b] - rig.vert[a]
        length = np.linalg.norm(axis)
        if length == 0.0:
            continue
        axis /= length
        helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * w)
        base = len(verts)
        verts.extend(rig.vert[a] + ring)
        verts.extend(rig.vert[b] + ring)
        rows.extend([a] * TUBE_SIDES + [b] * TUBE_SIDES)
        for i in range(TUBE_SIDES):
            j = (i + 1) % TUBE_SIDES
            a_i, a_j = base + i, base + j
            b_i, b_j = a_i + TUBE_SIDES, a_j + TUBE_SIDES
            # Counter-clockwise seen from outside: (u, w, axis) is right-handed.
            tris.extend([(a_i, a_j, b_i), (a_j, b_j, b_i)])
    return np.asarray(verts).reshape(-1, 3), np.asarray(tris).reshape(-1, 3), np.asarray(rows, int)


def _flex_uv(model: mujoco.MjModel, flex: int, nvert: int) -> np.ndarray | None:
    """The flex's texture coordinates when it has exactly one per vertex, else None."""
    adr = int(model.flex_texcoordadr[flex])
    if adr < 0:
        return None
    ends = [int(a) for a in model.flex_texcoordadr if int(a) > adr]
    end = min(ends) if ends else int(model.nflextexcoord)
    if end - adr != nvert:
        return None
    return model.flex_texcoord[adr:end]


def _flexes(model: mujoco.MjModel, bind: mujoco.MjData, logger: logging.Logger) -> list[Skin]:
    """Every drawn flex as a skin, so a viewer that animates skins needs nothing new.

    A flex's bones are the bodies its vertices follow, with measured weights
    (:mod:`roqsim.flex_skin`), bound at :func:`exported_state` (``bind``). What is drawn:

    * a solid (``dim=3``): its boundary triangles, ``flex_shell``, facing outward;
    * a sheet (``dim=2``): its elements, twice -- once per side, over a second copy of the vertices,
      because a viewer draws front faces only and a vertex shared by two opposite windings would get
      a normal of zero;
    * a line (``dim=1``): an open tube of ``flex_radius`` around each edge (:func:`_tube`).

    Colour and material are ``flex_rgba`` / ``flex_matid``, and texture coordinates travel when the
    flex has one per vertex. A flex in the collision group is skipped, as a geom is, and one whose
    bones are not all named is skipped with a warning, since a viewer binds bones by name.
    """
    skins: list[Skin] = []
    if not model.nflex:
        return skins
    for f in range(model.nflex):
        name = flex_skin.flex_name(model, f)
        if not drawn_group(model.flex_group[f]):
            continue
        rig = flex_skin.rig(model, bind, f)
        bone_names = [model.body(b).name for b in rig.bones]
        if not all(bone_names):
            logger.warning(
                "flex %r not exported: it follows %d unnamed body/bodies, and a viewer binds a "
                "skin's bones by name. Name the bodies (a <flexcomp> names its own).",
                name,
                sum(1 for n in bone_names if not n),
            )
            continue
        dim = int(model.flex_dim[f])
        faces = flex_skin.surface(model, f)
        nvert = len(rig.vert)
        rows = np.arange(nvert)  # the flex vertex each exported vertex follows
        uv = _flex_uv(model, f, nvert)
        if dim == 3:
            verts = rig.vert
        elif dim == 2:
            verts = np.vstack([rig.vert, rig.vert])
            faces = np.vstack([faces, faces[:, ::-1] + nvert])
            rows = np.concatenate([rows, rows])
            uv = None if uv is None else np.vstack([uv, uv])
        else:
            radius = float(model.flex_radius[f])
            if radius <= 0.0:
                logger.warning("flex %r not exported: a line flex of radius 0 has no surface", name)
                continue
            verts, faces, rows = _tube(rig, faces, radius)
            uv = None
        reduced = int(rig.reduced[np.unique(rows[np.unique(faces)])].sum())
        if reduced:
            logger.warning(
                "flex %r: %d drawn vertices follow more than %d node bodies (dof=quadratic). Each is "
                "drawn from its %d largest weights -- exact at rest and under any affine deformation, "
                "approximate under curvature between nodes",
                name,
                reduced,
                flex_skin.MAX_INFLUENCES,
                flex_skin.MAX_INFLUENCES,
            )
        skins.append(
            Skin(
                Mesh(verts, faces, uv),
                bone_names,
                [int(b) for b in rig.bones],
                rig.bind_pos.tolist(),  # world bind pose
                rig.bind_quat.tolist(),  # wxyz
                rig.index[rows],
                rig.weight[rows],
                int(model.flex_matid[f]),
                model.flex_rgba[f].tolist(),
                flex=name,
            )
        )
    return skins


# -- the walk --------------------------------------------------------------------------------------


def walk(model: mujoco.MjModel, data: mujoco.MjData, logger: logging.Logger) -> SceneContent:
    """What the scene of ``model`` at the state in ``data`` contains."""
    scene = SceneContent()

    # Geoms first: they say which meshes are drawn, so unreferenced and collision-only mesh data is
    # skipped. Mesh ids are remapped to a dense 0..N-1 in reference order.
    used_meshes: dict[int, int] = {}
    for g in range(model.ngeom):
        if not geom_drawn(model, g):
            continue
        gtype = GEOM_TYPE[int(model.geom_type[g])]
        mesh_ref = None
        if gtype == "mesh":
            mesh_ref = used_meshes.setdefault(int(model.geom_dataid[g]), len(used_meshes))
        scene.geoms.append(
            Geom(
                g,
                model.geom(g).name,
                int(model.geom_bodyid[g]),
                gtype,
                model.geom_pos[g].tolist(),
                model.geom_quat[g].tolist(),
                model.geom_size[g].tolist(),
                int(model.geom_matid[g]),
                model.geom_rgba[g].tolist(),
                mesh_ref,
            )
        )
    mesh_ids = [mid for mid, _ in sorted(used_meshes.items(), key=lambda kv: kv[1])]
    scene.meshes = [mesh_geometry(model, mid, logger) for mid in mesh_ids]

    # Skins (deformable character meshes) are rigged to bones, not walked as geoms; flexes have no
    # mesh at all and are drawn as skins whose bones are the bodies their vertices follow.
    state = exported_state(model, data)
    scene.skins = _skins(model, logger) + _flexes(model, state, logger)

    # Materials carry only their RGB-role texture. Textures are pruned to those an RGB role
    # references, remapped to a dense 0..N-1 -- normal maps and unused textures are dropped.
    used_tex: dict[int, int] = {}
    for i in range(model.nmat):
        rgb_tid = int(model.mat_texid[i, TEXROLE_RGB])
        tex_ref = used_tex.setdefault(rgb_tid, len(used_tex)) if rgb_tid >= 0 else -1
        scene.materials.append(
            Material(
                i,
                model.mat_rgba[i].tolist(),
                tex_ref,
                model.mat_texrepeat[i].tolist(),
                bool(model.mat_texuniform[i]),
            )
        )
    for tid, _ in sorted(used_tex.items(), key=lambda kv: kv[1]):
        scene.textures.append(
            Texture(
                tid,
                int(model.tex_width[tid]),
                int(model.tex_height[tid]),
                int(model.tex_nchannel[tid]),
                model_path(model, int(model.tex_pathadr[tid])),
            )
        )
    scene.joints, scene.initial_joints = joints(model, data)
    scene.bodies = body_poses(model, data)
    scene.world_poses = [
        (state.xpos[i].tolist(), state.xquat[i].tolist()) for i in range(model.nbody)
    ]
    return scene
