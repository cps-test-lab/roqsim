# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A world as the bake sees it: what is baked, what moves, and the light sources, in world coordinates.

:func:`export_world` reads a compiled model at its reset pose (whatever built it: a baked scene, a
floorplan, hand-written MJCF, props spawned at load) and splits its drawn geoms three ways:

* **baked** -- *static* (its body is welded to the world, ``body_weldid == 0``, and is not mocap),
  drawn (geom group below 3, alpha above 0), not a light source and not excluded. Its triangles go to
  an *entity*'s light map;
* **moving** -- every other drawn geom (a door's leaf, a robot, a free-jointed prop, a pooled body).
  It is lit at run time. What hangs on a joint to the world (a door's leaf) is an *occluder* in the
  bake at its reset pose, so its shadow is there; what floats free (a robot) is not
  (:func:`is_occluder`);
* **emitters** -- a static geom that is a light source (:mod:`.lights`): not baked, it lights the bake
  and glows at run time;
* **see-through** -- a static geom drawn with an alpha below 1 (glass): light passes it, so it is
  neither baked nor an occluder, and it is drawn as it is.

An **entity** is what shares a light map: the part of a geom's name (else its body's) before the first
``/`` -- what a spawned prop is prefixed with -- or else the room's shell. The shell's triangles that face
up at the lowest static height form the entity ``floor``, which gets the floor's resolution (the
contact shadows are there); a shell larger than one light map is split by geom.

The geometry is fixed for a bake as a bake needs it: a closed mesh wound inside
out is turned round (MuJoCo draws it, a bake lights the side a face points to); a per-corner normal
more than 60 degrees off its face takes the face's (a box of eight shared corners); a sheet drawn from
both sides is pulled 1 mm apart.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .. import tessellate
from .settings import BakeSettings

#: A geom in this group or above is not drawn by default (collision geoms, absent entities).
DRAWN_GROUPS = 3
#: Segments a curved primitive is tessellated with.
SEGMENTS = 32
#: Light-map packing loses about this much of the square.
PACKING_LOSS = 1.4
#: A light map's side is a multiple of this.
ROUND_TO = 64


@dataclass
class GeomRecord:
    """A drawn geom of the world, as the bake names it."""

    id: int
    name: str
    body: str
    #: Its index among its body's geoms (an unnamed geom is found by it).
    index_in_body: int
    #: How a bake names it: its name, else ``<body>#<index>``.
    key: str
    kind: str  # "baked" | "moving" | "emitter" | "see_through"
    collides: bool
    geom_group: int
    material: int
    #: The light maps its triangles lie in (a shell geom may be split between the floor and the rest).
    entities: list = field(default_factory=list)
    emitter: int | None = None


@dataclass
class Entity:
    """The triangles that share one light map, in world coordinates."""

    name: str
    cls: str  # "shell" | "floor" | "props"
    pos: np.ndarray  # (n, 3, 3)
    nrm: np.ndarray  # (n, 3, 3)
    uv1: np.ndarray  # (n, 3, 2) albedo UV, MuJoCo's convention; zeros where untextured
    mat: np.ndarray  # (n,) into Export.materials
    geom: np.ndarray  # (n,) into Export.geoms
    size: int = 0

    @property
    def area(self) -> float:
        return float(_areas(self.pos).sum())

    @property
    def two_sided(self) -> bool:
        """A prop's mesh is not always wound one way, so both sides are baked and each face keeps the
        one that sees the light; the shell is made to face the room and baked from that side."""
        return self.cls == "props"


@dataclass
class Export:
    geoms: list[GeomRecord]
    materials: list[dict]
    entities: dict[str, Entity]
    occluders: np.ndarray  # (m, 3, 3)
    emitters: dict = field(default_factory=dict)  # geom id -> (pos, nrm)
    textures: dict = field(default_factory=dict)  # tex id -> (h, w, 3) uint8, MuJoCo's rows
    bounds: tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    notes: list[str] = field(default_factory=list)

    def geom(self, key: str) -> GeomRecord | None:
        return next((g for g in self.geoms if g.key == key), None)


# -- geometry -------------------------------------------------------------------------------------


def _areas(tri: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def face_normals(tri: np.ndarray) -> np.ndarray:
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(fn, axis=1, keepdims=True)
    return np.where(a > 1e-12, fn / np.maximum(a, 1e-12), 0.0)


def _primitive(gtype: int, size) -> tuple[np.ndarray, np.ndarray]:
    """Triangles ``(n, 3, 3)`` and per-corner normals of a primitive geom in its own frame."""
    s = np.asarray(size, float)
    G = mujoco.mjtGeom
    if gtype == G.mjGEOM_BOX:
        v, f = tessellate.box(s[:3])
        tri = v[f]
        return tri, np.repeat(face_normals(tri)[:, None], 3, axis=1)
    if gtype == G.mjGEOM_SPHERE:
        v, f = tessellate.sphere(s[0], SEGMENTS)
        tri = v[f]
        return tri, tri / np.maximum(np.linalg.norm(tri, axis=-1, keepdims=True), 1e-12)
    if gtype == G.mjGEOM_ELLIPSOID:
        v, f = tessellate.ellipsoid(s[:3], SEGMENTS)
        tri = v[f]
        n = tri / np.maximum(s[:3] ** 2, 1e-12)
        return tri, n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    if gtype == G.mjGEOM_CAPSULE:
        v, f = tessellate.capsule(s[0], s[1], SEGMENTS)
        tri = v[f]
        axis = tri.copy()
        axis[..., :2] = 0.0
        axis[..., 2] = np.clip(axis[..., 2], -s[1], s[1])
        n = tri - axis
        return tri, n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    if gtype == G.mjGEOM_CYLINDER:
        v, f = tessellate.cylinder(s[0], s[1], SEGMENTS)
        tri = v[f]
        fn = face_normals(tri)
        n = np.repeat(fn[:, None], 3, axis=1)
        side = np.abs(fn[:, 2]) < 0.5
        radial = tri[side].copy()
        radial[..., 2] = 0.0
        n[side] = radial / np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-12)
        return tri, n
    raise ValueError(f"geom type {gtype}: not a primitive the bake knows")


def _box_uv(tri: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """A 2D texture on a box: each face 0..1 over its extent along the two axes it spans."""
    uv = np.zeros(tri.shape[:2] + (2,))
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    span = np.maximum(hi - lo, 1e-9)
    for f in range(len(tri)):
        ax = int(np.argmax(np.abs(nrm[f, 0])))
        a, b = [i for i in range(3) if i != ax]
        uv[f] = (tri[f][:, [a, b]] - lo[[a, b]]) / span[[a, b]]
    return uv


def _mesh(m: mujoco.MjModel, g: int):
    k = int(m.geom_dataid[g])
    va, vn = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
    fa, fnum = int(m.mesh_faceadr[k]), int(m.mesh_facenum[k])
    tri = m.mesh_vert[va : va + vn][m.mesh_face[fa : fa + fnum]].astype(float)
    na = int(m.mesh_normaladr[k])
    nrm = m.mesh_normal[na : na + int(m.mesh_normalnum[k])][m.mesh_facenormal[fa : fa + fnum]].astype(float)
    uv = None
    if int(m.mesh_texcoordnum[k]) > 0:
        ta = int(m.mesh_texcoordadr[k])
        uv = m.mesh_texcoord[ta : ta + int(m.mesh_texcoordnum[k])][m.mesh_facetexcoord[fa : fa + fnum]]
        uv = uv.astype(float)
    return tri, nrm, uv


def outward(tri: np.ndarray, nrm: np.ndarray):
    """A closed mesh wound inside out, with its winding and normals turned round.

    Inside out: its signed volume is negative and its own normals do not say otherwise. An open mesh
    (a chair's pad) can have a negative volume and still be wound as its normals say; it is left."""
    c = tri.reshape(-1, 3).mean(0)
    vol = np.einsum("ij,ij->i", tri[:, 0] - c, np.cross(tri[:, 1] - c, tri[:, 2] - c)).sum() / 6.0
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    box = float(np.prod(np.maximum(hi - lo, 1e-9)))
    agree = float(((np.asarray(nrm).mean(1) * face_normals(tri)).sum(1) > 0).mean())
    if vol < -0.05 * box and agree < 0.9:
        return tri[:, ::-1].copy(), -np.asarray(nrm)[:, ::-1], True
    return tri, nrm, False


def sane_normals(tri: np.ndarray, nrm: np.ndarray, max_deg: float = 60.0) -> np.ndarray:
    """Per corner, the face's own normal where the given one is more than *max_deg* off it."""
    fn = face_normals(tri)
    n = np.asarray(nrm, float)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    off = (n * fn[:, None, :]).sum(-1) < math.cos(math.radians(max_deg))
    off &= (np.linalg.norm(fn, axis=1) > 0)[:, None]
    return np.where(off[..., None], fn[:, None, :], n)


def separate_sheets(tri: np.ndarray, gap: float = 1e-3) -> np.ndarray:
    """Push each of two coincident, opposite triangles half of *gap* along its own normal."""
    fn = face_normals(tri)
    a = _areas(tri)
    key = np.sort(np.round(tri, 4), axis=1).reshape(len(tri), -1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    shared = (cnt[inv.ravel()] > 1) & (a > 1e-12)
    if not shared.any():
        return tri
    out = tri.copy()
    out[shared] += 0.5 * gap * fn[shared][:, None, :]
    return out


def geom_triangles(m: mujoco.MjModel, g: int, plane_half=None):
    """``(tri, nrm, uv)`` of geom *g* in its own frame; *uv* in MuJoCo's convention or ``None``."""
    gt = int(m.geom_type[g])
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        return _mesh(m, g)
    if gt == mujoco.mjtGeom.mjGEOM_PLANE:
        sx, sy = (float(v) for v in m.geom_size[g][:2])
        if sx <= 0 or sy <= 0:
            sx, sy = plane_half
        loc = np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]], float)
        tri = loc[[[0, 1, 2], [0, 2, 3]]]
        nrm = np.tile([0.0, 0.0, 1.0], (2, 3, 1))
        mat = int(m.geom_matid[g])
        rep = np.asarray(m.mat_texrepeat[mat], float) if mat >= 0 else np.ones(2)
        if mat >= 0 and bool(m.mat_texuniform[mat]):
            uv = tri[..., :2] * rep  # the repeat is per metre
        else:
            uv = (tri[..., :2] / (2 * np.array([sx, sy])) + 0.5) * rep
        return tri, nrm, uv
    tri, nrm = _primitive(gt, m.geom_size[g])
    uv = _box_uv(tri, nrm) if gt == mujoco.mjtGeom.mjGEOM_BOX else None
    return tri, nrm, uv


# -- the split ------------------------------------------------------------------------------------


def entity_of(geom_name: str, body_name: str) -> str:
    """The entity a static geom's light map is shared by: the name before the first ``/`` (a spawned
    prop's prefix), of the geom else of its body, else the shell."""
    for name in (geom_name, body_name):
        if name and "/" in name:
            return name.split("/", 1)[0].replace(" ", "_")
    return "shell"


def resolved_rgba(m: mujoco.MjModel, g: int) -> np.ndarray:
    mat = int(m.geom_matid[g])
    return np.array(m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g], float)


def texture_of(m: mujoco.MjModel, g: int) -> int:
    """The geom's 2D colour texture, or -1."""
    mat = int(m.geom_matid[g])
    if mat < 0:
        return -1
    tex = int(m.mat_texid[mat][mujoco.mjtTextureRole.mjTEXROLE_RGB])
    if tex < 0:
        return -1
    return tex if int(m.tex_type[tex]) == int(mujoco.mjtTexture.mjTEXTURE_2D) else -1


def texture_array(m: mujoco.MjModel, tex: int) -> np.ndarray:
    """A texture's pixels as ``(h, w, 3)`` uint8, MuJoCo's rows (row 0 is v = 0)."""
    w, h, c = int(m.tex_width[tex]), int(m.tex_height[tex]), int(m.tex_nchannel[tex])
    a = int(m.tex_adr[tex])
    data = np.asarray(m.tex_data[a : a + w * h * c]).reshape(h, w, c)
    if c >= 3:
        return np.ascontiguousarray(data[..., :3])
    return np.repeat(data[..., :1], 3, axis=-1)


def texture_hash(m: mujoco.MjModel, tex: int) -> str:
    w, h, c = int(m.tex_width[tex]), int(m.tex_height[tex]), int(m.tex_nchannel[tex])
    a = int(m.tex_adr[tex])
    return hashlib.sha256(bytes(m.tex_data[a : a + w * h * c])).hexdigest()[:16]


def _drawn(m: mujoco.MjModel, g: int) -> bool:
    return int(m.geom_group[g]) < DRAWN_GROUPS and resolved_rgba(m, g)[3] > 0


def is_static(m: mujoco.MjModel, g: int) -> bool:
    b = int(m.geom_bodyid[g])
    return int(m.body_weldid[b]) == 0 and int(m.body_mocapid[b]) < 0


def free_floating(m: mujoco.MjModel, g: int) -> bool:
    """Whether geom *g* is part of a body tree that hangs on a free joint (a robot, a loose crate)."""
    root = int(m.body_rootid[int(m.geom_bodyid[g])])
    adr, num = int(m.body_jntadr[root]), int(m.body_jntnum[root])
    return any(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(adr, adr + num))


def is_occluder(m: mujoco.MjModel, g: int, occluders: tuple[str, ...] | None) -> bool:
    """Whether moving geom *g* throws its shadow into the bake.

    By default what moves on a joint to the world (a door's leaf, a drawer, an arm at its reset pose)
    does; what floats free (a robot, a loose crate) and mocap bodies do not: they are elsewhere in every
    run, and a robot added to a world must not make its bake stale. *occluders* (fnmatch patterns on the
    geom's name or its body's) replaces the rule: ``()`` for none.
    """
    if occluders is not None:
        names = (m.geom(g).name, m.body(int(m.geom_bodyid[g])).name)
        return any(fnmatch.fnmatch(n, p) for n in names if n for p in occluders)
    b = int(m.geom_bodyid[g])
    root = int(m.body_rootid[b])
    return not free_floating(m, g) and int(m.body_mocapid[root]) < 0 and int(m.body_mocapid[b]) < 0


def export_world(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    settings: BakeSettings,
    *,
    emitter_geoms: dict[int, int] | None = None,
    exclude: tuple[str, ...] = (),
    occluders: tuple[str, ...] | None = None,
) -> Export:
    """Split the compiled world (*d* at its reset pose, after ``mj_forward``) for the bake.

    *emitter_geoms* maps a geom id to the index of the light source it is (:mod:`.lights`); an
    *exclude* pattern (fnmatch, on the geom's name) leaves a static geom unbaked, lit live;
    *occluders* chooses what of the moving geoms throws its shadow into the bake (:func:`is_occluder`).
    """
    emitter_geoms = emitter_geoms or {}
    per_body: dict[int, int] = {}
    geoms: list[GeomRecord] = []
    materials: list[dict] = []
    matkey: dict = {}
    textures: dict = {}
    parts: dict[str, dict] = {}
    occl, emitters = [], {}
    static_lo, static_hi = np.full(3, np.inf), np.full(3, -np.inf)
    planes = []

    def material_index(g: int) -> int:
        rgba = resolved_rgba(m, g)
        tex = texture_of(m, g)
        mat = int(m.geom_matid[g])
        emission = float(m.mat_emission[mat]) if mat >= 0 else 0.0
        key = (tuple(np.round(rgba, 4)), tex, round(emission, 4))
        if key not in matkey:
            if tex >= 0 and tex not in textures:
                textures[tex] = texture_array(m, tex)
            matkey[key] = len(materials)
            materials.append(
                {
                    "rgba": [float(x) for x in rgba],
                    "texture": tex,
                    "texture_hash": texture_hash(m, tex) if tex >= 0 else "",
                    "emission": emission,
                }
            )
        return matkey[key]

    todo = []
    index_of: dict[int, int] = {}
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        k_in_body = per_body.get(b, 0)
        per_body[b] = k_in_body + 1
        if not _drawn(m, g):
            continue
        name = m.geom(g).name
        body = m.body(b).name
        key = name or f"{body}#{k_in_body}"
        static = is_static(m, g)
        if static and g in emitter_geoms:
            kind = "emitter"
        elif static and resolved_rgba(m, g)[3] < 1.0:
            # Glass: light passes it and the classic renderer blends it, so it is neither baked nor an
            # occluder; it is drawn as it is, lit live.
            kind = "see_through"
        elif static and not any(fnmatch.fnmatch(name, p) for p in exclude):
            kind = "baked"
        else:
            kind = "moving"
        rec = GeomRecord(
            id=g,
            name=name,
            body=body,
            index_in_body=k_in_body,
            key=key,
            kind=kind,
            collides=bool(m.geom_contype[g] or m.geom_conaffinity[g]),
            geom_group=int(m.geom_group[g]),
            material=material_index(g),
            emitter=emitter_geoms.get(g) if kind == "emitter" else None,
        )
        index_of[g] = len(geoms)
        geoms.append(rec)
        if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            planes.append((rec, g))
            continue
        todo.append((rec, g))
        if static:
            aabb = _world_aabb(m, d, g)
            static_lo, static_hi = np.minimum(static_lo, aabb[0]), np.maximum(static_hi, aabb[1])
    if not np.isfinite(static_lo).all():
        static_lo, static_hi = np.zeros(3), np.zeros(3)
    # A plane of no size (infinite) is drawn as far as the static geometry reaches, and a metre more.
    half = tuple(np.maximum(np.abs(np.stack([static_lo[:2], static_hi[:2]])).max(0) + 1.0, 1.0))
    for rec, g in planes:
        todo.append((rec, g))
        p = d.geom_xpos[g]
        static_lo = np.minimum(static_lo, [p[0] - half[0], p[1] - half[1], p[2]])
        static_hi = np.maximum(static_hi, [p[0] + half[0], p[1] + half[1], p[2]])

    turned = 0
    for rec, g in todo:
        tri, nrm, uv = geom_triangles(m, g, plane_half=half)
        tri, nrm, flipped = outward(tri, nrm)
        if flipped and uv is not None:
            uv = uv[:, ::-1]
        turned += int(flipped)
        nrm = sane_normals(tri, nrm)
        tri = separate_sheets(tri)
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        wtri, wnrm = tri @ R.T + p, nrm @ R.T
        if rec.kind == "see_through":
            continue
        if rec.kind == "moving":
            if is_occluder(m, g, occluders):
                occl.append(wtri)
            continue
        if rec.kind == "emitter":
            emitters[g] = (wtri, wnrm)
            continue
        if materials[rec.material]["texture"] >= 0 and uv is None:
            # A textured primitive other than a box or a plane: drawn in its material's colour.
            materials[rec.material] = {**materials[rec.material]}
            rec.material = _untextured(materials, matkey, rec.material)
        P = parts.setdefault(
            entity_of(rec.name, rec.body), {"pos": [], "nrm": [], "uv1": [], "mat": [], "geom": []}
        )
        P["pos"].append(wtri)
        P["nrm"].append(wnrm)
        P["uv1"].append(np.zeros(tri.shape[:2] + (2,)) if uv is None else np.asarray(uv, float))
        P["mat"].append(np.full(len(tri), rec.material))
        P["geom"].append(np.full(len(tri), index_of[g]))

    entities: dict[str, Entity] = {}
    for name, P in parts.items():
        ent = Entity(
            name=name,
            cls="shell" if name == "shell" else "props",
            pos=np.concatenate(P["pos"]),
            nrm=np.concatenate(P["nrm"]),
            uv1=np.concatenate(P["uv1"]),
            mat=np.concatenate(P["mat"]).astype(np.int32),
            geom=np.concatenate(P["geom"]).astype(np.int32),
        )
        entities[name] = ent
    if "shell" in entities:
        entities.pop("shell")
        for ent in _split_shell(
            Entity("shell", "shell", *[np.concatenate(parts["shell"][k]) for k in ("pos", "nrm", "uv1")],
                   np.concatenate(parts["shell"]["mat"]).astype(np.int32),
                   np.concatenate(parts["shell"]["geom"]).astype(np.int32)),
            settings,
        ):
            entities[ent.name] = ent
    for ent in entities.values():
        ent.size = map_size(ent.area, ent.cls, settings)
        for gi in np.unique(ent.geom):
            geoms[int(gi)].entities.append(ent.name)
    notes = [f"turned {turned} geom(s) wound inside out to face out"] if turned else []
    return Export(
        geoms=geoms,
        materials=materials,
        entities=dict(sorted(entities.items())),
        occluders=np.concatenate(occl) if occl else np.zeros((0, 3, 3)),
        emitters=emitters,
        textures=textures,
        bounds=(tuple(float(v) for v in static_lo), tuple(float(v) for v in static_hi)),
        notes=notes,
    )


def _untextured(materials, matkey, index) -> int:
    mat = materials[index]
    key = (tuple(np.round(mat["rgba"], 4)), -1, round(mat["emission"], 4))
    if key not in matkey:
        matkey[key] = len(materials)
        materials.append({**mat, "texture": -1, "texture_hash": ""})
    return matkey[key]


def _world_aabb(m, d, g):
    """The world-space bounding box of geom *g* (its bounding sphere where MuJoCo has no box)."""
    c = d.geom_xpos[g]
    r = float(m.geom_rbound[g])
    if r <= 0:
        return c, c
    if hasattr(m, "geom_aabb"):
        aabb = np.asarray(m.geom_aabb[g], float)  # centre (3), half sizes (3), in the geom's frame
        R = d.geom_xmat[g].reshape(3, 3)
        centre = c + R @ aabb[:3]
        half = np.abs(R) @ aabb[3:]
        return centre - half, centre + half
    return c - r, c + r


def map_size(area: float, cls: str, settings: BakeSettings) -> int:
    """A light map's side in texels for *area* m² of class *cls*."""
    side = math.sqrt(max(area, 1e-6) * PACKING_LOSS) * settings.texels_per_m[cls]
    cap = settings.prop_cap if cls == "props" else settings.map_cap
    return int(min(cap, max(ROUND_TO, math.ceil(side / ROUND_TO) * ROUND_TO)))


def floor_height(shell: Entity) -> float | None:
    """The height of the floor: of the shell's triangles that face up, the 2 cm band holding the most
    area (the floor is the largest surface facing up; what lies under it or on top of a wall is not)."""
    fn = face_normals(shell.pos)
    up = fn[:, 2] > 0.9
    if not up.any():
        return None
    z = shell.pos[up][..., 2].mean(1)
    a = _areas(shell.pos[up])
    bins = np.round(z / 0.02).astype(np.int64)
    keys, inv = np.unique(bins, return_inverse=True)
    return float(keys[np.argmax(np.bincount(inv.ravel(), weights=a))] * 0.02)


def _split_shell(shell: Entity, settings: BakeSettings) -> list[Entity]:
    """The shell as the entity ``floor`` (its triangles facing up at the floor's height,
    :func:`floor_height`) and the rest, the rest split by geom into light maps that fit ``map_cap``."""
    fn = face_normals(shell.pos)
    zf = floor_height(shell)
    mid = shell.pos[..., 2].mean(1)
    floor = (fn[:, 2] > 0.9) & (np.abs(mid - zf) <= 0.03) if zf is not None else np.zeros(len(fn), bool)
    out = []

    def take(name, cls, sel):
        return Entity(name, cls, shell.pos[sel], shell.nrm[sel], shell.uv1[sel], shell.mat[sel], shell.geom[sel])

    if floor.any():
        out.append(take("floor", "floor", floor))
    rest = ~floor
    if rest.any():
        # By geom, largest first, into as few maps as stay under the cap.
        areas = _areas(shell.pos)
        cap_area = (settings.map_cap / settings.texels_per_m["shell"]) ** 2 / PACKING_LOSS
        by_geom = sorted(
            ((float(areas[rest & (shell.geom == gi)].sum()), int(gi)) for gi in np.unique(shell.geom[rest])),
            reverse=True,
        )
        bins: list[list] = []
        for a, gi in by_geom:
            for b in bins:
                if b[0] + a <= cap_area:
                    b[0] += a
                    b[1].append(gi)
                    break
            else:
                bins.append([a, [gi]])
        for i, (_, members) in enumerate(bins):
            sel = rest & np.isin(shell.geom, members)
            out.append(take("shell" if i == 0 else f"shell_{i}", "shell", sel))
    return out
