"""Export a compiled MuJoCo world as one binary glTF 2.0 file (.glb).

glTF is the standard format for 3D scenes: three.js, Blender and every engine load it, so a world
exported this way is looked at, placed into or published without learning a format of roqsim's own.
What the scene contains is decided by :mod:`roqsim.scene_content`, the walk ``roqsim export web``
writes too, so the two exports agree on it; this module only spells it as glTF.

Usage::

    roqsim export gltf --world path/to/world.yaml --out scene.glb
    roqsim export gltf --mjcf  path/to/model.xml  --out model.glb

The file
========

**The frame.** glTF is Y-up and the world is Z-up. The root node, ``world``, carries the turn
between them (-90 degrees about x), so **a point in the frame of node ``world`` is a point in roqsim
world coordinates**, and the world point ``(x, y, z)`` is the glTF scene point ``(x, z, -y)``. A
viewer that reports points in a node's frame reports them, for ``world``, in the world's.

**Bodies.** Every body is a node, the child of its parent body's node, in MuJoCo's body tree, with
the body's pose relative to its parent at the exported state: a free body where its world placed
it, a mocap body where its plugin posed it, a jointed link at its configured joint value (glTF has
no joints, so the angle is in the node), and a link on an unnamed joint at that joint's rest, as
``export web`` shows it (:func:`roqsim.scene_content.exported_state`). A node is named exactly as
its body is, and its ``extras`` repeat the name as ``body`` beside ``body_id``: a loader that
rewrites names it finds awkward (three.js removes ``.``, ``:``, ``/`` and brackets) still leaves the
exact name in the node's user data.

**Geoms.** A drawn geom is an unnamed child node of its body's node, at the geom's pose, holding
one mesh; its ``extras`` carry ``geom`` (its name, where it has one), ``body`` and ``geom_id``.
Primitives are tessellated (``--segments``) with their exact normals, MuJoCo meshes keep MuJoCo's
own normals, and a plane is a quad of its size (an infinite plane one of the model's extent, which
the export logs).

**View.** A world with an authored ``sim.view`` gets a perspective camera, on a root node of its own
outside ``world`` with ``extras: {"roqsim": "view"}``, placed where ``roqsim render`` puts its
default camera.

**Skins.** A MuJoCo skin, and a flex drawn as one (as ``export web`` draws it), is a glTF skinned
mesh whose joints are its bone bodies' nodes, so a viewer that moves bodies by name deforms it. Its
node is a root of the scene (glTF ignores a skinned mesh's node transform), with ``extras`` naming
the skin's index and, for a flex, the flex.

Left out: lights and the skybox (a viewer brings its own), joint metadata (``export web`` and
``export urdf`` carry the kinematics), and collision-only geometry (group 3), as for ``export web``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
import sys
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

from . import exit_status, logging_setup, tessellate
from .rendering import view_forward
from .scene_content import Geom, Mesh, SceneContent, Skin, walk
from .scene_source import (
    add_manifest_option,
    add_source_options,
    add_world_options,
    compile_source,
    refuse_options_for_mjcf,
    write_manifest,
)

logger = logging.getLogger(__name__)

#: The world node's rotation, glTF ``[x, y, z, w]``: -90 degrees about x, Z-up onto Y-up.
WORLD_ROTATION = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]

#: Angle between two faces above which a shared vertex is split, so an edge reads as an edge.
CREASE_DEG = 45.0

_FLOAT = 5126
_USHORT = 5123
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942


class GltfExportError(RuntimeError):
    """Something in the model the export refuses rather than draw wrong; the message names it."""


def _xyzw(wxyz) -> list[float]:
    """MuJoCo's ``(w, x, y, z)`` quaternion in glTF's ``(x, y, z, w)`` order."""
    w, x, y, z = (float(v) for v in wxyz)
    return [x, y, z, w]


def local_pose(parent: tuple, child: tuple) -> tuple[list[float], list[float]]:
    """``child``'s world (pos, wxyz) expressed in ``parent``'s frame."""
    ppos, pquat = (np.asarray(v, float) for v in parent)
    cpos, cquat = (np.asarray(v, float) for v in child)
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, pquat)
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, cpos - ppos, inv)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, inv, cquat)
    quat /= np.linalg.norm(quat)
    return pos.tolist(), quat.tolist()


def zup_to_yup(points: np.ndarray) -> np.ndarray:
    """World points ``(x, y, z)`` as glTF scene points ``(x, z, -y)``."""
    p = np.asarray(points, dtype=float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


# -- geometry --------------------------------------------------------------------------------------


def _weld_exact(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices at the same position and drop the triangles that collapse.

    The tessellators emit a pole as a ring of coincident vertices; welded, the pole is one vertex
    and its normal the average of the faces around it rather than one tilted face's.
    """
    key = np.round(verts, 9)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return unique, faces[keep]


def crease_normals(
    verts: np.ndarray, faces: np.ndarray, crease_deg: float = CREASE_DEG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``verts`` along creases and give every vertex a normal: ``(verts, normals, faces)``.

    A corner's normal is the area-weighted mean of the normals of the faces around its vertex that
    lie within ``crease_deg`` of its own face, so a sphere is smooth and a box or a cylinder's rim
    keeps its edge -- how MuJoCo shades its primitives. Corners with the same vertex and normal share
    one output vertex.
    """
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 * area
    area = np.linalg.norm(cross, axis=1)
    unit = cross / np.where(area > 0, area, 1.0)[:, None]
    cos_crease = math.cos(math.radians(crease_deg))
    corner_normals = np.zeros((len(faces), 3, 3))
    by_vertex: dict[int, list[int]] = {}
    for f, face in enumerate(faces):
        for v in face:
            by_vertex.setdefault(int(v), []).append(f)
    for v, around in by_vertex.items():
        around = np.asarray(around)
        n = unit[around]
        w = cross[around]
        close = (n @ n.T) >= cos_crease  # row i: the faces that smooth with face i at v
        summed = close.astype(float) @ w
        summed /= np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-30)
        for k, f in enumerate(around):
            corner = int(np.nonzero(faces[f] == v)[0][0])
            corner_normals[f, corner] = summed[k]
    flat_v = faces.reshape(-1)
    flat_n = np.round(corner_normals.reshape(-1, 3), 6)
    key = np.concatenate([flat_v[:, None].astype(float), flat_n], axis=1)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    out_v = verts[unique[:, 0].astype(int)]
    out_n = unique[:, 1:]
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-30)
    return out_v, out_n, inverse.reshape(-1, 3)


def _plane(size, extent: float) -> tuple[np.ndarray, np.ndarray]:
    sx = float(size[0]) if size[0] > 0 else extent
    sy = float(size[1]) if size[1] > 0 else extent
    verts = np.array([[-sx, -sy, 0.0], [sx, -sy, 0.0], [sx, sy, 0.0], [-sx, sy, 0.0]])
    return verts, np.array([[0, 1, 2], [0, 2, 3]])


def primitive_triangles(
    gtype: str, size, segments: int, extent: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A primitive geom of ``size`` as ``(verts, normals, faces)`` in the geom's frame."""
    if gtype == "plane":
        verts, faces = _plane(size, extent)
        return verts, np.tile([0.0, 0.0, 1.0], (4, 1)), faces
    if gtype == "box":
        verts, faces = tessellate.box(size[:3])
    elif gtype == "sphere":
        verts, faces = tessellate.sphere(float(size[0]), segments)
    elif gtype == "capsule":
        verts, faces = tessellate.capsule(float(size[0]), float(size[1]), segments)
    elif gtype == "cylinder":
        verts, faces = tessellate.cylinder(float(size[0]), float(size[1]), segments)
    elif gtype == "ellipsoid":
        verts, faces = tessellate.ellipsoid(size[:3], segments)
    else:  # pragma: no cover - scene_content draws no other type
        raise GltfExportError(f"no triangle spelling for a {gtype} geom")
    verts, faces = _weld_exact(np.asarray(verts, float), np.asarray(faces, int))
    verts, normals, faces = crease_normals(verts, faces)
    # The creases decide where a vertex splits; a round surface then takes its exact normal, as
    # MuJoCo shades it, rather than the mean of a coarse tessellation's faces.
    if gtype == "sphere":
        normals = verts.copy()
    elif gtype == "ellipsoid":
        normals = verts / np.asarray(size[:3], float) ** 2
    elif gtype == "capsule":
        half = float(size[1])
        normals = verts - np.c_[np.zeros((len(verts), 2)), np.clip(verts[:, 2], -half, half)]
    elif gtype == "cylinder":
        side = np.abs(normals[:, 2]) < 0.5
        normals[side] = np.c_[verts[side, :2], np.zeros(int(side.sum()))]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return verts, normals, faces


def mesh_triangles(
    model: mujoco.MjModel, mesh: Mesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """A MuJoCo mesh with MuJoCo's own normals: ``(verts, normals, faces, uv)``.

    MuJoCo indexes positions, normals and texture coordinates separately (OBJ-style); one GPU index
    takes the unique (position, normal, texcoord) triples the faces use. The normals are MuJoCo's,
    so its smoothing and creases are what a viewer shades.
    """
    from .scene_content import reindex

    mid = mesh.source
    va = int(model.mesh_vertadr[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    na = int(model.mesh_normaladr[mid])
    faces = model.mesh_face[fa : fa + fn]
    fnorm = model.mesh_facenormal[fa : fa + fn]
    tca = int(model.mesh_texcoordadr[mid])
    if tca >= 0:
        ftex = model.mesh_facetexcoord[fa : fa + fn]
        unique, index = reindex(faces, fnorm, ftex)
        uv = model.mesh_texcoord[tca + unique[:, 2]]
    else:
        unique, index = reindex(faces, fnorm)
        uv = None
    verts = model.mesh_vert[va + unique[:, 0]]
    normals = model.mesh_normal[na + unique[:, 1]]
    return verts, normals, index, uv


# -- the glTF document -----------------------------------------------------------------------------


class _Document:
    """The glTF JSON and its one binary buffer, built up as the scene is walked."""

    def __init__(self) -> None:
        self.gltf: dict = {
            "asset": {"version": "2.0", "generator": f"roqsim {_producer_version()}"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
        }
        self.bin = bytearray()

    def _view(self, data: bytes, target: int | None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.bin.extend(data)
        self.gltf["bufferViews"].append(view)
        return len(self.gltf["bufferViews"]) - 1

    def accessor(self, array: np.ndarray, kind: str, *, minmax: bool = False) -> int:
        """Add ``array`` as a float vertex attribute of ``kind`` (VEC2/VEC3/VEC4)."""
        data = np.ascontiguousarray(array, dtype="<f4")
        acc = {
            "bufferView": self._view(data.tobytes(), _ARRAY_BUFFER),
            "componentType": _FLOAT,
            "count": int(data.shape[0]),
            "type": kind,
        }
        if minmax:
            acc["min"] = data.min(axis=0).tolist()
            acc["max"] = data.max(axis=0).tolist()
        self.gltf["accessors"].append(acc)
        return len(self.gltf["accessors"]) - 1

    def indices(self, faces: np.ndarray, nvert: int) -> int:
        dtype, ctype = ("<u2", _USHORT) if nvert <= 0xFFFF else ("<u4", _UINT)
        data = np.ascontiguousarray(faces, dtype=dtype).ravel()
        self.gltf["accessors"].append(
            {
                "bufferView": self._view(data.tobytes(), _ELEMENT_ARRAY_BUFFER),
                "componentType": ctype,
                "count": int(data.size),
                "type": "SCALAR",
            }
        )
        return len(self.gltf["accessors"]) - 1

    def raw_accessor(self, data: np.ndarray, ctype: int, kind: str, target: int | None) -> int:
        """Add ``data`` (already in its final little-endian dtype) as an accessor of ``kind``."""
        data = np.ascontiguousarray(data)
        self.gltf["accessors"].append(
            {
                "bufferView": self._view(data.tobytes(), target),
                "componentType": ctype,
                "count": int(data.shape[0]),
                "type": kind,
            }
        )
        return len(self.gltf["accessors"]) - 1

    def node(self, node: dict) -> int:
        self.gltf["nodes"].append(node)
        return len(self.gltf["nodes"]) - 1

    def add(self, key: str, item: dict) -> int:
        self.gltf.setdefault(key, []).append(item)
        return len(self.gltf[key]) - 1

    def glb(self) -> bytes:
        while len(self.bin) % 4:
            self.bin.append(0)
        self.gltf["buffers"] = [{"byteLength": len(self.bin)}]
        gltf = {k: v for k, v in self.gltf.items() if v != []}
        text = json.dumps(gltf, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        total = 12 + 8 + len(text) + 8 + len(self.bin)
        return b"".join(
            [
                struct.pack("<III", _GLB_MAGIC, 2, total),
                struct.pack("<II", len(text), _CHUNK_JSON),
                text,
                struct.pack("<II", len(self.bin), _CHUNK_BIN),
                bytes(self.bin),
            ]
        )


def _producer_version() -> str:
    try:
        return metadata.version("roqsim")
    except metadata.PackageNotFoundError:  # pragma: no cover - a source checkout
        return "unknown"


# -- materials -------------------------------------------------------------------------------------


class _Materials:
    """glTF materials, one per distinct look, created on first use."""

    def __init__(self, doc: _Document, model: mujoco.MjModel) -> None:
        self.doc = doc
        self.model = model
        self._index: dict[tuple, int] = {}
        self._clamped: set[str] = set()

    def _label(self, matid: int) -> str:
        return (
            self.model.material(matid).name or f"material {matid}" if matid >= 0 else "geom colour"
        )

    def get(self, matid: int, rgba) -> int:
        """The material for a geom of material ``matid`` (-1 for none) and colour ``rgba``.

        A material's colour overrides the geom's own, as MuJoCo resolves it.
        """
        model = self.model
        if matid >= 0:
            rgba = model.mat_rgba[matid]
            emission = float(model.mat_emission[matid])
            shininess = float(model.mat_shininess[matid])
        else:
            emission, shininess = 0.0, 0.5
        rgba = [float(c) for c in rgba]
        key = (matid, tuple(round(c, 6) for c in rgba))
        if key in self._index:
            return self._index[key]
        if any(c > 1.0 for c in rgba[:3]):
            label = self._label(matid)
            if label not in self._clamped:
                self._clamped.add(label)
                logger.warning(
                    "%s has a colour above 1 (%s); glTF's base colour stops at 1, so it is clamped",
                    label,
                    ", ".join(f"{c:g}" for c in rgba[:3]),
                )
        base = [min(max(c, 0.0), 1.0) for c in rgba]
        mat: dict = {
            "pbrMetallicRoughness": {
                "baseColorFactor": base,
                "metallicFactor": 0.0,
                "roughnessFactor": min(max(1.0 - shininess, 0.0), 1.0),
            }
        }
        if matid >= 0 and model.material(matid).name:
            mat["name"] = model.material(matid).name
        if emission > 0:
            mat["emissiveFactor"] = [min(emission * c, 1.0) for c in base[:3]]
        if base[3] < 1.0:
            mat["alphaMode"] = "BLEND"
        self._index[key] = self.doc.add("materials", mat)
        return self._index[key]


# -- the export ------------------------------------------------------------------------------------


def _camera_node(view: dict, model: mujoco.MjModel) -> dict | None:
    """The authored free camera as a glTF camera node (Y-up scene coordinates), or None."""
    if not view or not all(k in view for k in ("lookat", "distance", "azimuth", "elevation")):
        return None
    forward = view_forward(float(view["azimuth"]), float(view["elevation"]))
    eye = np.asarray(view["lookat"], float) - float(view["distance"]) * forward
    f = zup_to_yup(forward)
    right = np.cross(f, [0.0, 1.0, 0.0])
    if np.linalg.norm(right) < 1e-9:  # looking straight up or down: MuJoCo's x stays right
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    rot = np.column_stack([right, up, -f])  # camera x, y, z (it looks down its -z)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return {
        "translation": zup_to_yup(eye).tolist(),
        "rotation": _xyzw(quat),
        "extras": {"roqsim": "view"},
        "camera_def": {
            "type": "perspective",
            "perspective": {
                "yfov": math.radians(float(model.vis.global_.fovy)),
                "znear": float(model.vis.map.znear * model.stat.extent),
                "zfar": float(model.vis.map.zfar * model.stat.extent),
            },
        },
    }


def export_gltf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    out: Path,
    *,
    segments: int = 32,
    view: dict | None = None,
    log: logging.Logger | None = None,
) -> dict:
    """Write the scene of ``model`` at the state in ``data`` to ``out`` as GLB; return the glTF JSON."""
    log = log or logger
    content = walk(model, data, log)
    doc = _Document()
    materials = _Materials(doc, model)
    segments = max(3, int(segments))
    extent = float(model.stat.extent)

    # Bodies: body b is node body_node[b], placed where the exported state puts it relative to its
    # parent -- a hinged link at its configured angle, since glTF has no joints to apply one.
    body_node: list[int] = []
    for b, body in enumerate(content.bodies):
        if b == 0:
            node = {"name": body.name or "world", "rotation": WORLD_ROTATION}
        else:
            pos, quat = local_pose(content.world_poses[body.parent], content.world_poses[b])
            node = {"translation": pos, "rotation": _xyzw(quat)}
            if body.name:
                node["name"] = body.name
        node["extras"] = {"body": body.name, "body_id": b}
        body_node.append(doc.node(node))
        if b > 0:
            doc.gltf["nodes"][body_node[body.parent]].setdefault("children", []).append(
                body_node[b]
            )
    doc.gltf["scenes"][0]["nodes"].append(body_node[0])

    meshes: dict[tuple, int] = {}
    infinite = []
    for geom in content.geoms:
        mesh_index = _geom_mesh(doc, model, content, geom, materials, meshes, segments, extent)
        if mesh_index is None:
            continue
        if geom.type == "plane" and (geom.size[0] <= 0 or geom.size[1] <= 0):
            infinite.append(geom.name or f"geom {geom.id}")
        extras = {"body": content.bodies[geom.body].name, "geom_id": geom.id}
        if geom.name:
            extras = {"geom": geom.name, **extras}
        child = doc.node(
            {
                "translation": [float(v) for v in geom.pos],
                "rotation": _xyzw(geom.quat),
                "mesh": mesh_index,
                "extras": extras,
            }
        )
        doc.gltf["nodes"][body_node[geom.body]].setdefault("children", []).append(child)
    if infinite:
        log.info(
            "infinite plane(s) %s drawn as a square of half-side %.3g m (the model's extent)",
            ", ".join(infinite),
            extent,
        )

    for i, skin in enumerate(content.skins):
        _skin(doc, skin, i, body_node, materials)

    cam = _camera_node(view or {}, model)
    if cam is not None:
        cam["camera"] = doc.add("cameras", cam.pop("camera_def"))
        doc.gltf["scenes"][0]["nodes"].append(doc.node(cam))

    blob = doc.glb()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    log.info(
        "exported %d bodies, %d geoms, %d skins (%d of them flexes), %d meshes, %d materials "
        "-> %s (%d KiB)",
        len(content.bodies),
        len(content.geoms),
        len(content.skins),
        content.flex_count,
        len(doc.gltf["meshes"]),
        len(doc.gltf["materials"]),
        out,
        len(blob) // 1024,
    )
    return doc.gltf


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals: the area-weighted mean of the faces around each vertex."""
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(verts, dtype=float)
    for k in range(3):
        np.add.at(normals, faces[:, k], cross)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    # A vertex no face uses (a solid flex's interior) gets any unit normal; nothing draws it.
    normals[length[:, 0] == 0] = [0.0, 0.0, 1.0]
    return normals / np.where(length > 0, length, 1.0)


def _skin(doc: _Document, skin: Skin, i: int, body_node: list[int], materials: _Materials) -> None:
    """One skin (or flex) as a glTF skinned mesh whose joints are its bone bodies' nodes.

    The vertices are written turned into the scene's Y-up frame, and each bone's inverse bind
    matrix is ``inv(W @ B)``, ``W`` the ``world`` node's turn and ``B`` the bone's world bind pose,
    so a vertex lands at ``J @ inv(W @ B) @ (W @ v) = W @ (J' @ inv(B) @ v)``: the bone's motion since
    the bind, in world coordinates, seen in the scene. The node is a root of the scene, because glTF
    ignores the transform of a skinned mesh's node and its parents.
    """
    mesh = skin.mesh
    faces = np.asarray(mesh.index, int).reshape(-1, 3)
    verts = np.asarray(mesh.vert, float).reshape(-1, 3)
    normals = vertex_normals(verts, faces)
    joints, weights, dropped = skin_attributes(skin.skin_index, skin.skin_weight)
    if dropped:
        logger.warning(
            "%s: %d negative skin weights set to 0 (glTF allows none); exact at rest, "
            "approximate under deformation",
            f"flex {skin.flex!r}" if skin.flex is not None else f"skin {i}",
            dropped,
        )
    attributes = {
        "POSITION": doc.accessor(zup_to_yup(verts), "VEC3", minmax=True),
        "NORMAL": doc.accessor(zup_to_yup(normals), "VEC3"),
        "JOINTS_0": doc.raw_accessor(joints, _USHORT, "VEC4", _ARRAY_BUFFER),
        "WEIGHTS_0": doc.accessor(weights, "VEC4"),
    }
    if mesh.uv is not None:
        attributes["TEXCOORD_0"] = doc.accessor(np.asarray(mesh.uv).reshape(-1, 2), "VEC2")
    primitive = {
        "attributes": attributes,
        "indices": doc.indices(faces, len(verts)),
        "material": materials.get(skin.matid, skin.rgba),
    }
    turn = np.eye(4)
    turn[:3, :3] = _quat_to_mat([WORLD_ROTATION[3], *WORLD_ROTATION[:3]])
    inverse_bind = []
    for pos, quat in zip(skin.bindpos, skin.bindquat, strict=True):
        bind = np.eye(4)
        bind[:3, :3] = _quat_to_mat(quat)
        bind[:3, 3] = pos
        inverse_bind.append(np.linalg.inv(turn @ bind).T.ravel())  # glTF matrices are column-major
    skin_index = doc.add(
        "skins",
        {
            "joints": [body_node[b] for b in skin.bone_ids],
            "inverseBindMatrices": doc.raw_accessor(
                np.asarray(inverse_bind, "<f4"), _FLOAT, "MAT4", None
            ),
            "skeleton": body_node[0],
        },
    )
    extras = {"skin": i} if skin.flex is None else {"skin": i, "flex": skin.flex}
    node = doc.node(
        {
            "mesh": doc.add("meshes", {"primitives": [primitive]}),
            "skin": skin_index,
            "extras": extras,
        }
    )
    doc.gltf["scenes"][0]["nodes"].append(node)


def skin_attributes(index, weight) -> tuple[np.ndarray, np.ndarray, int]:
    """``JOINTS_0`` and ``WEIGHTS_0`` as glTF requires them, and how many negative weights it dropped.

    glTF wants each weight in [0, 1], each row summing to 1 in float32, and no joint named by a zero
    weight. A flex with ``dof="quadratic"`` has truly negative weights (its basis functions dip
    below zero) and float noise leaves others a hair below zero; both are set to 0 and the row
    renormalised. That keeps the rest shape exact -- at the bind pose any weights summing to 1 give
    back the vertex -- and makes a deformation approximate, as the four-bone cap already does.
    """
    joints = np.asarray(index, np.int64).reshape(-1, 4).copy()
    w = np.asarray(weight, np.float64).reshape(-1, 4).copy()
    dropped = int((w < -1e-6).sum())
    w[w < 0] = 0.0
    total = w.sum(axis=1, keepdims=True)
    w = w / np.where(total > 0, total, 1.0)
    w32 = w.astype(np.float32)
    # Put the float32 rounding of each row on its largest weight, so the row sums to 1 exactly.
    rows = np.arange(len(w32))
    big = np.argmax(w32, axis=1)
    w32[rows, big] += np.float32(1.0) - w32.sum(axis=1, dtype=np.float32)
    joints[w32 == 0] = 0
    return joints.astype("<u2"), w32, dropped


def _quat_to_mat(wxyz) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(wxyz, dtype=float))
    return m.reshape(3, 3)


def _geom_mesh(
    doc: _Document,
    model: mujoco.MjModel,
    content: SceneContent,
    geom: Geom,
    materials: _Materials,
    cache: dict[tuple, int],
    segments: int,
    extent: float,
) -> int | None:
    """The glTF mesh drawing ``geom``, shared by every geom that looks the same; None if empty."""
    material = materials.get(geom.matid, geom.rgba)
    if geom.type == "mesh":
        key = ("mesh", content.meshes[geom.mesh].source, material)
    else:
        key = (geom.type, tuple(round(float(s), 9) for s in geom.size), material)
    if key in cache:
        return cache[key]
    if geom.type == "mesh":
        verts, normals, faces, uv = mesh_triangles(model, content.meshes[geom.mesh])
    else:
        verts, normals, faces = primitive_triangles(geom.type, geom.size, segments, extent)
        uv = None
    if len(faces) == 0:
        cache[key] = None
        return None
    attributes = {
        "POSITION": doc.accessor(verts, "VEC3", minmax=True),
        "NORMAL": doc.accessor(normals, "VEC3"),
    }
    if uv is not None:
        attributes["TEXCOORD_0"] = doc.accessor(uv, "VEC2")
    primitive = {
        "attributes": attributes,
        "indices": doc.indices(faces, len(verts)),
        "material": material,
    }
    cache[key] = doc.add("meshes", {"primitives": [primitive]})
    return cache[key]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export gltf",
        description="Export a compiled MuJoCo world as one binary glTF 2.0 file (.glb).",
        epilog=(
            "The file is Y-up. Its root node `world` carries the turn from roqsim's Z-up world, so a "
            "point in the frame of node `world` is a point in roqsim world coordinates. Every body is "
            "a node named as the body is.\n\n" + exit_status.epilog(exit_status.BAD_INPUT)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_source_options(parser)
    parser.add_argument("--out", required=True, help="the .glb file to write")
    add_world_options(parser)
    parser.add_argument(
        "--segments",
        type=int,
        default=32,
        help="segments around a sphere, capsule, cylinder or ellipsoid (default 32)",
    )
    add_manifest_option(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    refuse_options_for_mjcf(parser, args)
    if Path(args.out).suffix.lower() != ".glb":
        parser.error(f"--out must name a .glb file, got {args.out!r}")

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_gltf")

    model, data, view = compile_source(args, log)
    try:
        export_gltf(model, data, Path(args.out), segments=args.segments, view=view, log=log)
    except GltfExportError as err:
        return exit_status.fail(parser.prog, err)
    write_manifest(args, log)
    return exit_status.OK


if __name__ == "__main__":
    sys.exit(main())
