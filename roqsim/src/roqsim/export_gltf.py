"""Export a compiled MuJoCo world as one binary glTF 2.0 file (.glb).

glTF is the standard format for 3D scenes: three.js, Blender and every engine load it, so a world
exported this way is looked at, placed into or published without learning a format of roqsim's own.
What the scene contains is decided by :mod:`roqsim.scene_content`, the walk every scene export
shares; this module only spells it as glTF.

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
no joints, so the angle is in the node), and a link on an unnamed joint at that joint's rest
(:func:`roqsim.scene_content.exported_state`). A node is named exactly as
its body is, and its ``extras`` repeat the name as ``body`` beside ``body_id``: a loader that
rewrites names it finds awkward (three.js removes ``.``, ``:``, ``/`` and brackets) still leaves the
exact name in the node's user data.

**Joints.** A body's joints are listed in its node's ``extras.joints``, in the order MuJoCo
composes them: ``name``, ``type`` (``hinge``, ``slide``, ``ball`` or ``free``), ``axis`` and
``pos`` (the anchor) in the body's frame, ``qposadr``, and for a hinge or a slide ``ref``, the value
at which the body sits at its ``body_pos``/``body_quat``. The scene's ``extras.initialJoints`` give
the value of each named hinge and slide the node is posed at (an unnamed one is at ``ref``), so a
viewer recovers the body's pose at ``ref`` from the node and moves it by ``q - ref`` for a joint
value ``q`` -- turning about the axis through the anchor, or sliding along the axis -- exactly as
MuJoCo does.

**The scene's extras** state the contract they are written to, ``format`` (:data:`EXTRAS_FORMAT`)
and ``version`` (:data:`EXTRAS_VERSION`), with ``initialJoints`` and, for a world with an authored
``sim.view``, ``view`` (its ``lookat``, ``distance``, ``azimuth`` and ``elevation``).

**Geoms.** A drawn geom is an unnamed child node of its body's node, at the geom's pose, holding
one mesh; its ``extras`` carry ``geom`` (its name, where it has one), ``body`` and ``geom_id``.
Primitives are tessellated (``--segments``) with their exact normals, MuJoCo meshes keep MuJoCo's
own normals, and a plane is a quad of its size (an infinite plane one of the model's extent, which
the export logs). A surface MuJoCo shades with its faces' normals -- a box, a triangulated STL or
OBJ -- carries none: glTF draws a primitive without normals flat, so its corners are welded into
one vertex per position (:func:`flat_shaded`).

**Colours.** MuJoCo writes a colour to the screen as it stands, so an ``rgba`` is an sRGB-encoded
colour; glTF states a colour factor in linear space, so it is converted (:func:`srgb_to_linear`)
and a viewer with sRGB output shows the colour MuJoCo shows.

**Compression.** With ``--meshopt``, vertex and index data are written with
``EXT_meshopt_compression`` (:mod:`roqsim.meshopt`): positions to :data:`POSITION_BITS` bits below
each mesh's extent, normals and texture coordinates to fixed powers of two. ``--texture-format
webp`` stores every image as WebP (``EXT_texture_webp``). Both are required extensions, so a viewer
without them refuses the file rather than drawing it wrong.

**Sensor cameras.** Every imaging sensor the world's plugins declare
(:meth:`roqsim.plugin.Plugin.scene_cameras`) is a perspective camera on a node of its own, the child
of the node of the body it is mounted on, at its mount pose, so it follows that body wherever a
viewer moves it. MuJoCo's camera frame is glTF's (it looks down -z with +y up), so the pose is the
MuJoCo camera's as it stands. The node and the camera are named after the sensor's address, and the
node's ``extras`` carry ``sensor`` (that address), ``kind`` (``rgb`` or ``depth``) and the published
image's ``width`` and ``height``.

**View.** A world with an authored ``sim.view`` gets a perspective camera, on a root node of its own
outside ``world`` with ``extras: {"roqsim": "view"}``, placed where ``roqsim render`` puts its
default camera.

**Skins.** A MuJoCo skin, and a flex drawn as one (:mod:`roqsim.flex_skin`), is a glTF skinned
mesh whose joints are its bone bodies' nodes, so a viewer that moves bodies by name deforms it. Its
node is a root of the scene (glTF ignores a skinned mesh's node transform), with ``extras`` naming
the skin's index and, for a flex, the flex.

Left out: lights and the skybox (a viewer brings its own), and collision-only geometry (group 3).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import struct
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

from . import exit_status, logging_setup, meshopt, tessellate
from .rendering import view_forward
from .scene_content import (
    Geom,
    Mesh,
    SceneContent,
    SensorCamera,
    Skin,
    reindex,
    resolved_rgba,
    walk,
)
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

#: The longest side an embedded image keeps by default.
DEFAULT_MAX_TEX_DIM = 1024

_FLOAT = 5126
_USHORT = 5123
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_MESHOPT = "EXT_meshopt_compression"
_WEBP = "EXT_texture_webp"

#: What the scene's ``extras`` declare: the contract of the extras this file carries, so a reader
#: refuses a file written to one it has not seen. Bumped when a key changes meaning, never when one
#: is added.
EXTRAS_FORMAT = "roqsim.scene"
EXTRAS_VERSION = 1

#: Compressed, a position keeps this many bits below the largest coordinate of its mesh along each
#: axis: some fifteen micrometres on a metre-sized part, a millimetre on a fifty-metre wall.
POSITION_BITS = 16
#: Compressed, a normal is stored to this power of two, and a texture coordinate to that one: a
#: quarter of a texel of a 1024-pixel image, however often the texture repeats.
NORMAL_STEP = -10
TEXCOORD_STEP = -12


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


def texture_scale(gtype: str, size, texrepeat, texuniform: bool, explicit: bool) -> np.ndarray:
    """What MuJoCo's renderer multiplies a geom's texture coordinates by (``settexture``).

    ``explicit`` is a geom with texture coordinates of its own -- a primitive's, a mesh's or a
    skin's: they scale by ``texrepeat`` (1 where it is 0), and by the geom's size along x and y when
    ``texuniform`` is set, so a texture repeats per metre rather than per object. A mesh without
    any gets them generated from its vertices instead (:func:`generated_uv`).
    """
    rep = np.asarray(texrepeat, float)[:2]
    scl = np.where(rep > 0, rep, 1.0) if explicit else rep.copy()
    vis = _vis_size(gtype, size)
    if texuniform and gtype not in ("skin", "flex"):
        scl = np.where(vis[:2] > 0, scl * vis[:2], scl)
    return scl


def _vis_size(gtype: str, size) -> np.ndarray:
    """The size MuJoCo's renderer gives a geom (``mjv_initGeom``): a sphere ``(r, r, r)``, a capsule
    or a cylinder ``(r, r, h)``, everything else its own."""
    size = np.asarray(size, float)
    if gtype == "sphere":
        return np.array([size[0], size[0], size[0]])
    if gtype in ("capsule", "cylinder"):
        return np.array([size[0], size[0], size[1]])
    return size[:3]


def generated_uv(verts: np.ndarray, size, texrepeat, texuniform: bool) -> np.ndarray:
    """The texture coordinates MuJoCo generates for a mesh that has none (``GL_OBJECT_LINEAR``).

    ``s = scl_x / 2 * x - 1/2`` and ``t = -scl_y / 2 * y - 1/2`` in the mesh's frame, ``scl`` being
    ``texrepeat`` over the geom's size, times the size again under ``texuniform`` -- so a uniform
    texture repeats ``texrepeat / 2`` times per metre. This is how a UV-less wall mesh is textured.
    """
    size = np.asarray(size, float)
    scl = np.asarray(texrepeat, float)[:2].copy()
    for k in range(2):
        if size[k] > 0:
            scl[k] /= max(size[k], mujoco.mjMINVAL)
            if texuniform:
                scl[k] *= size[k]
    verts = np.asarray(verts, float)
    return np.stack([0.5 * scl[0] * verts[:, 0] - 0.5, -0.5 * scl[1] * verts[:, 1] - 0.5], axis=1)


def mesh_triangles(
    model: mujoco.MjModel, mesh: Mesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """A MuJoCo mesh with MuJoCo's own normals: ``(verts, normals, faces, uv)``.

    MuJoCo indexes positions, normals and texture coordinates separately (OBJ-style); one GPU index
    takes the unique (position, normal, texcoord) triples the faces use. The normals are MuJoCo's,
    so its smoothing and creases are what a viewer shades.
    """
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


#: A mesh is drawn flat (:func:`flat_shaded`) when its corner normals follow their faces to within
#: :data:`FLAT_DEGREES` over all but :data:`FLAT_SLACK` of its surface area.
FLAT_DEGREES = 2.0
FLAT_SLACK = 1e-3


def flat_shaded(verts, normals, faces, uv):
    """A mesh MuJoCo shades with its faces' normals, welded and without normals; else None.

    glTF draws a primitive without ``NORMAL`` with flat normals, so such a mesh needs none of its
    own, and the corners MuJoCo kept apart only to give each face its normal are one vertex again
    (equal position and texture coordinate). Returns ``(verts, faces, uv)`` with the vertices welded.
    A triangulated STL or OBJ export, where every face has vertices of its own, is most of a robot's
    meshes. The test is by area, since the sliver triangles of such a file carry normals rounded far
    from their own faces' and are too thin to show how they are shaded.
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, np.int64).reshape(-1, 3)
    tri = verts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(face_n, axis=1)
    solid = length > 1e-12 * max(1.0, float(np.abs(verts).max(initial=0.0))) ** 2
    if not solid.any():
        return None
    area = length[solid]
    face_n = face_n[solid] / area[:, None]
    corner = np.asarray(normals, float)[faces[solid]]
    corner = corner / np.maximum(np.linalg.norm(corner, axis=2, keepdims=True), 1e-300)
    off = np.einsum("ijk,ik->ij", corner, face_n).min(axis=1) < math.cos(math.radians(FLAT_DEGREES))
    if area[off].sum() > FLAT_SLACK * area.sum():
        return None
    rows = verts if uv is None else np.c_[verts, np.asarray(uv, float)]
    _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    welded_uv = None if uv is None else np.asarray(uv)[first]
    return verts[first], inverse.reshape(-1)[faces], welded_uv


# -- the glTF document -----------------------------------------------------------------------------


class _Document:
    """The glTF JSON and its one binary buffer, built up as the scene is walked."""

    def __init__(self, *, compress: bool = False) -> None:
        #: Whether vertex and index data go through ``EXT_meshopt_compression`` (:mod:`roqsim.meshopt`).
        self.compress = compress
        #: Bytes of the decoded views, laid out in the fallback buffer a compressed view names.
        self.fallback = 0
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

    def _compressed_view(
        self, raw: bytes, encoded: bytes, target: int, stride: int, count: int, mode: str, filt
    ) -> int:
        """A view whose bytes are ``encoded`` in the BIN chunk and decode to ``raw``."""
        while len(self.bin) % 4:
            self.bin.append(0)
        ext = {
            "buffer": 0,
            "byteOffset": len(self.bin),
            "byteLength": len(encoded),
            "byteStride": stride,
            "mode": mode,
            "count": count,
        }
        if filt:
            ext["filter"] = filt
        self.bin.extend(encoded)
        self.fallback += -self.fallback % 4
        view = {"buffer": 1, "byteOffset": self.fallback, "byteLength": len(raw), "target": target}
        if target == _ARRAY_BUFFER:
            view["byteStride"] = stride
        view["extensions"] = {_MESHOPT: ext}
        self.fallback += len(raw)
        self.gltf["bufferViews"].append(view)
        return len(self.gltf["bufferViews"]) - 1

    def _vertex_view(self, data: np.ndarray, words: np.ndarray | None = None) -> int:
        """A vertex attribute's view: plain, or compressed (``words`` its filtered form)."""
        raw = np.ascontiguousarray(data)
        if not self.compress:
            return self._view(raw.tobytes(), _ARRAY_BUFFER)
        stored = raw if words is None else np.ascontiguousarray(words)
        rows = stored.reshape(len(stored), -1).view(np.uint8)
        return self._compressed_view(
            raw.tobytes(),
            meshopt.encode_vertex_buffer(rows),
            _ARRAY_BUFFER,
            rows.shape[1],
            len(stored),
            "ATTRIBUTES",
            None if words is None else "EXPONENTIAL",
        )

    def accessor(
        self, array: np.ndarray, kind: str, *, minmax: bool = False, step: int | str | None = None
    ) -> int:
        """Add ``array`` as a float vertex attribute of ``kind`` (VEC2/VEC3/VEC4).

        Compressed, an attribute given a ``step`` is stored at that precision: a power-of-two
        exponent, or ``"position"`` for :data:`POSITION_BITS` below the extent of each component.
        The accessor then holds the values a reader decodes, and so do its bounds.
        """
        data = np.ascontiguousarray(array, dtype="<f4")
        words = None
        if self.compress and step is not None:
            if step == "position":
                peak = np.abs(data).max(axis=0) if len(data) else np.ones(data.shape[1])
                step = np.ceil(np.log2(np.maximum(peak, 2.0**-20))).astype(int) - POSITION_BITS
            words, data = meshopt.exponential_filter(data, step)
            data = np.ascontiguousarray(data, dtype="<f4")
        acc = {
            "bufferView": self._vertex_view(data, words),
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
        if self.compress:
            view = self._compressed_view(
                data.tobytes(),
                meshopt.encode_index_buffer(faces),
                _ELEMENT_ARRAY_BUFFER,
                data.itemsize,
                int(data.size),
                "TRIANGLES",
                None,
            )
        else:
            view = self._view(data.tobytes(), _ELEMENT_ARRAY_BUFFER)
        self.gltf["accessors"].append(
            {
                "bufferView": view,
                "componentType": ctype,
                "count": int(data.size),
                "type": "SCALAR",
            }
        )
        return len(self.gltf["accessors"]) - 1

    def raw_accessor(self, data: np.ndarray, ctype: int, kind: str, target: int | None) -> int:
        """Add ``data`` (already in its final little-endian dtype) as an accessor of ``kind``."""
        data = np.ascontiguousarray(data)
        view = (
            self._vertex_view(data) if target == _ARRAY_BUFFER else self._view(data.tobytes(), None)
        )
        self.gltf["accessors"].append(
            {
                "bufferView": view,
                "componentType": ctype,
                "count": int(data.shape[0]),
                "type": kind,
            }
        )
        return len(self.gltf["accessors"]) - 1

    def require(self, extension: str) -> None:
        """Name ``extension`` as used and required: a reader without it cannot draw the file."""
        for key in ("extensionsUsed", "extensionsRequired"):
            if extension not in self.gltf.setdefault(key, []):
                self.gltf[key].append(extension)

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
        if self.fallback:
            self.gltf["buffers"].append(
                {"byteLength": self.fallback, "extensions": {_MESHOPT: {"fallback": True}}}
            )
            self.require(_MESHOPT)
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


#: glTF sampler constants.
_LINEAR, _LINEAR_MIPMAP_LINEAR, _REPEAT = 9729, 9987, 10497

#: Channel count -> Pillow mode.
_PIL_MODE = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}


def srgb_to_linear(c: float) -> float:
    """A colour channel as MuJoCo draws it (straight onto the screen, so sRGB-encoded) in the linear
    space glTF states a colour factor in."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class _Materials:
    """glTF materials and the images they sample, one per distinct look, created on first use.

    A material's colour is MuJoCo's resolved colour (:func:`roqsim.scene_content.resolved_rgba`);
    its texture is its RGB-role texture, multiplied by that colour as MuJoCo does. glTF's base
    colour stops at 1 while roqsim's tint does not (``docs/textures.rst``), so a textured tint above
    1 is multiplied into a copy of the image instead, and an untextured one is clamped with a
    warning naming the material.
    """

    def __init__(
        self,
        doc: _Document,
        model: mujoco.MjModel,
        content: SceneContent,
        *,
        max_tex_dim: int,
        texture_format: str,
        texture_quality: int,
    ) -> None:
        self.doc = doc
        self.model = model
        self.content = content
        self.max_tex_dim = int(max_tex_dim)
        self.texture_format = texture_format
        self.texture_quality = int(texture_quality)
        self._index: dict[tuple, int] = {}
        self._images: dict[tuple, int] = {}
        self._clamped: set[str] = set()
        self.image_bytes = 0

    def _label(self, matid: int) -> str:
        if matid < 0:
            return "a geom colour"
        return f"material {self.model.material(matid).name or matid!r}"

    def texture(self, matid: int):
        """The 2-D texture material ``matid`` draws with, or None; refuses any other kind."""
        if matid < 0:
            return None
        ref = self.content.materials[matid].texture
        if ref < 0:
            return None
        tex = self.content.textures[ref]
        kind = int(self.model.tex_type[tex.id])
        if kind != int(mujoco.mjtTexture.mjTEXTURE_2D):
            name = self.model.texture(tex.id).name or f"texture {tex.id}"
            raise GltfExportError(
                f"{self._label(matid)} draws with {name!r}, a "
                f"{mujoco.mjtTexture(kind).name.removeprefix('mjTEXTURE_').lower()} texture; glTF "
                "has no cube mapping, and this export draws only 2-D textures rather than guess"
            )
        return tex

    def get(self, matid: int, rgba, textured: bool) -> int:
        """The material for a surface of material ``matid`` (-1 for none) and geom colour ``rgba``.

        ``textured``: the surface has texture coordinates, so the material's texture applies.
        """
        model = self.model
        rgba = resolved_rgba(model, matid, rgba)
        if matid >= 0:
            emission = float(model.mat_emission[matid])
            shininess = float(model.mat_shininess[matid])
        else:
            emission, shininess = 0.0, 0.5
        tex = self.texture(matid) if textured else None
        key = (matid, tuple(round(c, 6) for c in rgba), tex is not None)
        if key in self._index:
            return self._index[key]
        bright = any(c > 1.0 for c in rgba[:3])
        tint = None
        if tex is not None and bright:
            tint, rgba = tuple(rgba[:3]), [1.0, 1.0, 1.0, rgba[3]]
        elif bright and (label := self._label(matid)) not in self._clamped:
            self._clamped.add(label)
            logger.warning(
                "%s has a colour above 1 (%s) and no texture to carry it; glTF's base colour stops "
                "at 1, so it is clamped",
                label,
                ", ".join(f"{c:g}" for c in rgba[:3]),
            )
        base = [min(max(c, 0.0), 1.0) for c in rgba]
        pbr: dict = {
            "baseColorFactor": [*map(srgb_to_linear, base[:3]), base[3]],
            "metallicFactor": 0.0,
            "roughnessFactor": min(max(1.0 - shininess, 0.0), 1.0),
        }
        if tex is not None:
            pbr["baseColorTexture"] = {"index": self._image(tex, tint)}
        mat: dict = {"pbrMetallicRoughness": pbr}
        if matid >= 0 and model.material(matid).name:
            mat["name"] = model.material(matid).name
        if emission > 0:
            mat["emissiveFactor"] = [srgb_to_linear(min(emission * c, 1.0)) for c in base[:3]]
        if base[3] < 1.0:
            mat["alphaMode"] = "BLEND"
        self._index[key] = self.doc.add("materials", mat)
        return self._index[key]

    def _image(self, tex, tint) -> int:
        """The glTF texture of ``tex``'s compiled pixels, ``tint`` multiplied in when given."""
        key = (tex.id, tint)
        if key in self._images:
            return self._images[key]
        from PIL import Image

        pixels = tex.pixels(self.model)
        mode = _PIL_MODE.get(tex.channels)
        if mode is None:
            raise GltfExportError(
                f"texture {self.model.texture(tex.id).name or tex.id!r} has {tex.channels} channels, "
                "which no glTF image format holds"
            )
        img = Image.fromarray(pixels[..., 0] if tex.channels == 1 else pixels)
        assert img.mode == mode, (img.mode, mode)
        if tint is not None:
            rgb = np.asarray(img.convert("RGB"), float) * np.asarray(tint)
            tinted = Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB")
            if "A" in img.getbands():
                tinted.putalpha(img.getchannel("A"))
            img = tinted
        if self.max_tex_dim and max(img.size) > self.max_tex_dim:
            scale = self.max_tex_dim / max(img.size)
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                Image.Resampling.LANCZOS,
            )
        opaque = "A" not in img.getbands() or img.getchannel("A").getextrema() == (255, 255)
        buf = io.BytesIO()
        if self.texture_format == "jpeg" and opaque:
            img.convert("L" if img.mode in ("L", "LA") else "RGB").save(
                buf, "JPEG", quality=self.texture_quality
            )
            mime = "image/jpeg"
        elif self.texture_format == "webp":
            img.convert("RGB" if opaque else "RGBA").save(
                buf, "WEBP", quality=self.texture_quality, method=6
            )
            mime = "image/webp"
        else:
            img.save(buf, "PNG", optimize=True)
            mime = "image/png"
        data = buf.getvalue()
        self.image_bytes += len(data)
        image = self.doc.add("images", {"bufferView": self.doc._view(data, None), "mimeType": mime})
        if not self.doc.gltf.get("samplers"):
            self.doc.add(
                "samplers",
                {
                    "magFilter": _LINEAR,
                    "minFilter": _LINEAR_MIPMAP_LINEAR,
                    "wrapS": _REPEAT,
                    "wrapT": _REPEAT,
                },
            )
        if mime == "image/webp":
            # No PNG beside it: a second copy of every image is what this format is chosen to avoid.
            self.doc.require(_WEBP)
            texture = {"sampler": 0, "extensions": {_WEBP: {"source": image}}}
        else:
            texture = {"sampler": 0, "source": image}
        self._images[key] = self.doc.add("textures", texture)
        return self._images[key]


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
    segments: int | None = None,
    view: dict | None = None,
    max_tex_dim: int = DEFAULT_MAX_TEX_DIM,
    texture_format: str = "png",
    texture_quality: int = 85,
    compress: bool = False,
    log: logging.Logger | None = None,
    cameras: Sequence[SensorCamera] = (),
) -> dict:
    """Write the scene of ``model`` at the state in ``data`` to ``out`` as GLB; return the glTF JSON.

    ``segments`` is the count around a round primitive; by default the model's own
    ``visual/quality numslices``, so a texture lands on the facets MuJoCo draws.
    ``max_tex_dim`` caps an embedded image's longest side (0 keeps every image whole);
    ``texture_format`` ``"jpeg"`` writes opaque images as JPEG and ``"webp"`` every image as WebP,
    at ``texture_quality``; ``compress`` writes vertex and index data with
    ``EXT_meshopt_compression`` (:mod:`roqsim.meshopt`). ``cameras``
    are the sensor cameras to place (:attr:`roqsim.scene_source.CompiledSource.cameras`).
    """
    log = log or logger
    content = walk(model, data, log, cameras)
    doc = _Document(compress=compress)
    materials = _Materials(
        doc,
        model,
        content,
        max_tex_dim=max_tex_dim,
        texture_format=texture_format,
        texture_quality=texture_quality,
    )
    segments = max(3, int(segments if segments else model.vis.quality.numslices))
    extent = float(model.stat.extent)

    # Bodies: body b is node body_node[b], placed where the exported state puts it relative to its
    # parent -- a hinged link at its configured angle, since glTF has no joints to apply one.
    joints_of: list[list[dict]] = [[] for _ in content.bodies]
    for joint in content.joints:
        entry = {
            "name": joint.name,
            "type": joint.type,
            "axis": joint.axis,
            "pos": joint.pos,
            "qposadr": joint.qposadr,
        }
        if joint.ref is not None:
            entry["ref"] = joint.ref
        joints_of[joint.body].append(entry)
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
        if joints_of[b]:
            node["extras"]["joints"] = joints_of[b]
        body_node.append(doc.node(node))
        if b > 0:
            doc.gltf["nodes"][body_node[body.parent]].setdefault("children", []).append(
                body_node[b]
            )
    doc.gltf["scenes"][0]["nodes"].append(body_node[0])
    extras: dict = {
        "format": EXTRAS_FORMAT,
        "version": EXTRAS_VERSION,
        "initialJoints": content.initial_joints,
    }
    authored = {
        k: view[k] for k in ("lookat", "distance", "azimuth", "elevation") if k in (view or {})
    }
    if authored:
        extras["view"] = authored
    doc.gltf["scenes"][0]["extras"] = extras

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

    for sensor in content.cameras:
        camera = doc.add(
            "cameras",
            {
                "name": sensor.name,
                "type": "perspective",
                "perspective": {
                    "yfov": sensor.yfov,
                    "aspectRatio": sensor.aspect_ratio,
                    "znear": sensor.znear,
                    "zfar": sensor.zfar,
                },
            },
        )
        child = doc.node(
            {
                "name": sensor.name,
                "translation": [float(v) for v in sensor.pos],
                "rotation": _xyzw(sensor.quat),
                "camera": camera,
                "extras": {
                    "sensor": sensor.name,
                    "kind": sensor.kind,
                    "width": sensor.width,
                    "height": sensor.height,
                },
            }
        )
        doc.gltf["nodes"][body_node[sensor.body]].setdefault("children", []).append(child)

    cam = _camera_node(view or {}, model)
    if cam is not None:
        cam["camera"] = doc.add("cameras", cam.pop("camera_def"))
        doc.gltf["scenes"][0]["nodes"].append(doc.node(cam))

    blob = doc.glb()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    log.info(
        "exported %d bodies, %d geoms, %d skins (%d of them flexes), %d meshes, %d materials, "
        "%d images -> %s (%d KiB, %d%% of it images; --max-tex-dim and --texture-format webp "
        "make them smaller)",
        len(content.bodies),
        len(content.geoms),
        len(content.skins),
        content.flex_count,
        len(doc.gltf["meshes"]),
        len(doc.gltf["materials"]),
        len(doc.gltf.get("images", [])),
        out,
        len(blob) // 1024,
        round(100 * materials.image_bytes / max(len(blob), 1)),
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
    uv = None if mesh.uv is None else np.asarray(mesh.uv).reshape(-1, 2)
    if doc.compress:
        order, faces = meshopt.fetch_order(faces, len(verts))
        verts, normals, joints, weights = (
            verts[order],
            normals[order],
            joints[order],
            weights[order],
        )
        uv = None if uv is None else uv[order]
    attributes = {
        "POSITION": doc.accessor(zup_to_yup(verts), "VEC3", minmax=True, step="position"),
        "NORMAL": doc.accessor(zup_to_yup(normals), "VEC3", step=NORMAL_STEP),
        "JOINTS_0": doc.raw_accessor(joints, _USHORT, "VEC4", _ARRAY_BUFFER),
        # Not filtered: glTF wants each row to sum to 1, which a rounded weight would not.
        "WEIGHTS_0": doc.accessor(weights, "VEC4"),
    }
    textured = uv is not None and materials.texture(skin.matid) is not None
    if textured:
        rep = materials.model.mat_texrepeat[skin.matid]
        scl = texture_scale("skin", [0.0, 0.0, 0.0], rep, False, explicit=True)
        attributes["TEXCOORD_0"] = doc.accessor(uv * scl, "VEC2", step=TEXCOORD_STEP)
    primitive = {
        "attributes": attributes,
        "indices": doc.indices(faces, len(verts)),
        "material": materials.get(skin.matid, skin.rgba, textured),
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
    """The glTF mesh drawing ``geom``, shared by every geom that looks the same; None if empty.

    A geom whose material has a texture carries the texture coordinates MuJoCo's renderer gives it:
    a primitive's own, a mesh's own, or for a mesh with none the ones MuJoCo generates from its
    vertices -- each scaled by ``texrepeat`` and ``texuniform`` as MuJoCo scales them.
    """
    tex = materials.texture(geom.matid)
    material = materials.get(geom.matid, geom.rgba, textured=tex is not None)
    mat = content.materials[geom.matid] if geom.matid >= 0 else None
    look = (material,) if tex is None else (material, tuple(mat.texrepeat), mat.texuniform)
    size = tuple(round(float(v), 9) for v in geom.size)
    if geom.type == "mesh":
        # The geom's size enters the texture coordinates, so it is part of what the mesh looks like.
        key = ("mesh", content.meshes[geom.mesh].source, *look, size if tex is not None else None)
    else:
        key = (geom.type, size, *look)
    if key in cache:
        return cache[key]
    if geom.type == "mesh":
        verts, normals, faces, uv = mesh_triangles(model, content.meshes[geom.mesh])
        if tex is not None:
            if uv is None:
                uv = generated_uv(verts, geom.size, mat.texrepeat, mat.texuniform)
            else:
                uv = uv * texture_scale("mesh", geom.size, mat.texrepeat, mat.texuniform, True)
    else:
        verts, normals, uv, faces = tessellate.textured_surface(
            geom.type, geom.size, segments, extent
        )
        if tex is None:
            uv = None
        else:
            uv = uv * texture_scale(geom.type, geom.size, mat.texrepeat, mat.texuniform, True)
            if geom.type == "plane" and (geom.size[0] <= 0 or geom.size[1] <= 0):
                uv = uv - 0.5  # MuJoCo re-centres an infinite plane's texture
    if len(faces) == 0:
        cache[key] = None
        return None
    if tex is None:
        uv = None
    if (flat := flat_shaded(verts, normals, faces, uv)) is not None:
        verts, faces, uv = flat
        normals = None
    if doc.compress:
        order, faces = meshopt.fetch_order(faces, len(verts))
        verts = np.asarray(verts)[order]
        normals = None if normals is None else np.asarray(normals)[order]
        uv = None if uv is None else np.asarray(uv)[order]
    attributes = {"POSITION": doc.accessor(verts, "VEC3", minmax=True, step="position")}
    if normals is not None:
        attributes["NORMAL"] = doc.accessor(normals, "VEC3", step=NORMAL_STEP)
    if uv is not None:
        attributes["TEXCOORD_0"] = doc.accessor(uv, "VEC2", step=TEXCOORD_STEP)
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
            "The file is Y-up. Its root node `world` carries the turn from roqsim's Z-up\n"
            "world, so a point in the frame of node `world` is a point in roqsim world\n"
            "coordinates. Every body is a node named as the body is.\n\n"
            + exit_status.epilog(exit_status.BAD_INPUT)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_source_options(parser)
    parser.add_argument("--out", required=True, help="the .glb file to write")
    add_world_options(parser)
    parser.add_argument(
        "--segments",
        type=int,
        default=None,
        help="segments around a sphere, capsule, cylinder or ellipsoid (default: the model's "
        "visual/quality numslices, the count MuJoCo draws them with)",
    )
    parser.add_argument(
        "--max-tex-dim",
        type=int,
        default=DEFAULT_MAX_TEX_DIM,
        help=f"downscale every embedded image whose longest side exceeds this (0 keeps them whole). "
        f"Default {DEFAULT_MAX_TEX_DIM}. Textures are most of a large world's file: lower this, or "
        "use --texture-format webp or jpeg, where a world is too large for the device that loads "
        "it.",
    )
    parser.add_argument(
        "--texture-format",
        choices=("png", "jpeg", "webp"),
        default="png",
        help="how to embed images: png (lossless, the default), jpeg (opaque images; images with "
        "transparency stay PNG) or webp (every image, through EXT_texture_webp). Both lossy "
        "formats are several times smaller for photographic textures.",
    )
    parser.add_argument(
        "--texture-quality",
        type=int,
        default=85,
        help="quality of a jpeg or webp image, 1-95 (default 85)",
    )
    parser.add_argument(
        "--meshopt",
        action="store_true",
        help="compress vertex and index data with EXT_meshopt_compression, which a loader needs a "
        "decoder for (three.js: GLTFLoader.setMeshoptDecoder). Positions keep "
        f"{POSITION_BITS} bits below each mesh's extent, normals and texture coordinates "
        "1/1024 and 1/4096.",
    )
    add_manifest_option(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    refuse_options_for_mjcf(parser, args)
    if Path(args.out).suffix.lower() != ".glb":
        parser.error(f"--out must name a .glb file, got {args.out!r}")
    if not 1 <= args.texture_quality <= 95:
        parser.error(f"--texture-quality takes 1-95, got {args.texture_quality}")
    if args.max_tex_dim < 0:
        parser.error(f"--max-tex-dim takes 0 or more, got {args.max_tex_dim}")
    if args.segments is not None and args.segments < 3:
        parser.error(f"--segments takes 3 or more, got {args.segments}")

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_gltf")

    src = compile_source(args, log)
    try:
        export_gltf(
            src.model,
            src.data,
            Path(args.out),
            segments=args.segments,
            view=src.view,
            cameras=src.cameras,
            max_tex_dim=args.max_tex_dim,
            texture_format=args.texture_format,
            texture_quality=args.texture_quality,
            compress=args.meshopt,
            log=log,
        )
    except GltfExportError as err:
        return exit_status.fail(parser.prog, err)
    write_manifest(args, log)
    return exit_status.OK


if __name__ == "__main__":
    sys.exit(main())
