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
import io
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
from .scene_content import Geom, Mesh, SceneContent, Skin, resolved_rgba, walk
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


def primitive_triangles(
    gtype: str, size, segments: int, extent: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A primitive geom of ``size`` as ``(verts, normals, uv, faces)`` in the geom's frame.

    The surface, its exact normals and MuJoCo's own texture coordinates
    (:func:`roqsim.tessellate.textured_surface`), before ``texrepeat`` scales them.
    """
    return tessellate.textured_surface(gtype, size, segments, extent)


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


#: glTF sampler constants.
_LINEAR, _LINEAR_MIPMAP_LINEAR, _REPEAT = 9729, 9987, 10497

#: Channel count -> Pillow mode.
_PIL_MODE = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}


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
        jpeg_quality: int,
    ) -> None:
        self.doc = doc
        self.model = model
        self.content = content
        self.max_tex_dim = int(max_tex_dim)
        self.texture_format = texture_format
        self.jpeg_quality = int(jpeg_quality)
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
            "baseColorFactor": base,
            "metallicFactor": 0.0,
            "roughnessFactor": min(max(1.0 - shininess, 0.0), 1.0),
        }
        if tex is not None:
            pbr["baseColorTexture"] = {"index": self._image(tex, tint)}
        mat: dict = {"pbrMetallicRoughness": pbr}
        if matid >= 0 and model.material(matid).name:
            mat["name"] = model.material(matid).name
        if emission > 0:
            mat["emissiveFactor"] = [min(emission * c, 1.0) for c in base[:3]]
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
                buf, "JPEG", quality=self.jpeg_quality
            )
            mime = "image/jpeg"
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
        self._images[key] = self.doc.add("textures", {"sampler": 0, "source": image})
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
    jpeg_quality: int = 85,
    log: logging.Logger | None = None,
) -> dict:
    """Write the scene of ``model`` at the state in ``data`` to ``out`` as GLB; return the glTF JSON.

    ``segments`` is the count around a round primitive; by default the model's own
    ``visual/quality numslices``, so a texture lands on the facets MuJoCo draws.
    ``max_tex_dim`` caps an embedded image's longest side (0 keeps every image whole);
    ``texture_format`` ``"jpeg"`` writes opaque images as JPEG at ``jpeg_quality``.
    """
    log = log or logger
    content = walk(model, data, log)
    doc = _Document()
    materials = _Materials(
        doc,
        model,
        content,
        max_tex_dim=max_tex_dim,
        texture_format=texture_format,
        jpeg_quality=jpeg_quality,
    )
    segments = max(3, int(segments if segments else model.vis.quality.numslices))
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
        "exported %d bodies, %d geoms, %d skins (%d of them flexes), %d meshes, %d materials, "
        "%d images -> %s (%d KiB, %d%% of it images; --max-tex-dim and --texture-format jpeg "
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
    attributes = {
        "POSITION": doc.accessor(zup_to_yup(verts), "VEC3", minmax=True),
        "NORMAL": doc.accessor(zup_to_yup(normals), "VEC3"),
        "JOINTS_0": doc.raw_accessor(joints, _USHORT, "VEC4", _ARRAY_BUFFER),
        "WEIGHTS_0": doc.accessor(weights, "VEC4"),
    }
    textured = mesh.uv is not None and materials.texture(skin.matid) is not None
    if textured:
        rep = materials.model.mat_texrepeat[skin.matid]
        scl = texture_scale("skin", [0.0, 0.0, 0.0], rep, False, explicit=True)
        attributes["TEXCOORD_0"] = doc.accessor(np.asarray(mesh.uv).reshape(-1, 2) * scl, "VEC2")
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
        verts, normals, uv, faces = primitive_triangles(geom.type, geom.size, segments, extent)
        if tex is None:
            uv = None
        else:
            uv = uv * texture_scale(geom.type, geom.size, mat.texrepeat, mat.texuniform, True)
            if geom.type == "plane" and (geom.size[0] <= 0 or geom.size[1] <= 0):
                uv = uv - 0.5  # MuJoCo re-centres an infinite plane's texture
    if len(faces) == 0:
        cache[key] = None
        return None
    attributes = {
        "POSITION": doc.accessor(verts, "VEC3", minmax=True),
        "NORMAL": doc.accessor(normals, "VEC3"),
    }
    if uv is not None and tex is not None:
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
        "use --texture-format jpeg, where a world is too large for the device that loads it.",
    )
    parser.add_argument(
        "--texture-format",
        choices=("png", "jpeg"),
        default="png",
        help="how to embed opaque images: png (lossless, the default) or jpeg (several times "
        "smaller for photographic textures). Images with transparency are always PNG.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality, 1-95, with --texture-format jpeg (default 85)",
    )
    add_manifest_option(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    refuse_options_for_mjcf(parser, args)
    if Path(args.out).suffix.lower() != ".glb":
        parser.error(f"--out must name a .glb file, got {args.out!r}")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error(f"--jpeg-quality takes 1-95, got {args.jpeg_quality}")
    if args.max_tex_dim < 0:
        parser.error(f"--max-tex-dim takes 0 or more, got {args.max_tex_dim}")

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_gltf")

    model, data, view = compile_source(args, log)
    try:
        export_gltf(
            model,
            data,
            Path(args.out),
            segments=args.segments,
            view=view,
            max_tex_dim=args.max_tex_dim,
            texture_format=args.texture_format,
            jpeg_quality=args.jpeg_quality,
            log=log,
        )
    except GltfExportError as err:
        return exit_status.fail(parser.prog, err)
    write_manifest(args, log)
    return exit_status.OK


if __name__ == "__main__":
    sys.exit(main())
