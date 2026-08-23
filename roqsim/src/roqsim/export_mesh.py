"""Export a model as ONE merged mesh in a chosen body's frame (STL / OBJ / PLY / 3MF).

Why this exists
===============
``export urdf`` and ``export web`` both keep a model as a body tree, because their consumers plan or
animate it. A second class of consumer wants the opposite: one rigid mesh and nothing else. Model-based
6D pose estimators (render-and-compare and ICP alike) take a single mesh and return the pose *of that
mesh's frame*; CAD tools import one solid or one mesh body and know nothing about joints. Neither can
use a link tree, and assembling one by hand from the shipped meshes means redoing the geom and body
transforms the compiler already did.

The frame is the contract
=========================
Vertices come out in the frame of one chosen body (``--frame``), so a pose estimated against the
exported mesh IS the pose of that body -- no further correction. That is why every geom's transform is
composed back through the body tree from the COMPILED model, never taken from ``data``: a robot is
compiled at whatever pose its world spawned it in, and baking that in would put a constant offset into
every estimate. Exporting the same robot from a bare model and from a world that spawns it somewhere
odd therefore produces the same file, which is asserted in the tests.

What travels
============
* Every geom in ``--groups`` below the frame body: meshes verbatim from the compiled model, primitives
  tessellated (``--segments``). Group 3 is excluded by default -- the repo convention is that group-3
  geometry is collision-only, and a chassis-swallowing collision cylinder is not a shape the robot has.
* Colours, where the format can carry them: a ``usemtl`` group per material in OBJ, per-vertex colours
  in PLY, base materials in 3MF. STL cannot carry any, which is the one real reason to prefer another
  format for an estimator that matches on appearance rather than on silhouette.
* 3MF alone keeps the parts apart -- one named, coloured object per geom rather than one merged mesh --
  and states its unit in the file. That makes it the format to hand a CAD tool: a wheel arrives as a
  selectable named part, and there is no mm-versus-m guess to get wrong.
* Nothing else. There is no joint state, no articulation and no unit metadata inside an STL, so the
  chosen frame, the unit and the tessellation are reported in the JSON summary (and in the file's own
  header comment where the format has one) rather than left for the reader to guess.

Watertightness is reported, not claimed
=======================================
A merged robot mesh is usually not a closed solid: shipped visual meshes may have boundary or
non-manifold edges, and two geoms that touch do not fuse. That is fine for rendering and for pose
estimation, which need a silhouette and an appearance, and it matters for CAD, which may need a repair
pass before ``mesh -> solid``. So the summary counts boundary and non-manifold edges per geom instead
of asserting a solid. ``--weld`` merges coincident vertices first, which closes the common case of a
mesh whose seam vertices are merely duplicated.

Usage::

    roqsim export mesh --model turtlebot4 --out robot.obj --sidecar robot.json
    roqsim export mesh --model turtlebot4 --out robot.3mf --units mm     # for CAD
    roqsim export mesh --world w.yaml --prefix ur10e_ --frame base_link --out arm.ply
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import mujoco
import numpy as np

from . import logging_setup

logger = logging.getLogger(__name__)

_COLLISION_GROUP = 3  # the repo convention: group-3 geoms are collision-only, never drawn

#: Unit name -> multiplier applied to metres. MuJoCo is metric; CAD tools conventionally read
#: millimetres, and an STL carries no unit at all, so a model imported as 0.35 mm tall is the classic
#: symptom. The scale is applied in exactly one place (:meth:`MeshExporter.export`).
_UNITS = {"m": 1.0, "mm": 1000.0}

#: Formats, keyed by the extension that selects them.
_FORMATS = {".stl": "stl", ".obj": "obj", ".ply": "ply", ".3mf": "3mf"}

#: 3MF states its unit in the file, which is the one thing STL cannot do.
_UNIT_3MF = {"m": "meter", "mm": "millimeter"}


def _name(model, objtype, idx: int) -> str:
    return mujoco.mj_id2name(model, objtype, idx) or ""


def _quat_to_mat(quat) -> np.ndarray:
    """(w,x,y,z) -> 3x3 rotation matrix.

    Lives here rather than in each exporter: going through the matrix instead of composing quaternion
    components by hand is what keeps the degenerate cases (a 90 deg turn about y, all over this
    substrate's arms) out of every consumer -- see ``export_urdf._quat_to_rpy`` for what that cost.
    """
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, dtype=float))
    return m.reshape(3, 3)


# --------------------------------------------------------------------------------------------------
# Primitive tessellation. Poles and seams are emitted as duplicated vertices and left for the weld
# pass to merge; the degenerate triangles that produces are dropped there too, so each generator stays
# a readable ring-and-cap construction instead of carrying its own special cases.
# --------------------------------------------------------------------------------------------------


def _rings(rows, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a stack of ``(radius, z)`` rings into (vertices, faces), closing in longitude."""
    lon = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = np.concatenate(
        [
            np.stack([r * np.cos(lon), r * np.sin(lon), np.full(segments, z)], axis=1)
            for r, z in rows
        ]
    )
    faces = []
    for i in range(len(rows) - 1):
        for j in range(segments):
            k = (j + 1) % segments
            a, b = i * segments + j, i * segments + k
            c, d = (i + 1) * segments + j, (i + 1) * segments + k
            # Wound so the normal points AWAY from the axis: with rows ordered bottom-to-top and
            # longitude increasing, (a,c,d) faces inward. An inverted primitive renders as a hole and
            # CAD reads it as a void, and nothing about the file looks wrong -- test_closed_geoms
            # _wind_outward is the guard.
            faces += [[a, d, c], [a, b, d]]
    return verts, np.asarray(faces, dtype=int)


def _cylinder(radius: float, half_length: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed cylinder about local z (MuJoCo's convention), as (vertices, faces)."""
    rows = [(0.0, -half_length), (radius, -half_length), (radius, half_length), (0.0, half_length)]
    return _rings(rows, segments)


def _box(size) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box of half-extents ``size``, as (vertices, faces)."""
    sx, sy, sz = (float(v) for v in size)
    verts = np.array(
        [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 7, 5],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=int,
    )
    return verts, faces


def _sphere(radius: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-np.pi / 2, np.pi / 2, max(3, segments // 2 + 1))
    rows = [(radius * np.cos(a), radius * np.sin(a)) for a in lat]
    return _rings(rows, segments)


def _capsule(radius: float, half_length: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """A cylinder about local z closed by two hemispheres -- MuJoCo's capsule."""
    n = max(2, segments // 4)
    lower = np.linspace(-np.pi / 2, 0.0, n + 1)
    upper = np.linspace(0.0, np.pi / 2, n + 1)
    rows = [(radius * np.cos(a), -half_length + radius * np.sin(a)) for a in lower]
    rows += [(radius * np.cos(a), half_length + radius * np.sin(a)) for a in upper]
    return _rings(rows, segments)


def _ellipsoid(size, segments: int) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = _sphere(1.0, segments)
    return verts * np.asarray(size, dtype=float)[:3], faces


def _weld(verts: np.ndarray, faces: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices closer than ``tol`` and drop the faces that collapse.

    Not an optimisation. A shipped visual mesh whose seam vertices are duplicated has boundary edges
    everywhere along that seam and can never be closed, however good the exporter is; welding fixes it
    deterministically and without touching geometry. It also lets the primitive generators above emit
    poles as ordinary rings.
    """
    if tol <= 0:
        keep = faces[
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        ]
        return verts, keep
    quantized = np.round(verts / tol).astype(np.int64)
    _, first, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    # Keep the FIRST occurrence's own coordinates rather than the rounded grid point: the weld decides
    # which vertices are the same vertex, it does not move the survivor onto a lattice.
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    welded = verts[first[order]]
    refaced = remap[inverse.reshape(-1)][faces]
    keep = refaced[
        (refaced[:, 0] != refaced[:, 1])
        & (refaced[:, 1] != refaced[:, 2])
        & (refaced[:, 0] != refaced[:, 2])
    ]
    return welded, keep


def _edge_stats(faces: np.ndarray) -> dict:
    """Boundary / manifold / non-manifold edge counts -- the watertightness report."""
    if len(faces) == 0:
        return {"edges": 0, "boundary": 0, "manifold": 0, "non_manifold": 0, "closed": False}
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())
    return {
        "edges": int(len(counts)),
        "boundary": boundary,
        "manifold": int((counts == 2).sum()),
        "non_manifold": non_manifold,
        "closed": boundary == 0 and non_manifold == 0,
    }


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Divergence-theorem volume. Meaningful only for a closed shell; its SIGN reports the winding."""
    if len(faces) == 0:
        return 0.0
    tri = verts[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


class MeshExportError(RuntimeError):
    """The selection cannot be exported as a mesh -- reported loudly, never worked around."""


def _chain(model, body_id: int) -> list[int]:
    """``body_id`` and every ancestor up to the world body, nearest first."""
    out, node = [], int(body_id)
    while True:
        out.append(node)
        if node == 0:
            return out
        node = int(model.body_parentid[node])


def _common_ancestor(model, bodies) -> int:
    """The deepest body that is an ancestor of (or equal to) all of ``bodies``.

    This is what makes ``--frame`` optional without hardcoding a link name: for a single robot the
    answer is its own root body (``base_link`` for a mobile base, because both wheels hang off it), and
    for a whole world it falls back to the world body, which is the only honest answer there.
    """
    chains = [_chain(model, b) for b in bodies]
    shared = set(chains[0]).intersection(*(set(c) for c in chains[1:]))
    return max(shared, key=lambda b: len(_chain(model, b))) if shared else 0


def _body_to_frame(model, body_id: int, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` taking a point in ``body_id``'s frame into ``frame_id``'s, via the body tree.

    Composes the COMPILED ``body_pos``/``body_quat``, never ``data.xpos`` -- see the module docstring
    on why the export must not depend on where a world spawned the robot.
    """
    rot, trans, node = np.eye(3), np.zeros(3), int(body_id)
    while node != frame_id:
        if node == 0:
            raise MeshExportError(
                f"body {_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)!r} is not below the frame "
                f"body {_name(model, mujoco.mjtObj.mjOBJ_BODY, frame_id)!r}"
            )
        r = _quat_to_mat(model.body_quat[node])
        trans = model.body_pos[node] + r @ trans
        rot = r @ rot
        node = int(model.body_parentid[node])
    return rot, trans


class MeshExporter:
    """Merges the selected geoms of a compiled model into one mesh in a chosen body's frame."""

    def __init__(
        self,
        model,
        *,
        frame: str = "",
        prefix: str = "",
        groups=None,
        exclude=(),
        segments: int = 32,
        weld: float = 1e-6,
    ) -> None:
        self.model = model
        self.prefix = prefix
        self.exclude = tuple(exclude)
        self.segments = max(3, int(segments))
        self.weld = float(weld)
        # "Every group except the collision one" is read off the MODEL rather than assumed to be 0..5:
        # geom_group is a plain int, so a model that numbers a group above the conventional range
        # would otherwise lose that geometry silently.
        self.groups = (
            {int(g) for g in groups}
            if groups
            else {int(g) for g in set(model.geom_group.tolist()) if int(g) != _COLLISION_GROUP}
        )
        self.skipped: list[str] = []
        self.geoms: list[dict] = []
        self.materials: list[tuple[str, tuple]] = []
        self._selection = self._select()
        self.frame_id = self._resolve_frame(frame)
        self.frame = _name(model, mujoco.mjtObj.mjOBJ_BODY, self.frame_id) or "world"

    # -- selection ---------------------------------------------------------------------------------

    def _select(self) -> list[int]:
        model = self.model
        out = []
        for gid in range(model.ngeom):
            if int(model.geom_group[gid]) not in self.groups:
                continue
            body = int(model.geom_bodyid[gid])
            body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if self.prefix and not body_name.startswith(self.prefix):
                continue
            if any(pattern in body_name for pattern in self.exclude):
                self.skipped.append(f"{body_name} (excluded)")
                continue
            # A plane is unbounded and is scenery rather than part of a model, so it is dropped here
            # and reported. Dropping it at SELECTION time (not later) matters: the default frame is
            # the root of the selection, and a world's ground plane would otherwise drag that frame
            # up to the world body and quietly re-express the whole export.
            if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                self.skipped.append(
                    f"{_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or 'plane'} (plane: unbounded)"
                )
                continue
            out.append(gid)
        if not out:
            raise MeshExportError(
                f"no geoms in groups {sorted(self.groups)}"
                + (f" below bodies matching prefix {self.prefix!r}" if self.prefix else "")
                + " -- nothing to export"
            )
        return out

    def _resolve_frame(self, frame: str) -> int:
        if frame:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame)
            if fid < 0:
                raise MeshExportError(f"frame body {frame!r} not found in the model")
            return int(fid)
        bodies = {int(self.model.geom_bodyid[g]) for g in self._selection}
        return _common_ancestor(self.model, sorted(bodies))

    # -- geometry ----------------------------------------------------------------------------------

    def _local(self, gid: int) -> tuple[np.ndarray, np.ndarray]:
        """The geom's own triangles in its own frame."""
        model = self.model
        gtype = int(model.geom_type[gid])
        size = model.geom_size[gid]
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[gid])
            start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fstart, fcount = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            verts = model.mesh_vert[start : start + count].reshape(-1, 3).astype(float)
            faces = model.mesh_face[fstart : fstart + fcount].reshape(-1, 3).astype(int)
            return verts, faces
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            return _box(size[:3])
        if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return _cylinder(float(size[0]), float(size[1]), self.segments)
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return _sphere(float(size[0]), self.segments)
        if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            return _capsule(float(size[0]), float(size[1]), self.segments)
        if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return _ellipsoid(size[:3], self.segments)
        raise MeshExportError(
            f"geom {_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or gid!r} on body "
            f"{_name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[gid]))!r} is a "
            f"{mujoco.mjtGeom(gtype).name} and has no mesh spelling. Exclude it (--exclude) or drop "
            f"its group (--groups) -- a stand-in shape labelled as the model would be worse than "
            f"refusing."
        )

    def _geom_label(self, gid: int, gtype: int) -> str:
        """A name a reader can act on: the geom's own, else its mesh's, else its body's plus the type.

        Visual geoms are commonly unnamed -- the MJCF names the mesh instead -- and the name is what
        makes both the watertightness report and a 3MF part list usable: "shell is open" rather than
        "mesh_3 is open", and "left_wheel_cylinder" rather than "cylinder_12".
        """
        own = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if own:
            return own
        kind = mujoco.mjtGeom(gtype).name.removeprefix("mjGEOM_").lower()
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = _name(self.model, mujoco.mjtObj.mjOBJ_MESH, int(self.model.geom_dataid[gid]))
            if mesh:
                return mesh
        body = _name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[gid]))
        return f"{body}_{kind}" if body else f"{kind}_{gid}"

    def _material(self, gid: int) -> int:
        """Index into :attr:`materials` for this geom's colour (MJCF material first, then geom rgba)."""
        model = self.model
        matid = int(model.geom_matid[gid])
        if matid >= 0:
            rgba = tuple(round(float(v), 6) for v in model.mat_rgba[matid])
            name = _name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid) or f"material_{matid}"
        else:
            rgba = tuple(round(float(v), 6) for v in model.geom_rgba[gid])
            name = "color_{:02x}{:02x}{:02x}".format(*(int(round(v * 255)) for v in rgba[:3]))
        for i, (known, known_rgba) in enumerate(self.materials):
            if known == name and known_rgba == rgba:
                return i
        self.materials.append((name, rgba))
        return len(self.materials) - 1

    def collect(self) -> None:
        """Walk the selection, transform every geom into the frame, and weld each one."""
        model = self.model
        for gid in self._selection:
            gtype = int(model.geom_type[gid])
            body = int(model.geom_bodyid[gid])
            verts, faces = self._local(gid)
            g_rot = _quat_to_mat(model.geom_quat[gid])
            g_pos = np.asarray(model.geom_pos[gid], dtype=float)
            b_rot, b_pos = _body_to_frame(model, body, self.frame_id)
            verts = (b_rot @ (g_rot @ verts.T + g_pos[:, None]) + b_pos[:, None]).T
            verts, faces = _weld(verts, faces, self.weld)
            self.geoms.append(
                {
                    "name": self._geom_label(gid, gtype),
                    "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                    "type": mujoco.mjtGeom(gtype).name.removeprefix("mjGEOM_").lower(),
                    "group": int(model.geom_group[gid]),
                    "material": self._material(gid),
                    "verts": verts,
                    "faces": faces,
                }
            )
        if not self.geoms:
            raise MeshExportError("every selected geom was skipped -- nothing to export")

    def merge(self, scale: float) -> dict:
        """One vertex array, one face array, and the per-face / per-vertex material indices."""
        verts, faces, face_mat, vert_mat, used = [], [], [], [], 0
        for geom in self.geoms:
            verts.append(geom["verts"] * scale)
            faces.append(geom["faces"] + used)
            face_mat.append(np.full(len(geom["faces"]), geom["material"], dtype=int))
            vert_mat.append(np.full(len(geom["verts"]), geom["material"], dtype=int))
            used += len(geom["verts"])
        return {
            "verts": np.concatenate(verts),
            "faces": np.concatenate(faces),
            "face_material": np.concatenate(face_mat),
            "vertex_material": np.concatenate(vert_mat),
        }


# --------------------------------------------------------------------------------------------------
# Writers. Every header carries the model, the frame, the unit and the tool -- and nothing else: a
# generated mesh is often committed, and a path or a command line baked into its header outlives the
# machine it came from.
# --------------------------------------------------------------------------------------------------


def _face_normals(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)


def write_stl(path: Path, mesh: dict, name: str, header: str, binary: bool = True) -> None:
    tri = mesh["verts"][mesh["faces"]]
    normals = _face_normals(tri)
    if binary:
        with path.open("wb") as fh:
            # The 80-byte header is the only place a binary STL can say what it is.
            fh.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
            fh.write(struct.pack("<I", len(tri)))
            for normal, (a, b, c) in zip(normals, tri, strict=True):
                fh.write(struct.pack("<12fH", *normal, *a, *b, *c, 0))
        return
    with path.open("w", encoding="ascii", errors="replace") as fh:
        fh.write(f"solid {name}\n")
        for normal, (a, b, c) in zip(normals, tri, strict=True):
            fh.write("  facet normal {:.6e} {:.6e} {:.6e}\n    outer loop\n".format(*normal))
            for vertex in (a, b, c):
                fh.write("      vertex {:.6e} {:.6e} {:.6e}\n".format(*vertex))
            fh.write("    endloop\n  endfacet\n")
        fh.write(f"endsolid {name}\n")


def write_obj(path: Path, mesh: dict, name: str, header: list[str], materials, color: bool) -> None:
    mtl_path = path.with_suffix(".mtl")
    with path.open("w", encoding="utf-8") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        if color:
            fh.write(f"mtllib {mtl_path.name}\n")
        fh.write(f"o {name}\n")
        for x, y, z in mesh["verts"]:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        faces, face_mat = mesh["faces"] + 1, mesh["face_material"]  # OBJ is 1-indexed
        if not color:
            for a, b, c in faces:
                fh.write(f"f {a} {b} {c}\n")
        else:
            for m in range(len(materials)):
                selected = faces[face_mat == m]
                if not len(selected):
                    continue
                fh.write(f"usemtl {materials[m][0]}\n")
                for a, b, c in selected:
                    fh.write(f"f {a} {b} {c}\n")
    if not color:
        return
    with mtl_path.open("w", encoding="utf-8") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        for mat_name, rgba in materials:
            r, g, b = (float(v) for v in rgba[:3])
            fh.write(f"newmtl {mat_name}\nKd {r:.6f} {g:.6f} {b:.6f}\n")
            fh.write(f"Ka 0 0 0\nKs 0 0 0\nd {float(rgba[3]):.6f}\nillum 1\n")


def write_ply(path: Path, mesh: dict, header: list[str], materials, color: bool) -> None:
    verts, faces = mesh["verts"], mesh["faces"]
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {line}" for line in header]
    lines += [
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if color:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines += [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    with path.open("wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode("ascii", "replace"))
        if color:
            rgb = np.array(
                [[int(round(v * 255)) for v in rgba[:3]] for _, rgba in materials], dtype=np.uint8
            )[mesh["vertex_material"]]
            for (x, y, z), (r, g, b) in zip(verts, rgb, strict=True):
                fh.write(struct.pack("<3fBBB", x, y, z, int(r), int(g), int(b)))
        else:
            for x, y, z in verts:
                fh.write(struct.pack("<3f", x, y, z))
        for a, b, c in faces:
            fh.write(struct.pack("<B3i", 3, int(a), int(b), int(c)))


_3MF_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_3MF_RELTYPE = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"

_3MF_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" \
ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" \
ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

_3MF_RELS = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rel0" Target="/3D/3dmodel.model" Type="{_3MF_RELTYPE}"/>
</Relationships>
"""


def _unique(names) -> list[str]:
    """Suffix repeats, so four identical standoffs arrive in CAD as four distinguishable parts."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def write_3mf(
    path: Path,
    geoms,
    materials,
    *,
    name: str,
    unit: str,
    color: bool,
    scale: float,
    header: list[str],
) -> None:
    """Write a 3MF package: one named, coloured OBJECT per geom, assembled into one build item.

    3MF is the one format here that a CAD tool can read *and* that carries both colour and the unit --
    an STL states no unit anywhere, which is how a 0.35 m robot arrives 0.35 mm tall. It is an OPC
    package (a zip) of three members, and the geometry is plain XML, so the whole writer is stdlib.

    Unlike the single-mesh formats this emits a part per geom rather than one merged blob, because that
    is what the format is for: a wheel arrives as a selectable, named, coloured part instead of
    anonymous triangles inside a lump. Geometry is still baked into the export frame, so the components
    carry no transform -- 3MF *could* instance the four identical standoffs through one object placed
    four times, but only by un-baking their poses, which would give up the frame contract.

    The mesh is unchanged from the other writers, including its defects: 3MF's spec wants manifold
    objects, and a shipped visual mesh with boundary edges does not become manifold by being packaged.
    Consumers repair it; a strict conformance checker will say so.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model unit="{unit}" xml:lang="en-US" xmlns="{_3MF_CORE}">',
        f'  <metadata name="Title">{escape(name)}</metadata>',
        '  <metadata name="Application">roqsim export mesh</metadata>',
        # Not an XML comment: every header line contains "--", which a comment may not.
        f'  <metadata name="Description">{escape(" ".join(header))}</metadata>',
        "  <resources>",
    ]
    if color and materials:
        lines.append('    <basematerials id="1">')
        for mat_name, rgba in materials:
            r, g, b = (int(round(float(v) * 255)) for v in rgba[:3])
            alpha = int(round(float(rgba[3]) * 255))
            lines.append(
                f"      <base name={quoteattr(mat_name)} "
                f'displaycolor="#{r:02X}{g:02X}{b:02X}{alpha:02X}"/>'
            )
        lines.append("    </basematerials>")

    ids = []
    for object_id, (geom, part) in enumerate(
        zip(geoms, _unique([g["name"] for g in geoms]), strict=True), start=2
    ):
        material = f' pid="1" pindex="{geom["material"]}"' if color and materials else ""
        lines.append(f'    <object id="{object_id}" type="model" name={quoteattr(part)}{material}>')
        lines.append("      <mesh>")
        lines.append("        <vertices>")
        lines += [
            f'          <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>'
            for x, y, z in geom["verts"] * scale
        ]
        lines.append("        </vertices>")
        lines.append("        <triangles>")
        # 3MF wants each triangle counter-clockwise seen from outside -- the same outward winding the
        # exporter already asserts per geom.
        lines += [f'          <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in geom["faces"]]
        lines.append("        </triangles>")
        lines.append("      </mesh>")
        lines.append("    </object>")
        ids.append(object_id)

    root = ids[-1] + 1
    lines.append(f'    <object id="{root}" type="model" name={quoteattr(name)}>')
    lines.append("      <components>")
    lines += [f'        <component objectid="{i}"/>' for i in ids]
    lines.append("      </components>")
    lines.append("    </object>")
    lines.append("  </resources>")
    lines.append(f'  <build><item objectid="{root}"/></build>')
    lines.append("</model>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        for member, text in (
            ("[Content_Types].xml", _3MF_CONTENT_TYPES),
            ("_rels/.rels", _3MF_RELS),
            ("3D/3dmodel.model", "\n".join(lines) + "\n"),
        ):
            # A fixed timestamp, because a zip stores one per member and the export is otherwise
            # byte-reproducible -- see the determinism test.
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(info, text)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export mesh",
        description="Export a model as one merged mesh in a chosen body's frame (STL/OBJ/PLY).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="model reference (name, package:name, or path to an MJCF)")
    source.add_argument("--world", help="path to a world YAML (compiled via the plugin pipeline)")
    source.add_argument("--mjcf", help="path to a bare MJCF file (compiled directly)")
    parser.add_argument("--out", required=True, help="output .stl / .obj / .ply / .3mf path")
    parser.add_argument(
        "--frame",
        default="",
        help="body whose frame the mesh is expressed in (default: the root body of the selection). A "
        "pose estimated against the mesh is the pose of THIS body",
    )
    parser.add_argument(
        "--prefix", default="", help="MJCF name prefix selecting one robot in a world"
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=None,
        help=f"geom groups to include (default: every group except {_COLLISION_GROUP}, which is "
        f"collision-only by convention)",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="drop bodies whose name contains this"
    )
    parser.add_argument(
        "--units",
        choices=sorted(_UNITS),
        default="m",
        help="m (default, MuJoCo native, no rescaling) or mm (the CAD convention)",
    )
    parser.add_argument(
        "--format",
        choices=["stl", "stl-ascii", "obj", "ply", "3mf"],
        default=None,
        help="output format (default: from the --out extension)",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="carry materials (OBJ usemtl + .mtl, PLY vertex colours, 3MF base materials). STL "
        "cannot carry any",
    )
    parser.add_argument(
        "--segments", type=int, default=32, help="tessellation of round primitives (default 32)"
    )
    parser.add_argument(
        "--weld",
        type=float,
        default=1e-6,
        help="merge vertices closer than this many METRES (default 1e-6; 0 disables). Closes meshes "
        "whose seam vertices are merely duplicated",
    )
    parser.add_argument("--sidecar", help="also write the JSON summary to this path")
    parser.add_argument("--manifest", help="also write {'inputs': [...]} for campaign caching")
    parser.add_argument(
        "--skip-plugins",
        default="",
        help="--world: extra plugin names/refs to drop before compiling (transport plugins always are)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_mesh")

    out = Path(args.out)
    fmt = args.format or _FORMATS.get(out.suffix.lower())
    if fmt is None:
        log.error(
            "cannot tell the format from %r -- use one of %s as the extension, or pass --format",
            out.name,
            ", ".join(sorted(_FORMATS)),
        )
        return 2

    if args.model:
        from .models import apply_assets, resolve_model

        asset = resolve_model(args.model)
        spec = mujoco.MjSpec.from_file(str(asset.path))
        apply_assets(spec, asset)
        model = spec.compile()
        # The resolved file's stem, not the reference: "roqsim_mobile:turtlebot4" would otherwise carry
        # its colon into the object name inside the file.
        label = asset.path.stem
        inputs = [str(asset.path)]
    else:
        from .export_web import _compile_from_mjcf, _compile_from_world

        if args.mjcf:
            model, _data, _view = _compile_from_mjcf(Path(args.mjcf))
            label, inputs = Path(args.mjcf).stem, [str(Path(args.mjcf).resolve())]
        else:
            skip = {s.strip() for s in args.skip_plugins.split(",") if s.strip()}
            model, _data, _view = _compile_from_world(args.world, skip, {}, log)
            label, inputs = Path(args.world).stem, None

    try:
        exporter = MeshExporter(
            model,
            frame=args.frame,
            prefix=args.prefix,
            groups=args.groups,
            exclude=args.exclude,
            segments=args.segments,
            weld=args.weld,
        )
        exporter.collect()
    except MeshExportError as exc:
        log.error("%s", exc)
        return 1

    name = (args.prefix or label).strip("_") or label
    mesh = exporter.merge(_UNITS[args.units])
    stats = _edge_stats(mesh["faces"])
    lo, hi = mesh["verts"].min(axis=0), mesh["verts"].max(axis=0)
    header = [
        f"{name}, expressed in the {exporter.frame} frame ({args.units}).",
        "GENERATED by `roqsim export mesh` -- do not edit.",
        # Neither OBJ nor STL states an axis convention, so consumers assume one. Blender's OBJ
        # importer assumes Y-up and silently lays a Z-up model on its side (its STL importer does
        # not convert at all, so the same geometry stands). Saying it here is the cheapest fix.
        "Z-up, right-handed -- the model's own frame. A Y-up importer (Blender's OBJ default) "
        "rotates this; import with up=Z, forward=Y.",
        f"{len(mesh['verts'])} vertices, {len(mesh['faces'])} triangles from "
        f"{len(exporter.geoms)} geom(s), groups {sorted(exporter.groups)}.",
        f"extent x [{lo[0]:.4f}, {hi[0]:.4f}]  y [{lo[1]:.4f}, {hi[1]:.4f}]  "
        f"z [{lo[2]:.4f}, {hi[2]:.4f}]",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    color = args.color and fmt in ("obj", "ply", "3mf")
    if fmt in ("stl", "stl-ascii"):
        if args.color:
            log.info(
                "STL carries neither materials nor a unit; OBJ, PLY and 3MF carry colour, and 3MF "
                "also states the unit in the file"
            )
        write_stl(out, mesh, name, " ".join(header[0].split()), binary=(fmt == "stl"))
    elif fmt == "obj":
        write_obj(out, mesh, name, header, exporter.materials, color)
    elif fmt == "ply":
        write_ply(out, mesh, header, exporter.materials, color)
    else:
        write_3mf(
            out,
            exporter.geoms,
            exporter.materials,
            name=name,
            unit=_UNIT_3MF[args.units],
            color=color,
            scale=_UNITS[args.units],
            header=header,
        )

    summary = {
        "out": out.name,
        "format": fmt,
        # Every format but 3MF merges the selection into one mesh; 3MF keeps one object per geom.
        "objects": len(exporter.geoms) if fmt == "3mf" else 1,
        "model": name,
        "frame": exporter.frame,
        "units": args.units,
        "vertices": int(len(mesh["verts"])),
        "triangles": int(len(mesh["faces"])),
        "bbox": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
        "groups": sorted(exporter.groups),
        "segments": exporter.segments,
        "weld": exporter.weld,
        "materials": [
            {"name": mat, "rgba": [float(v) for v in rgba]} for mat, rgba in exporter.materials
        ]
        if color
        else [],
        "watertight": stats,
        "geoms": [
            {
                "name": g["name"],
                "body": g["body"],
                "type": g["type"],
                "group": g["group"],
                "triangles": int(len(g["faces"])),
                "material": exporter.materials[g["material"]][0],
                "watertight": _edge_stats(g["faces"]),
            }
            for g in exporter.geoms
        ],
        "skipped": exporter.skipped,
    }
    print(json.dumps(summary))
    if args.sidecar:
        Path(args.sidecar).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sidecar).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log.info(
        "wrote %s (%d triangles, %d vertices, frame %s, %s)%s",
        out,
        len(mesh["faces"]),
        len(mesh["verts"]),
        exporter.frame,
        args.units,
        ""
        if stats["closed"]
        else f"; NOT closed: {stats['boundary']} boundary and {stats['non_manifold']} non-manifold "
        f"edge(s) -- fine for rendering and pose estimation, a CAD solid may need a repair pass",
    )
    for entry in exporter.skipped:
        log.info("skipped %s", entry)

    if args.manifest:
        from .config import world_sources

        sources = inputs if inputs is not None else [str(p) for p in world_sources(args.world)]
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps({"inputs": sources}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
