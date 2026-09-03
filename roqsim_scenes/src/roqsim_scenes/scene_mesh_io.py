"""Mesh read / transform / write for the scene import pipeline (no third-party mesh deps).

Stage-1 importers must emit **world-space OBJ in metres, MuJoCo Z-up**, because ``scene_to_mjcf.py``
places every scene geom at a single ``origin`` and lets the mesh carry the placement (see its
``g.pos = origin``). So the pose chain is baked into the vertices here, not deferred.

Readers cover what SDF worlds actually ship: Collada (``.dae``), STL (binary + ascii) and OBJ. The
Collada reader is generalised from ``external/convert/convert_husky_meshes.py`` — a direct
geometry copy via lxml, deliberately not Blender: these are static visual/collision triangles that
MuJoCo re-normals anyway, so a faithful vertex/index copy is both simpler and higher-fidelity than a
round-trip through an importer that may add its own axis conversion. If a mesh ever needs real work
(decimation, normals, UVs), do that in Blender and feed the result back in.

Primitives (``<box>``/``<cylinder>``/``<sphere>``) are tessellated here so that everything downstream
is uniformly a mesh — ``scene.json`` has no primitive representation.

**Materials.** A reader returns a list of :class:`Submesh`, one **per material**, carrying UVs and the
resolved diffuse-texture path. The split is not cosmetic bookkeeping: a MuJoCo geom has exactly one
material, so a warehouse DAE binding 8 materials across 13 triangle groups can only be textured as 8
geoms. Merging it into one mesh throws the UVs and the material binding away and leaves an
untextured grey shell that no `scene.yaml` can repaint.

UVs force **de-indexing**: Collada and OBJ index position and texcoord *separately* (one corner is a
``(pos_idx, uv_idx)`` pair), while MuJoCo wants one flat vertex array. So a corner whose position is
shared but whose UV differs — every seam on every texture atlas — must become two vertices. Readers
here do that per material group.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lxml import etree


@dataclass
class Submesh:
    """One material's worth of triangles, in the frame its reader documents.

    ``uv``/``texture`` are ``None`` for geometry that legitimately has no texture (STL, a colour-only
    Collada material, a tessellated primitive) — that is not an error, it just renders with ``rgba``.
    """

    verts: np.ndarray  # (N,3) float
    faces: np.ndarray  # (M,3) int, indexing verts
    uv: np.ndarray | None = None  # (N,2) float, per-vertex; parallel to verts
    texture: Path | None = None  # resolved diffuse image on disk
    rgba: list[float] | None = None  # material diffuse colour, when the source states one


# ---------------------------------------------------------------- transforms


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF ``<pose>`` RPY -> 3x3. SDF uses fixed-axis XYZ, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def pose_to_matrix(pose: str | None) -> np.ndarray:
    """``"x y z roll pitch yaw"`` -> 4x4 homogeneous. Missing/short poses default to identity parts."""
    m = np.eye(4)
    if not pose:
        return m
    v = [float(x) for x in pose.split()]
    v += [0.0] * (6 - len(v))
    m[:3, 3] = v[:3]
    m[:3, :3] = rpy_to_matrix(v[3], v[4], v[5])
    return m


def transform(verts: np.ndarray, mat: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Apply optional per-axis scale then a 4x4 transform to Nx3 vertices."""
    v = verts * scale if scale is not None else verts
    return (mat[:3, :3] @ v.T).T + mat[:3, 3]


# ---------------------------------------------------------------- readers


def _localname(el) -> str:
    # Comments/PIs have a callable .tag and are not QNames.
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _findall(el, name: str) -> list:
    return [c for c in el.iter() if _localname(c) == name]


def _accessor_stride(src, default: int) -> int:
    acc = next((c for c in src.iter() if _localname(c) == "accessor"), None)
    if acc is not None and acc.get("stride"):
        return int(acc.get("stride"))
    return default


def _collada_sources(root) -> dict[str, tuple[np.ndarray, int]]:
    """``#source_id -> (flat float array, stride)``. Stride matters: POSITION is 3-wide but TEXCOORD
    is usually 2 and occasionally 3, so it cannot be assumed."""
    out: dict[str, tuple[np.ndarray, int]] = {}
    for src in _findall(root, "source"):
        fa = next((c for c in src if _localname(c) == "float_array"), None)
        if fa is None or not fa.text:
            continue
        arr = np.array([float(x) for x in fa.text.split()], dtype=np.float64)
        out["#" + src.get("id")] = (arr, _accessor_stride(src, 3))
    return out


def _resolve_image(ref: str, base: Path) -> Path:
    """A Collada/OBJ image reference -> a real file. Fails loudly: a texture the source *names* but we
    cannot find is a broken import, not a reason to fall back to grey."""
    from urllib.parse import unquote, urlparse

    raw = unquote(urlparse(ref).path if ref.startswith("file:") else ref)
    cand = Path(raw)
    tries = [cand] if cand.is_absolute() else [base / cand]
    # Fuel models are inconsistent about where a mesh says its textures live: some point at a sibling
    # file, some at ../materials/textures/ (Jersey Barrier, Casual female).
    name = cand.name
    tries += [
        base / name,
        base.parent / "materials" / "textures" / name,
        base.parent / "textures" / name,
        base.parent / name,
    ]
    for t in tries:
        if t.is_file():
            return t.resolve()
    raise ValueError(
        f"texture {ref!r} referenced by a material under {base} was not found (tried: "
        + ", ".join(str(t) for t in tries)
        + ")"
    )


def _collada_materials(root, base: Path) -> dict[str, tuple[Path | None, list[float] | None]]:
    """``material id -> (diffuse texture path | None, diffuse rgba | None)``.

    Walks the standard chain material -> instance_effect -> effect -> newparam/surface -> image, and
    also accepts effects that name an image id straight in ``<texture texture=...>``.
    """
    images = {
        im.get("id"): t.strip()
        for im in root.iter()
        if _localname(im) == "image"
        for t in [next((c.text for c in im if _localname(c) == "init_from" and c.text), None)]
        if t
    }

    effects: dict[str, tuple[Path | None, list[float] | None]] = {}
    for eff in root.iter():
        if _localname(eff) != "effect":
            continue
        surfaces: dict[str, str] = {}
        samplers: dict[str, str] = {}
        for prm in eff.iter():
            if _localname(prm) != "newparam":
                continue
            sid = prm.get("sid")
            surf = next((c for c in prm if _localname(c) == "surface"), None)
            if surf is not None:
                ini = next((c.text for c in surf if _localname(c) == "init_from" and c.text), None)
                if ini:
                    surfaces[sid] = ini.strip()
            s2d = next((c for c in prm if _localname(c) == "sampler2D"), None)
            if s2d is not None:
                srcs = next((c.text for c in s2d if _localname(c) == "source" and c.text), None)
                if srcs:
                    samplers[sid] = srcs.strip()

        tex_ref, rgba = None, None
        for dif in eff.iter():
            if _localname(dif) != "diffuse":
                continue
            t = next((c for c in dif if _localname(c) == "texture"), None)
            if t is not None:
                tex_ref = t.get("texture")
            col = next((c for c in dif if _localname(c) == "color"), None)
            if col is not None and col.text:
                v = [float(x) for x in col.text.split()]
                rgba = (v + [1.0] * 4)[:4]
        img = None
        if tex_ref:
            sid = samplers.get(tex_ref, tex_ref)
            img = _resolve_image(images.get(surfaces.get(sid, sid), surfaces.get(sid, sid)), base)
        effects["#" + eff.get("id")] = (img, rgba)

    out: dict[str, tuple[Path | None, list[float] | None]] = {}
    for mat in root.iter():
        if _localname(mat) != "material":
            continue
        ie = next((c.get("url") for c in mat if _localname(c) == "instance_effect"), None)
        if ie in effects:
            out[mat.get("id")] = effects[ie]
    # <triangles material=SYMBOL> names a binding symbol, resolved by the visual scene's
    # <instance_material symbol=.. target=..>. Usually symbol == target id, but not by rule.
    for im in root.iter():
        if _localname(im) == "instance_material":
            tgt = (im.get("target") or "").lstrip("#")
            if tgt in out:
                out[im.get("symbol")] = out[tgt]
    return out


def read_collada(path: Path, submesh: str | None = None, center: bool = False) -> list[Submesh]:
    """Submeshes (one per material) **in metres, Z-up**, placed through the visual scene graph.

    *submesh* selects a single named node's subtree, mirroring SDF ``<mesh><submesh><name>``. Ignoring
    it is not a cosmetic loss: the Warehouse model draws its drop zone by re-using ``warehouse.dae``
    with ``<submesh>drop1</submesh>`` at z+0.101, so loading the whole file instead stacks a second
    copy of the entire building 10 cm above the first — whose floor then swallows the feet of every
    object standing on the real one. *center* re-origins that selection on its bounding-box centre, as
    ``<center>true</center>`` asks.

    Honours ``<triangles>``/``<polylist>`` input offsets so interleaved index streams de-interleave
    correctly, ``<up_axis>`` (Y_UP files are rotated to Z-up), and — critically —
    ``<asset><unit meter="..."/>``.

    That unit is not optional bookkeeping. Fuel ships a mix: a Gazebo ``Chair`` is authored in metres,
    while ``shelf_big``/``Jersey Barrier``/``foldable_chair`` are authored in centimetres and declare
    ``meter="0.01"``. Ignore it and those models come out exactly 100x too large — a 50 m shelf — while
    their neighbours look right, which is the failure mode that reads later as "the planner behaves
    strangely in scene 1" rather than "the importer was wrong".
    """
    root = etree.parse(str(path)).getroot()
    sources = _collada_sources(root)
    mats = _collada_materials(root, path.parent)

    geoms: dict[str, list[Submesh]] = {}
    for geo in _findall(root, "geometry"):
        gid = "#" + (geo.get("id") or "")
        mesh = next((c for c in geo if _localname(c) == "mesh"), None)
        if mesh is None:
            continue
        got = _parse_mesh(mesh, sources, mats)
        if got:
            geoms[gid] = got

    if not geoms:
        raise ValueError(f"no triangles found in {path}")

    # Instance the geometries through the visual scene graph. Skipping this is a real bug, not an
    # optimisation: Fuel DAEs routinely carry per-node scales (a shelf whose `boards` node scales x9),
    # so merging raw <library_geometries> yields a mesh that is wrong non-uniformly -- which looks
    # like a modelling quirk rather than an importer fault. Files whose nodes are all identity (e.g.
    # the husky meshes) are the lucky case, not the norm.
    parts: list[Submesh] = []
    for vscene in _findall(root, "visual_scene"):
        for node in [c for c in vscene if _localname(c) == "node"]:
            _walk_node(node, np.eye(4), geoms, parts, select=submesh, inside=submesh is None)
    if (
        not parts and submesh is None
    ):  # no visual scene: fall back to merging every geometry unplaced
        parts = [s for subs in geoms.values() for s in subs]
    if not parts:
        raise ValueError(
            f"<submesh> {submesh!r} not found in {path} -- it names a node of the visual scene; "
            f"available: {sorted(_node_names(root))}"
        )

    out = _finish_collada(root, _merge_by_material(parts))
    if center:
        # <center>true</center>: re-origin on the bbox centre of the SELECTION, so the SDF <pose> then
        # places that centre. Computed across all submeshes at once -- per-material would shear them.
        allv = np.vstack([s.verts for s in out])
        mid = (allv.min(axis=0) + allv.max(axis=0)) / 2.0
        for s in out:
            s.verts = s.verts - mid
    return out


def _node_names(root) -> set[str]:
    return {n.get("name") or n.get("id") for n in root.iter() if _localname(n) == "node"}


def _merge_by_material(parts: list[Submesh]) -> list[Submesh]:
    """Concatenate submeshes sharing a material. One geom per material beats one per triangle group
    (the warehouse instances `wall-material` four times); MuJoCo cost tracks geom count."""
    order: list[tuple] = []
    groups: dict[tuple, list[Submesh]] = {}
    for s in parts:
        key = (str(s.texture) if s.texture else None, tuple(s.rgba) if s.rgba else None)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)

    out: list[Submesh] = []
    for key in order:
        subs = groups[key]
        vs, fs, uvs, base = [], [], [], 0
        for s in subs:
            vs.append(s.verts)
            fs.append(s.faces + base)
            # A textured group must stay UV-complete: one untextured member would misalign the rest.
            uvs.append(s.uv if s.uv is not None else np.zeros((len(s.verts), 2)))
            base += len(s.verts)
        has_uv = any(s.uv is not None for s in subs)
        out.append(
            Submesh(
                verts=np.vstack(vs),
                faces=np.vstack(fs),
                uv=np.vstack(uvs) if has_uv else None,
                texture=subs[0].texture,
                rgba=subs[0].rgba,
            )
        )
    return out


def _parse_mesh(
    mesh,
    sources: dict[str, tuple[np.ndarray, int]],
    mats: dict[str, tuple[Path | None, list[float] | None]],
) -> list[Submesh]:
    """One <mesh> -> one Submesh per <triangles>/<polylist> group, in its own geometry frame."""
    out: list[Submesh] = []
    # <vertices id=..><input semantic="POSITION" source="#pos"/></vertices>
    vmap: dict[str, str] = {}
    for vert in [c for c in mesh if _localname(c) == "vertices"]:
        for inp in vert:
            if inp.get("semantic") == "POSITION":
                vmap["#" + vert.get("id")] = inp.get("source")

    for prim in [c for c in mesh if _localname(c) in ("triangles", "polylist")]:
        pos_off = uv_off = None
        pos_src = uv_src = None
        stride = 0
        for inp in [c for c in prim if _localname(c) == "input"]:
            off = int(inp.get("offset", 0))
            stride = max(stride, off + 1)
            sem = inp.get("semantic")
            if sem == "VERTEX":
                pos_off, pos_src = off, vmap.get(inp.get("source"), inp.get("source"))
            elif sem == "POSITION" and pos_off is None:
                pos_off, pos_src = off, inp.get("source")
            elif sem == "TEXCOORD" and uv_off is None:  # set 0 only; extra sets are lightmaps
                uv_off, uv_src = off, inp.get("source")
        p = next((c for c in prim if _localname(c) == "p"), None)
        if p is None or pos_src is None or pos_src not in sources or not p.text:
            continue

        idx = np.array([int(x) for x in p.text.split()], dtype=np.int64).reshape(-1, max(stride, 1))
        if _localname(prim) == "polylist":
            counts = next((c for c in prim if _localname(c) == "vcount"), None)
            if counts is not None and counts.text:
                vc = [int(x) for x in counts.text.split()]
                if any(c != 3 for c in vc):  # fan-triangulate n-gons
                    rows, off = [], 0
                    for c in vc:
                        for k in range(1, c - 1):
                            rows += [idx[off], idx[off + k], idx[off + k + 1]]
                        off += c
                    idx = np.array(rows, dtype=np.int64)

        pos = sources[pos_src][0].reshape(-1, 3)
        tex, rgba = mats.get(prim.get("material") or "", (None, None))
        if uv_src is not None and uv_src in sources:
            arr, ust = sources[uv_src]
            uvsrc = arr.reshape(-1, ust)[:, :2]
            # De-index: a corner is a (position, texcoord) pair, so positions shared across a UV seam
            # must be duplicated. MuJoCo takes one flat vertex array, not Collada's parallel streams.
            pairs = np.stack([idx[:, pos_off], idx[:, uv_off]], axis=1)
            uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
            verts, uv = pos[uniq[:, 0]], uvsrc[uniq[:, 1]]
            faces = inv.reshape(-1, 3)
        else:
            verts, uv = pos, None
            faces = idx[:, pos_off].reshape(-1, 3)
        out.append(Submesh(verts=verts, faces=faces, uv=uv, texture=tex, rgba=rgba))
    return out


def _node_matrix(node) -> np.ndarray:
    """Compose a Collada node's transform elements, in document order (matrix/translate/rotate/scale)."""
    m = np.eye(4)
    for c in node:
        t = _localname(c)
        txt = (c.text or "").split()
        if t == "matrix" and len(txt) >= 16:
            m = m @ np.array([float(x) for x in txt[:16]]).reshape(4, 4)  # Collada is row-major
        elif t == "translate" and len(txt) >= 3:
            tm = np.eye(4)
            tm[:3, 3] = [float(x) for x in txt[:3]]
            m = m @ tm
        elif t == "rotate" and len(txt) >= 4:
            ax = np.array([float(x) for x in txt[:3]], dtype=float)
            n = np.linalg.norm(ax)
            if n == 0:
                continue
            ax = ax / n
            a = math.radians(float(txt[3]))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            rm = np.eye(4)
            rm[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
            m = m @ rm
        elif t == "scale" and len(txt) >= 3:
            sm = np.eye(4)
            sm[[0, 1, 2], [0, 1, 2]] = [float(x) for x in txt[:3]]
            m = m @ sm
    return m


def _walk_node(
    node,
    parent: np.ndarray,
    geoms: dict[str, list[Submesh]],
    out: list,
    select: str | None = None,
    inside: bool = True,
) -> None:
    m = parent @ _node_matrix(node)
    # Ancestors' transforms still apply to a selected node, so the walk starts at the root either way
    # and only gates what it *emits*.
    inside = inside or select in (node.get("name"), node.get("id"))
    for c in node:
        t = _localname(c)
        if t == "instance_geometry" and inside:
            for s in geoms.get(c.get("url", ""), []):
                # Placement is per instance; UVs and material ride along untouched.
                out.append(
                    Submesh(
                        verts=transform(s.verts, m),
                        faces=s.faces,
                        uv=s.uv,
                        texture=s.texture,
                        rgba=s.rgba,
                    )
                )
        elif t == "node":
            _walk_node(c, m, geoms, out, select, inside)


def _finish_collada(root, subs: list[Submesh]) -> list[Submesh]:
    # <asset><unit meter="0.01"/> -- author units. Fuel mixes metre- and centimetre-authored models
    # (a Gazebo Chair in metres next to a Jersey Barrier in centimetres), so ignoring this yields
    # models exactly 100x too large sitting beside correct ones.
    unit = next((c for c in root.iter() if _localname(c) == "unit"), None)
    scale = float(unit.get("meter")) if unit is not None and unit.get("meter") else None

    up = next(
        (c.text for c in root.iter() if _localname(c) == "up_axis" and c.text), "Z_UP"
    ).strip()
    rot = (
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
        if up == "Y_UP"
        else None
    )

    for s in subs:
        if scale is not None:
            s.verts = s.verts * scale
        if rot is not None:  # rotate +90 about X so Y-up becomes Z-up
            s.verts = transform(s.verts, rot)
    return subs


def read_stl(path: Path) -> list[Submesh]:
    """STL carries no UVs or materials by format — a single untextured Submesh is the honest result."""
    data = path.read_bytes()
    is_ascii = data[:5].lower() == b"solid" and b"facet" in data[:512].lower()
    tris: list[list[float]] = []
    if is_ascii:
        for line in data.decode("utf-8", "replace").splitlines():
            s = line.strip().split()
            if len(s) == 4 and s[0] == "vertex":
                tris.append([float(s[1]), float(s[2]), float(s[3])])
    else:
        n = struct.unpack("<I", data[80:84])[0]
        for i in range(n):
            off = 84 + i * 50 + 12  # skip normal
            for k in range(3):
                tris.append(list(struct.unpack("<3f", data[off + k * 12 : off + k * 12 + 12])))
    verts = np.array(tris, dtype=np.float64)
    if len(verts) == 0:
        raise ValueError(f"no triangles in {path}")
    return [Submesh(verts=verts, faces=np.arange(len(verts), dtype=np.int64).reshape(-1, 3))]


def _read_mtl(path: Path) -> dict[str, tuple[Path | None, list[float] | None]]:
    """``newmtl`` name -> (map_Kd path, Kd rgba). Missing .mtl is fine (plain-geometry OBJ)."""
    out: dict[str, tuple[Path | None, list[float] | None]] = {}
    if not path.is_file():
        return out
    cur, tex, rgba = None, None, None

    def flush() -> None:
        if cur is not None:
            out[cur] = (tex, rgba)

    for line in path.read_text(errors="replace").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "newmtl":
            flush()
            cur, tex, rgba = s[1], None, None
        elif s[0] == "map_Kd" and len(s) > 1:
            # map_Kd may carry options (-s, -o ...) before the filename; the path is the last token.
            tex = _resolve_image(s[-1], path.parent)
        elif s[0] == "Kd" and len(s) >= 4:
            rgba = [float(s[1]), float(s[2]), float(s[3]), 1.0]
    flush()
    return out


def read_obj(path: Path) -> list[Submesh]:
    """One Submesh per ``usemtl`` group, with UVs de-indexed from ``v/vt`` corners."""
    pos: list[list[float]] = []
    uvs: list[list[float]] = []
    mats: dict[str, tuple[Path | None, list[float] | None]] = {}
    groups: dict[str, list[tuple[int, int]]] = {}  # material -> corner (pos_idx, uv_idx) triples
    cur = ""

    for line in path.read_text(errors="replace").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            pos.append([float(x) for x in s[1:4]])
        elif s[0] == "vt":
            uvs.append([float(x) for x in s[1:3]])
        elif s[0] == "mtllib":
            mats.update(_read_mtl(path.parent / " ".join(s[1:])))
        elif s[0] == "usemtl":
            cur = s[1]
        elif s[0] == "f":
            corners = []
            for tok in s[1:]:
                bits = tok.split("/")
                vi = int(bits[0])
                ti = int(bits[1]) if len(bits) > 1 and bits[1] else 0
                corners.append(
                    (
                        vi - 1 if vi > 0 else len(pos) + vi,
                        (ti - 1 if ti > 0 else len(uvs) + ti) if ti else -1,
                    )
                )
            for k in range(1, len(corners) - 1):  # fan-triangulate
                groups.setdefault(cur, []).extend([corners[0], corners[k], corners[k + 1]])

    if not groups:
        raise ValueError(f"no faces in {path}")

    apos = np.array(pos, dtype=np.float64)
    auv = np.array(uvs, dtype=np.float64) if uvs else None
    out: list[Submesh] = []
    for name, corners in groups.items():
        tex, rgba = mats.get(name, (None, None))
        arr = np.array(corners, dtype=np.int64)
        if auv is not None and (arr[:, 1] >= 0).all():
            uniq, inv = np.unique(arr, axis=0, return_inverse=True)
            out.append(
                Submesh(
                    verts=apos[uniq[:, 0]],
                    faces=inv.reshape(-1, 3),
                    uv=auv[uniq[:, 1]],
                    texture=tex,
                    rgba=rgba,
                )
            )
        else:
            out.append(Submesh(verts=apos, faces=arr[:, 0].reshape(-1, 3), texture=tex, rgba=rgba))
    return out


_READERS = {".dae": read_collada, ".stl": read_stl, ".obj": read_obj}


def read_mesh(path: Path, submesh: str | None = None, center: bool = False) -> list[Submesh]:
    fn = _READERS.get(path.suffix.lower())
    if fn is None:
        raise ValueError(
            f"unsupported mesh format {path.suffix} ({path}) -- add a reader, do not substitute"
        )
    if submesh is not None:
        if fn is not read_collada:
            # Silently loading the whole file would duplicate geometry the SDF asked to narrow.
            raise ValueError(
                f"<submesh>{submesh}</submesh> requested for {path.name}, but only Collada supports "
                f"named submesh selection here -- add it for {path.suffix} rather than ignoring it"
            )
        return read_collada(path, submesh=submesh, center=center)
    return fn(path)


# ---------------------------------------------------------------- primitives


def box(size: str) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = ([float(x) for x in size.split()] + [0, 0, 0])[:3]
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array(
        [[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=np.float64
    )
    f = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return v, f


# (normal, u, v) unit axes per box face, ordered so (u, v, normal) is right-handed -- i.e. the corner
# order below winds counter-clockwise seen from *outside* the box.
_BOX_FACES = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
)


def box_uv(size: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A box unwrapped per face: 24 corners (4 per face) with **UVs in metres** in the face plane.

    :func:`box` shares its 8 corners between three faces, which leaves the OBJ without texcoords --
    and MuJoCo then projects a texture along a single axis, smearing it into stripes across a floor
    or a wall. Unwrapping per face costs 16 extra vertices on a static slab and gives every face a
    true planar mapping. UVs are metric, so a material tiling at a real-world ``physical_size``
    scales them by ``1 / physical_size`` (see :class:`roqsim.textures.UVScaler`) and the tile lands at
    that size on every face, whatever the box measures.
    """
    half = np.array(([float(x) for x in size.split()] + [0, 0, 0])[:3], dtype=np.float64) / 2
    verts, uvs, faces = [], [], []
    for n, u, v in _BOX_FACES:
        n, u, v = (np.array(a, dtype=np.float64) for a in (n, u, v))
        hu, hv = float(np.abs(u) @ half), float(np.abs(v) @ half)
        base = len(verts)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(n * half + su * hu * u + sv * hv * v)
            uvs.append((su * hu, sv * hv))
        faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return (
        np.array(verts, dtype=np.float64),
        np.array(faces, dtype=np.int64),
        np.array(uvs, dtype=np.float64),
    )


def cylinder(radius: float, length: float, seg: int = 24) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    hz = length / 2
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    v = np.vstack(
        [
            np.hstack([ring, np.full((seg, 1), -hz)]),
            np.hstack([ring, np.full((seg, 1), hz)]),
            np.array([[0, 0, -hz], [0, 0, hz]]),
        ]
    )
    bc, tc = 2 * seg, 2 * seg + 1
    f = []
    for i in range(seg):
        j = (i + 1) % seg
        f += [[i, j, seg + j], [i, seg + j, seg + i], [bc, j, i], [tc, seg + i, seg + j]]
    return v, np.array(f, dtype=np.int64)


def sphere(radius: float, seg: int = 16) -> tuple[np.ndarray, np.ndarray]:
    us = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    vs = np.linspace(-math.pi / 2, math.pi / 2, seg // 2 + 1)
    v, f = [], []
    for vi in vs:
        for ui in us:
            v.append(
                [
                    radius * math.cos(vi) * math.cos(ui),
                    radius * math.cos(vi) * math.sin(ui),
                    radius * math.sin(vi),
                ]
            )
    rows = len(vs)
    for r in range(rows - 1):
        for c in range(seg):
            a, b = r * seg + c, r * seg + (c + 1) % seg
            d, e = (r + 1) * seg + c, (r + 1) * seg + (c + 1) % seg
            f += [[a, b, e], [a, e, d]]
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


# ---------------------------------------------------------------- writer


def split_components(
    verts: np.ndarray, faces: np.ndarray, weld_tol: float = 1e-5, uv: np.ndarray | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Split a mesh into connected components (welding coincident vertices first).

    **This is not an optimisation — it is what makes an imported scene collidable.** MuJoCo collides a
    mesh by its convex hull, so one merged building mesh hulls into a solid block that fills its own
    interior and swallows the robot. ``usd_to_scene.py`` never needs this because USD hands it separate
    prims (a USD lab import: 57 objects); SDF hands you one ``<visual>`` for an entire warehouse, so the split
    has to be recovered here. Each wall/slab then hulls into a sane solid on its own.

    Components are found over face adjacency after welding vertices that coincide within *weld_tol* —
    exported meshes usually duplicate vertices per-face, which would otherwise make every triangle its
    own component.
    """
    if len(faces) == 0:
        return []
    # Weld: map each vertex to a representative by quantised position.
    keys = np.round(verts / weld_tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)

    # Union-find over welded vertex ids, joined through each face.
    parent = np.arange(inv.max() + 1)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    wf = inv[faces]
    for a, b, c in wf:
        union(a, b)
        union(a, c)

    roots = np.array([find(a) for a in wf[:, 0]])
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for r in np.unique(roots):
        f = faces[roots == r]
        used = np.unique(f)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append((verts[used], remap[f], uv[used] if uv is not None else None))
    return out


def split_convex_parts(
    verts: np.ndarray, faces: np.ndarray, uv: np.ndarray | None = None, max_parts: int = 256
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Cut a mesh at its **reflex edges** into near-convex pieces.

    ``split_components`` is not enough on its own. MuJoCo collides a mesh by its convex hull, and a
    building's shell is a single *connected* component: the warehouse's four walls plus roof are 552
    welded triangles whose hull is an 18,750 m^3 solid block filling the room. A robot spawned inside
    then starts 8 cm deep in "solid" geometry and gets pushed down through the floor -- which reads as
    "the robot sinks into the floor" rather than "the wall mesh is a brick".

    The rule: two triangles sharing an edge belong to the same convex piece only if that edge is
    convex. For a closed shell with outward normals, every vertex of a convex solid lies on or behind
    each face plane, so ``dot(n1, v2 - p1) > 0`` marks the edge reflex. Cutting exactly there splits an
    assembly of boxes (walls, slabs, beams) back into boxes, whose hulls are then faithful. Genuinely
    convex geometry has no reflex edges and survives as one piece.

    *max_parts* is a loud tripwire, not a fallback: an organic mesh is reflex nearly everywhere and
    would shatter into per-triangle geoms, and silently keeping the swallowing hull instead is the
    bug this exists to prevent.
    """
    if len(faces) == 0:
        return []
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)

    # Edge key -> incident faces, over WELDED vertex ids so split triangles still register as adjacent.
    keys = np.round(verts / 1e-5).astype(np.int64)
    _, wid = np.unique(keys, axis=0, return_inverse=True)
    wf = wid[faces]
    edges: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(wf):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(fi)

    parent = np.arange(len(faces))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    tol = 1e-6
    for (u, v), fs in edges.items():
        if len(fs) != 2:
            continue  # boundary or non-manifold: cutting there is the safe read
        f1, f2 = fs
        # Vertices of each face that are not on the shared edge decide the dihedral's sense.
        o1 = [w for w in wf[f1] if w not in (u, v)]
        o2 = [w for w in wf[f2] if w not in (u, v)]
        if not o1 or not o2:
            continue
        p1, p2 = tri[f1][0], tri[f2][0]
        q2 = verts[faces[f2][[i for i, w in enumerate(wf[f2]) if w == o2[0]][0]]]
        q1 = verts[faces[f1][[i for i, w in enumerate(wf[f1]) if w == o1[0]][0]]]
        if np.dot(n[f1], q2 - p1) <= tol and np.dot(n[f2], q1 - p2) <= tol:
            union(f1, f2)

    roots = np.array([find(i) for i in range(len(faces))])
    uniq = np.unique(roots)
    if len(uniq) > max_parts:
        raise ValueError(
            f"reflex-edge split produced {len(uniq)} convex parts (> max_parts={max_parts}). This mesh "
            "is not an assembly of convex pieces; give it a real convex decomposition or a primitive "
            "collision shape rather than accepting a hull that swallows its own interior."
        )
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for r in uniq:
        f = faces[roots == r]
        used = np.unique(f)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append((verts[used], remap[f], uv[used] if uv is not None else None))
    return out


def write_obj(
    path: Path, verts: np.ndarray, faces: np.ndarray, uv: np.ndarray | None = None
) -> None:
    """Write world-space OBJ. With *uv*, emit ``vt`` and ``f v/vt`` corners: MuJoCo reads texcoords
    from the OBJ itself and ignores any ``.mtl``, so this is the only channel a texture's UVs survive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if uv is not None and len(uv) != len(verts):
        raise ValueError(f"uv/vertex count mismatch writing {path}: {len(uv)} vs {len(verts)}")
    with open(path, "w") as fh:
        fh.write("# generated by roqsim_scenes/tools (world-space, metres, Z-up)\n")
        for x, y, z in verts:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        if uv is not None:
            for u, v in uv:
                fh.write(f"vt {u:.6f} {v:.6f}\n")
            for a, b, c in faces + 1:
                fh.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        else:
            for a, b, c in faces + 1:
                fh.write(f"f {a} {b} {c}\n")
