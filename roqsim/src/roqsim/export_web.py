"""Export a compiled MuJoCo world to a browser scene descriptor (scene.json + scene.bin).

This is the build-time half of "render MuJoCo's geometry in the web UI" (see
``mujoco-web-render-plan.md``, direction B). Rather than authoring the cell twice (MJCF for physics,
URDF for the web), we compile the *same* world the sim runs and walk the resulting :class:`MjModel`,
emitting a compact descriptor a small three.js loader (``ts_web/src/lib/mujocoSceneLoader.ts``)
renders. Because we compile the whole world, the conveyor, floorplan walls, furniture, turtlebot and
pedestrian all export for free -- the old URDF path could only ever show the arm.

Usage::

    roqsim export web --world path/to/world.yaml --out ts_web/public/scene/<name>/
    roqsim export web --mjcf  path/to/model.xml  --out /tmp/scene/

Output (all in ``--out``):
  - ``scene.json`` -- tree + joints + geoms + materials + mesh/texture index (offsets into scene.bin)
  - ``scene.bin``  -- concatenated Float32/Uint32/Uint8 buffers referenced by byte offset + count
  - ``tex_<i>.png``-- one PNG per *image* texture: copied verbatim when the MJCF's recorded path
                      resolves, else re-encoded from the compiled pixels (a baked scene's paths are
                      relative to ``texturedir`` and so never resolve). Procedural textures have no
                      source image and are packed raw into scene.bin as a DataTexture instead

What we deliberately do NOT export: lighting (the browser keeps its own three.js lights) and
collision-only geoms (``geom_group == 3``). FK metadata (joint axis/anchor/qposadr) rides in
scene.json so the browser animates the arm from ``/joint_states`` exactly as the URDF loader did.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import logging_setup
from .config import (
    deep_merge,
    drop_transport_plugins,
    load_config,
    overrides_from_dotlist,
    overrides_from_files,
    world_sources,
)
from .engine import Engine

# MuJoCo joint types (mjtJoint) -> the string the web loader switches on.
_JOINT_TYPE = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}

# MuJoCo geom types (mjtGeom) we emit as primitives -> a name the loader builds three geometry from.
# Mesh geoms are handled separately (geom_dataid -> mesh buffer). Types not listed are skipped.
_GEOM_TYPE = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}

_COLLISION_GROUP = 3  # the repo convention: group-3 geoms are collision-only, never drawn
_TEXROLE_RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)


class _BinWriter:
    """Accumulates typed buffers into one ``scene.bin`` blob, 4-byte aligned.

    Each :meth:`add` returns ``{"off": byte_offset, "count": num_elements}``; the loader reads
    ``count`` elements of the field's known dtype (float32 for verts/normals/uv, uint32 for indices,
    uint8 for raw texture data) starting at ``off``. Alignment to 4 bytes keeps ``Float32Array`` /
    ``Uint32Array`` views valid after a uint8 (texture) append.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def add(self, arr: np.ndarray, dtype) -> dict:
        flat = np.ascontiguousarray(arr, dtype=dtype).ravel()
        while len(self._buf) % 4 != 0:
            self._buf.append(0)
        off = len(self._buf)
        self._buf.extend(flat.tobytes())
        return {"off": off, "count": int(flat.size)}

    def bytes(self) -> bytes:
        return bytes(self._buf)


def _model_string(model: mujoco.MjModel, adr: int) -> str | None:
    """Read a NUL-terminated string at ``adr`` in ``model.paths`` (the packed texture file paths)."""
    if adr < 0:
        return None
    paths = model.paths
    end = paths.find(b"\x00", adr)
    return paths[adr:end].decode() if end >= 0 else paths[adr:].decode()


def _free_body_poses(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, tuple[list, list]]:
    """Map each free-jointed body -> its initial (pos, quat) from ``data.qpos``.

    A free body's ``body_pos``/``body_quat`` is just a static offset (usually identity); its real
    initial placement lives in ``qpos`` (3 pos + 4 quat). We read the *configured* state
    (``data.qpos`` after ``setup()``, which applies each spawn plugin's initial pose) -- not
    ``model.qpos0``, which is the bare model default (often all zeros). Baking that in seats the
    turtlebot base, the belt package, etc. at their rest pose. Phase 2 will stream live poses.
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
    Reading the world transform seats the exported rest pose at the configured stance instead of the
    park pose -- which is what the browser shows before live bone poses stream in over ``/tf``. Mocap
    bodies are world children, so the world transform is also the local (parent-relative) transform.
    """
    out: dict[int, tuple[list, list]] = {}
    for i in range(model.nbody):
        if int(model.body_mocapid[i]) >= 0:
            out[i] = (data.xpos[i].tolist(), data.xquat[i].tolist())
    return out


def _export_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
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
        bodies.append(
            {
                "name": model.body(i).name,
                "parent": int(model.body_parentid[i]),
                "pos": pos,
                "quat": quat,  # wxyz, MuJoCo order
            }
        )
    return bodies


def _export_joints(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[list[dict], dict[str, float]]:
    """Return (joint FK metadata, {name: home value}) for hinge/slide joints from ``data.qpos``.

    The home map lets the browser show the rest pose (arm ``home:``) before any ``/joint_states``
    arrive -- the same role ``parseInitialJointPositions`` played for the URDF. We read the configured
    ``data.qpos`` (set by the spawn plugins during ``setup()``), not ``model.qpos0`` (the bare model
    default, usually zeros).
    """
    joints = []
    initial: dict[str, float] = {}
    for i in range(model.njnt):
        jtype = _JOINT_TYPE.get(int(model.jnt_type[i]), "unknown")
        name = model.joint(i).name
        qadr = int(model.jnt_qposadr[i])
        joints.append(
            {
                "name": name,
                "body": int(model.jnt_bodyid[i]),
                "type": jtype,
                "axis": model.jnt_axis[i].tolist(),
                "pos": model.jnt_pos[i].tolist(),
                "qposadr": qadr,
            }
        )
        if jtype in ("hinge", "slide"):
            initial[name] = float(data.qpos[qadr])
    return joints, initial


def _export_meshes(
    model: mujoco.MjModel, mesh_ids: list[int], binw: _BinWriter, logger: logging.Logger
) -> list[dict]:
    """Emit indexed geometry (positions + faces + UVs) for each mesh, in ``mesh_ids`` order.

    Face indices are mesh-local (verified against MuJoCo 3.10), so slicing ``mesh_vert`` /
    ``mesh_face`` per mesh yields a self-contained indexed ``BufferGeometry``. Normals are computed
    in the browser (``computeVertexNormals``) -- MuJoCo indexes normals separately from vertices, so
    recomputing avoids a vert/normal index mismatch and keeps the payload small.

    Texcoords need the opposite treatment: they must be emitted, because a texture atlas -- what
    every ``roqsim_assets`` prop and every baked scene mesh uses -- cannot be reconstructed from
    geometry, and the loader's triplanar fallback would sample arbitrary regions of it. MuJoCo
    indexes them OBJ-style (``mesh_facetexcoord``), which for some meshes is already per-vertex
    (``mesh_texcoordnum == mesh_vertnum`` with identical face indices, e.g. every baked depot mesh)
    and for others is not (the TurtleBot 4 body: 3995 texcoords over 2002 vertices). A GPU buffer
    needs one index, so the second case is re-indexed over the unique (vertex, texcoord) pairs the
    faces actually use -- the standard OBJ-to-GPU split, and far smaller than de-indexing outright.

    UVs are in MuJoCo's convention (v measured from the image's top row), the same convention the
    skin UVs use; the loader compensates with ``flipY = false``.
    """
    out = []
    for mid in mesh_ids:
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
                pairs = np.stack([faces.ravel(), ftex.ravel()], axis=1)
                unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
                verts = verts[unique[:, 0]]
                uv = texcoord[unique[:, 1]]
                faces = inverse.reshape(-1, 3)
                logger.debug(
                    "mesh %r: split %d vertices into %d to carry its %d texcoords",
                    model.mesh(mid).name,
                    vn,
                    len(unique),
                    tcn,
                )

        entry = {
            "vert": binw.add(verts, np.float32),
            "index": binw.add(faces, np.uint32),
        }
        if uv is not None:
            entry["uv"] = binw.add(uv, np.float32)
        out.append(entry)
    return out


def _export_skins(
    model: mujoco.MjModel, binw: _BinWriter, mesh_base: int, logger: logging.Logger
) -> tuple[list[dict], list[dict], list[dict]]:
    """Emit each drawn skin as a deformable ``THREE.SkinnedMesh``: geometry + a bind block.

    A MuJoCo ``<skin>`` (``model.skin_*``) is a mesh rigged to bones (bodies). ``_export_meshes`` only
    walks geoms, so skins are exported here separately. Returns ``(skins, geoms, meshes)`` to append to
    the descriptor: one ``meshes`` entry (bind-pose verts/faces/uv -- the browser re-skins live), one
    ``geoms`` entry (``body 0``, carrying ``mesh`` + ``skin`` indices), and one ``skins`` bind block.

    The bind block transposes MuJoCo's per-bone vertex lists (``skin_bonevert*``) into three's
    per-vertex ``skinIndex``/``skinWeight`` (<=4 bones/vertex, normalised), and carries the bone body
    **names** (== the exported body names, so the loader binds to those existing nodes) plus each bone's
    bind-pose world transform (``skin_bonebind{pos,quat}`` -> three ``boneInverses``).
    """
    skins: list[dict] = []
    geoms: list[dict] = []
    meshes: list[dict] = []
    over_cap = 0
    for i in range(model.nskin):
        if int(model.skin_group[i]) == _COLLISION_GROUP:
            continue
        va, vn = int(model.skin_vertadr[i]), int(model.skin_vertnum[i])
        fa, fn = int(model.skin_faceadr[i]), int(model.skin_facenum[i])
        # Bind-pose verts + mesh-local faces (verified skin-local, like mesh_face), UV per vertex.
        mesh_entry = {
            "vert": binw.add(model.skin_vert[va : va + vn], np.float32),
            "index": binw.add(model.skin_face[fa : fa + fn], np.uint32),
        }
        tca = int(model.skin_texcoordadr[i])
        if tca >= 0:  # skin_texcoord holds one (u, v) per vertex
            mesh_entry["uv"] = binw.add(model.skin_texcoord[tca : tca + vn], np.float32)

        # Transpose MuJoCo per-bone vert lists into three per-vertex (skinIndex, skinWeight) slots.
        ba, bn = int(model.skin_boneadr[i]), int(model.skin_bonenum[i])
        bones, bindpos, bindquat = [], [], []
        skin_index = np.zeros((vn, 4), np.uint16)
        skin_weight = np.zeros((vn, 4), np.float32)
        slot = np.zeros(vn, np.int32)  # next free (index, weight) slot per vertex
        for j in range(bn):
            bid = int(model.skin_bonebodyid[ba + j])
            bones.append(model.body(bid).name)
            bindpos.append(model.skin_bonebindpos[ba + j].tolist())  # world bind pose
            bindquat.append(model.skin_bonebindquat[ba + j].tolist())  # wxyz
            bva = int(model.skin_bonevertadr[ba + j])
            bvn = int(model.skin_bonevertnum[ba + j])
            vids = model.skin_bonevertid[bva : bva + bvn]
            wts = model.skin_bonevertweight[bva : bva + bvn]
            for vid, w in zip(vids.tolist(), wts.tolist(), strict=True):
                s = int(slot[vid])
                if s < 4:  # three's SkinnedMesh supports up to 4 bones per vertex
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
            {
                "bones": bones,
                "bindpos": bindpos,
                "bindquat": bindquat,
                "skinIndex": binw.add(skin_index, np.uint16),
                "skinWeight": binw.add(skin_weight, np.float32),
            }
        )
        geoms.append(
            {
                "body": 0,  # skin verts + bind poses are world-frame; bind to the world body node
                "type": "mesh",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "size": [0.0, 0.0, 0.0],
                "matid": int(model.skin_matid[i]),
                "rgba": model.skin_rgba[i].tolist(),
                "mesh": mesh_base + len(meshes),
                "skin": len(skins) - 1,
            }
        )
        meshes.append(mesh_entry)
    if over_cap:
        logger.warning(
            "skin export: %d bone-weight entries past the 4-bones/vertex cap were dropped "
            "(weights renormalised)",
            over_cap,
        )
    return skins, geoms, meshes


def _write_texture(
    model: mujoco.MjModel, tid: int, out_name: str, out_dir: Path, binw: _BinWriter, max_dim: int
) -> dict:
    """Emit one texture: a PNG file beside ``scene.json`` when it came from an image, else raw pixels.

    An **image** texture is one MuJoCo recorded a path for, and it ships as a PNG:

    * the path resolves on disk (an absolute ``file=``) -- copy those bytes verbatim, so the artifact
      carries the author's own encoding;
    * it does not resolve -- re-encode the compiled pixels from ``tex_data``. This is the path a
      **baked scene** takes, and it is the reason this branch exists: MuJoCo records ``tex_pathadr``
      as the path *written in the MJCF*, never resolved against ``<compiler texturedir>``, so
      ``depot.xml``'s ``file="ROOF_Albedo.png"`` is stored verbatim and the copy above can never fire
      for the scenes roqsim bakes. Re-encoding is equivalent (``tex_data`` is byte-identical to the
      source PNG's rows, verified against ``WALLS_Albedo.png``) and it is what keeps the artifact
      small: 14 textures at 2048x2048x3 are 176 MB of *uncompressed* RGB in ``scene.bin`` against
      ~11 MB of PNG for the whole scene.

    A **procedural** texture (builtin checker/gradient -- no recorded path, nothing to re-encode
    faithfully to) stays packed raw into ``scene.bin`` as before, for the loader to upload as a
    DataTexture. Raw is also the fallback for an image texture when Pillow is missing or the channel
    count has no PNG equivalent: large, but a correct export rather than a failed one.
    """
    h, w, nch = int(model.tex_height[tid]), int(model.tex_width[tid]), int(model.tex_nchannel[tid])
    src = _model_string(model, int(model.tex_pathadr[tid]))
    dst = out_dir / out_name
    adr = int(model.tex_adr[tid])
    raw = model.tex_data[adr : adr + h * w * nch]
    if src:
        if Path(src).is_file():
            if not (max_dim and _downscale_png(Path(src), dst, max_dim)):
                shutil.copyfile(src, dst)
            return {"file": dst.name}
        if _encode_png(raw, w, h, nch, dst, max_dim):
            return {"file": dst.name}
    return {"raw": binw.add(raw, np.uint8), "width": w, "height": h, "channels": nch}


#: MuJoCo channel count -> Pillow mode. A count not listed here has no obvious PNG mode, so it takes
#: the raw path rather than being guessed at.
_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def _encode_png(raw, width: int, height: int, channels: int, dst: Path, max_dim: int) -> bool:
    """Write ``raw`` (row-major, ``channels`` bytes per pixel) to ``dst`` as PNG, capped at ``max_dim``.

    Returns False when it cannot -- Pillow missing, or a channel count with no PNG equivalent -- so
    the caller falls back to packing the pixels into ``scene.bin``.
    """
    mode = _PIL_MODE.get(channels)
    if mode is None:
        return False
    try:
        from PIL import Image
    except ImportError:
        return False
    img = Image.frombytes(mode, (width, height), bytes(np.asarray(raw, dtype=np.uint8)))
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    img.save(dst)
    return True


def _downscale_png(src: Path, dst: Path, max_dim: int) -> bool:
    """Downscale ``src`` to fit ``max_dim`` and write to ``dst``; return False if Pillow is missing.

    Optional: without Pillow we just copy the source verbatim (caller falls back). Web viewers don't
    need 8K character skins, so a cap keeps the committed artifact small.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(src) as img:
        if max(img.size) <= max_dim:
            img.save(dst)
        else:
            scale = max_dim / max(img.size)
            img.resize((round(img.width * scale), round(img.height * scale))).save(dst)
    return True


def export_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    out_dir: Path,
    logger: logging.Logger,
    max_tex_dim: int = 2048,
    view: dict | None = None,
) -> dict:
    """Walk ``model`` and write ``scene.json`` + ``scene.bin`` + ``tex_*.png`` into ``out_dir``.

    ``data`` supplies the configured initial state (joint ``home`` pose, free-body placement) read
    from ``data.qpos``. ``max_tex_dim`` caps copied image textures' longest side (needs Pillow; 0
    disables). Web viewers don't need 8K character skins -- capping keeps the committed artifact small.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    binw = _BinWriter()

    # Geoms first: they tell us which meshes are actually drawn (so we skip unreferenced /
    # collision-only mesh data). Mesh ids are remapped to a dense 0..N-1 in reference order.
    used_meshes: dict[int, int] = {}
    geoms = []
    for g in range(model.ngeom):
        if int(model.geom_group[g]) == _COLLISION_GROUP:
            continue
        gtype = _GEOM_TYPE.get(int(model.geom_type[g]))
        if gtype is None:
            continue
        mesh_ref = None
        if gtype == "mesh":
            dataid = int(model.geom_dataid[g])
            if dataid < 0:
                continue
            mesh_ref = used_meshes.setdefault(dataid, len(used_meshes))
        matid = int(model.geom_matid[g])
        geoms.append(
            {
                "body": int(model.geom_bodyid[g]),
                "type": gtype,
                "pos": model.geom_pos[g].tolist(),
                "quat": model.geom_quat[g].tolist(),
                "size": model.geom_size[g].tolist(),
                "matid": matid,
                "rgba": model.geom_rgba[g].tolist(),
                "mesh": mesh_ref,
            }
        )

    mesh_ids = [mid for mid, _ in sorted(used_meshes.items(), key=lambda kv: kv[1])]
    meshes = _export_meshes(model, mesh_ids, binw, logger)

    # Skins (deformable character meshes, e.g. pedestrians) are rigged to bones, not walked as geoms;
    # export them as SkinnedMesh geoms whose bind data rides once in scene.bin (live bone poses stream).
    skins, skin_geoms, skin_meshes = _export_skins(model, binw, len(meshes), logger)
    meshes.extend(skin_meshes)
    geoms.extend(skin_geoms)

    # Materials carry only their RGB-role texture (the loader ignores normal/other roles). Textures
    # are pruned to those an RGB role actually references, remapped to a dense 0..N-1 -- this drops
    # normal maps and any unused textures (often the bulk of a character/robot's texture payload).
    used_tex: dict[int, int] = {}
    materials = []
    for i in range(model.nmat):
        rgb_tid = int(model.mat_texid[i, _TEXROLE_RGB])
        tex_ref = used_tex.setdefault(rgb_tid, len(used_tex)) if rgb_tid >= 0 else -1
        materials.append(
            {
                "rgba": model.mat_rgba[i].tolist(),
                "texture": tex_ref,
                "texrepeat": model.mat_texrepeat[i].tolist(),
                "texuniform": bool(model.mat_texuniform[i]),
            }
        )
    tex_ids = [tid for tid, _ in sorted(used_tex.items(), key=lambda kv: kv[1])]
    textures = [
        _write_texture(model, tid, f"tex_{new_id}.png", out_dir, binw, max_tex_dim)
        for new_id, tid in enumerate(tex_ids)
    ]
    joints, initial_joints = _export_joints(model, data)

    scene = {
        "up": "z",  # MuJoCo is Z-up (like ROS); the web wrapper group rotates it into three's Y-up
        "bodies": _export_bodies(model, data),
        "joints": joints,
        "initialJoints": initial_joints,
        "geoms": geoms,
        "meshes": meshes,
        "skins": skins,
        "materials": materials,
        "textures": textures,
    }
    # Bake the world's authored initial camera view (MuJoCo free camera) so any viewer that loads this
    # scene frames it the way the world author intended -- no per-deployment web config needed.
    if view:
        cam = {k: view[k] for k in ("lookat", "distance", "azimuth", "elevation") if k in view}
        if cam:
            scene["view"] = cam

    (out_dir / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    (out_dir / "scene.bin").write_bytes(binw.bytes())
    logger.info(
        "exported %d bodies, %d joints, %d geoms, %d meshes, %d skins, %d materials, %d textures -> %s "
        "(scene.bin %d KiB)",
        len(scene["bodies"]),
        len(scene["joints"]),
        len(geoms),
        len(meshes),
        len(skins),
        len(materials),
        len(textures),
        out_dir,
        len(binw.bytes()) // 1024,
    )
    return scene


def _compile_from_mjcf(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile a bare MJCF file directly (no plugins / world YAML). Initial state is the model default."""
    model = mujoco.MjSpec.from_file(str(path)).compile()
    return model, mujoco.MjData(model), {}


def _compile_from_world(
    world: str | Path,
    skip: set[str],
    overrides: dict,
    logger: logging.Logger,
    settle_steps: int = 0,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile a world YAML through the real build pipeline, so the export == the simulated scene.

    Transport plugins are dropped first -- a geometry export needs no bridge, and one may not even be
    loadable here (``roqsim.config.drop_transport_plugins``). ``skip`` then drops *further* plugins by
    ``name`` or plugin ref before the engine is built. Everything that contributes geometry
    (floorplan, spawn_robot, walker, conveyor, ...) stays. ``overrides``
    (from ``--set``) is deep-merged into the world first -- e.g. to supply a floorplan ``mesh`` that a
    scenario would normally inject at run time.

    ``setup()`` alone never runs ``on_reset``, so a mocap-driven walker's bones stay at their park
    pose (z=-50) and free bodies keep only their configure-time seating. We ``reset()`` (which runs
    every plugin's ``on_reset`` -- re-posing the walker's mocap and re-seating robot bases) then a
    second ``mj_forward`` to propagate the re-posed mocap into ``data.xpos``/``xquat`` (reset()'s own
    forward runs *before* the walker re-poses). ``settle_steps`` optionally steps physics a little
    further (e.g. to let a dropped free body settle) before the state is captured.
    """
    cfg = load_config(world, overrides or None)
    transport, unavailable = drop_transport_plugins(cfg)
    if transport:
        logger.info("skipping transport plugins: %s", ", ".join(transport))
    if unavailable:
        logger.warning(
            "skipping plugin(s) this environment cannot load: %s. They build no geometry, so the "
            "export is unaffected -- but check the spelling if you expected one.",
            ", ".join(unavailable),
        )
    if skip:
        kept = [p for p in cfg.plugins if p.ref not in skip and (p.name or "") not in skip]
        dropped = [p.name or p.ref for p in cfg.plugins if p not in kept]
        if dropped:
            logger.info("skipping plugins: %s", ", ".join(dropped))
        cfg.plugins = kept
    engine = Engine(cfg)
    engine.setup()  # build + compile + configure (each spawn plugin's initial pose applied)
    engine.reset()  # on_reset: re-pose mocap walkers, re-seat robot bases
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)  # propagate re-posed mocap into data.xpos
    for _ in range(max(0, settle_steps)):
        engine.step()
    return engine.ctx.model, engine.ctx.data, cfg.view


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export web",
        description="Export a compiled MuJoCo world to a browser scene descriptor.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--world", help="path to the world YAML (compiled via the plugin pipeline)")
    source.add_argument("--mjcf", help="path to a bare MJCF file (compiled directly)")
    parser.add_argument("--out", required=True, help="output directory for scene.json/scene.bin")
    parser.add_argument(
        "--skip-plugins",
        default="",
        help="comma-separated plugin names/refs to drop before compiling, on top of the "
        "transport/bridge plugins (which contribute no geometry and are always dropped)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="path.to.key=value",
        help="override a world value before compiling, e.g. "
        "--set plugins.floorplan.mesh=/abs/rooms.stl (repeatable)",
    )
    parser.add_argument(
        "--override",
        dest="override_files",
        action="append",
        default=[],
        metavar="FILE",
        help="a YAML file of world overrides -- the file spelling of --set, for anything "
        "structured enough that flattening it onto a command line loses it (repeatable; "
        "later files and --set win). The same flag, and the same loader, as `roqsim sim`: a "
        "campaign whose overrides are a nested tree (a list of obstacle instances, say) can "
        "hand this exporter exactly what it handed the run",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="after reset(), step physics this many times before capturing state (e.g. to let a "
        "dropped free body settle). Default 0 -- capture the reset pose.",
    )
    parser.add_argument(
        "--max-tex-dim",
        type=int,
        default=2048,
        help="downscale copied image textures whose longest side exceeds this (needs Pillow; "
        "0 disables). Default 2048 -- keeps 8K character skins from bloating the committed artifact.",
    )
    parser.add_argument(
        "--manifest",
        help='also write a JSON source manifest to this path: {"inputs": [<file>, ...]} '
        "listing every file the world is defined by (the YAML, its 'extends' ancestors, the "
        "MJCF and its mesh/texture assets). A build system caching this export re-runs it when "
        "one of them changes -- the leaf world alone is not enough, since an inherited scene or "
        "a replaced mesh changes the result without touching it.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(verbose=args.verbose)
    logger = logging.getLogger("roqsim.export_web")

    skip = {s.strip() for s in args.skip_plugins.split(",") if s.strip()}
    # Files first, then --set, so the two spell one thing and the flat one wins on a
    # collision -- identical to `roqsim sim`, because an export that resolved overrides
    # differently from the run would compile geometry the run never had.
    overrides = deep_merge(
        overrides_from_files(args.override_files), overrides_from_dotlist(args.overrides)
    )
    model, data, view = (
        _compile_from_mjcf(Path(args.mjcf))
        if args.mjcf
        else _compile_from_world(
            args.world, skip, overrides, logger, settle_steps=args.settle_steps
        )
    )
    export_scene(model, data, Path(args.out), logger, max_tex_dim=args.max_tex_dim, view=view)
    if args.manifest:
        sources = (
            [str(p) for p in world_sources(args.world)]
            if args.world
            else [str(Path(args.mjcf).resolve())]
        )
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump({"inputs": sources}, fh, indent=2)
        logger.info("wrote source manifest (%d files) to %s", len(sources), args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
