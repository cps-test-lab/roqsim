"""Export a compiled MuJoCo world to a browser scene descriptor (scene.json + scene.bin).

Rather than authoring a cell twice (MJCF for physics, URDF for the web), this compiles the *same*
world the sim runs and writes what it contains as a compact descriptor that a small three.js loader
renders. Because the whole world is compiled, the conveyor, floorplan walls, furniture, robots and
pedestrians all export for free -- a URDF path could only ever show the arm.

What the scene contains is decided by :mod:`roqsim.scene_content`, which ``roqsim export gltf``
reads too; this module only spells it as ``roqsim.web_scene``. The source options and the compile
are :mod:`roqsim.scene_source`'s.

Usage::

    roqsim export web --world path/to/world.yaml --out site/scene/<name>/
    roqsim export web --mjcf  path/to/model.xml  --out /tmp/scene/

Output (all in ``--out``):
  - ``scene.json`` -- tree + joints + geoms + materials + mesh/texture index (offsets into scene.bin),
                      headed by ``format``/``version`` (:data:`FORMAT`, :data:`FORMAT_VERSION`) so a
                      reader can refuse a descriptor written to a contract it has not seen
  - ``scene.bin``  -- concatenated Float32/Uint32/Uint8 buffers referenced by byte offset + count
  - ``tex_<i>.png``-- one PNG per *image* texture: copied verbatim when the MJCF's recorded path
                      resolves, else re-encoded from the compiled pixels (a baked scene's paths are
                      relative to ``texturedir`` and so never resolve). Procedural textures have no
                      source image and are packed raw into scene.bin as a DataTexture instead

Deformable geometry travels as **skins**: a MuJoCo ``<skin>`` as itself, and a **flex** (a
``<flexcomp>``: a soft solid, a sheet, a cable) as a skin whose bones are the bodies its vertices
follow (``roqsim.flex_skin``). A viewer that animates skins from bone poses therefore replays a flex's
deformation from the run capture's pose tracks for those bodies, with no flex-specific code.

What is deliberately NOT exported: lighting (the browser keeps its own three.js lights) and
collision-only geoms (``geom_group == 3``). FK metadata (joint axis/anchor/qposadr) rides in
scene.json so the browser animates an arm from ``/joint_states``. Normals are left to the loader:
MuJoCo indexes them separately from vertices, and recomputing them keeps the payload small.
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
from PIL import Image

from . import exit_status, logging_setup
from .scene_content import Mesh, SceneContent, Texture, walk
from .scene_source import (
    add_manifest_option,
    add_source_options,
    add_world_options,
    compile_source,
    refuse_options_for_mjcf,
    write_manifest,
)

#: What ``scene.json`` declares itself to be. The other ``scene.json`` in this tree -- a
#: ``roqsim_scenes`` scene manifest, a bill of meshes and bounds -- shares the file name and nothing
#: else, and a reader given the wrong one otherwise finds out from a missing key deep in a loader.
FORMAT = "roqsim.web_scene"
#: The descriptor's format version. Bumped when a key changes MEANING, never when one is added: a
#: reader takes what it knows by name, so an additive key (``skins`` arrived that way) costs nothing,
#: while a changed one would be read with confidence and drawn wrong. A reader refuses a version
#: above the one it implements, and reads an absent stamp as version 1.
FORMAT_VERSION = 1


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


def _mesh_entry(mesh: Mesh, binw: _BinWriter) -> dict:
    entry = {
        "vert": binw.add(mesh.vert, np.float32),
        "index": binw.add(mesh.index, np.uint32),
    }
    if mesh.uv is not None:
        entry["uv"] = binw.add(mesh.uv, np.float32)
    return entry


def _skin_blocks(
    content: SceneContent, binw: _BinWriter, mesh_base: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Each skin as a ``THREE.SkinnedMesh``: ``(skins, geoms, meshes)`` to append to the descriptor.

    One ``meshes`` entry (bind-pose verts/faces/uv -- the browser re-skins live), one ``geoms`` entry
    (``body 0``, carrying ``mesh`` + ``skin`` indices) and one ``skins`` bind block carrying the bone
    body **names** (== the exported body names, so the loader binds to those nodes) and each bone's
    bind-pose world transform (-> three ``boneInverses``). A skin drawing a flex also carries a
    ``flex`` key naming it, which a viewer may ignore.
    """
    skins: list[dict] = []
    geoms: list[dict] = []
    meshes: list[dict] = []
    for skin in content.skins:
        mesh_entry = _mesh_entry(skin.mesh, binw)
        block = {} if skin.flex is None else {"flex": skin.flex}
        block.update(
            {
                "bones": skin.bones,
                "bindpos": skin.bindpos,
                "bindquat": skin.bindquat,
                "skinIndex": binw.add(skin.skin_index, np.uint16),
                "skinWeight": binw.add(skin.skin_weight, np.float32),
            }
        )
        skins.append(block)
        geoms.append(
            {
                "body": 0,  # skin verts + bind poses are world-frame; bind to the world body node
                "type": "mesh",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "size": [0.0, 0.0, 0.0],
                "matid": skin.matid,
                "rgba": skin.rgba,
                "mesh": mesh_base + len(meshes),
                "skin": len(skins) - 1,
            }
        )
        meshes.append(mesh_entry)
    return skins, geoms, meshes


def _write_texture(
    model: mujoco.MjModel,
    tex: Texture,
    out_name: str,
    out_dir: Path,
    binw: _BinWriter,
    max_dim: int,
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
      source PNG's rows) and it is what keeps the artifact small: a 2048x2048 RGB texture is 12 MB
      raw in ``scene.bin`` and a fraction of that as PNG.

    A **procedural** texture (builtin checker/gradient -- no recorded path, nothing to re-encode
    faithfully to) stays packed raw into ``scene.bin``, for the loader to upload as a DataTexture.
    Raw is also the path for an image texture whose channel count has no PNG equivalent.
    """
    dst = out_dir / out_name
    raw = tex.pixels(model).ravel()
    if tex.path:
        src = tex.source_file()
        if src is not None:
            if max_dim:
                _downscale_png(src, dst, max_dim)
            else:
                shutil.copyfile(src, dst)
            return {"file": dst.name}
        if _encode_png(raw, tex.width, tex.height, tex.channels, dst, max_dim):
            return {"file": dst.name}
    return {
        "raw": binw.add(raw, np.uint8),
        "width": tex.width,
        "height": tex.height,
        "channels": tex.channels,
    }


#: MuJoCo channel count -> Pillow mode. A count not listed here has no obvious PNG mode, so it takes
#: the raw path rather than being guessed at.
_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def _encode_png(raw, width: int, height: int, channels: int, dst: Path, max_dim: int) -> bool:
    """Write ``raw`` (row-major, ``channels`` bytes per pixel) to ``dst`` as PNG, capped at ``max_dim``.

    Returns False for a channel count with no PNG equivalent, so the caller packs the pixels into
    ``scene.bin`` instead.
    """
    mode = _PIL_MODE.get(channels)
    if mode is None:
        return False
    img = Image.frombytes(mode, (width, height), bytes(np.asarray(raw, dtype=np.uint8)))
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    img.save(dst)
    return True


def _downscale_png(src: Path, dst: Path, max_dim: int) -> None:
    """Write ``src`` to ``dst`` as PNG, downscaled to fit ``max_dim``.

    Web viewers don't need 8K character skins, so a cap keeps the committed artifact small.
    """
    with Image.open(src) as img:
        if max(img.size) <= max_dim:
            img.save(dst)
        else:
            scale = max_dim / max(img.size)
            img.resize((round(img.width * scale), round(img.height * scale))).save(dst)


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
    from ``data.qpos``. ``max_tex_dim`` caps image textures' longest side (0 disables). Web viewers
    don't need 8K character skins -- capping keeps the committed artifact small.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    content = walk(model, data, logger)
    binw = _BinWriter()

    geoms = [
        {
            "body": g.body,
            "type": g.type,
            "pos": g.pos,
            "quat": g.quat,
            "size": g.size,
            "matid": g.matid,
            "rgba": g.rgba,
            "mesh": g.mesh,
        }
        for g in content.geoms
    ]
    meshes = [_mesh_entry(m, binw) for m in content.meshes]
    skins, skin_geoms, skin_meshes = _skin_blocks(content, binw, len(meshes))
    meshes.extend(skin_meshes)
    geoms.extend(skin_geoms)

    materials = [
        {
            "rgba": m.rgba,
            "texture": m.texture,
            "texrepeat": m.texrepeat,
            "texuniform": m.texuniform,
        }
        for m in content.materials
    ]
    textures = [
        _write_texture(model, tex, f"tex_{new_id}.png", out_dir, binw, max_tex_dim)
        for new_id, tex in enumerate(content.textures)
    ]

    scene = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "up": "z",  # MuJoCo is Z-up (like ROS); the web wrapper group rotates it into three's Y-up
        "bodies": [
            {"name": b.name, "parent": b.parent, "pos": b.pos, "quat": b.quat}  # quat wxyz
            for b in content.bodies
        ],
        "joints": [
            {
                "name": j.name,
                "body": j.body,
                "type": j.type,
                "axis": j.axis,
                "pos": j.pos,
                "qposadr": j.qposadr,
            }
            for j in content.joints
        ],
        "initialJoints": content.initial_joints,
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
        "exported %d bodies, %d joints, %d geoms, %d meshes, %d skins (%d of them flexes), "
        "%d materials, %d textures -> %s (scene.bin %d KiB)",
        len(scene["bodies"]),
        len(scene["joints"]),
        len(geoms),
        len(meshes),
        len(skins),
        content.flex_count,
        len(materials),
        len(textures),
        out_dir,
        len(binw.bytes()) // 1024,
    )
    return scene


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export web",
        description="Export a compiled MuJoCo world to a browser scene descriptor.",
        epilog=exit_status.epilog(exit_status.BAD_INPUT),
    )
    add_source_options(parser)
    parser.add_argument("--out", required=True, help="output directory for scene.json/scene.bin")
    add_world_options(parser)
    parser.add_argument(
        "--max-tex-dim",
        type=int,
        default=2048,
        help="downscale image textures whose longest side exceeds this (0 disables). Default 2048 "
        "-- keeps 8K character skins from bloating the committed artifact.",
    )
    add_manifest_option(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    refuse_options_for_mjcf(parser, args)

    logging_setup.configure(verbose=args.verbose)
    logger = logging.getLogger("roqsim.export_web")

    model, data, view = compile_source(args, logger)
    export_scene(model, data, Path(args.out), logger, max_tex_dim=args.max_tex_dim, view=view)
    write_manifest(args, logger)
    return exit_status.OK


if __name__ == "__main__":
    sys.exit(main())
