"""Bake an imported static scene (scene.json + meshes) into a plain MuJoCo MJCF world.

Stage 2 of the import pipeline (after ``usd_to_scene.py`` produces ``scene.json`` + per-object OBJs):
emits a self-contained ``<scene>.xml`` you load with any MuJoCo tool -- no roqsim runtime, no
plugin. It bakes what the old ``static_scene`` plugin did at runtime:

- one mesh geom per object (rendered from its true triangles, collided by its convex hull),
- per-object colours or textured materials (textures resolved via :mod:`roqsim.textures`, UVs scaled
  to honour ``physical_size`` -- MuJoCo ignores ``texrepeat`` on a UV'd mesh; a ``materials`` entry
  may also set ``reflectance`` and ``emission``),
- a ground plane + hemispherical light,
- optional extra props dropped in with ``--prop``.

The look/collision/lighting come from a ``scene.yaml`` next to ``scene.json`` (``--config`` to override).
All referenced meshes/textures are copied next to the XML into ``assets/`` (relative paths), so the
output dir is a portable world. No git.

Usage::

    roqsim scenes scene-to-mjcf --scene depot \
        --prop ../roqsim_assets/src/roqsim_assets/models/industrial_table/industrial_table.obj,12.9,10.4

    python -m mujoco.viewer --mjcf worlds/depot/depot.xml
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import mujoco
import yaml

from roqsim import surfaces
from roqsim.textures import UVScaler, resolve_texture, texture_manifest
from roqsim_scenes import scene_mesh_io as mio

_HERE = os.path.dirname(os.path.abspath(__file__))
# The bundled scenes ship inside the package, so resolve them through it rather than by walking up
# from this file: a relative path is only correct while the file stays where it was written.
_SCENES_DIR = os.path.join(os.path.dirname(_HERE), "scenes")

_DEFAULT_PHYSICAL_SIZE = 2.0
_LIGHT_DEFAULTS = {
    "height": 4.0,
    "diffuse": [0.4, 0.4, 0.4],
    "cutoff": 90.0,
    "fill": [0.3, 0.3, 0.3],
}


def _resolve_scene(scene: str) -> str:
    """A bundled scene name (``depot``) -> ``scenes/<name>/scene.json``; else a path to a scene.json."""
    bundled = os.path.join(_SCENES_DIR, scene, "scene.json")
    if os.path.isfile(bundled):
        return bundled
    path = os.path.abspath(scene)
    return os.path.join(path, "scene.json") if os.path.isdir(path) else path


def _load_config(scene_json: str, explicit: str | None) -> dict:
    """Load the generation config: ``--config`` if given, else ``scene.yaml`` beside scene.json, else {}."""
    path = explicit or os.path.join(os.path.dirname(scene_json), "scene.yaml")
    if os.path.isfile(path):
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _physical_size(entry: dict) -> float:
    png = resolve_texture(entry["texture"])
    return float(
        entry.get(
            "physical_size", texture_manifest(png).get("physical_size", _DEFAULT_PHYSICAL_SIZE)
        )
    )


def _make_material(
    spec: mujoco.MjSpec, idx: int, entry: dict, tex_names: dict[str, str]
) -> tuple[str, float]:
    """Create material ``scene_mat_{idx}`` (deduping textures by file); return (name, uv_scale)."""
    mat_name = f"scene_mat_{idx}"
    texture = entry.get("texture")
    if not texture:
        mat = spec.add_material()
        mat.name = mat_name
        mat.rgba = [float(c) for c in entry["rgba"]]
        if "reflectance" in entry:
            mat.reflectance = float(entry["reflectance"])
        if "emission" in entry:
            mat.emission = float(entry["emission"])
        return mat_name, 1.0

    png = str(resolve_texture(texture))
    if png not in tex_names:
        tex = spec.add_texture()
        tex.name = f"scene_tex_{len(tex_names)}"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.file = png
        tex_names[png] = tex.name

    size = _physical_size(entry)
    mat = spec.add_material()
    mat.name = mat_name
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex_names[png]
    mat.reflectance = float(entry.get("reflectance", texture_manifest(png).get("reflectance", 0.0)))
    mat.texuniform = (
        True  # scales UV-less / auto-projected meshes; a UV'd mesh is scaled via its UVs
    )
    mat.texrepeat = [1.0 / size, 1.0 / size]
    if entry.get("rgba") is not None:
        mat.rgba = [float(c) for c in entry["rgba"]]
    # Self-illumination. A ceiling soffit faces away from every light in the room and would otherwise
    # render near-black; a little emission is what makes it read as the lit concrete it is.
    if "emission" in entry:
        mat.emission = float(entry["emission"])
    return mat_name, (1.0 / size if size > 0 else 1.0)


def _scene_material(
    spec: mujoco.MjSpec,
    png: str,
    rgba: list[float] | None,
    tex_names: dict[str, str],
    mat_names: dict[str, str],
) -> str:
    """Material for a texture the *importer* pinned beside scene.json (``obj['texture']``), deduped by file.

    Distinct from :func:`_make_material`, which serves ``scene.yaml`` entries: those repaint UV-less
    geometry and are auto-projected (``texuniform``), whereas this mesh arrived with the author's own
    UVs. Projecting over them would ignore the unwrap and smear the atlas — so texrepeat stays 1:1.
    """
    if png not in tex_names:
        tex = spec.add_texture()
        tex.name = f"scene_tex_src_{len(tex_names)}"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.file = png
        tex_names[png] = tex.name
    if png not in mat_names:
        mat = spec.add_material()
        mat.name = f"scene_mat_src_{len(mat_names)}"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex_names[png]
        mat.texuniform = False
        mat.texrepeat = [1.0, 1.0]
        mat.rgba = [1.0, 1.0, 1.0, float(rgba[3])] if rgba else [1.0, 1.0, 1.0, 1.0]
        mat_names[png] = mat.name
    return mat_names[png]


def _bounds(manifest: dict, origin: list[float]) -> tuple[list[float], list[float]]:
    lo = [manifest["bounds_min"][i] + origin[i] for i in range(3)]
    hi = [manifest["bounds_max"][i] + origin[i] for i in range(3)]
    return lo, hi


#: How far under everything visible the drawn floor sits. Big enough that no depth buffer confuses it
#: with a scene's own floor (which is what made a single coplanar geom z-fight), small enough that the
#: step at the scene's edge is not a visible cliff.
_FLOOR_VISUAL_DROP = 0.002

#: Beyond this much drop the backdrop is far enough below the walkable floor to look wrong at the
#: scene's edge, so the bake says so instead of leaving it to be discovered by looking.
_FLOOR_VISUAL_DROP_WARN = 0.1

#: Default look of the drawn floor -- the same light-gray checker the floorplan plugin uses, so a baked
#: scene and a generated room put the same floor under the robot.
_FLOOR_DEFAULTS = {
    "rgb1": [0.85, 0.85, 0.85],
    "rgb2": [0.78, 0.78, 0.79],
    "reflectance": 0.2,
    "texture": None,
}


def _add_ground_plane(
    spec: mujoco.MjSpec, manifest: dict, origin: list[float], config: dict, meshdir: str = ""
) -> None:
    """The ground, as **two** geoms: one the robot stands on, one a viewer sees.

    They are separate because they answer to different constraints. The collider must sit exactly at
    the ground height; the visual must never hide the floor a scene brought of its own. One geom
    doing both is what made a drawn plane z-fight with a scene's own floor mesh across the whole room.
    """
    if not config.get("ground_plane", True):
        return
    lo, hi = _bounds(manifest, origin)
    # Precedence: explicit scene.yaml override -> the ground height the SOURCE world stated (importers
    # record it from the <plane> they skip) -> the scene's lowest point. That last one is only a guess,
    # and it is wrong for any scene with geometry below its walkable floor (an outdoor apron, a drain,
    # a loading dock): the floor lands low and everything standing on it appears sunk into it.
    ground_z = config.get("ground_z")
    if ground_z is None:
        ground_z = manifest.get("ground_z")
        if ground_z is not None:
            ground_z = float(ground_z) + origin[2]
    stated = ground_z is not None
    z = float(ground_z) if stated else lo[2]
    centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]
    size = [max((hi[0] - lo[0]) / 2, 1.0), max((hi[1] - lo[1]) / 2, 1.0), 0.05]

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [*centre, z]
    floor.size = size
    floor.rgba = [0.0, 0.0, 0.0, 0.0]  # the collider; `floor_visual` below is what anyone sees
    floor.group = 3
    floor.friction = [2.0, 0.005, 0.0001]

    if not stated:
        # Drawing a floor at a guessed height is worse than drawing none: everything standing on it
        # appears to hover or sink. Say so, though -- an unexplained void under the robot in the run
        # view is exactly the report this whole feature came from.
        print(
            "  note: this scene states no ground height, so no floor is DRAWN (it still collides).\n"
            "        The run view, `roqsim render` and the viewer will show the void under the robot.\n"
            "        Fix by setting `ground_z` in the scene config, or by recording it in the "
            "importer that wrote this scene.json."
        )
        return

    visual = spec.worldbody.add_geom()
    visual.name = "floor_visual"
    visual.type = mujoco.mjtGeom.mjGEOM_PLANE
    # Under EVERY renderable vertex, not a fixed drop below `z`: a scene whose own floor is modelled
    # below its stated ground height (a recessed slab, a floor with thickness downward) would be
    # covered by a plane at a fixed offset. Under the minimum it cannot be, whatever the scene's shape.
    visual.pos = [
        *centre,
        min(z, _lowest_renderable_z(manifest, origin, meshdir)) - _FLOOR_VISUAL_DROP,
    ]
    visual.size = size
    visual.contype = 0
    visual.conaffinity = 0
    visual.material = surfaces.surface_material(
        spec, "ground_tex", "ground_mat", config.get("floor") or {}, _FLOOR_DEFAULTS
    )
    drop = z - visual.pos[2]
    if drop > _FLOOR_VISUAL_DROP_WARN:
        print(
            f"  note: the drawn floor sits {drop:.2f} m below the ground height, because the scene has "
            f"geometry that far down. It shows only past the scene's own floor, so a large drop reads "
            f"as a step at the edge."
        )


def _lowest_renderable_z(manifest: dict, origin: list[float], meshdir: str) -> float:
    """The lowest point of anything a viewer *draws*, so the drawn floor can go under all of it.

    Renderable objects only. A collision-only part routinely reaches below the floor -- a wall's
    footing, a plinth -- and it cannot be covered up by definition, so letting it decide the height
    would drop the backdrop for nothing and put a visible step at the scene's edge.

    The manifest bounds are the fallback for a mesh that will not read, and only then: they cover every
    object including the ones that do not render, so using them unconditionally would reintroduce
    exactly the footing problem. Falling back keeps the guarantee that matters -- the floor lands lower
    than it needed to, never higher, so it can still hide nothing.
    """
    lowest = math.inf
    unreadable = False
    for obj in manifest["objects"]:
        if not obj.get("render", True):
            continue
        try:
            subs = mio.read_mesh(Path(meshdir) / obj["mesh"])
        except Exception:
            unreadable = True
            continue
        for sub in subs:
            if len(sub.verts):
                lowest = min(lowest, float(sub.verts[:, 2].min()) + origin[2])
    if unreadable or lowest is math.inf:
        lo, _ = _bounds(manifest, origin)
        return min(lowest, lo[2])
    return lowest


def _add_lights(spec: mujoco.MjSpec, manifest: dict, origin: list[float], config: dict) -> None:
    cfg = {**_LIGHT_DEFAULTS, **(config.get("light") or {})}
    fill = [float(v) for v in cfg["fill"]]
    if any(fill):
        spec.visual.headlight.ambient = fill
    lo, hi = _bounds(manifest, origin)
    light = spec.worldbody.add_light()
    light.pos = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + float(cfg["height"])]
    light.dir = [0, 0, -1]
    light.diffuse = [float(v) for v in cfg["diffuse"]]
    light.exponent = 0.0
    light.cutoff = float(cfg["cutoff"])


def _obj_footprint(path: str) -> tuple[float, float, float]:
    """(center_x, center_y, min_z) of an OBJ's vertices -- to centre a prop at (x, y) on the floor."""
    xs, ys, zs = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(zs)


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_") or "prop"


def _add_prop(spec: mujoco.MjSpec, spec_str: str) -> str:
    parts = spec_str.split(",")
    path = os.path.abspath(parts[0])
    x, y = float(parts[1]), float(parts[2])
    yaw = math.radians(float(parts[3])) if len(parts) > 3 else 0.0
    if not os.path.isfile(path):
        sys.exit(f"prop mesh not found: {path}")
    cx, cy, zmin = _obj_footprint(path)
    name = f"prop_{_slug(os.path.splitext(os.path.basename(path))[0])}"

    m = spec.add_mesh()
    m.name = name
    m.file = path
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # imported props can be non-watertight
    g = spec.worldbody.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = name
    g.pos = [x - cx, y - cy, -zmin]  # footprint centred at (x, y), base on the floor
    g.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    g.rgba = [0.62, 0.5, 0.38, 1.0]  # MuJoCo ignores OBJ .mtl; give the prop a neutral wood tone
    return name


def build_spec(
    scene_json: str, config: dict, props: list[str], uv_scaler: UVScaler
) -> mujoco.MjSpec:
    """Build the MjSpec for a scene + props. ``uv_scaler`` must outlive the caller's asset relocation."""
    with open(scene_json) as fh:
        manifest = json.load(fh)
    meshdir = os.path.dirname(scene_json)
    materials = config.get("materials") or []
    collide_scene = config.get("collision", "convex") == "convex"
    origin = [float(v) for v in config.get("origin", [0.0, 0.0, 0.0])]

    spec = mujoco.MjSpec()
    # Name the model after the scene so the viewer never shows MuJoCo's default "MuJoCo Model" title.
    spec.modelname = str(manifest.get("name") or "scene")
    tex_names: dict[str, str] = {}
    mat_names: dict[int, str] = {}
    uv_scales: dict[int, float] = {}
    src_mat_names: dict[str, str] = {}

    for obj in manifest["objects"]:
        name = f"scene_{obj['name']}"
        mesh_file = os.path.join(meshdir, obj["mesh"])
        mat_idx = next(
            (i for i, e in enumerate(materials) if fnmatch.fnmatch(obj["name"], e["match"])), None
        )
        if mat_idx is not None and mat_idx not in mat_names:
            mat_names[mat_idx], uv_scales[mat_idx] = _make_material(
                spec, mat_idx, materials[mat_idx], tex_names
            )
        if mat_idx is not None and uv_scales[mat_idx] != 1.0:
            mesh_file = uv_scaler.scaled(mesh_file, uv_scales[mat_idx], name)

        m = spec.add_mesh()
        m.name = name
        m.file = mesh_file
        # Scene geoms hang off worldbody and are static, so their inertia is never used -- but MuJoCo
        # still computes it, and refuses a zero-volume mesh. Splitting a visual per material isolates
        # exactly such meshes (a floor decal is a flat quad), so ask for shell inertia instead.
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        g = spec.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = name
        g.pos = origin
        if mat_idx is not None:
            g.material = mat_names[mat_idx]
        elif obj.get("texture"):
            # The importer carried the source model's own texture + UVs through. A scene.yaml `materials`
            # entry still wins above: that is the deliberate override, this is the faithful default.
            g.material = _scene_material(
                spec,
                os.path.join(meshdir, obj["texture"]),
                obj.get("rgba"),
                tex_names,
                src_mat_names,
            )
        else:
            g.rgba = [float(c) for c in obj.get("rgba", [0.7, 0.7, 0.72, 1.0])]
        if not (collide_scene and obj.get("collide", True)):
            g.contype = 0
            g.conaffinity = 0
        if not obj.get("render", True):
            # A collision stand-in with a distinct visual mesh: keep it in group 3 so it still
            # collides and can be toggled on for debugging, but does not box in the mesh it stands for.
            g.group = 3

    _add_ground_plane(spec, manifest, origin, config, meshdir)
    _add_lights(spec, manifest, origin, config)
    for p in props:
        print(f"+ prop {_add_prop(spec, p)}  <- {p}")
    return spec


def _relocate_assets(spec: mujoco.MjSpec, assets_dir: str) -> None:
    """Copy every referenced mesh/texture into ``assets_dir``, rewriting refs to relative basenames.

    ``to_xml()`` opens the files (relative to ``meshdir``) while serialising, so the dirs are set to the
    absolute ``assets_dir`` here and :func:`main` rewrites that prefix to a relative ``assets/``.
    """
    os.makedirs(assets_dir, exist_ok=True)
    for coll in (spec.meshes, spec.textures):
        for asset in coll:
            src = getattr(asset, "file", "") or ""
            if src:
                shutil.copyfile(src, os.path.join(assets_dir, os.path.basename(src)))
                asset.file = os.path.basename(src)
    spec.meshdir = assets_dir
    spec.texturedir = assets_dir


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="depot", help="bundled scene name or path to a scene.json")
    ap.add_argument(
        "--config", help="generation config YAML (default: scene.yaml beside scene.json)"
    )
    ap.add_argument("--out", help="output MJCF path (default: worlds/<scene>/<scene>.xml)")
    ap.add_argument("--prop", action="append", default=[], help="prop to add: 'PATH,X,Y[,YAW]'")
    args = ap.parse_args(argv)

    scene_json = _resolve_scene(args.scene)
    if not os.path.isfile(scene_json):
        sys.exit(f"scene.json not found: {scene_json}")
    config = _load_config(scene_json, args.config)
    name = os.path.basename(os.path.dirname(scene_json))
    out = os.path.abspath(
        args.out or os.path.join(os.path.dirname(_HERE), "worlds", name, f"{name}.xml")
    )

    uv_scaler = UVScaler(prefix="scene_to_mjcf_uv_")  # kept alive until the assets are copied
    spec = build_spec(scene_json, config, args.prop, uv_scaler)

    outdir = os.path.dirname(out)
    assets_dir = os.path.join(outdir, "assets")
    os.makedirs(outdir, exist_ok=True)
    _relocate_assets(spec, assets_dir)

    xml = spec.to_xml().replace(assets_dir, "assets")  # abs meshdir -> relative 'assets/'
    with open(out, "w") as fh:
        fh.write(xml)

    model = mujoco.MjModel.from_xml_path(out)  # verify it loads as a plain MJCF
    print(f"wrote {out}\n  loads OK: ngeom={model.ngeom} nmesh={model.nmesh} ntex={model.ntex}")


if __name__ == "__main__":
    main()
