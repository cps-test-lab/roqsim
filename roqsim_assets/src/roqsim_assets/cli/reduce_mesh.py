"""Reduce a mesh's triangle count with Blender's Decimate and export a MuJoCo-friendly OBJ.

Step 2 of the prop pipeline, after ``sketchfab_helper download`` and before ``roqsim assets inspect-prop``
(look at the result with ``roqsim render out.obj --out check.png``). Imports glTF/GLB/OBJ/FBX/STL,
collapses to a target triangle budget (UVs/materials preserved), and writes a triangulated ``.obj``
(+ ``.mtl`` and textures) in MuJoCo's Z-up frame. No git.

**Dual-mode** so the Blender binary is a CLI parameter -- ``--blender`` defaults to ``blender`` on
``PATH`` (checked at startup) and accepts an explicit path otherwise:

    # plain Python: re-invokes Blender on itself (uses 'blender' on PATH by default)
    roqsim assets reduce-mesh in.gltf out.obj --target-faces 20000
    roqsim assets reduce-mesh --blender ~/blender/blender in.gltf out.obj   # explicit path

    # (equivalently, if you prefer to call Blender yourself -- point --python at THIS module's file)
    <blender> --background --python .../roqsim_assets/cli/reduce_mesh.py -- in.gltf out.obj --target-faces 20000

``--target-faces`` is a triangle budget; the collapse ratio is ``target / current`` (never upscales).
Use ``--scale`` if the source is not in metres (glTF is usually metres; many CAD exports are mm/cm).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

try:
    import bpy  # available only when running inside Blender

    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def _script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def blender_exe(path: str = "blender") -> str | None:
    """Resolve ``path`` to a working Blender binary (on PATH or absolute), or None if unavailable.

    Verifies it actually runs and reports as Blender, so a wrong/broken path fails fast at startup with
    a clear message rather than deep inside a subprocess.
    """
    exe = shutil.which(path)
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if out.returncode == 0 and "Blender" in (out.stdout + out.stderr) else None


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="source mesh (.gltf/.glb/.obj/.fbx/.stl/.dae)")
    ap.add_argument("output", help="output .obj path")
    ap.add_argument(
        "--target-faces", type=int, default=20000, help="triangle budget (default 20000)"
    )
    ap.add_argument(
        "--scale", type=float, default=1.0, help="uniform scale to metres (default 1.0)"
    )
    ap.add_argument(
        "--no-materials",
        action="store_true",
        help="omit the .mtl and its reference. MuJoCo ignores OBJ materials, so for a mesh whose "
        "colours are declared in the MJCF the sidecar is dead weight and its mtllib line is a "
        "dangling reference. Leave it off for props, whose textures finalize-mujoco consumes",
    )
    ap.add_argument(
        "--blender",
        default="blender",
        help="Blender binary: on PATH as 'blender' (default) or an explicit path",
    )
    return ap.parse_args(argv)


def _run_outside_blender(args: argparse.Namespace) -> None:
    exe = blender_exe(args.blender)
    if exe is None:
        sys.exit(
            f"Blender not available as {args.blender!r} -- put it on PATH as 'blender' or pass "
            f"--blender /path/to/blender"
        )
    passthrough = [
        args.input,
        args.output,
        "--target-faces",
        str(args.target_faces),
        "--scale",
        str(args.scale),
    ]
    if args.no_materials:
        passthrough.append("--no-materials")
    cmd = [exe, "--background", "--python", __file__, "--", *passthrough]
    print("+", " ".join(cmd))
    raise SystemExit(subprocess.run(cmd).returncode)


def _import(path: str) -> None:
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("gltf", "glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == "stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == "dae":
        # Collada, which is what ROS description packages ship. Note this importer HONOURS the file's
        # <up_axis>, unlike external/convert/dae2obj.py, which loads vertices verbatim because for the
        # arm meshes that tag is vestigial and obeying it wrecks them. So the two paths agree only on
        # Z_UP files -- check a converted mesh's bounding box against the source before trusting it.
        bpy.ops.wm.collada_import(filepath=path)
    else:
        sys.exit(f"unsupported input extension: .{ext}")


def _run_in_blender(args: argparse.Namespace) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import(args.input)

    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.polygons]
    if not meshes:
        sys.exit("no mesh geometry imported")

    # MuJoCo loads a single mesh per OBJ file (a multi-object OBJ loads as ~1 face), so join every
    # imported part into one mesh -- the prop becomes one MuJoCo mesh geom.
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    total = sum(len(p.vertices) - 2 for p in obj.data.polygons)  # triangles
    ratio = min(1.0, args.target_faces / total) if total else 1.0
    if args.scale != 1.0:
        obj.scale = (args.scale, args.scale, args.scale)
    if ratio < 1.0:
        mod = obj.modifiers.new("decimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio

    # Export ONLY the joined mesh. Selecting everything would drag along any polygon-less stray the
    # join skipped (a 2-vertex loose object, a leftover empty) as a second `o` block -- MuJoCo loads
    # just one mesh per OBJ, so that stray would silently drop most of the prop (single_mesh FAIL).
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=args.output,
        export_selected_objects=True,
        apply_modifiers=True,  # bakes the Decimate + scale
        export_materials=not args.no_materials,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",  # MuJoCo is Z-up
    )

    if ratio < 1.0:
        note = f"decimated to ratio {ratio:.3f} (~{round(100 * ratio)}%)"
    else:
        note = "already within budget -- not decimated"
    print(f"REDUCE_OK in_tris={total} target={args.target_faces}: {note} -> {args.output}")


def main(argv: list | None = None) -> None:
    # Inside Blender the arguments arrive after `--`, because Blender owns sys.argv. Outside it they
    # come from the caller: the `roqsim` command tree hands them over explicitly, and sys.argv there still
    # holds `assets reduce-mesh`, which reading sys.argv here would consume as `input` and `output`.
    if IN_BLENDER:
        _run_in_blender(_parse(_script_args()))
    else:
        _run_outside_blender(_parse(sys.argv[1:] if argv is None else list(argv)))


if __name__ == "__main__":
    main()
