"""Make a reduced OBJ prop MuJoCo-ready (step 3, after `roqsim assets reduce-mesh`).

MuJoCo can't use an OBJ's ``.mtl`` (a mesh geom takes a single material declared in the MJCF) and its
built-in texture loader **only reads PNG** -- so a Sketchfab prop with a JPEG ``baseColor`` renders
untextured. This finalizer fixes both, non-destructively re-usable on any prop folder:

1. transcode every non-PNG texture in the folder to PNG (drop the original);
2. rewrite the ``.mtl`` to reference the PNGs by correct relative path (for OBJ viewers/Blender);
3. write ``<name>.xml`` -- a self-contained MJCF that loads the mesh with an explicit
   ``<texture>``/``<material>`` so it renders textured in MuJoCo (or a plain neutral geom if the prop
   has no colour texture).

Usage::

    roqsim assets finalize-mujoco path/to/models/fire_extinguisher            # infer <name> from the .obj
    roqsim assets finalize-mujoco path/to/models/fire_extinguisher --scale 0.25   # bake a display scale

``--scale`` writes a ``scale`` on the ``<mesh>`` (leaves the OBJ untouched) -- handy to correct a prop
imported at the wrong size without re-running the whole pipeline. The pipeline itself bakes scale into
the geometry via ``sketchfab_helper import --scale``, so it calls this with the default 1.0.

Needs Pillow (already a dep of the preview step). No git.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Images MuJoCo can't load and we transcode to PNG. (.png passes through untouched.)
_RASTER_EXT = (".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".gif", ".webp")


def _find_obj(prop_dir: str, name: str) -> str:
    """The prop's OBJ: ``<name>.obj`` if present, else the sole ``.obj`` in the folder."""
    named = os.path.join(prop_dir, f"{name}.obj")
    if os.path.isfile(named):
        return named
    objs = [f for f in os.listdir(prop_dir) if f.lower().endswith(".obj")]
    if len(objs) != 1:
        sys.exit(f"expected exactly one .obj in {prop_dir} (found {objs or 'none'})")
    return os.path.join(prop_dir, objs[0])


def _transcode_textures(prop_dir: str) -> None:
    """Convert every non-PNG image under ``prop_dir`` to PNG in place, deleting the original."""
    from PIL import Image

    for dp, _, fs in os.walk(prop_dir):
        for f in fs:
            if not f.lower().endswith(_RASTER_EXT):
                continue
            src = os.path.join(dp, f)
            dst = os.path.splitext(src)[0] + ".png"
            img = Image.open(src)
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGB")  # e.g. CMYK JPEG / palettised
            img.save(dst, optimize=True)
            if os.path.abspath(dst) != os.path.abspath(src):
                os.remove(src)
            print(
                f"  transcoded {os.path.relpath(src, prop_dir)} -> {os.path.relpath(dst, prop_dir)}"
            )


def _repoint_gltf(gltf_path: str) -> None:
    """Rewrite a ``.gltf``'s image URIs from a non-PNG extension to ``.png`` (in step with
    :func:`_transcode_textures`), so re-importing it in the reduce loop still finds the textures."""
    if not gltf_path.lower().endswith(".gltf"):
        return  # a binary .glb embeds its images; nothing to repoint
    with open(gltf_path) as fh:
        data = json.load(fh)
    changed = False
    for img in data.get("images", []):
        uri = img.get("uri")
        if uri and uri.lower().endswith(_RASTER_EXT):
            img["uri"] = os.path.splitext(uri)[0] + ".png"
            changed = True
    if changed:
        with open(gltf_path, "w") as fh:
            json.dump(data, fh)


def pngify(prop_dir: str, gltf_path: str | None = None) -> None:
    """Convert every texture to PNG **once, up front on import** (MuJoCo loads only PNG), and repoint a
    sibling ``.gltf`` at the PNGs. Run before the reduce/preview loop so the loop, the preview, and the
    final asset all read PNGs -- no per-render transcoding anywhere downstream."""
    _transcode_textures(prop_dir)
    if gltf_path:
        _repoint_gltf(gltf_path)


def _image_index(prop_dir: str) -> dict[str, str]:
    """Map each image's lower-cased stem -> path relative to ``prop_dir`` (post-transcode)."""
    index: dict[str, str] = {}
    for dp, _, fs in os.walk(prop_dir):
        for f in fs:
            if f.lower().endswith(".png"):
                stem = os.path.splitext(f)[0].lower()
                index[stem] = os.path.relpath(os.path.join(dp, f), prop_dir)
    return index


def _resolve(ref: str, index: dict[str, str]) -> str | None:
    """Resolve an .mtl texture reference (any path/extension) to a folder-relative PNG path."""
    return index.get(os.path.splitext(os.path.basename(ref))[0].lower())


def _rewrite_mtl(mtl_path: str, index: dict[str, str]) -> str | None:
    """Fix every ``map_*`` path in the MTL to its PNG; return the first ``map_Kd`` (colour) PNG."""
    color: str | None = None
    out_lines: list[str] = []
    with open(mtl_path) as fh:
        for line in fh:
            toks = line.split()
            if toks and toks[0].startswith("map_"):
                png = _resolve(
                    toks[-1], index
                )  # last token is the filename (after any -bm/-o flags)
                if png:
                    toks[-1] = png
                    line = " ".join(toks) + "\n"
                    if toks[0] == "map_Kd" and color is None:
                        color = png
            out_lines.append(line)
    with open(mtl_path, "w") as fh:
        fh.writelines(out_lines)
    return color


def _write_xml(prop_dir: str, name: str, obj_rel: str, color_png: str | None, scale: float) -> str:
    """Write ``<name>.xml``: an MJCF loading the mesh, textured if a colour PNG was found."""
    scale_attr = f' scale="{scale} {scale} {scale}"' if scale != 1.0 else ""
    if color_png:
        asset = (
            f'    <texture name="{name}_color" type="2d" file="{color_png}"/>\n'
            f'    <material name="{name}" texture="{name}_color"/>\n'
            f'    <mesh name="{name}" file="{obj_rel}"{scale_attr}/>\n'
        )
        geom = f'      <geom type="mesh" mesh="{name}" material="{name}"/>\n'
        note = (
            "MuJoCo can't read OBJ .mtl textures and only loads PNG, so the colour map is declared\n"
            "       here explicitly. Load standalone or <include> it into a scene."
        )
    else:
        asset = f'    <mesh name="{name}" file="{obj_rel}"{scale_attr}/>\n'
        geom = f'      <geom type="mesh" mesh="{name}" rgba="0.7 0.7 0.72 1"/>\n'
        note = "This prop has no colour texture; the geom gets a neutral rgba."
    xml = (
        f'<mujoco model="{name}">\n'
        f"  <!-- {note} -->\n"
        f'  <compiler meshdir="." texturedir="."/>\n'
        f"  <asset>\n{asset}  </asset>\n"
        f"  <worldbody>\n"
        f'    <body name="{name}">\n{geom}    </body>\n'
        f"  </worldbody>\n"
        f"</mujoco>\n"
    )
    out = os.path.join(prop_dir, f"{name}.xml")
    with open(out, "w") as fh:
        fh.write(xml)
    return out


def finalize(prop_dir: str, name: str | None = None, scale: float = 1.0) -> str:
    """Transcode textures, fix the MTL, and emit ``<name>.xml``. Returns the XML path."""
    prop_dir = os.path.abspath(prop_dir)
    name = name or os.path.basename(prop_dir.rstrip("/"))
    obj = _find_obj(prop_dir, name)
    obj_rel = os.path.relpath(obj, prop_dir)

    _transcode_textures(prop_dir)
    index = _image_index(prop_dir)
    color_png: str | None = None
    for f in os.listdir(prop_dir):
        if f.lower().endswith(".mtl"):
            found = _rewrite_mtl(os.path.join(prop_dir, f), index)
            color_png = color_png or found
    xml = _write_xml(prop_dir, name, obj_rel, color_png, scale)
    tex = f"textured ({color_png})" if color_png else "no colour texture -> neutral geom"
    print(f"finalized {name}: {tex} -> {os.path.relpath(xml, prop_dir)}")
    return xml


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prop_dir", help="prop folder (containing the .obj/.mtl/textures)")
    ap.add_argument("--name", help="asset name (default: folder name)")
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="uniform display scale baked into the <mesh> (leaves the OBJ untouched)",
    )
    args = ap.parse_args(argv)
    finalize(args.prop_dir, args.name, args.scale)


if __name__ == "__main__":
    main()
