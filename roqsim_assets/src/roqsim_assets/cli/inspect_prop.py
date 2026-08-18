"""Sanity-check an imported prop before it is added to the repo -- and mechanically fix its origin.

The prop pipeline (``sketchfab_helper import``) gives clean, reduced geometry but does **not** check that the
result is actually usable as a scene object. Two things routinely go wrong and are invisible until the
prop is placed:

* **Origin.** A Sketchfab model's local origin is wherever the author left it -- often metres away
  from the geometry. Placed at a body pose of ``(0,0,0)`` such a prop appears off to the side or
  buried in / floating above the floor. The substrate convention is **origin at the footprint centre,
  base on the floor**: ``x``/``y`` centred on 0 and ``min z == 0``, so ``<body pos="x y z">`` puts the
  prop exactly there, standing on the ground. ``--fix-origin`` bakes that in by translating the OBJ
  vertices.

  This is the check a *render* cannot make: MuJoCo internally recentres a mesh to its centre of mass at
  compile time, so anything reading the compiled model's ``mesh_vert`` sees a nicely centred bounding
  box even when the prop is metres off origin in the actual world.
  The recentre is cancelled by a compensating geom offset, so **world position == raw OBJ coords ×
  scale + geom pos**. This tool reads the raw OBJ (world truth), which is why it catches offsets the
  preview hides.
* **Textures.** MuJoCo binds a **single material per mesh**, so a multi-material prop (several
  ``usemtl`` / ``newmtl``) renders with one texture for the whole thing and the rest flat -- it cannot
  be fixed here, only flagged (prefer a single-material source). It also loads **PNG only**, and needs
  the textured ``<name>.xml`` that ``roqsim assets finalize-mujoco`` writes; a prop that was never finalized has no
  textures wired at all.

Also checks scale plausibility, up-axis (is it standing?), leftover download intermediates, and that a
redistributable ``CREDITS.txt`` is present.

Pure stdlib, no MuJoCo/GL -- geometry comes straight from the OBJ (honouring any ``scale`` on the
``<mesh>`` in the MJCF). Pair it with ``roqsim render <prop> --out check.png`` for the visual eyeball;
this is the deterministic ground truth the ``model-import`` skill drives.

Usage::

    roqsim assets inspect-prop path/to/models/free_chipboard_shelf            # report only
    roqsim assets inspect-prop path/to/models/free_chipboard_shelf --fix-origin   # ground + centre, then re-report
    roqsim assets inspect-prop path/to/models/free_chipboard_shelf --json     # machine-readable only

Exits non-zero if any check is FAIL (so a CI/agent step fails loudly on a bad prop).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

# Redistributable licence slugs (mirrors roqsim_assets.sketchfab._REDISTRIBUTABLE); anything else is a WARN.
_REDISTRIBUTABLE = {"cc0", "by", "by-sa"}
# Images MuJoCo cannot load (must be PNG). Mirrors finalize_mujoco._RASTER_EXT.
_NON_PNG_EXT = (".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".gif", ".webp")
_INTERMEDIATE_EXT = (".gltf", ".glb", ".bin")

# Grounding/centring tolerance: the larger of an absolute floor (2 cm) and 2% of the relevant extent,
# so a big prop isn't held to sub-millimetre precision and a tiny one isn't rejected for 2 cm.
_ABS_TOL = 0.02
_REL_TOL = 0.02
# Plausible real-world extent for a single scene prop (metres). Outside this, scale is probably wrong.
_MIN_DIM, _MAX_DIM = 0.02, 5.0


class Check:
    """One named verdict: status in {PASS, WARN, FAIL} plus a human message."""

    def __init__(self, name: str, status: str, msg: str):
        self.name, self.status, self.msg = name, status, msg

    def as_dict(self) -> dict:
        return {"check": self.name, "status": self.status, "detail": self.msg}


def _locate(target: str) -> tuple[str, str, str]:
    """Resolve the argument (a prop dir or an .obj) to (prop_dir, name, obj_path)."""
    target = os.path.abspath(target)
    if target.lower().endswith(".obj"):
        return os.path.dirname(target), os.path.splitext(os.path.basename(target))[0], target
    if not os.path.isdir(target):
        sys.exit(f"not a prop directory or .obj: {target}")
    name = os.path.basename(target.rstrip("/"))
    named = os.path.join(target, f"{name}.obj")
    if os.path.isfile(named):
        return target, name, named
    objs = [f for f in os.listdir(target) if f.lower().endswith(".obj")]
    if len(objs) != 1:
        sys.exit(f"expected exactly one .obj in {target} (found {objs or 'none'})")
    return target, os.path.splitext(objs[0])[0], os.path.join(target, objs[0])


def _obj_geometry(obj_path: str) -> dict:
    """Raw bounds + structure straight from the OBJ text (pre-scale, in the file's own units)."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    n_faces = n_objects = n_usemtl = 0
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                x, y, z = (float(t) for t in line.split()[1:4])
                for i, c in enumerate((x, y, z)):
                    lo[i], hi[i] = min(lo[i], c), max(hi[i], c)
            elif line.startswith("f "):
                n_faces += 1
            elif line.startswith("o "):
                n_objects += 1
            elif line.startswith("usemtl "):
                n_usemtl += 1
    if hi[0] < lo[0]:
        sys.exit(f"no vertices found in {obj_path}")
    return {"lo": lo, "hi": hi, "faces": n_faces, "objects": n_objects, "usemtl": n_usemtl}


def _mjcf_info(prop_dir: str, name: str) -> dict:
    """Read the finalized ``<name>.xml``: mesh scale + whether the geom is textured. Absent => not finalized."""
    xml = os.path.join(prop_dir, f"{name}.xml")
    if not os.path.isfile(xml):
        return {"exists": False, "scale": 1.0, "textured": False, "pos": [0.0, 0.0, 0.0]}
    root = ET.parse(xml).getroot()
    scale = 1.0
    mesh = root.find(".//asset/mesh")
    if mesh is not None and mesh.get("scale"):
        scale = float(mesh.get("scale").split()[0])  # uniform scale: first component
    geom = root.find(".//worldbody//geom")
    textured = geom is not None and geom.get("material") is not None
    pos = [0.0, 0.0, 0.0]
    if geom is not None and geom.get("pos"):
        pos = [float(t) for t in geom.get("pos").split()[:3]]
    return {"exists": True, "scale": scale, "textured": textured, "pos": pos}


def _license_slug(prop_dir: str) -> str | None:
    """Extract the licence slug from ``CREDITS.txt`` (the ``(slug)`` on the 'licensed under' line)."""
    credits = os.path.join(prop_dir, "CREDITS.txt")
    if not os.path.isfile(credits):
        return None
    with open(credits) as fh:
        for line in fh:
            m = re.search(r"\(([a-z0-9-]+)\)", line)
            if "licensed under" in line and m:
                return m.group(1)
    return None


def _mtl_has_map_kd(prop_dir: str) -> bool:
    """True if any ``.mtl`` in the folder declares a ``map_Kd`` (a colour texture the mesh will show).

    A Blender glTF->OBJ export can drop the PBR baseColour binding, leaving textures on disk that no
    material references -- the prop then finalizes untextured however many PNGs are present.
    """
    for f in os.listdir(prop_dir):
        if f.lower().endswith(".mtl"):
            with open(os.path.join(prop_dir, f)) as fh:
                if any(line.split()[:1] == ["map_Kd"] for line in fh if line.strip()):
                    return True
    return False


def _texture_files(prop_dir: str) -> tuple[list[str], list[str]]:
    """(non-PNG raster images, PNG images) anywhere under the prop folder."""
    non_png, png = [], []
    for dp, _, fs in os.walk(prop_dir):
        for f in fs:
            low = f.lower()
            if low.endswith(_NON_PNG_EXT):
                non_png.append(os.path.relpath(os.path.join(dp, f), prop_dir))
            elif low.endswith(".png") and not low.endswith(".thumb.png"):
                png.append(os.path.relpath(os.path.join(dp, f), prop_dir))
    return non_png, png


def _intermediates(prop_dir: str) -> list[str]:
    return [
        os.path.relpath(os.path.join(dp, f), prop_dir)
        for dp, _, fs in os.walk(prop_dir)
        for f in fs
        if f.lower().endswith(_INTERMEDIATE_EXT)
    ]


def _analyze(prop_dir: str, name: str, obj_path: str) -> tuple[list[Check], dict]:
    geo = _obj_geometry(obj_path)
    mj = _mjcf_info(prop_dir, name)
    scale = mj["scale"]
    # World bounds: raw OBJ x scale, plus any geom pos already baked into the MJCF (MuJoCo's internal
    # CoM recentre cancels out -- see the module docstring). This is the position the prop actually
    # renders at, which is what the origin/scale checks must judge.
    lo = [c * scale + mj["pos"][i] for i, c in enumerate(geo["lo"])]
    hi = [c * scale + mj["pos"][i] for i, c in enumerate(geo["hi"])]
    size = [hi[i] - lo[i] for i in range(3)]
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    checks: list[Check] = []

    # --- origin: footprint centred on x/y, base on the floor (min z == 0) ---
    # x/y off-centre is a real defect: the origin is off the geometry, so a scene placing the body over
    # a point puts the prop somewhere else (the shelf-at-y=-5.86 bug). => FAIL.
    # z-not-grounded is a *convention* question: an origin at the centroid (base = -height/2) is correct
    # for a wall/ceiling-mounted prop and wrong for floor-standing furniture -- the tool can't tell
    # which, so it WARNs and points at --fix-origin (which grounds it) rather than failing.
    # FAIL only when the origin is genuinely off the footprint (offset past half the half-extent, or
    # >15 cm) -- the "origin nowhere near the geometry" bug. A few-cm offset is cosmetic => WARN.
    z_tol = max(_ABS_TOL, _REL_TOL * size[2])
    warn_xy = max(_ABS_TOL, _REL_TOL * max(size[0], size[1]))
    fail_x = max(0.15, 0.25 * size[0])
    fail_y = max(0.15, 0.25 * size[1])
    off_center_fail = abs(center[0]) > fail_x or abs(center[1]) > fail_y
    off_center_warn = abs(center[0]) > warn_xy or abs(center[1]) > warn_xy
    off_ground = abs(lo[2]) > z_tol
    xy_msg = f"footprint centre at ({center[0]:+.3f}, {center[1]:+.3f}) m, not (0,0)"
    if off_center_fail:
        checks.append(
            Check("origin", "FAIL", xy_msg + " -- origin is off the geometry; run --fix-origin")
        )
    elif off_center_warn:
        checks.append(Check("origin", "WARN", xy_msg + " -- run --fix-origin to centre it"))
    elif off_ground:
        where = "below" if lo[2] < 0 else "above"
        checks.append(
            Check(
                "origin",
                "WARN",
                f"base {where} floor at z={lo[2]:+.3f} m (origin at centroid) -- fine for a "
                f"wall/ceiling mount; run --fix-origin for a floor-standing prop",
            )
        )
    else:
        checks.append(Check("origin", "PASS", "footprint centred on (0,0), base on floor"))

    # --- scale plausibility (can't know intent; flag physically unlikely extents) ---
    bad = [f"{'xyz'[i]}={size[i]:.3f}" for i in range(3) if not (_MIN_DIM <= size[i] <= _MAX_DIM)]
    dims = f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m (w x d x h)"
    if bad:
        checks.append(
            Check("scale", "WARN", f"{dims}; implausible: {', '.join(bad)} -- check --scale/units")
        )
    else:
        checks.append(Check("scale", "PASS", dims))

    # --- up-axis: tallest extent should be z for a standing prop (heuristic; a rug/table is a valid WARN) ---
    tallest = max(range(3), key=lambda i: size[i])
    if tallest != 2:
        checks.append(
            Check(
                "upright",
                "WARN",
                f"tallest axis is {'xyz'[tallest]}, not z -- lying down / wrong up-axis?",
            )
        )
    else:
        checks.append(Check("upright", "PASS", "tallest extent is z"))

    # --- single mesh: MuJoCo loads one mesh per OBJ; multiple objects => only ~one face survives ---
    if geo["objects"] > 1:
        checks.append(
            Check(
                "single_mesh",
                "FAIL",
                f"{geo['objects']} objects in OBJ -- MuJoCo loads one; re-run reduce_mesh (it joins)",
            )
        )
    else:
        checks.append(Check("single_mesh", "PASS", f"{geo['faces']} faces, one object"))

    # --- textures: finalized? PNG-only? single material? ---
    non_png, png = _texture_files(prop_dir)
    if not mj["exists"]:
        checks.append(
            Check(
                "finalized",
                "FAIL",
                "no <name>.xml -- run `roqsim assets finalize-mujoco` to wire textures + MJCF",
            )
        )
    elif not mj["textured"] and png:
        checks.append(
            Check(
                "finalized",
                "WARN",
                "MJCF present but geom has no material though a texture PNG exists",
            )
        )
    else:
        checks.append(
            Check(
                "finalized",
                "PASS",
                "textured MJCF present" if mj["textured"] else "MJCF present (no colour texture)",
            )
        )

    if non_png:
        checks.append(
            Check(
                "png_only",
                "WARN",
                f"non-PNG textures (MuJoCo can't load): {', '.join(non_png)} -- finalize transcodes",
            )
        )
    # Textures on disk that no material references: the export lost the binding, so the prop finalizes
    # untextured no matter how many PNGs sit in textures/ -- the "textures are missing" symptom.
    if png and not _mtl_has_map_kd(prop_dir):
        checks.append(
            Check(
                "texture_bound",
                "WARN",
                f"{len(png)} texture PNG(s) on disk but no map_Kd in the MTL -- the OBJ export "
                f"dropped the binding; prop will render untextured. Wire map_Kd or re-source",
            )
        )
    if geo["usemtl"] > 1:
        checks.append(
            Check(
                "single_material",
                "WARN",
                f"{geo['usemtl']} materials -- MuJoCo binds one per mesh, so only one is textured "
                f"and other parts render flat; accept it, or prefer a single-material source",
            )
        )
    else:
        checks.append(Check("single_material", "PASS", "one material"))

    # --- licence + intermediates ---
    slug = _license_slug(prop_dir)
    if slug is None:
        checks.append(
            Check(
                "license",
                "FAIL",
                "no CREDITS.txt / licence slug -- provenance missing, do not commit",
            )
        )
    elif slug not in _REDISTRIBUTABLE:
        checks.append(
            Check(
                "license",
                "WARN",
                f"licence '{slug}' not in {sorted(_REDISTRIBUTABLE)} -- local-only, don't commit",
            )
        )
    else:
        checks.append(Check("license", "PASS", f"licence '{slug}' redistributable"))

    leftovers = _intermediates(prop_dir)
    if leftovers:
        checks.append(
            Check(
                "clean",
                "WARN",
                f"leftover download intermediates: {', '.join(leftovers)} -- delete before commit",
            )
        )
    else:
        checks.append(Check("clean", "PASS", "no leftover intermediates"))

    summary = {
        "name": name,
        "scale": scale,
        "bounds_min": [round(c, 4) for c in lo],
        "bounds_max": [round(c, 4) for c in hi],
        "size": [round(c, 4) for c in size],
        "center": [round(c, 4) for c in center],
        "faces": geo["faces"],
        "objects": geo["objects"],
        "materials": geo["usemtl"],
    }
    return checks, summary


def _fix_origin(obj_path: str) -> tuple[list[float], list[float]]:
    """Translate the OBJ's vertices so footprint centre -> (0,0) and base -> z=0. Returns the applied offset.

    Edits only ``v`` lines (vertex positions); normals (``vn``) and UVs (``vt``) are unaffected by a pure
    translation. Idempotent: a re-run on an already-grounded prop applies a ~zero offset.
    """
    geo = _obj_geometry(obj_path)
    lo, hi = geo["lo"], geo["hi"]
    offset = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]  # centre x/y, ground z
    out_lines: list[str] = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = (float(parts[i]) + offset[i - 1] for i in range(1, 4))
                rest = (
                    " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                )  # keep optional vertex colour/w
                out_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}{rest}\n")
            else:
                out_lines.append(line)
    with open(obj_path, "w") as fh:
        fh.writelines(out_lines)
    return offset, [hi[i] - lo[i] for i in range(3)]


_STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2}


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prop", help="prop directory (models/<name>/) or a .obj path")
    ap.add_argument(
        "--fix-origin",
        action="store_true",
        help="translate the OBJ so footprint centre -> (0,0) and base -> z=0, then re-report",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args(argv)

    prop_dir, name, obj_path = _locate(args.prop)

    if args.fix_origin:
        # The fix bakes the origin into the geometry and assumes the geom sits at pos 0. A prop that
        # already carries a manual geom pos would end up offset by it -- refuse rather than half-fix.
        pos = _mjcf_info(prop_dir, name)["pos"]
        if any(abs(c) > 1e-9 for c in pos):
            sys.exit(
                f"{name}.xml already has a geom pos {pos}; --fix-origin bakes the origin into the "
                f"OBJ and expects pos 0. Remove the pos (or re-finalize) first."
            )
        offset, _ = _fix_origin(obj_path)
        if not args.json:
            print(
                f"fixed origin: translated OBJ by ({offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f}) m\n"
            )

    checks, summary = _analyze(prop_dir, name, obj_path)
    worst = min((_STATUS_ORDER[c.status] for c in checks), default=2)

    if args.json:
        print(json.dumps({"summary": summary, "checks": [c.as_dict() for c in checks]}, indent=2))
    else:
        print(
            f"prop: {name}   size {summary['size']} m   faces {summary['faces']}   materials {summary['materials']}"
        )
        for c in sorted(checks, key=lambda c: _STATUS_ORDER[c.status]):
            mark = {"PASS": "ok  ", "WARN": "WARN", "FAIL": "FAIL"}[c.status]
            print(f"  [{mark}] {c.name}: {c.msg}")

    sys.exit(1 if worst == _STATUS_ORDER["FAIL"] else 0)


if __name__ == "__main__":
    main()
