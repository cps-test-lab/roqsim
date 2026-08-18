"""Bake an already-generated MJCF building into a packaged roqsim world.

Sibling of ``scene_to_mjcf.py``. That baker starts from a ``scene.json`` (the USD/SDF import path);
this one starts from an MJCF that an external exporter already produced -- a building shell that
ships its own semantic materials (glass, timber, concrete, terrazzo) and lighting baked into the XML.
We keep that look verbatim and only layer on what a runnable world needs and the source shell lacks:

- an invisible **ground-plane collider** (``friction=2``, group 3, name ``floor``) sized to the scene,
- optional **props** dropped in on the floor (e.g. the office table the UR10e mounts on),
- a self-contained ``assets/`` dir (every referenced mesh copied next to the XML, relative paths).

The floor/prop/relocation logic is reused from :mod:`scene_to_mjcf` so both bakers stay in sync. The
source MJCF's own bodies, materials, lights and ``<custom>`` metadata pass through untouched.

Usage::

    roqsim scenes mjcf-to-world \
        --mjcf path/to/building.xml \
        --prop path/to/assets/industrial_table.obj,12.9,10.4 \
        --out src/roqsim_scenes/worlds/mylab/mylab.xml

    python -m mujoco.viewer --mjcf worlds/mylab/mylab.xml
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

from .scene_to_mjcf import _add_ground_plane, _add_prop, _relocate_assets


def _load_spec(mjcf: str) -> mujoco.MjSpec:
    """Load the source MJCF and absolutise its mesh paths so :func:`_relocate_assets` can copy them.

    ``_relocate_assets`` copies each mesh by ``asset.file``; from_file leaves those relative to the
    XML's ``meshdir``, so resolve them here and clear ``meshdir`` (the paths are now absolute).
    """
    spec = mujoco.MjSpec.from_file(mjcf)
    meshdir = os.path.abspath(os.path.join(os.path.dirname(mjcf), spec.meshdir or "."))
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = os.path.join(meshdir, mesh.file)
    spec.meshdir = ""
    return spec


def _scene_bounds(spec: mujoco.MjSpec) -> tuple[list[float], list[float]]:
    """World-space AABB of the source shell (before we add the floor/props).

    ``_add_ground_plane`` centres and sizes the plane from a manifest's ``bounds_min``/``bounds_max``;
    the source MJCF has no such manifest, so derive it from the compiled geometry. Use each mesh's
    actual vertices, not ``geom_aabb`` -- that field is a padded broadphase box, ~1.5 m loose here.
    A building shell is all meshes; a primitive geom would be ignored (none ship in these exports).
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[i]
        adr = model.mesh_vertadr[mesh_id]
        vert = model.mesh_vert[adr : adr + model.mesh_vertnum[mesh_id]]
        rot = data.geom_xmat[i].reshape(3, 3)
        world = data.geom_xpos[i] + (rot @ vert.T).T
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return lo.tolist(), hi.tolist()


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--mjcf", required=True, help="source building MJCF (semantic materials + lights)"
    )
    ap.add_argument(
        "--out", required=True, help="output world MJCF path (assets/ created beside it)"
    )
    ap.add_argument("--prop", action="append", default=[], help="prop to add: 'PATH,X,Y[,YAW]'")
    ap.add_argument(
        "--ground-z", type=float, help="ground-plane height (default: scene's lowest point)"
    )
    ap.add_argument(
        "--no-ground-plane", action="store_true", help="skip the collidable ground plane"
    )
    ap.add_argument(
        "--headlight-ambient",
        type=float,
        default=0.3,
        help="ambient headlight fill to lift the source lights' shadows (0 to disable)",
    )
    ap.add_argument(
        "--shadow-lights",
        help="comma-separated light names that keep shadows; all others stop casting "
        "(each extra shadow-casting light is a full shadow-map pass -- ~2x render "
        "cost per light). Omit to leave the source lights untouched.",
    )
    args = ap.parse_args(argv)

    if not os.path.isfile(args.mjcf):
        sys.exit(f"source MJCF not found: {args.mjcf}")

    spec = _load_spec(os.path.abspath(args.mjcf))
    # The source shell bakes its own directional lights but no <visual>; a small ambient headlight fill
    # lifts the shadows the way the scene_to_mjcf baker did, without washing out the materials.
    if args.headlight_ambient > 0:
        spec.visual.headlight.ambient = [args.headlight_ambient] * 3
    if args.shadow_lights is not None:
        keep = {n for n in args.shadow_lights.split(",") if n}
        names = {light.name for light in spec.lights}
        missing = keep - names
        if missing:
            sys.exit(
                f"--shadow-lights names no such light: {', '.join(sorted(missing))} "
                f"(have: {', '.join(sorted(names))})"
            )
        for light in spec.lights:
            light.castshadow = light.name in keep
    lo, hi = _scene_bounds(spec)
    manifest = {"bounds_min": lo, "bounds_max": hi}
    config = {"ground_plane": not args.no_ground_plane}
    if args.ground_z is not None:
        config["ground_z"] = args.ground_z

    _add_ground_plane(spec, manifest, [0.0, 0.0, 0.0], config)
    for p in args.prop:
        print(f"+ prop {_add_prop(spec, p)}  <- {p}")

    out = os.path.abspath(args.out)
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
