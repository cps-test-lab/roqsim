#!/usr/bin/env python3
"""Build roqsim's UFACTORY xArm 7 MJCF from MuJoCo Menagerie's ``ufactory_xarm7``.

Menagerie already ships a curated, inertially sound xArm 7, so this converter does not author a
model -- it *transforms* one, and every mass, inertia tensor, joint limit, link offset and actuator
gain below is passed through from upstream verbatim. Generating rather than hand-editing is what
keeps that claim checkable: re-run this script against the pinned commit and the result must be
byte-identical to what is committed.

Four deltas are applied, each a roqsim convention rather than a fix to upstream:

1. **The arm-with-hand variant is not used.** ``xarm7_nohand.xml`` is the source; the UFACTORY
   gripper (``hand.xml``) is a mimic-driven parallel gripper and belongs in its own model beside
   ``robotiq_2f85`` / ``schunk_pg70``, verified against the gripper battery. An arm here carries an
   ``attachment_site`` and nothing distal of it, exactly as ``ur10e`` does.
2. **No ``<option>``.** Upstream pins ``integrator="implicitfast"``. Timestep, integrator, solver and
   contact overrides are properties of the EXPERIMENT and live in the world YAML's ``sim:`` block --
   a model that pinned them would silently override every world it is spawned into.
3. **Base neutralised to the origin.** Upstream offsets ``link_base`` by ``z=0.12`` to stand it on a
   plinth in its own ``scene.xml``. The ``spawn_arm`` plugin places the arm via an attach frame, so
   scene placement belongs in the world, not baked into the model (same edit ``ur10e`` carries).
4. **Visual/collision split.** Upstream uses one mesh geom per link for both roles, so MuJoCo
   collides against nine convex hulls of full-detail CAD. Every sibling arm in
   ``roqsim_manipulation_assets`` instead separates a visual class (``contype=0``, group 2) from a
   collision class of primitives (group 3), which is both faster and more stable at campaign
   timestep. The primitives are FITTED HERE from the actual mesh vertices rather than typed in by
   hand -- see ``fit_collision_primitive``.

Usage::

    python external/convert/build_xarm7_mjcf.py           # fetch, build, write into the package
    python external/convert/build_xarm7_mjcf.py --check   # rebuild and diff against what is committed
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Pinned. If this moves, the port log and roqsim_manipulation_assets/THIRD_PARTY.md move with it --
# a converter that quietly builds from another revision makes the port log lie about its provenance.
MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
MENAGERIE_SUBDIR = "ufactory_xarm7"

# external/ is a sibling of the packages, so anchor back through it rather than assuming a CWD.
PKG = (
    Path(__file__).resolve().parents[2]
    / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/xarm7"
)

MESHES = [
    "link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "end_tool",
]

# A capsule is the right primitive for a limb and the wrong one for a hub. Fit a capsule only when
# the link is meaningfully longer than it is wide; otherwise a box bounds a stubby link far more
# tightly than a capsule swollen to contain it.
CAPSULE_ASPECT = 1.3
# CALIBRATED, not a fudge -- a capsule's circular cross-section is a poor envelope for the xArm's
# prismatic links, so the radius that merely bounds the widest point over-reports self-collision
# badly. Measured self-collision rate over 3000 uniformly sampled joint configurations:
#
#     upstream, full-detail convex hulls   8.0%     (the ground truth this approximates)
#     RADIUS_SCALE 1.00                   35.3%
#     RADIUS_SCALE 0.93                   21.1%
#     RADIUS_SCALE 0.86                   16.8%     <- chosen
#
# 0.86 puts this arm level with the package's own reference arms (ur10e 17.4%, ur5e 17.5%) measured
# the same way, i.e. at the conservatism the substrate already accepts, rather than at some rate
# invented here. SENSITIVITY: high for anything that reasons about self-collision -- a motion planner
# will refuse configurations that upstream's hulls allow. Re-measure if the meshes change. A tighter
# envelope needs several primitives per link (ur10e uses two on its long links), not a smaller scale:
# a single axis-aligned box was tried and is worse (30.2%), because these links are bent.
RADIUS_SCALE = 0.86

AXIS_QUAT = {0: "0.7071068 0 0.7071068 0", 1: "0.7071068 -0.7071068 0 0", 2: None}


def fit_collision_primitive(verts: np.ndarray) -> dict[str, str]:
    """Fit one collision primitive to a link's mesh vertices, in the body frame.

    Returns the MJCF geom attributes. Bounding-box based on purpose: the point is a conservative,
    cheap collision volume, not a faithful one -- a tight fit to concave CAD is exactly what the
    visual mesh is for.
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre, extent = (lo + hi) / 2, hi - lo
    axis = int(np.argmax(extent))
    perp = [i for i in range(3) if i != axis]
    radius = float(max(extent[perp]) / 2 * RADIUS_SCALE)
    half = float(extent[axis] / 2)

    pos = " ".join(f"{v:.5g}" for v in centre)
    if half / radius >= CAPSULE_ASPECT:
        # MuJoCo capsules run along local z, so rotate z onto the link's long axis. size is
        # (radius, cylinder half-length): subtract the radius so the caps stay inside the bbox.
        attrs = {"type": "capsule", "size": f"{radius:.5g} {max(half - radius, 1e-4):.5g}",
                 "pos": pos}
        if (quat := AXIS_QUAT[axis]) is not None:
            attrs["quat"] = quat
        return attrs
    return {"type": "box", "size": " ".join(f"{v / 2 * RADIUS_SCALE:.5g}" for v in extent),
            "pos": pos}


def mesh_vertices(model, body_name: str) -> np.ndarray:
    """Vertices of every mesh geom on *body_name*, expressed in that body's frame."""
    import mujoco

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    chunks = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_dataid[g] < 0:
            continue
        mid = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        chunks.append(verts @ rot.reshape(3, 3).T + model.geom_pos[g])
    if not chunks:
        raise RuntimeError(f"{body_name} carries no mesh geom to fit a collision primitive to")
    return np.concatenate(chunks)


def build(source: Path) -> str:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(source / "xarm7_nohand.xml"))
    tree = ET.parse(source / "xarm7_nohand.xml")
    root = tree.getroot()

    root.set("model", "xarm7")
    root.find("compiler").set("meshdir", "meshes")

    # Delta 2: physics options belong to the world.
    for option in root.findall("option"):
        root.remove(option)

    # Delta 4a: the visual/collision classes the sibling arms use.
    cls = root.find('default/default[@class="xarm7"]')
    for name, geom in (
        ("visual", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"}),
        ("collision", {"group": "3", "rgba": "0.6 0.1 0.1 0.35"}),
    ):
        node = ET.SubElement(cls, "default", {"class": name})
        ET.SubElement(node, "geom", geom)

    # Delta 3: neutralise the plinth offset.
    base = root.find('worldbody/body[@name="link_base"]')
    base.set("pos", "0 0 0")

    # Delta 4b: tag upstream geoms visual, then append a fitted collision primitive per link.
    for body in root.iter("body"):
        name = body.get("name")
        geoms = [g for g in body.findall("geom") if g.get("mesh")]
        if not geoms:
            continue
        for geom in geoms:
            geom.set("class", "visual")
        attrs = fit_collision_primitive(mesh_vertices(model, name))
        # Insert after the last visual geom so the file reads visual-then-collision per link.
        body.insert(list(body).index(geoms[-1]) + 1,
                    ET.Element("geom", {"class": "collision", **attrs}))

    ET.indent(tree, space="  ")
    body_xml = ET.tostring(root, encoding="unicode")
    return HEADER + body_xml[body_xml.index(">") + 1:].lstrip("\n")


HEADER = f"""<mujoco model="xarm7">
  <!--
    UFACTORY xArm 7 (7-DoF serial arm) for MuJoCo.

    GENERATED by external/convert/build_xarm7_mjcf.py from MuJoCo Menagerie's `ufactory_xarm7`
    @ {MENAGERIE_COMMIT} (BSD-3-Clause - see XARM7_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia tensor, joint limit, link offset
    and actuator gain below is Menagerie's own value, passed through unchanged.

    Four documented deltas, all roqsim conventions (the generator's docstring says why):
      1. built from xarm7_nohand.xml - the UFACTORY gripper is a separate model, not bundled here
      2. no <option> - integrator/timestep/cone belong to the world YAML's `sim:` block
      3. link_base neutralised to the origin - spawn_arm places the arm via an attach frame
      4. visual/collision split - upstream collides against nine full-detail convex hulls; the
         collision primitives here are fitted from the mesh vertices by the generator
  -->
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                       help="rebuild and diff against the committed model instead of writing it")
    args = parser.parse_args()

    source = resolve_source("mujoco_menagerie", MENAGERIE_URL, MENAGERIE_COMMIT,
                            sparse=MENAGERIE_SUBDIR, subdir=MENAGERIE_SUBDIR)
    xml = build(source)
    target = PKG / "xarm7.xml"

    if args.check:
        if not target.exists():
            print(f"{target} does not exist", file=sys.stderr)
            return 1
        if target.read_text() != xml:
            print(f"{target} differs from a fresh build -- was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {MENAGERIE_COMMIT[:12]}")
        return 0

    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for mesh in MESHES:
        shutil.copy2(source / "assets" / f"{mesh}.stl", PKG / "meshes" / f"{mesh}.stl")
    shutil.copy2(source / "LICENSE", PKG / "XARM7_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + {len(MESHES)} meshes + XARM7_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
