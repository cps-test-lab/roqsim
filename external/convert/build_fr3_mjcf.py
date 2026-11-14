#!/usr/bin/env python3
"""Build roqsim's Franka Research 3 MJCF (``fr3``) from Franka's own description package.

Source: ``frankarobotics/franka_description`` (Apache-2.0, (c) Franka Robotics GmbH), at the commit
``roqsim_manipulation/tests/upstream/franka/SOURCES.toml`` pins, and the URDF its
``robots/fr3/fr3.urdf.xacro`` produces with ``hand:=false`` (vendored there as ``fr3.urdf`` by
``external/franka_upstream.py``). The tests compare the built model against that URDF
(``roqsim_manipulation/tests/test_franka_model.py``) and its meshes against the digest of the DAE
visuals beside it (``fr3_visual.json``, ``test_franka_visual.py``), so a rebuild that drifts fails
them.

Passed through from the URDF by MuJoCo's own URDF compiler: every link offset, mass, centre of mass,
full inertia tensor, joint axis and position limit, the effort limit (as the actuator's force range),
and the flange (``fr3_link8``, 0.107 m beyond link 7). The accelerometer frames (massless, no
geometry) are dropped.

Not from the URDF, and said so in ``CREDITS.txt``:

- **Joint dynamics.** franka_description's ``dynamics.yaml`` gives a damping of 0.003 and a friction
  of 0.2 per joint and no armature in MuJoCo's form; its motor inertia (4.206e-5 kg m^2 at a gear
  ratio of 120) reflects to 0.606 kg m^2. MuJoCo Menagerie's ``franka_fr3`` carries values
  "identified in torque control" (its CHANGELOG, 2025-04-25): armature 0.195 (joints 1-4) and 0.074
  (5-7), damping 0.21, frictionloss 1.137 (1-4), 0.763, 0.44 and 0.248. These are used, since a
  torque-controlled arm is what the model is for; neither is these arms' own measurement.
- **Actuators.** One torque motor per joint (``control: effort``), force range the
  URDF's effort limit (87 N m on joints 1-4, 12 N m on 5-7). Nothing compensates gravity: on the
  robot the controller adds it from its model and the configured load, and so does ``franka_fci``.
- **Visual meshes.** franka_description's COLLADA visuals, read with pycollada (node transforms
  applied), one OBJ per DAE primitive, which is one material: the triangles verbatim, exact duplicate
  vertices merged, nothing remeshed or decimated. A decimated piece pulls its outline in on its own,
  so neighbouring pieces of different colours stop meeting and the surface tears. Each piece's
  material is the DAE's: ``rgba`` its diffuse colour, ``emission`` its emission over its diffuse
  (the lit status LEDs), ``specular`` its reflectivity where the DAE gives one.
- **Collision.** franka_description's collision STL per link, split into convex pieces (CoACD,
  seeded). MuJoCo collides a mesh as its convex hull, and the hull of one link's STL bridges the
  link's recesses by up to 70 mm, so a contact would land on empty air beside the elbow and the
  wrist. Where the STL itself leaves out a visual part (the base's rear strip, a recess on links 3
  and 4), that part has no collider, as upstream.
- **Contacts.** Each link's pieces are excluded against its neighbour's, which they meet at the
  joint between them.
- **Sites.** ``flange`` and ``attachment_site`` at ``fr3_link8``, where a tool is mounted.
- **Home.** A ``home`` keyframe at libfranka's examples' start pose ``{0, -pi/4, 0, -3 pi/4, 0, pi/2, pi/4}``
  (``examples/generate_joint_position_motion.cpp`` moves there first); all zeros is outside joints 4
  and 6's ranges.

    pip install pycollada trimesh coacd
    python external/convert/build_fr3_mjcf.py --description <franka_description checkout at the pin>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
from pathlib import Path

import collada
import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/fr3"
URDF = ROOT / "roqsim_manipulation/tests/upstream/franka/franka_description/fr3.urdf"

#: MuJoCo Menagerie franka_fr3 (v2026.11.3, fr3.xml): identified in torque control.
ARMATURE = [0.195] * 4 + [0.074] * 3
FRICTIONLOSS = [1.137] * 4 + [0.763, 0.44, 0.248]
DAMPING = 0.21
EFFORT = [87.0] * 4 + [12.0] * 3
#: libfranka's examples start every motion here (examples/generate_joint_position_motion.cpp).
HOME = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
#: CoACD's concavity threshold and seed for the collision pieces.
SPLIT_THRESHOLD = 0.05
SPLIT_SEED = 0
#: Vertices per collision piece: enough for the STL's curvature, few enough for the narrow phase.
MAX_HULL_VERTICES = 64


def dae_pieces(dae: Path) -> list[dict]:
    """Every primitive of a COLLADA file in the link frame, with its material, as pycollada reads it.

    One primitive is one material. The vertices are the file's, node transforms applied; the
    triangles keep the file's winding, which agrees with its authored normals.
    """
    pieces = []
    for geom in collada.Collada(str(dae)).scene.objects("geometry"):
        for prim in geom.primitives():
            tris = prim.triangleset() if hasattr(prim, "triangleset") else prim
            corners = np.asarray(tris.vertex, float)[np.asarray(tris.vertex_index)].reshape(-1, 3)
            vertices, faces = np.unique(corners, axis=0, return_inverse=True)
            effect = prim.material.effect
            pieces.append(
                {
                    "material": prim.material.id.removesuffix("-material"),
                    "diffuse": [float(c) for c in effect.diffuse],
                    "emission": [float(c) for c in effect.emission],
                    "reflectivity": None
                    if effect.reflectivity is None
                    else float(effect.reflectivity),
                    "vertices": vertices,
                    "faces": faces.reshape(-1, 3),
                }
            )
    return pieces


def mjcf_material(piece: dict) -> dict:
    """The MuJoCo material of a DAE piece: diffuse as rgba, emission as MuJoCo's fraction of it."""
    rgba = piece["diffuse"]
    lit = float(np.mean(rgba[:3]))
    emission = float(np.mean(piece["emission"][:3])) / lit if lit > 0 else 0.0
    out = {"rgba": rgba, "emission": round(min(emission, 1.0), 4)}
    if piece["reflectivity"] is not None:
        out["specular"] = piece["reflectivity"]
    return out


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    body = "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices)
    body += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
    path.write_text(body, encoding="utf-8")


def convert_visuals(
    description: Path, meshes: Path
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, dict]]:
    """Each link's visual pieces as OBJ files: ``{link: [(mesh, material)]}``, and the materials by
    name."""
    pieces: dict[str, list[tuple[str, str]]] = {}
    materials: dict[str, dict] = {}
    for i in range(8):
        for j, piece in enumerate(
            dae_pieces(description / f"meshes/robots/fr3/visual/link{i}.dae")
        ):
            name = f"link{i}_{j}"
            write_obj(meshes / f"{name}.obj", piece["vertices"], piece["faces"])
            # Named by the DAE's colour; a colour that two files give different values (white,
            # lit on the wrist) is told apart by its link.
            material = f"fr3_{piece['material'].split('_color_')[-1]}"
            values = mjcf_material(piece)
            if materials.setdefault(material, values) != values:
                material = f"{material}_link{i}"
                materials[material] = values
            pieces.setdefault(f"fr3_link{i}", []).append((name, material))
    return pieces, materials


def convert_collision(description: Path, meshes: Path) -> dict[str, list[str]]:
    """Each link's collision STL as convex pieces, one OBJ each; returns ``{link: [mesh]}``."""
    import coacd

    coacd.set_log_level("error")
    pieces: dict[str, list[str]] = {}
    for i in range(8):
        stl = trimesh.load(description / f"meshes/robots/fr3/collision/link{i}.stl", process=True)
        parts = coacd.run_coacd(
            coacd.Mesh(stl.vertices, stl.faces),
            threshold=SPLIT_THRESHOLD,
            seed=SPLIT_SEED,
            max_ch_vertex=MAX_HULL_VERTICES,
        )
        for k, (v, f) in enumerate(parts):
            hull = trimesh.Trimesh(v, f).convex_hull
            name = f"link{i}_collision_{k}"
            write_obj(meshes / f"{name}.obj", hull.vertices, hull.faces)
            pieces.setdefault(f"fr3_link{i}", []).append(name)
    return pieces


def build(description: Path) -> None:
    meshes = OUT / "meshes"
    if meshes.exists():
        for old in meshes.iterdir():
            old.unlink()
    meshes.mkdir(parents=True, exist_ok=True)
    visuals, materials = convert_visuals(description, meshes)
    collision = convert_collision(description, meshes)

    urdf = URDF.read_text(encoding="utf-8")
    urdf = re.sub(r"<visual[ >].*?</visual>", "", urdf, flags=re.S)
    urdf = re.sub(r"<collision[ >].*?</collision>", "", urdf, flags=re.S)
    urdf = urdf.replace(
        '<robot name="fr3">',
        '<robot name="fr3"><mujoco><compiler fusestatic="false" '
        'balanceinertia="false" discardvisual="false" strippath="false"/></mujoco>',
        1,
    )
    spec = mujoco.MjSpec.from_string(urdf)
    spec.meshdir = str(meshes)
    for body in list(spec.bodies):
        if "accelerometer" in body.name:
            spec.delete(body)
    for i in range(7):
        joint = spec.joint(f"fr3_joint{i + 1}")
        joint.armature = ARMATURE[i]
        joint.damping = [DAMPING, 0, 0]
        joint.frictionloss = FRICTIONLOSS[i]
    for name, values in materials.items():
        material = spec.add_material(name=name)
        material.rgba = values["rgba"]
        material.emission = values["emission"]
        if "specular" in values:
            material.specular = values["specular"]
    for link, parts in visuals.items():
        body = spec.body(link)
        for name, material in parts:
            spec.add_mesh(name=name, file=f"{name}.obj")
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                material=material,
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
            )
            geom.name = f"fr3_{name}_visual"
    for link, parts in collision.items():
        body = spec.body(link)
        for name in parts:
            spec.add_mesh(name=name, file=f"{name}.obj")
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, group=3, density=0)
            geom.name = f"fr3_{name}"
            geom.rgba = [0.6, 0.6, 0.6, 1]
    # Neighbouring links' pieces meet at their joint. MuJoCo filters a body's contacts with its
    # parent, but not once the parent is welded to the world, as link0 is: exclude every pair.
    for i in range(7):
        spec.add_exclude(bodyname1=f"fr3_link{i}", bodyname2=f"fr3_link{i + 1}")
    flange = spec.body("fr3_link8")
    flange.add_site(name="flange", size=[0.005] * 3, group=4)
    flange.add_site(name="attachment_site", size=[0.001] * 3, group=4)
    for i in range(7):
        act = spec.add_actuator(
            name=f"fr3_joint{i + 1}", target=f"fr3_joint{i + 1}", trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        act.gainprm[0] = 1.0
        # The force range is the torque limit; no control range, so a world may switch the joint to
        # another law (roqsim.actuators) without a range in the wrong unit clamping it.
        act.forcelimited = 1
        act.ctrllimited = 0
        act.forcerange = [-EFFORT[i], EFFORT[i]]
    spec.add_key(name="home", qpos=HOME)
    spec.compile()
    # MuJoCo writes the keyframe count as <size nkey>, which every world the arm is attached to
    # then warns about; the <key> element alone defines it.
    xml = spec.to_xml().replace(f'meshdir="{meshes}"', 'meshdir="meshes"')
    size = '  <size nkey="1"/>\n\n'
    if size not in xml:
        raise SystemExit(f"fr3.xml: MuJoCo no longer writes {size.strip()!r}")
    xml = xml.replace(size, "", 1)
    # The comment that opens <mujoco> is the model's catalog description: its first sentence says
    # what the model is.
    header = (
        "  <!--\n"
        "    Franka Research 3 without a hand, with seven torque motors and no gravity compensation.\n"
        "\n"
        "    Built by external/convert/build_fr3_mjcf.py from franka_description 2.9.0 (Apache-2.0);\n"
        f"    collision pieces by CoACD {importlib.metadata.version('coacd')}, threshold "
        f"{SPLIT_THRESHOLD}, seed {SPLIT_SEED}.\n"
        "    See CREDITS.txt.\n"
        "  -->\n"
    )
    opening = '<mujoco model="fr3">\n'
    if not xml.startswith(opening):
        raise SystemExit(f"fr3.xml: MuJoCo no longer opens the file with {opening!r}")
    (OUT / "fr3.xml").write_text(opening + header + xml[len(opening) :], encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--description",
        type=Path,
        required=True,
        help="a franka_description checkout at the pinned commit",
    )
    build(parser.parse_args(argv).description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
