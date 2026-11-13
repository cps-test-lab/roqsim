#!/usr/bin/env python3
"""Build roqsim's Franka Research 3 MJCF (``fr3``) from Franka's own description package.

Source: ``frankarobotics/franka_description`` (Apache-2.0, (c) Franka Robotics GmbH), at the commit
``roqsim_manipulation/tests/upstream/franka/SOURCES.toml`` pins, and the URDF its
``robots/fr3/fr3.urdf.xacro`` produces with ``hand:=false`` (vendored there as ``fr3.urdf`` by
``roqsim arm franka-upstream``). The tests compare the built model against that URDF
(``roqsim_manipulation/tests/test_franka_model.py``), so a rebuild that drifts fails them.

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
- **Meshes.** franka_description's COLLADA visuals, converted per material to OBJ with trimesh
  (pycollada, node transforms applied) and decimated (fast-simplification) to at most a sixth of
  their faces, not below 3000; the material colours become geom colours. Collision: franka_description's
  own STL per link, unchanged.
- **Sites.** ``flange`` and ``attachment_site`` at ``fr3_link8``, where a tool is mounted.
- **Home.** A ``home`` keyframe at libfranka's examples' start pose ``{0, -pi/4, 0, -3 pi/4, 0, pi/2, pi/4}``
  (``examples/generate_joint_position_motion.cpp`` moves there first); all zeros is outside joints 4
  and 6's ranges.

    pip install xacro pycollada fast-simplification trimesh
    python external/convert/build_fr3_mjcf.py --description <franka_description checkout at the pin>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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


def convert_visuals(description: Path, meshes: Path) -> dict[str, list[tuple[str, list[float]]]]:
    """Each link's visual pieces as OBJ files, with their colours."""
    pieces: dict[str, list[tuple[str, list[float]]]] = {}
    for i in range(8):
        parts = trimesh.load(description / f"meshes/robots/fr3/visual/link{i}.dae").dump()
        for j, part in enumerate(parts):
            colour = np.array(part.visual.material.baseColorFactor, float)
            colour = colour / 255 if colour.max() > 1 else colour
            faces = len(part.faces)
            if faces > 3000:
                part = part.simplify_quadric_decimation(face_count=max(3000, faces // 6))
            name = f"link{i}_{j}"
            part.export(meshes / f"{name}.obj", include_texture=False)
            pieces.setdefault(f"fr3_link{i}", []).append((name, [float(c) for c in colour]))
    return pieces


def build(description: Path) -> None:
    meshes = OUT / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    for stl in sorted((description / "meshes/robots/fr3/collision").glob("link*.stl")):
        (meshes / stl.name).write_bytes(stl.read_bytes())
    pieces = convert_visuals(description, meshes)

    urdf = URDF.read_text(encoding="utf-8")
    urdf = re.sub(r"<visual[ >].*?</visual>", "", urdf, flags=re.S)
    urdf = urdf.replace("package://franka_description/meshes/robots/fr3/collision/", "")
    urdf = urdf.replace(
        '<robot name="fr3">',
        f'<robot name="fr3"><mujoco><compiler meshdir="{meshes}" fusestatic="false" '
        'balanceinertia="false" discardvisual="false" strippath="false"/></mujoco>',
        1,
    )
    spec = mujoco.MjSpec.from_string(urdf)
    for body in list(spec.bodies):
        if "accelerometer" in body.name:
            spec.delete(body)
    for i in range(7):
        joint = spec.joint(f"fr3_joint{i + 1}")
        joint.armature = ARMATURE[i]
        joint.damping = [DAMPING, 0, 0]
        joint.frictionloss = FRICTIONLOSS[i]
    for geom in list(spec.geoms):  # the URDF's collision meshes
        geom.group = 3
        geom.rgba = [0.6, 0.6, 0.6, 1]
    for link, parts in pieces.items():
        body = spec.body(link)
        for name, colour in parts:
            spec.add_mesh(name=name, file=f"{name}.obj")
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
            )
            geom.name = f"fr3_{name}_visual"
            geom.rgba = colour
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
    xml = spec.to_xml().replace(f'meshdir="{meshes}"', 'meshdir="meshes"')
    header = (
        "<!-- Franka Research 3, built by external/convert/build_fr3_mjcf.py from "
        "franka_description 2.9.0 (Apache-2.0); see CREDITS.txt. -->\n"
    )
    (OUT / "fr3.xml").write_text(header + xml, encoding="utf-8")


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
