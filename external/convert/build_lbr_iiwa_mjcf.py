#!/usr/bin/env python3
"""Build roqsim's KUKA LBR iiwa 7 R800 and 14 R820 MJCFs from lbr-stack's description packages.

A xacro port, like the M1013's. Two variants, one generator: ``lbr_iiwa7_r800_description`` and
``lbr_iiwa14_r820_description`` are the descriptions `lbr_fri_ros2_stack` -- the ROS 2 stack KUKA FRI
users run -- ships, so the frames and joint names here are the ones a ROS 2 user of the real arm
already has. MuJoCo Menagerie carries only the iiwa 14 (``kuka_iiwa_14``); it is the independent
cross-check in the tests, not the source.

Passed through verbatim from the expanded xacro: every link offset, mass, centre of mass, full
inertia tensor, joint axis and joint position limit. The link masses sum to KUKA's published arm
weights (23.9 kg / 29.9 kg) to the gram.

**Not** taken from the source: the effort limit. ``config/joint_limits.yaml`` sets ``effort: 200`` on
every axis of both arms, a placeholder; KUKA publishes per-axis maximum torques, and the hardware
wins. ``MAX_TORQUE`` below is that datasheet table, as Drake's ``iiwa_description/README.md`` records
it with its sources. It becomes each actuator's ``forcerange``.

Visuals are Collada (fusion2urdf exports), converted through ``dae2obj.py``/pycollada and decimated
with ``reduce-mesh``; the URDF places each visual with a pure translation (the meshes are authored in
the arm's zero pose, world frame), carried here as the geom ``pos``. Collision is lbr-stack's own
decimated STL collision mesh per link, which MuJoCo hulls -- seven small, already-simplified meshes,
so no fitting is needed.

Usage::

    python external/convert/build_lbr_iiwa_mjcf.py           # fetch, expand, convert, write both
    python external/convert/build_lbr_iiwa_mjcf.py --check   # rebuild and diff against what is committed
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, write_license  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models"

#: The descriptions' Apache-2.0 grant. The standalone repositories declare it in package.xml but ship
#: no LICENSE file; the files were split out of this repository, under whose root LICENSE they were
#: published until the commit after this one.
STACK_URL = "https://github.com/lbr-stack/lbr_fri_ros2_stack.git"
STACK_LICENSE_COMMIT = "e41261fd63bc8d6a6a282eb1747f1ed329cbddbb"

JOINTS = [f"A{i}" for i in range(1, 8)]
LINKS = [f"link_{i}" for i in range(8)]
#: Per-material triangle budget for the visual meshes (the sources are ~1.2 MB of CAD per link).
TARGET_FACES = 6000

VARIANTS = {
    "lbr_iiwa7": {
        "title": "KUKA LBR iiwa 7 R800",
        "package": "lbr_iiwa7_r800_description",
        "url": "https://github.com/lbr-stack/lbr_iiwa7_r800_description.git",
        "commit": "64a0cc38708988c631873b74070f3ee418327c68",  # tag v2.5.0
        # KUKA "LBR iiwa technical data", axis table (via Drake iiwa_description/README.md), N*m.
        "max_torque": [176, 176, 110, 110, 110, 40, 40],
    },
    "lbr_iiwa14": {
        "title": "KUKA LBR iiwa 14 R820",
        "package": "lbr_iiwa14_r820_description",
        "url": "https://github.com/lbr-stack/lbr_iiwa14_r820_description.git",
        "commit": "86ac0532841a90694afe6a65c300f08b55eb1296",  # tag v2.5.0
        # KUKA "Sensitive robotics LBR iiwa" brochure, p. 30 (via Drake iiwa_description/README.md), N*m.
        "max_torque": [320, 320, 176, 176, 110, 40, 40],
    },
}

#: Servo class per axis. The torques fall 8x along the chain, and so does the inertia each joint
#: moves; one stiff setting chatters on the wrist (the M1013 lesson). Indexed by axis.
SERVO_CLASS = ["shoulder", "shoulder", "elbow", "elbow", "wrist", "hand", "hand"]


def fmt(v: float) -> str:
    return f"{v:.6g}"


def convert_meshes(description: Path, pkg: Path) -> dict[str, list]:
    """Collada -> per-material OBJ (pycollada) -> decimated OBJ; STL collision copied. Returns materials."""
    meshes = pkg / "meshes"
    if meshes.exists():
        shutil.rmtree(meshes)
    meshes.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "dae2obj.py"),
             str(description / "meshes/visual"), str(raw)],
            check=True, capture_output=True,
        )
        found = json.loads((raw / "materials.json").read_text())
        materials: dict[str, list] = {}
        for stem in sorted(p.stem for p in (description / "meshes/visual").glob("*.dae")):
            # A single-material mesh has no sub-meshes and no entry; its colour is the one material.
            parts = found.get(stem) or [[stem, None]]
            materials[stem] = []
            for sub, rgb in parts:
                out = f"{stem}_visual_{len(materials[stem])}"
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(TARGET_FACES), "--no-materials",
                     str(raw / f"{sub}.obj"), str(meshes / f"{out}.obj")],
                    check=True, cwd=ROOT, capture_output=True,
                )
                materials[stem].append([out, rgb])
    for stl in sorted((description / "meshes/collision").glob("*.stl")):
        shutil.copyfile(stl, meshes / f"{stl.stem}_collision.stl")
    return materials


def _origin_xyz(element: ET.Element) -> str:
    origin = element.find("origin")
    rpy = [float(v) for v in ((origin.get("rpy") if origin is not None else None) or "0 0 0").split()]
    if any(abs(a) > 1e-12 for a in rpy):
        # Every joint and visual in both descriptions is a pure translation; a rotation would need
        # carrying through, so refuse rather than silently drop it.
        raise ValueError(f"unexpected rotation {rpy} in {ET.tostring(element)[:120]!r}")
    return " ".join(fmt(float(v)) for v in ((origin.get("xyz") if origin is not None else None)
                                             or "0 0 0").split())


def build(name: str, urdf: ET.Element, materials: dict[str, list]) -> str:
    variant = VARIANTS[name]
    links = {link.get("name").removeprefix("lbr_"): link for link in urdf.findall("link")}
    joints = {j.get("name").removeprefix("lbr_"): j for j in urdf.findall("joint")}

    palette: dict[str, list[float]] = {}
    assets = []
    for parts in materials.values():
        for sub, rgb in parts:
            if rgb is not None:
                palette[f"mat_{sub}"] = rgb
            assets.append(f'    <mesh file="{sub}.obj"/>\n')
    for link in LINKS:
        assets.append(f'    <mesh file="{link}_collision.stl"/>\n')
    material_block = "".join(
        f'    <material name="{n}" rgba="{" ".join(fmt(c) for c in rgb)} 1"/>\n'
        for n, rgb in sorted(palette.items())
    )

    body = ""
    closing = []
    for i, link_name in enumerate(LINKS):
        link = links[link_name]
        indent = "    " + "  " * i
        if i == 0:
            body += f'{indent}<body name="{link_name}" childclass="{name}">\n'
        else:
            joint = joints[JOINTS[i - 1]]
            assert joint.find("child").get("link").removeprefix("lbr_") == link_name
            body += f'{indent}<body name="{link_name}" pos="{_origin_xyz(joint)}">\n'
        inert = inertial(link, full=True)
        body += (f'{indent}  <inertial pos="{inert["pos"]}" mass="{inert["mass"]}"'
                 f' fullinertia="{inert["fullinertia"]}"/>\n')
        if i > 0:
            joint = joints[JOINTS[i - 1]]
            limit = joint.find("limit")
            axis = " ".join(fmt(float(v)) for v in joint.find("axis").get("xyz").split())
            body += (f'{indent}  <joint name="{JOINTS[i - 1]}" class="{SERVO_CLASS[i - 1]}"'
                     f' axis="{axis}"'
                     f' range="{fmt(float(limit.get("lower")))} {fmt(float(limit.get("upper")))}"/>\n')
        visual = link.find("visual")
        stem = Path(visual.find("geometry/mesh").get("filename")).stem
        vpos = _origin_xyz(visual)
        for sub, rgb in materials[stem]:
            mat = f' material="mat_{sub}"' if rgb is not None else ""
            body += f'{indent}  <geom class="visual" mesh="{sub}" pos="{vpos}"{mat}/>\n'
        collision = link.find("collision")
        body += (f'{indent}  <geom class="collision" mesh="{link_name}_collision"'
                 f' pos="{_origin_xyz(collision)}"/>\n')
        closing.append(indent)
    ee = joints["joint_ee"]
    body += f'{indent}  <site name="attachment_site" pos="{_origin_xyz(ee)}"/>\n'
    for indent in reversed(closing):
        body += f"{indent}</body>\n"

    actuators = ""
    for axis, (joint_name, torque) in enumerate(zip(JOINTS, variant["max_torque"])):
        limit = joints[joint_name].find("limit")
        actuators += (
            f'    <position class="{SERVO_CLASS[axis]}" name="{joint_name}" joint="{joint_name}"'
            f' ctrlrange="{fmt(float(limit.get("lower")))} {fmt(float(limit.get("upper")))}"'
            f' forcerange="-{torque} {torque}"/>\n'
        )
    excludes = "".join(
        f'    <exclude body1="{a}" body2="{b}"/>\n' for a, b in zip(LINKS, LINKS[1:])
    )
    velocities = ", ".join(
        f"{fmt(math.degrees(float(joints[j].find('limit').get('velocity'))))}" for j in JOINTS
    )
    return TEMPLATE.format(
        name=name, title=variant["title"], package=variant["package"], commit=variant["commit"],
        torques=", ".join(str(t) for t in variant["max_torque"]), velocities=velocities,
        materials=material_block, assets="".join(assets), body=body, actuators=actuators,
        excludes=excludes,
    )


TEMPLATE = """<mujoco model="{name}">
  <!--
    {title} for MuJoCo.

    GENERATED by external/convert/build_lbr_iiwa_mjcf.py from lbr-stack's {package}
    @ {commit} (Apache-2.0, see the LICENSE file beside this model). Do not hand-edit: re-run the
    generator. Link offsets, masses, centres of mass, full inertia tensors, joint axes and position
    limits are the description's own values.

    Names follow the description with its `lbr_` robot-name prefix removed (link_0..link_7, A1..A7):
    spawn_arm adds the world's prefix, and the manifest's arm_controller reports the joints as
    lbr_A1..lbr_A7, the names lbr_fri_ros2_stack uses.

    forcerange is KUKA's per-axis maximum torque ({torques} N*m), not the description's `effort`,
    which is a uniform placeholder. The datasheet's maximum speeds ({velocities} deg/s, which the
    description does carry) are enforced by the manifest's arm_controller `max_velocity`; MJCF has no
    joint velocity limit.

    Collision is the description's own decimated STL mesh per link, hulled by MuJoCo.

    No <option>: timestep, integrator, solver and contact overrides belong to the experiment
    (the world YAML's `sim:` block), not to the arm.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="{name}">
      <!-- Servo classes follow the torque classes of the axes, which fall 8x along the chain with
           the inertia each joint moves. Sized so kv*dt/I stays well below the explicit-damping
           stability bound at a 2 ms step on the lightest-loaded axis of each class, and stiff enough
           that the rated payload at the flange, arm stretched out horizontally, sags no axis by more
           than a degree (a proportional servo has a steady-state error under load; KUKA's controller
           does not show one). Armature stands in for the reflected rotor inertia of the harmonic
           drives, which the description does not carry. Both are substrate choices. -->
      <default class="shoulder">
        <joint damping="20" armature="0.5"/>
        <position kp="10000" kv="300"/>
      </default>
      <default class="elbow">
        <joint damping="10" armature="0.3"/>
        <position kp="6000" kv="150"/>
      </default>
      <default class="wrist">
        <joint damping="5" armature="0.2"/>
        <position kp="3000" kv="60"/>
      </default>
      <default class="hand">
        <joint damping="2" armature="0.1"/>
        <position kp="2000" kv="30"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" material="iiwa_default"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <site size="0.005" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
  </default>

  <asset>
    <material name="iiwa_default" rgba="0.75 0.75 0.75 1"/>
{materials}{assets}  </asset>

  <worldbody>
{body}  </worldbody>

  <contact>
    <!-- Chain neighbours. MuJoCo's filterparent is skipped when the parent is the world body, and
         link_0 is welded to world, so every parent/child pair here would be measured; adjacent links
         are mechanically constrained and the hulls of neighbouring housings overlap at the joint. -->
{excludes}    <!-- link_5 / link_7, across A6. Their hulls meet once A6 passes ~113 deg, even with every
         other joint at zero, and the real arm drives A6 to its 120 deg limit there: the hull of
         link_5 closes over the recess link_7 turns in. The only non-adjacent pair whose hulls meet
         in a single-joint sweep within KUKA's limits, so the only one excluded; every other
         self-collision stays live. -->
    <exclude body1="link_5" body2="link_7"/>
  </contact>

  <actuator>
{actuators}  </actuator>

  <keyframe>
    <!-- Elbow bent over the workspace: A2 30 deg forward, A4 -60 deg, A6 60 deg, which keeps the
         flange in front of and above the base, clear of the joint limits. -->
    <key name="home" qpos="0 0.523599 0 -1.0472 0 1.0472 0" ctrl="0 0.523599 0 -1.0472 0 1.0472 0"/>
  </keyframe>
</mujoco>
"""


def license_header(name: str) -> list[str]:
    variant = VARIANTS[name]
    return [
        f"{variant['title']} -- vendored geometry and description.",
        "",
        f"Upstream:   {variant['url'].removesuffix('.git')}",
        f"Commit:     {variant['commit']} (tag v2.5.0)",
        "Author:     Martin Huber (lbr-stack)",
        f"License:    Apache License 2.0, as {variant['package']}/package.xml declares. That",
        "            repository ships no LICENSE file: its URDF and meshes were split out of",
        f"            {STACK_URL.removesuffix('.git')} (lbr_description),",
        "            published there under the root LICENSE reproduced below, taken at",
        f"            {STACK_LICENSE_COMMIT}, the last commit carrying them.",
        "            lbr_description credits iiwa_stack (BSD-3-Clause) as its starting point.",
        "",
        "Not from upstream: the actuator forceranges are KUKA's published per-axis torques.",
        "Regenerate with: external/convert/build_lbr_iiwa_mjcf.py",
        "The full text of the grant follows.",
    ]


def expand(name: str, description: Path, work: Path) -> ET.Element:
    variant = VARIANTS[name]
    return expand_xacro(
        {variant["package"]: description},
        description / f"urdf/{variant['package'].removesuffix('_description')}.urdf.xacro",
        work,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", choices=sorted(VARIANTS))
    args = parser.parse_args()

    stack = resolve_source("lbr_fri_ros2_stack", STACK_URL, STACK_LICENSE_COMMIT)
    failed = False
    for name in [args.only] if args.only else sorted(VARIANTS):
        variant = VARIANTS[name]
        description = resolve_source(variant["package"], variant["url"], variant["commit"])
        pkg = MODELS / name
        target = pkg / f"{name}.xml"
        if args.check:
            materials = json.loads((pkg / "meshes" / f"{name}.materials.json").read_text())
        else:
            materials = convert_meshes(description, pkg)
            (pkg / "meshes" / f"{name}.materials.json").write_text(
                json.dumps(materials, indent=2, sort_keys=True) + "\n"
            )
        with tempfile.TemporaryDirectory() as tmp:
            xml = build(name, expand(name, description, Path(tmp)), materials)
        if args.check:
            if not target.exists() or target.read_text() != xml:
                print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
                failed = True
            else:
                print(f"{target}: up to date with {variant['commit'][:12]}")
            continue
        write_license(stack / "LICENSE", pkg / f"{name.upper()}_LICENSE", license_header(name))
        target.write_text(xml)
        print(f"wrote {target} + meshes + {name.upper()}_LICENSE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
