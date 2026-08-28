#!/usr/bin/env python3
"""Build roqsim's Doosan M1013 MJCF from `DoosanRobotics/doosan-robot2`'s `dsr_description2`.

A xacro-tree port, like the ROSbot's: there is no upstream MJCF. The vendor's own ``dsr_mujoco``
package is *not* one -- opening it (rather than trusting the name) shows two small Python helpers and
three controller YAMLs, a runtime scene builder over the URDF. That discovery is what moved this
port's estimate from 3 iterations to 6 before any of it was written.

Every mass, inertia tensor, centre of mass, link offset, joint limit, velocity limit and effort limit
below is Doosan's own value, read out of the expanded xacro. What the source does *not* give usable
is collision geometry:

**The vendor's ``*_collision`` meshes are byte-for-byte copies of its visual CAD** -- same ten files,
same sizes, ~7 MB. Colliding against those is the porting playbook's first anti-pattern, so collision
here is the **convex hull of each decimated visual mesh** (MuJoCo hulls a mesh geom for collision),
which the playbook allows alongside primitives.

Fitted primitives were tried first, as the xArm's are, and measured against the vendor meshes as
ground truth over 1500 uniformly sampled configurations:

    vendor meshes (ground truth)   11.7%
    one capsule per mesh, s=0.90   62.4%
    ... s=0.80 / 0.70 / 0.60       50.0% / 37.4% / 30.3%
    ... s=0.50                     21.9%   <- and already non-conservative

A single capsule is simply a poor envelope for this arm's long links (0.62 m upper arm, 0.559 m
forearm), and shrinking the radius to compensate stops containing the mesh long before it reaches the
right rate -- which would let a planner drive links through each other. The playbook says as much:
a tighter envelope needs *several* primitives per link, not a smaller scale. Hulls of the decimated
meshes give the ground-truth rate exactly, at 4000 triangles per part rather than the vendor's full
CAD. Multi-primitive fitting is the better long-term answer and is noted in the port log.

Three further things the tree makes awkward, all handled here:

1. **The top-level xacro needs a sibling package.** ``m1013.urdf.xacro`` instantiates a Gazebo macro
   that resolves ``$(find dsr_controller2)``, so both packages are fetched and put on a private ament
   index. Expanding only the visual macro does not avoid it -- the macro includes the Gazebo block
   itself.
2. **Collada, which MuJoCo cannot load and stock Linux Blender cannot import** (no OpenCOLLADA, as
   the porting playbook warns). Conversion goes through ``dae2obj.py``/pycollada, which loads
   vertices verbatim and splits per material, then through ``reduce-mesh`` to decimate.
3. **The meshes are in millimetres**; the URDF carries ``scale="0.001"`` on every visual. Baked in at
   conversion time so the committed OBJ is metres like every other mesh in the substrate.

Usage::

    python external/convert/build_doosan_mjcf.py           # fetch, expand, convert, write
    python external/convert/build_doosan_mjcf.py --check   # rebuild and diff against what is committed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402

DOOSAN_URL = "https://github.com/DoosanRobotics/doosan-robot2.git"
DOOSAN_COMMIT = "816ecb5d1c2599303eaf9540216afa03552f80ad"
#: The Gazebo macro in the description resolves this sibling; both must be on the ament index.
PACKAGES = ("dsr_description2", "dsr_controller2")

MODEL = "m1013"
COLOURWAY = "white"
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models" / MODEL

#: Per-mesh triangle budget after decimation. The vendor ships ~7 MB of full-detail CAD across ten
#: files; this is a visual-only budget, since collision is primitives.
TARGET_FACES = 4000
JOINTS = [f"joint_{i}" for i in range(1, 7)]


def expand_xacro(sources: dict[str, Path], work: Path) -> ET.Element:
    xacro = shutil.which("xacro") or "/opt/ros/jazzy/bin/xacro"
    if not Path(xacro).exists():
        raise RuntimeError(
            "xacro is required to expand dsr_description2 and was not found.\n"
            "Install ROS 2 (ros-jazzy-xacro) or `pip install xacro`, then re-run. Refusing to guess "
            "the expanded tree: every mass, inertia and joint limit comes from it."
        )
    share = work / "prefix/share"
    (share / "ament_index/resource_index/packages").mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        (share / f"ament_index/resource_index/packages/{name}").write_text("")
        if not (share / name).exists():
            (share / name).symlink_to(path)

    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = f"{work / 'prefix'}:{env.get('AMENT_PREFIX_PATH', '')}"
    ros_site = sorted(Path(xacro).resolve().parents[1].glob("lib/python3*/site-packages"))
    if ros_site:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ros_site[-1]), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    proc = subprocess.run(
        [xacro, str(sources["dsr_description2"] / f"xacro/{MODEL}.urdf.xacro"),
         f"color:={COLOURWAY}", "gripper:=none"],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"xacro failed:\n{proc.stderr.strip()}")
    return ET.fromstring(proc.stdout)


def convert_meshes(description: Path) -> dict[str, list]:
    """Collada -> per-material OBJ (pycollada) -> decimated, metre-scale OBJ. Returns materials."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in list((PKG / "meshes").glob("*.obj")):
        stale.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "dae2obj.py"),
             str(description / f"meshes/{MODEL}_{COLOURWAY}"), str(raw)],
            check=True, capture_output=True,
        )
        materials = json.loads((raw / "materials.json").read_text())
        for stem, parts in materials.items():
            for sub, _rgb in parts:
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(TARGET_FACES), "--no-materials", "--scale", "0.001",
                     str(raw / f"{sub}.obj"), str(PKG / "meshes" / f"{sub}.obj")],
                    check=True, cwd=ROOT, capture_output=True,
                )
    return materials


def _inertial(link: ET.Element) -> dict[str, str]:
    inertial = link.find("inertial")
    origin = inertial.find("origin")
    inertia = inertial.find("inertia")
    return {
        "pos": (origin.get("xyz") if origin is not None else "0 0 0").replace(",", " "),
        "mass": inertial.find("mass").get("value"),
        "fullinertia": " ".join(
            inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    }


def build(urdf: ET.Element, materials: dict[str, list]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    palette, assets = {}, []
    for parts in materials.values():
        for sub, rgb in parts:
            name = f"mat_{sub}"
            palette[name] = rgb
            assets.append(f'    <mesh file="{sub}.obj"/>\n')
    material_block = "".join(
        f'    <material name="{n}" rgba="{" ".join(f"{c:g}" for c in rgb)} 1"/>\n'
        for n, rgb in sorted(palette.items())
    )

    #: link -> the visual sub-meshes on it, in URDF order.
    per_link: dict[str, list[str]] = {}
    for name, link in links.items():
        for visual in link.findall("visual"):
            stem = Path(visual.find("geometry/mesh").get("filename")).stem
            per_link.setdefault(name, []).extend(sub for sub, _ in materials.get(stem, []))

    body = ""
    indent = "    "
    closing = []
    order = ["base_link"] + [joints[j].find("child").get("link") for j in JOINTS]
    for i, name in enumerate(order):
        indent = "    " + "  " * i
        if i == 0:
            body += f'{indent}<body name="{name}" childclass="{MODEL}">\n'
        else:
            joint = joints[JOINTS[i - 1]]
            origin = joint.find("origin")
            xyz = (origin.get("xyz") or "0 0 0").replace(",", " ")
            rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").replace(",", " ").split()]
            quat = _rpy_to_quat(*rpy)
            body += (f'{indent}<body name="{name}" pos="{xyz}" quat="{quat}">\n')
        inert = _inertial(links[name])
        body += (f'{indent}  <inertial pos="{inert["pos"]}" mass="{inert["mass"]}"'
                 f' fullinertia="{inert["fullinertia"]}"/>\n')
        if i > 0:
            limit = joints[JOINTS[i - 1]].find("limit")
            body += (f'{indent}  <joint name="{JOINTS[i - 1]}" class="{_gain_class(limit)}"'
                     f' range="{float(limit.get("lower")):.5g} {float(limit.get("upper")):.5g}"/>\n')
        for sub in per_link.get(name, []):
            body += f'{indent}  <geom class="visual" mesh="{sub}" material="mat_{sub}"/>\n'
        # Collision is the CONVEX HULL of each decimated mesh, not a fitted primitive. Measured, not
        # assumed -- see the module docstring for the numbers that decided it.
        for sub in per_link.get(name, []):
            body += f'{indent}  <geom class="collision" mesh="{sub}"/>\n'
        closing.append(indent)
    body += f'{indent}  <site name="attachment_site" pos="0 0 0"/>\n'
    for indent in reversed(closing):
        body += f"{indent}</body>\n"

    actuators = "".join(
        f'    <position class="{_gain_class(joints[j].find("limit"))}" name="{j}" joint="{j}"'
        f' ctrlrange="{float(joints[j].find("limit").get("lower")):.5g}'
        f' {float(joints[j].find("limit").get("upper")):.5g}"'
        f' forcerange="-{joints[j].find("limit").get("effort")}'
        f' {joints[j].find("limit").get("effort")}"/>\n'
        for j in JOINTS
    )
    # Chain neighbours must be excluded explicitly. MuJoCo's `filterparent` would normally drop a
    # parent/child contact, but it does not apply when the parent is the world body -- and base_link
    # is welded to world, so the whole chain is measured against a world-welded root. Without this,
    # base_link vs link_1 penetrates in every configuration and self-collision reads 100%. Same fix
    # open_manipulator_x needed, for the same reason its port log records.
    chain = ["base_link"] + [joints[j].find("child").get("link") for j in JOINTS]
    excludes = "".join(
        f'    <exclude body1="{a}" body2="{b}"/>\n' for a, b in zip(chain, chain[1:])
    )
    return TEMPLATE.format(commit=DOOSAN_COMMIT, materials=material_block, assets="".join(assets),
                           body=body, actuators=actuators, excludes=excludes)


def _gain_class(limit: ET.Element) -> str:
    """Servo class from the joint's published effort limit -- see the default block's comment."""
    effort = float(limit.get("effort"))
    return "strong" if effort >= 300 else ("medium" if effort >= 100 else "weak")


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> str:
    import numpy as np

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    q = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
    return " ".join(f"{v:.7g}" for v in q)


TEMPLATE = """<mujoco model="m1013">
  <!--
    Doosan M1013 (6-axis collaborative arm) for MuJoCo.

    GENERATED by external/convert/build_doosan_mjcf.py from DoosanRobotics/doosan-robot2's
    dsr_description2 @ {commit} (BSD-3-Clause - see M1013_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia tensor, centre of mass, link offset,
    joint limit and effort limit below is Doosan's own value, read out of the expanded xacro.

    Collision is FITTED primitives, not the vendor's meshes: dsr_description2's *_collision files are
    byte-for-byte copies of its full-detail visual CAD (~7 MB), which is the porting playbook's first
    anti-pattern. See external/convert/collision_fit.py and the port log.

    Visual meshes are Collada converted through pycollada (MuJoCo cannot load .dae, and stock Linux
    Blender has no OpenCOLLADA), split per material and decimated. They are authored in MILLIMETRES;
    the URDF's 0.001 scale is baked into the committed OBJs.

    No <option>: timestep, integrator, solver and contact overrides belong to the EXPERIMENT
    (the world YAML's `sim:` block), not to the arm.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="m1013">
      <!-- Servo gains scale with the vendor's own effort limits, which fall 7x along the chain
           (346, 346, 163, 50, 50, 50 N*m). A single stiff setting chatters on the distal joints:
           at kv=150 against link_6's effective inertia of ~0.105 kg*m^2, kv*dt/I is 2.9 at a 2 ms
           timestep -- past the explicit-damping stability threshold -- and joints 5 and 6 sat at
           their force limit flipping sign every step while barely moving. Sized so kv*dt/I stays
           below ~0.5 for each class. -->
      <default class="joint">
        <joint axis="0 0 1" armature="0.1"/>
      </default>
      <default class="strong">
        <joint axis="0 0 1" damping="20" armature="0.1"/>
        <position kp="3000" kv="120"/>
      </default>
      <default class="medium">
        <joint axis="0 0 1" damping="10" armature="0.1"/>
        <position kp="3000" kv="90"/>
      </default>
      <default class="weak">
        <joint axis="0 0 1" damping="3" armature="0.1"/>
        <position kp="1200" kv="25"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
  </default>

  <asset>
{materials}{assets}  </asset>

  <worldbody>
{body}  </worldbody>

  <contact>
    <!-- Chain neighbours. MuJoCo's filterparent does not fire here: it is skipped when the parent is
         the world body, and base_link is welded to world, so every parent/child pair in this chain
         is measured. Adjacent links of a real arm cannot collide -- they are mechanically
         constrained -- and leaving these in puts self-collision at 100% of sampled configurations. -->
{excludes}  </contact>

  <actuator>
{actuators}  </actuator>

  <keyframe>
    <!-- Elbow-up over the workspace, the stance Doosan's own examples start from. -->
    <key name="home" qpos="0 0 1.5708 0 1.5708 0" ctrl="0 0 1.5708 0 1.5708 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sources = {
        name: resolve_source("doosan_robot2", DOOSAN_URL, DOOSAN_COMMIT, sparse=name, subdir=name)
        for name in PACKAGES
    }
    target = PKG / f"{MODEL}.xml"

    if args.check:
        materials = json.loads((PKG / "meshes" / f"{MODEL}.materials.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            xml = build(expand_xacro(sources, Path(tmp)), materials)
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {DOOSAN_COMMIT[:12]}")
        return 0

    materials = convert_meshes(sources["dsr_description2"])
    (PKG / "meshes" / f"{MODEL}.materials.json").write_text(
        json.dumps(materials, indent=2, sort_keys=True)
    )
    with tempfile.TemporaryDirectory() as tmp:
        xml = build(expand_xacro(sources, Path(tmp)), materials)
    shutil.copy2(sources["dsr_description2"].parent / "LICENSE", PKG / "M1013_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + meshes + M1013_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
