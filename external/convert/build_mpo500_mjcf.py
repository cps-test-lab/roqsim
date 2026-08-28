#!/usr/bin/env python3
"""Build roqsim's Neobotix MPO-500 MJCF from `neobotix/neo_simulation2`.

A 72.8 kg four-wheel **omnidirectional** base: all four wheels drive, none steers. Holonomic, so it
uses ``omni_drive``'s existing mecanum path -- no swerve inverse kinematics and no ``slip_factor``.

The ledger's recorded unknown was "whether its omni wheels are mecanum or Swedish-roller … the macro
was listed rather than opened". Opened: the only wheel macro is ``mpo_500_omni_wheel``, axis
``0 1 0``, and there is **no caster macro at all**, so nothing steers. Either roller type reduces to
the same planar model here, and the wheel spin that keeps ``joint_states`` honest is the mecanum
convention ``omni_drive`` already applies to the TIAGo Pro and the Ridgeback.

Structurally this is the MPO-700 minus the steering layer, and it shares that port's two source
facts: the MIT tree is `neo_simulation2` rather than the unlicensed `neo_mpo_500-2` the survey
ranked, and the meshes are millimetre-scale Collada. It differs in one welcome way -- its
``ODM_joint_type`` defaults to ``revolute`` and is forced to ``fixed`` only when an arm is mounted,
so unlike the MPO-700 the wheels are articulated without our intervention.

**Not consolidated with `build_mpo700_mjcf.py`, deliberately.** The two share the wrapper pattern and
the mesh pipeline but differ in body structure (steer layer or not), and the two remaining Neobotix
platforms are *differential*, a third shape again. Consolidating two of four shapes now and reworking
for the third is worse than consolidating once when all three are known -- see the port log.

Usage::

    python external/convert/build_mpo500_mjcf.py           # fetch, convert, write
    python external/convert/build_mpo500_mjcf.py --check    # rebuild and diff
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, mesh_scales, pose  # noqa: E402

NEO_URL = "https://github.com/neobotix/neo_simulation2.git"
NEO_COMMIT = "832041452c1a0199afea1e9b65adf37381e96214"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/mpo_500"
#: The wheel Collada is 8.3 MB -- every mecanum roller modelled -- so it gets a tighter budget than
#: the body. Both are cosmetic: the wheel's contact is a sphere, not this mesh.
TARGET_FACES = {"MPO-500-WHEEL": 2500}
DEFAULT_FACES = 4000

#: Our own top-level. The vendor's would work here (unlike the MPO-700's), but going through it
#: pulls in their Gazebo xacro and its ros2_control block; this keeps the expansion to geometry.
WRAPPER = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="mpo_500">
  <xacro:arg name="use_docking_adapter" default="false"/>
  <xacro:property name="ODM_joint_type" value="revolute"/>
  <xacro:property name="arm" value=""/>
  <xacro:property name="use_arm" value="false"/>
  <xacro:include filename="$(find neo_simulation2)/robots/mpo_500/urdf/mpo_500_body.urdf.xacro"/>
</robot>
"""

#: Corner order matching omni_drive's WHEEL_ORDER (front_left, front_right, rear_left, rear_right).
CORNERS = ("front_left", "front_right", "back_left", "back_right")
SENSOR_LINKS = ("lidar_1_link", "lidar_2_link")


def convert_meshes(source: Path, urdf: ET.Element) -> dict[str, str]:
    """DAE -> per-material OBJ -> decimated OBJ, with the palette kept beside the meshes."""
    scales = mesh_scales(urdf)
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (PKG / "meshes").glob("*.obj"):
        stale.unlink()
    wanted = {}
    for mesh in urdf.iter("mesh"):
        rel = mesh.get("filename").split("neo_simulation2/", 1)[1]
        wanted[Path(rel).stem] = source / rel
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "dae"
        staged.mkdir()
        for stem, path in wanted.items():
            shutil.copy2(path, staged / f"{stem}.dae")
        raw = Path(tmp) / "obj"
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "dae2obj.py"), str(staged), str(raw)],
            check=True, capture_output=True,
        )
        palette = json.loads((raw / "materials.json").read_text())
        (PKG / "meshes/mpo_500.materials.json").write_text(
            json.dumps(palette, indent=2, sort_keys=True) + "\n")
        for stem, parts in palette.items():
            budget = TARGET_FACES.get(stem, DEFAULT_FACES)
            for sub, _rgb in parts:
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(budget), "--no-materials",
                     str(raw / f"{sub}.obj"), str(PKG / "meshes" / f"{sub}.obj")],
                    check=True, cwd=ROOT, capture_output=True,
                )
    return scales


def build(urdf: ET.Element, shipped: set[str], scales: dict[str, str]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    colours = {
        sub: " ".join(f"{float(c):g}" for c in (*rgb[:3], 1.0))
        for parts in json.loads((PKG / "meshes/mpo_500.materials.json").read_text()).values()
        for sub, rgb in parts
    }

    def geoms(link: ET.Element, indent: str, wheel: bool = False) -> str:
        out = ""
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            stem = Path(mesh.get("filename")).stem
            xyz, quat = pose(visual)
            for sub in sorted(s for s in shipped if s == stem or s.startswith(f"{stem}__")):
                mat = f' material="{sub}_mat"' if sub in colours else ""
                out += f'{indent}<geom class="visual" mesh="{sub}"{mat} pos="{xyz}"{quat}/>\n'
        for collision in link.findall("collision"):
            shape = collision.find("geometry")[0]
            xyz, quat = pose(collision)
            if shape.tag == "sphere":
                out += (f'{indent}<geom class="wheel_collision" name="{link.get("name")}_tyre"'
                        f' size="{float(shape.get("radius")):g}" pos="{xyz}"/>\n')
            else:
                stem = Path(shape.get("filename")).stem
                sub = next(iter(sorted(s for s in shipped
                                       if s == stem or s.startswith(f"{stem}__"))))
                out += (f'{indent}<geom class="collision" mesh="{sub}"'
                        f' name="{link.get("name")}_collision" pos="{xyz}"{quat}/>\n')
        return out

    base = links["base_link"]
    sensors = ""
    for name in SENSOR_LINKS:
        pos, quat = pose(joints[name])
        sensors += SENSOR_BODY.format(
            name=name, body_pos=pos, body_quat=quat,
            geoms=geoms(links[name], "          "), **inertial(links[name]),
        )
    wheels = ""
    for corner in CORNERS:
        name = f"mpo_500_omni_wheel_{corner}_link"
        pos, quat = pose(joints[name])
        wheels += WHEEL_BODY.format(
            name=name, body_pos=pos, body_quat=quat, joint=joints[name].get("name"),
            geoms=geoms(links[name], "            ", wheel=True), **inertial(links[name]),
        )
    excludes = "".join(
        f'    <exclude body1="base_link" body2="mpo_500_omni_wheel_{c}_link"/>\n' for c in CORNERS
    )
    assets = "".join(
        f'    <material name="{sub}_mat" rgba="{rgba}"/>\n'
        for sub, rgba in sorted(colours.items()) if sub in shipped
    ) + "".join(
        f'    <mesh file="{sub}.obj" scale="{scales.get(sub.split("__")[0], "1 1 1")}"'
        f' inertia="shell"/>\n'
        for sub in sorted(shipped)
    )
    wheel_z = float(pose(joints[f"mpo_500_omni_wheel_{CORNERS[0]}_link"])[0].split()[2])
    radius = float(links[f"mpo_500_omni_wheel_{CORNERS[0]}_link"]
                   .find("collision/geometry/sphere").get("radius"))
    lidar_1, _ = pose(joints["lidar_1_link"])
    lidar_2, _ = pose(joints["lidar_2_link"])
    return TEMPLATE.format(
        commit=NEO_COMMIT, assets=assets, excludes=excludes,
        base_geoms=geoms(base, "        "), sensors=sensors, wheels=wheels,
        lidar_1=lidar_1, lidar_2=lidar_2, rest_height=f"{radius - wheel_z:g}",
        **{f"base_{k}": v for k, v in inertial(base).items()},
    )


SENSOR_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}        </body>
"""

WHEEL_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="{joint}" class="wheel"/>
{geoms}        </body>
"""

TEMPLATE = """<mujoco model="mpo_500">
  <!--
    Neobotix MPO-500 - a four-wheel OMNIDIRECTIONAL base, 72.8 kg. All four wheels drive; none
    steers.

    GENERATED by external/convert/build_mpo500_mjcf.py from neobotix/neo_simulation2 @ {commit}
    (MIT - see mpo_500_LICENSE). Do not hand-edit: re-run the generator.

    Drive: HOLONOMIC via omni wheels, so it uses `omni_drive`'s mecanum path - no swerve inverse
    kinematics (unlike the mpo_700, which steers) and no slip_factor (it does not turn by scrubbing).
    The vendor's only wheel macro is `mpo_500_omni_wheel` and there is no caster macro at all, which
    is what settles it.

    Wheel contact is the vendor's own SPHERE, not this 8.3 MB roller mesh: a sphere has no preferred
    rolling direction, which is what an omni wheel approximates. The mesh is cosmetic and decimated
    accordingly.

    One consequence to know: omni_drive integrates odometry from the ACHIEVED twist and models no
    wheel slip, so encoder odometry and ground truth coincide by construction. This platform cannot
    be used to study odometry drift.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <!--
    Every mesh is `inertia="shell"`. These are CAD surface exports split per material, so several
    sub-meshes are thin shells with no meaningful enclosed volume and MuJoCo refuses to integrate an
    inertia over them. It never needed to: every body carries the vendor's own explicit <inertial>.
  -->

  <default>
    <default class="mpo_500">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Near-frictionless spheres: the base is driven through the planar actuators below, and
             `priority` is what makes the low friction take effect at all - MuJoCo otherwise takes
             the MAXIMUM of the two contacting geoms' friction and the floor's value wins. -->
        <geom type="sphere" group="3" rgba="0.05 0.05 0.05 0.4"
              friction="0.02 0.005 0.0001" priority="2"/>
      </default>
      <default class="wheel">
        <!-- The vendor declares these `revolute` with +-1e16 limits, which is "unlimited" spelled
             awkwardly; emitted as an unlimited hinge so MuJoCo does not carry a limit constraint
             that can never engage. -->
        <joint axis="0 1 0" damping="1.0" armature="0.05" limited="false"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <contact>
    <!--
      The vendor's base collision IS the full body mesh, and MuJoCo convex-hulls a collision mesh, so
      the hull closes over the wheel arches and overlaps the wheels inside them. Here the wheels are
      direct children of base_link, so MuJoCo's automatic parent-child exclusion would cover them -
      these are kept for symmetry with the mpo_700 (where the steering link makes each wheel a
      grandchild and the automatic exclusion misses it) and so the pair set does not silently depend
      on tree depth.
    -->
{excludes}  </contact>

  <worldbody>
    <body name="base_link" childclass="mpo_500">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <!-- The vendor's own two scanner mounts, front and rear on the centreline. -->
      <site name="lidar_1" pos="{lidar_1}" size="0.01" rgba="1 0 0 0.6"/>
      <site name="lidar_2" pos="{lidar_2}" size="0.01" rgba="1 0 0 0.6"/>
{base_geoms}{sensors}{wheels}    </body>
  </worldbody>

  <actuator>
    <!-- The planar drive. Neobotix publishes 1.5 m/s for the MPO-500; forcerange is sized to move
         72.8 kg. -->
    <velocity name="base_vx" joint="base_free" gear="1 0 0 0 0 0" kv="9000" ctrlrange="-1.5 1.5" forcerange="-900 900"/>
    <velocity name="base_vy" joint="base_free" gear="0 1 0 0 0 0" kv="9000" ctrlrange="-1.5 1.5" forcerange="-900 900"/>
    <velocity name="base_wz" joint="base_free" gear="0 0 0 0 0 1" kv="4500" ctrlrange="-1.5 1.5" forcerange="-500 500"/>
    <!-- Observational wheel servos, driven from the mecanum inverse kinematics so the wheels turn,
         and turn differently when strafing. They are not the motive force. -->
    <velocity name="wheel_front_left_motor" joint="mpo_500_omni_wheel_front_left_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="wheel_front_right_motor" joint="mpo_500_omni_wheel_front_right_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="wheel_back_left_motor" joint="mpo_500_omni_wheel_back_left_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="wheel_back_right_motor" joint="mpo_500_omni_wheel_back_right_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
  </actuator>

  <keyframe>
    <!-- base_link rides at wheel radius minus the wheel-centre height, which is NEGATIVE here as on
         the mpo_700: the vendor's base_link frame sits below the tyre contact. -->
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("neo_simulation2", NEO_URL, NEO_COMMIT)
    target = PKG / "mpo_500.xml"
    with tempfile.TemporaryDirectory() as tmp:
        urdf = expand_xacro({"neo_simulation2": source}, Path("mpo_500.urdf.xacro"),
                            Path(tmp), wrapper=WRAPPER)

    if args.check:
        shipped = {p.stem for p in (PKG / "meshes").glob("*.obj")}
        if not target.exists() or target.read_text() != build(urdf, shipped, mesh_scales(urdf)):
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {NEO_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    scales = convert_meshes(source, urdf)
    shipped = {p.stem for p in (PKG / "meshes").glob("*.obj")}
    shutil.copy2(source / "LICENSE", PKG / "mpo_500_LICENSE")
    target.write_text(build(urdf, shipped, scales))
    print(f"wrote {target} + meshes + mpo_500_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
