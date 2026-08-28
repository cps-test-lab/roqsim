#!/usr/bin/env python3
"""Build roqsim's Interbotix WidowX XSeries turret MJCF from `Interbotix/interbotix_ros_turrets`.

A 2-DOF pan/tilt, 1.07 kg. The substrate's **smallest actuated mechanism** and its first
non-manipulator articulated payload: a sensor mount whose whole purpose is to aim something. It reuses
``spawn_arm`` + ``arm_controller`` unchanged -- an "arm" to those plugins is any chain of position-
controlled joints, and two is a chain.

**Two source facts the ledger recorded as corrections, both worth carrying.** The 2026-08-24 survey
listed this platform with 183 stars and evidence pointing at ``interbotix_ros_manipulators`` -- the
*arms* repository. The turret is not there: it lives in ``interbotix_ros_turrets``, which has 7 stars
and **no ROS 2 Jazzy branch at all** (humble, kinetic, main, melodic, noetic; ``main`` is still ROS 1
catkin). So this is built from ``humble``, the only ament branch, and the platform does not meet the
"fully supported ROS 2 Jazzy" premise the survey admitted it under. It ports regardless -- a URDF is a
URDF -- and that mismatch is recorded rather than smoothed over.

Six variants share one description shape (``pxxls``, ``pxxls_cam``, ``vxxmd``, ``vxxms``, ``wxxmd``,
``wxxms``), differing in servo model and mesh. ``wxxms`` is built because the substrate already
carries two Interbotix arms and a WidowX turret pairs with them; :data:`VARIANT` is the only thing
that changes to build another.

Usage::

    python external/convert/build_wxxms_mjcf.py           # fetch, copy meshes, write
    python external/convert/build_wxxms_mjcf.py --check    # rebuild and diff
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, mesh_scales, pose  # noqa: E402

TURRET_URL = "https://github.com/Interbotix/interbotix_ros_turrets.git"
#: The `humble` branch: the only ament one. See the module docstring.
TURRET_COMMIT = "bd5fd2df2a13a46bb51f12ac3db9df1852572300"
VARIANT = "wxxms"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / f"roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/{VARIANT}"
JOINTS = ("pan", "tilt")


def copy_meshes(description: Path) -> None:
    """STL straight through -- MuJoCo loads STL, and the three here total 168 kB."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (PKG / "meshes").glob("*.stl"):
        stale.unlink()
    for stl in sorted((description / f"meshes/{VARIANT}_meshes").glob("*.stl")):
        shutil.copy2(stl, PKG / "meshes" / stl.name)


def build(urdf: ET.Element) -> str:
    # Interbotix namespaces every link as `<robot_name>/<link>`. Stripped here: the prefix is a
    # deployment detail of their multi-robot launch files, and `spawn_arm` applies roqsim's own
    # prefix on top, which would otherwise read `t_wxxms/pan_link`.
    def short(name: str) -> str:
        return name.split("/", 1)[-1]

    links = {short(link.get("name")): link for link in urdf.findall("link")}
    joints = {short(j.find("child").get("link")): j for j in urdf.findall("joint")}
    scales = mesh_scales(urdf)

    def geoms(link: ET.Element, indent: str) -> str:
        out = ""
        for tag, cls in (("visual", "visual"), ("collision", "collision")):
            for element in link.findall(tag):
                mesh = element.find("geometry/mesh")
                if mesh is None:
                    continue
                stem = Path(mesh.get("filename")).stem
                xyz, quat = pose(element)
                name = f' name="{short(link.get("name"))}_collision"' if tag == "collision" else ""
                out += f'{indent}<geom class="{cls}" mesh="{stem}"{name} pos="{xyz}"{quat}/>\n'
        return out

    limits = {}
    for name in JOINTS:
        joint = joints[f"{name}_link"]
        limit = joint.find("limit")
        pos, quat = pose(joint)
        limits[name] = {
            "pos": pos, "quat": quat,
            "axis": joint.find("axis").get("xyz").replace(",", " "),
            "lower": float(limit.get("lower")), "upper": float(limit.get("upper")),
            "effort": float(limit.get("effort")), "velocity": float(limit.get("velocity")),
        }

    assets = "".join(
        f'    <mesh file="{stem}.stl" scale="{scale}"/>\n' for stem, scale in sorted(scales.items())
    )
    return TEMPLATE.format(
        commit=TURRET_COMMIT, variant=VARIANT, assets=assets,
        base_geoms=geoms(links["base_link"], "      "),
        pan_geoms=geoms(links["pan_link"], "        "),
        tilt_geoms=geoms(links["tilt_link"], "          "),
        surface_pos=pose(joints["surface_link"])[0],
        # The inertial COM is `<link>_com`, distinct from the joint origin's `<link>_pos`: both are
        # positions and both belong to the same link, so sharing one placeholder name silently
        # substitutes one for the other -- which is a body in the wrong place, not an error.
        **{f"{n}_{'com' if k == 'pos' else k}": v
           for n, link in (("base", "base_link"), ("pan", "pan_link"), ("tilt", "tilt_link"))
           for k, v in inertial(links[link]).items()},
        **{f"{n}_{k}": v for n, d in limits.items() for k, v in d.items()},
    )


TEMPLATE = """<mujoco model="{variant}">
  <!--
    Interbotix WidowX XSeries turret ({variant}) - a 2-DOF pan/tilt, 1.07 kg.

    GENERATED by external/convert/build_wxxms_mjcf.py from Interbotix/interbotix_ros_turrets @
    {commit} (BSD-3-Clause - see {variant}_LICENSE). Do not hand-edit: re-run the generator.
    Every mass, inertia, offset, limit and effort below is Interbotix's own value.

    The substrate's smallest actuated mechanism, and its first articulated thing that is not a
    manipulator: a sensor mount whose purpose is to AIM something. It uses `spawn_arm` +
    `arm_controller` unchanged, because an "arm" to those plugins is a chain of position-controlled
    joints and two is a chain. Mount a roqsim_sensors camera or lidar on the `surface` site to make
    it useful.

    Built from the vendor's `humble` branch - the only ament one. This platform has NO ROS 2 Jazzy
    branch, so it does not meet the "fully supported ROS 2 Jazzy" premise the corpus survey admitted
    it under; the URDF ports regardless. See the port log and the platform ledger.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="{variant}">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" material="{variant}_black"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="joint">
        <!-- armature 0.005 against a DYNAMIXEL XM430's reflected rotor inertia. The servo gains
             below are sized so kv*dt/I stays well under 1 at a 2 ms step: the tilt link's inertia is
             ~7e-4 kg*m^2, three orders of magnitude below an industrial arm's, so gains that are
             gentle on a UR10e would chatter here. This is the opposite end of the scale from the
             Doosan, whose distal joints needed gains LOWERED for the same reason. -->
        <joint armature="0.005" damping="0.5"/>
      </default>
      <default class="servo">
        <!-- forcerange is Interbotix's own 3 N*m effort limit, not a guess. -->
        <position kp="8" kv="1.2" forcerange="-3 3"/>
      </default>
    </default>
  </default>

  <asset>
    <material name="{variant}_black" rgba="0.15 0.15 0.15 1"/>
{assets}  </asset>

  <contact>
    <!-- Chain neighbours. MuJoCo's `filterparent` is skipped when the parent is the world body, and
         base_link is welded to world, so every parent/child pair in this chain is measured -- the
         same trap the m1013 port documents. The vendor's collision geometry IS its visual mesh, and
         the convex hulls of the base and the pan overlap where the pan seats into the base: measured
         2.6 mm of penetration at the home pose, whose constraint force exactly cancelled the pan
         servo's 3 N*m and left the joint immovable while reporting full actuator effort.

         base_link/tilt_link is a grandparent pair and was never auto-excluded either. -->
    <exclude body1="base_link" body2="pan_link"/>
    <exclude body1="pan_link" body2="tilt_link"/>
    <exclude body1="base_link" body2="tilt_link"/>
  </contact>

  <worldbody>
    <body name="base_link" childclass="{variant}">
      <inertial pos="{base_com}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
{base_geoms}      <body name="pan_link" pos="{pan_pos}"{pan_quat}>
        <inertial pos="{pan_com}" mass="{pan_mass}" diaginertia="{pan_diaginertia}"/>
        <joint name="pan" class="joint" axis="{pan_axis}" range="{pan_lower} {pan_upper}"/>
{pan_geoms}        <body name="tilt_link" pos="{tilt_pos}"{tilt_quat}>
          <inertial pos="{tilt_com}" mass="{tilt_mass}" diaginertia="{tilt_diaginertia}"/>
          <joint name="tilt" class="joint" axis="{tilt_axis}" range="{tilt_lower} {tilt_upper}"/>
{tilt_geoms}          <!-- The vendor's own mounting frame: what a camera or scanner bolts to, and the
               reason this mechanism exists. Its pose is the description's `surface` joint. -->
          <site name="surface" pos="{surface_pos}" size="0.005" rgba="0 0.6 0 0.6"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position class="servo" name="pan" joint="pan" ctrlrange="{pan_lower} {pan_upper}"/>
    <position class="servo" name="tilt" joint="tilt" ctrlrange="{tilt_lower} {tilt_upper}"/>
  </actuator>

  <keyframe>
    <!-- Level and forward: the pose a pan/tilt is calibrated from. -->
    <key name="home" qpos="0 0" ctrl="0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("interbotix_turrets", TURRET_URL, TURRET_COMMIT)
    description = source / "interbotix_ros_xsturrets/interbotix_xsturret_descriptions"
    target = PKG / f"{VARIANT}.xml"

    def fresh() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            urdf = expand_xacro({"interbotix_xsturret_descriptions": description},
                                description / f"urdf/{VARIANT}.urdf.xacro", Path(tmp))
        return build(urdf)

    if args.check:
        if not target.exists() or target.read_text() != fresh():
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {TURRET_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    copy_meshes(description)
    shutil.copy2(source / "LICENSE", PKG / f"{VARIANT}_LICENSE")
    target.write_text(fresh())
    print(f"wrote {target} + meshes + {VARIANT}_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
