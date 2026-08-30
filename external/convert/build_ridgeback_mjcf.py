#!/usr/bin/env python3
"""Build roqsim's Clearpath Ridgeback MJCF from `clearpathrobotics/clearpath_common`.

The batch's first **holonomic** base. Every other wheeled robot here is differential or skid-steer
and uses ``diff_drive``; the Ridgeback's four mecanum wheels strafe, so it uses ``omni_drive`` -- the
plugin written for PAL's OMNI base and, until now, used by nothing else. That is the whole reason
the platform ledger recorded this port as **no new capability**: the assessment checked
``roqsim_mobile/plugins/omni_drive.py`` rather than reasoning about mecanum, and found the capability
already there.

Same source package as ``husky_a200`` (a200) and ``clearpath_jackal`` (j100), so the conversion path
was proven before this started. Every mass, inertia, link offset and collision primitive below is
Clearpath's own value, read out of the expanded xacro.

Three things make this the cheapest port of the batch:

* **The meshes need no conversion at all.** They are STL, which MuJoCo loads directly, and the
  largest is 39 kB. No Collada, no Blender, no decimation, and therefore none of the axis and scale
  traps that cost the ROSbot and the Doosan their iterations.
* **The vendor ships a real collision mesh**, ``body-collision.stl`` (16 kB against the 29 kB
  visual), used as shipped.
* **The rocker suspension is fixed** in this description, so the tree is a chassis, two rockers and
  four wheels with nothing articulated but the wheels.

**The wheels are load carriers, not the drive.** ``omni_drive`` applies the commanded twist to three
velocity actuators on the base's free joint and spins the wheels only so the visuals and
``joint_states`` are right -- because modelling ~9 passive rollers per wheel would mean ~36 extra
bodies and MuJoCo's cylinder-on-plane contact frame is not roller-aligned anyway. The wheel geoms are
therefore given near-frictionless tyres, exactly as ``tiago_pro`` does.

Usage::

    python external/convert/build_ridgeback_mjcf.py           # fetch, expand, copy meshes, write
    python external/convert/build_ridgeback_mjcf.py --check   # rebuild and diff
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, link_visuals, pose, rpy_to_quat  # noqa: E402

CLEARPATH_URL = "https://github.com/clearpathrobotics/clearpath_common.git"
CLEARPATH_COMMIT = "b0f6d920422ad302372a1c65e31d61648da884ed"
#: clearpath_control is fetched only so the description's `$(find ...)` default resolves; nothing
#: from it is read into the model.
PACKAGES = ("clearpath_platform_description", "clearpath_control")

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/ridgeback"

#: Clearpath's own palette, from clearpath_platform_description/urdf/common.urdf.xacro.
PALETTE = {
    "clearpath_black": "0.15 0.15 0.15 1",
    "clearpath_yellow": "0.8 0.8 0.0 1",
    "clearpath_white": "1.0 1.0 1.0 1",
    "clearpath_red": "0.8 0.0 0.0 1",
}

WRAPPER = """<?xml version="1.0"?>
<robot name="ridgeback" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find clearpath_platform_description)/urdf/r100/r100.urdf.xacro"/>
  <xacro:r100 control="omni_4wd" front_wheels="mecanum" rear_wheels="mecanum"/>
</robot>
"""

#: Fixed links whose visuals ride on the chassis, with their joint offsets.
CHASSIS_PARTS = [
    "left_side_cover_link", "right_side_cover_link", "front_cover_link", "rear_cover_link",
    "front_lights_link", "rear_lights_link", "axle_link", "riser_link", "top_link",
]
WHEELS = ["front_left", "front_right", "rear_left", "rear_right"]


def copy_meshes(description: Path) -> None:
    """STL straight through. No conversion: MuJoCo loads STL and the largest here is 39 kB."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (PKG / "meshes").glob("*.stl"):
        stale.unlink()
    meshes = description / "meshes/r100"
    for stl in sorted(meshes.glob("*.stl")):
        shutil.copy2(stl, PKG / "meshes" / stl.name)
    shutil.copy2(meshes / "wheels/mecanum.stl", PKG / "meshes" / "mecanum.stl")


def build(urdf: ET.Element) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    def inertial(name):
        i = links[name].find("inertial")
        origin = i.find("origin")
        inertia = i.find("inertia")
        return {
            "pos": ((origin.get("xyz") if origin is not None else None) or "0 0 0").replace(",", " "),
            "mass": i.find("mass").get("value"),
            "diaginertia": " ".join(inertia.get(k) for k in ("ixx", "iyy", "izz")),
        }

    def offset_of(joint_name, base=(0.0, 0.0, 0.0)):
        origin = joints[joint_name].find("origin")
        xyz = ((origin.get("xyz") if origin is not None else None) or "0 0 0").replace(",", " ")
        return tuple(b + float(v) for b, v in zip(base, xyz.split(), strict=True))

    chassis = inertial("chassis_link")
    chassis_geoms = link_visuals(links["chassis_link"], "        ", (0.0, 0.0, 0.0),
                                  default_material="clearpath_black")
    for part in CHASSIS_PARTS:
        joint = next(j for j, e in joints.items() if e.find("child").get("link") == part)
        offset = offset_of(joint)
        if part == "top_link":            # hangs off the riser, not the chassis
            offset = offset_of(joint, offset_of("riser_link_joint"))
        chassis_geoms += link_visuals(links[part], "        ", offset, default_material="clearpath_black")

    axle_z = offset_of("axle_joint")[2]
    wheel_cyl = links["front_left_wheel_link"].find("collision/geometry/cylinder")
    radius = float(wheel_cyl.get("radius"))
    half_width = float(wheel_cyl.get("length")) / 2
    _, wheel_quat = pose(links["front_left_wheel_link"].find("collision"))

    rockers = ""
    for side in ("front", "rear"):
        rocker_x = offset_of(f"{side}_rocker")[0]
        rocker = inertial(f"{side}_rocker_link")
        wheels = ""
        for lr in ("left", "right"):
            name = f"{side}_{lr}"
            y = offset_of(f"{name}_wheel_joint")[1]
            wheels += WHEEL_BODY.format(
                name=name, y=f"{y:g}", radius=f"{radius:g}", half_width=f"{half_width:g}",
                quat=wheel_quat, geoms=link_visuals(links[f"{name}_wheel_link"], "            ",
                                                   default_material="clearpath_black"),
                **inertial(f"{name}_wheel_link"),
            )
        rockers += ROCKER_BODY.format(
            side=side, x=f"{rocker_x:g}", z=f"{axle_z:g}", wheels=wheels,
            geoms=link_visuals(links[f"{side}_rocker_link"], "          ",
                                 default_material="clearpath_black"),
            **rocker,
        )

    materials = "".join(
        f'    <material name="{n}" rgba="{rgba}"/>\n' for n, rgba in sorted(PALETTE.items())
    )
    assets = "".join(
        f'    <mesh file="{p.name}"/>\n' for p in sorted((PKG / "meshes").glob("*.stl"))
    )
    return TEMPLATE.format(
        commit=CLEARPATH_COMMIT, materials=materials, assets=assets,
        chassis_mass=chassis["mass"], chassis_pos=chassis["pos"],
        chassis_inertia=chassis["diaginertia"], chassis_geoms=chassis_geoms,
        rockers=rockers, rest_height=f"{radius - axle_z:g}",
    )


WHEEL_BODY = """          <body name="{name}_wheel_link" pos="0 {y} 0">
            <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
            <joint name="{name}_wheel_joint" class="wheel"/>
{geoms}            <geom class="wheel_collision" name="{name}_wheel_tyre"{quat}
                  size="{radius} {half_width}"/>
          </body>
"""

ROCKER_BODY = """        <body name="{side}_rocker_link" pos="{x} 0 {z}">
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}{wheels}        </body>
"""

TEMPLATE = """<mujoco model="ridgeback">
  <!--
    Clearpath Ridgeback (4-wheel mecanum, holonomic) for MuJoCo.

    GENERATED by external/convert/build_ridgeback_mjcf.py from clearpathrobotics/clearpath_common's
    clearpath_platform_description @ {commit} (BSD-3-Clause - see ridgeback_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia, link offset and collision primitive
    below is Clearpath's own value, from the expanded r100 xacro. Same package husky_a200 (a200) and
    clearpath_jackal (j100) came from.

    Drive: HOLONOMIC, and the only one in roqsim_mobile. The four mecanum wheels strafe, so this uses
    `omni_drive`, not `diff_drive` - and therefore no slip_factor, because it does not turn by
    scrubbing. The commanded twist is applied to three velocity actuators on the base free joint; the
    wheel servos exist so the visuals and joint_states are right, not as the motive force, which is
    why the tyres are near-frictionless. Modelling the ~9 passive rollers per wheel would mean ~36
    extra bodies, and MuJoCo's cylinder-on-plane contact frame is not roller-aligned anyway.

    One consequence worth knowing before using this model: omni_drive integrates odometry from the
    ACHIEVED twist and models no wheel slip, so encoder odometry and ground truth coincide by
    construction. This platform cannot be used to study odometry drift.

    Meshes are STL, copied as shipped - no conversion, no decimation, and none of the axis or scale
    traps that come with Collada. Body collision is Clearpath's own body-collision.stl.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="ridgeback">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Near-frictionless: an omni wheel IS this, in the directions that matter. The base is
             driven through the planar actuators below, not through these. Same treatment tiago_pro
             gives its mecanum tyres. -->
        <geom type="cylinder" group="3" rgba="0.05 0.05 0.05 0.4"
              friction="0.02 0.005 0.0001" priority="2"/>
      </default>
      <default class="wheel">
        <joint axis="0 1 0" damping="0.1" armature="0.05"/>
      </default>
    </default>
  </default>

  <asset>
{materials}{assets}  </asset>

  <worldbody>
    <body name="base_link" childclass="ridgeback">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <body name="chassis_link" pos="0 0 0">
        <inertial pos="{chassis_pos}" mass="{chassis_mass}" diaginertia="{chassis_inertia}"/>
{chassis_geoms}        <geom class="collision" mesh="body-collision"/>
        <!-- Ridgebacks are usually fitted with a scanner at each front corner; this is a single
             planar scan on the deck. Height is an assumption - see the port log. -->
        <site name="lidar" pos="0.3 0 0.29" size="0.008" rgba="1 0 0 0.6"/>
{rockers}      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- The planar drive. Limits are Clearpath's published figures for the Ridgeback: 1.1 m/s
         translation and 2.0 rad/s yaw. forcerange is sized to move ~196 kg. -->
    <velocity name="base_vx" joint="base_free" gear="1 0 0 0 0 0" kv="25000" ctrlrange="-1.1 1.1" forcerange="-2000 2000"/>
    <velocity name="base_vy" joint="base_free" gear="0 1 0 0 0 0" kv="25000" ctrlrange="-1.1 1.1" forcerange="-2000 2000"/>
    <velocity name="base_wz" joint="base_free" gear="0 0 0 0 0 1" kv="12500" ctrlrange="-2.0 2.0" forcerange="-1200 1200"/>
    <!-- Observational wheel servos: driven from the mecanum inverse kinematics so the wheels turn,
         and turn differently when strafing. They transmit almost nothing. -->
    <velocity name="front_left_wheel_motor" joint="front_left_wheel_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="front_right_wheel_motor" joint="front_right_wheel_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="rear_left_wheel_motor" joint="rear_left_wheel_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="rear_right_wheel_motor" joint="rear_right_wheel_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
  </actuator>

  <keyframe>
    <!-- base_link rides at wheel radius minus the axle height; spawn a hair above so it settles. -->
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sources = {
        name: resolve_source("clearpath_common", CLEARPATH_URL, CLEARPATH_COMMIT,
                             sparse=name, subdir=name)
        for name in PACKAGES
    }
    target = PKG / "ridgeback.xml"

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            xml = build(expand_xacro(sources,
                                     sources["clearpath_platform_description"]
                                     / "urdf/r100/r100.urdf.xacro", Path(tmp), wrapper=WRAPPER))
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {CLEARPATH_COMMIT[:12]}")
        return 0

    copy_meshes(sources["clearpath_platform_description"])
    with tempfile.TemporaryDirectory() as tmp:
        xml = build(expand_xacro(sources,
                                     sources["clearpath_platform_description"]
                                     / "urdf/r100/r100.urdf.xacro", Path(tmp), wrapper=WRAPPER))
    shutil.copy2(sources["clearpath_platform_description"].parent / "LICENSE",
                 PKG / "ridgeback_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + meshes + ridgeback_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
