#!/usr/bin/env python3
"""Build roqsim's Clearpath Warthog MJCF from `clearpathrobotics/clearpath_common`.

The fourth model from this package after ``husky_a200`` (a200), ``clearpath_jackal`` (j100) and
``ridgeback`` (r100), and by far the largest robot in ``roqsim_mobile``: 260 kg, 0.3 m wheels, a
1.136 m track. A 4-wheel outdoor skid-steer, so ``diff_drive`` with a ``slip_factor`` -- the
husky/jackal/rosbot/panther class rather than the Ridgeback's holonomic one.

Every mass, inertia, link offset and collision primitive below is Clearpath's own value, read out of
the expanded w200 xacro. The drive limits are Clearpath's own too, from
``clearpath_control/config/w200/control/diff_4wd.yaml``.

Three things this port did not have to pay for:

* **The meshes need no conversion.** STL, loaded by MuJoCo directly. No Collada, no Blender, and
  none of the axis or scale traps that cost the ROSbot and the Doosan their iterations.
* **The vendor ships real collision meshes.** ``chassis-collision.stl`` is 56 triangles against the
  visual's 2489, and ``fenders.stl`` is 60. Both are used as shipped.
* **Nothing is articulated but the wheels.** The rocker/differential suspension is a set of fixed
  joints in this description, so the tree is a chassis, two diff units and four wheels.

The one thing it did pay for is the yaw calibration -- see ``slip_factor`` in the manifest and the
port log. Clearpath publishes its own compensation here, and unusually it is legible: ``diff_4wd``
declares ``wheel_separation: 1.5`` for a robot whose URDF track is 1.13642 m, then multiplies it by
1.125. That is a 48% inflation of the geometric track, and it is the real driver's ICR compensation
under another name.

Usage::

    python external/convert/build_warthog_mjcf.py           # fetch, expand, copy meshes, write
    python external/convert/build_warthog_mjcf.py --check   # rebuild and diff
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
from urdf_source import expand_xacro, inertial, link_visuals, pose  # noqa: E402

CLEARPATH_URL = "https://github.com/clearpathrobotics/clearpath_common.git"
CLEARPATH_COMMIT = "b0f6d920422ad302372a1c65e31d61648da884ed"
#: clearpath_control is fetched so the description's `$(find ...)` default resolves -- and, unlike
#: the other Clearpath ports, one file in it IS read: config/w200/control/diff_4wd.yaml supplies the
#: published velocity and acceleration limits quoted in the manifest.
PACKAGES = ("clearpath_platform_description", "clearpath_control")

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/warthog"

#: Clearpath's own palette, from clearpath_platform_description/urdf/common.urdf.xacro, plus the
#: warthog_yellow the w200 macro selects by default.
PALETTE = {
    "clearpath_black": "0.15 0.15 0.15 1",
    "clearpath_dark_grey": "0.2 0.2 0.2 1",
    "clearpath_light_grey": "0.4 0.4 0.4 1",
    "clearpath_red": "0.8 0.0 0.0 1",
    "clearpath_white": "1.0 1.0 1.0 1",
    "warthog_yellow": "0.95 0.816 0.082 1",
}

WRAPPER = """<?xml version="1.0"?>
<robot name="warthog" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find clearpath_platform_description)/urdf/w200/w200.urdf.xacro"/>
  <xacro:w200 control="diff_4wd" front_wheels="outdoor" rear_wheels="outdoor"/>
</robot>
"""

#: Zero-mass links fixed to the chassis whose visuals ride on it.
CHASSIS_PARTS = ["diff_link"]
#: Zero-mass links fixed to a diff unit. Both lights sit at the unit's own origin and differ only by
#: a pi rotation, because light.stl is authored off-centre (x 0.486..0.668) -- the rotation is what
#: puts one at each end. Reproduced rather than "corrected".
DIFF_UNIT_PARTS = ["headlight_link", "taillight_link"]
SIDES = {"left": 1, "right": -1}


def copy_meshes(description: Path) -> None:
    """STL straight through. MuJoCo loads STL, and the vendor already ships collision meshes."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (PKG / "meshes").glob("*.stl"):
        stale.unlink()
    meshes = description / "meshes/w200"
    # Only what the default w200 (no attachments) actually references. bulkhead/generator/
    # arm-mount-plate belong to attachments this model does not instantiate, and shipping an unused
    # 215 kB mesh is what left the Raspberry Pi Mouse carrying a top plate nothing references.
    for name in ("chassis.stl", "chassis-collision.stl", "diff-link.stl", "e-stop.stl",
                 "fenders.stl", "light.stl", "rocker.stl", "susp-link.stl"):
        shutil.copy2(meshes / name, PKG / "meshes" / name)
    shutil.copy2(meshes / "wheels/outdoor.stl", PKG / "meshes" / "outdoor.stl")


def build(urdf: ET.Element) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    def offset_of(joint_name: str) -> tuple[float, ...]:
        origin = joints[joint_name].find("origin")
        xyz = ((origin.get("xyz") if origin is not None else None) or "0 0 0").replace(",", " ")
        return tuple(float(v) for v in xyz.split())

    chassis_geoms = link_visuals(links["chassis_link"], "          ",
                                 default_material="clearpath_black")
    for part in CHASSIS_PARTS:
        joint = next(j for j, e in joints.items() if e.find("child").get("link") == part)
        chassis_geoms += link_visuals(links[part], "          ", offset_of(joint),
                                      default_material="clearpath_light_grey")

    wheel_radius = float(links["front_left_wheel_link"].find("collision/geometry/cylinder")
                         .get("radius"))
    wheel_half = float(links["front_left_wheel_link"].find("collision/geometry/cylinder")
                       .get("length")) / 2
    _, wheel_quat = pose(links["front_left_wheel_link"].find("collision"))

    units = ""
    for side in SIDES:
        unit = f"{side}_diff_unit_link"
        unit_geoms = link_visuals(links[unit], "            ",
                                  default_material="warthog_yellow")
        for part in DIFF_UNIT_PARTS:
            name = f"{side}_diff_unit_{part}"
            joint = next(j for j, e in joints.items() if e.find("child").get("link") == name)
            unit_geoms += link_visuals(links[name], "            ", offset_of(joint),
                                       default_material="clearpath_white")
        wheels = ""
        for end in ("front", "rear"):
            wheel = f"{end}_{side}_wheel_link"
            x, _, z = offset_of(f"{end}_{side}_wheel_joint")
            wheels += WHEEL_BODY.format(
                name=f"{end}_{side}", x=f"{x:g}", z=f"{z:g}", quat=wheel_quat,
                radius=f"{wheel_radius:g}", half_width=f"{wheel_half:g}",
                geoms=link_visuals(links[wheel], "              ",
                                   default_material="clearpath_dark_grey"),
                **inertial(links[wheel]),
            )
        units += DIFF_UNIT_BODY.format(
            side=side, y=f"{offset_of(f'{side}_diff_unit_joint')[1]:g}",
            geoms=unit_geoms, wheels=wheels, **inertial(links[unit]),
        )

    materials = "".join(
        f'    <material name="{n}" rgba="{rgba}"/>\n' for n, rgba in sorted(PALETTE.items())
    )
    assets = "".join(
        f'    <mesh file="{p.name}"/>\n' for p in sorted((PKG / "meshes").glob("*.stl"))
    )
    chassis_z = offset_of("base_link_joint")[2]
    wheel_z = chassis_z + offset_of("front_left_wheel_joint")[2]
    return TEMPLATE.format(
        commit=CLEARPATH_COMMIT, materials=materials, assets=assets,
        chassis_z=f"{chassis_z:g}", chassis_geoms=chassis_geoms, units=units,
        rest_height=f"{wheel_radius - wheel_z:g}",
        **{f"chassis_{k}": v for k, v in inertial(links["chassis_link"]).items()},
    )


WHEEL_BODY = """              <body name="{name}_wheel_link" pos="{x} 0 {z}">
                <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
                <joint name="{name}_wheel_joint" class="wheel"/>
{geoms}                <geom class="wheel_collision" name="{name}_wheel_tyre"{quat}
                      size="{radius} {half_width}"/>
              </body>
"""

DIFF_UNIT_BODY = """            <body name="{side}_diff_unit_link" pos="0 {y} 0">
              <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}              <geom class="collision" mesh="fenders" name="{side}_fender_collision"/>
{wheels}            </body>
"""

TEMPLATE = """<mujoco model="warthog">
  <!--
    Clearpath Warthog W200 (4-wheel outdoor skid-steer) for MuJoCo.

    GENERATED by external/convert/build_warthog_mjcf.py from clearpathrobotics/clearpath_common's
    clearpath_platform_description @ {commit} (BSD-3-Clause - see warthog_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia, link offset and collision primitive
    below is Clearpath's own value, from the expanded w200 xacro. Same package as husky_a200 (a200),
    clearpath_jackal (j100) and ridgeback (r100).

    The largest robot in roqsim_mobile: 260 kg against the Husky's 50, on 0.3 m wheels over a
    1.136 m track. It is a skid-steer, so it turns by scrubbing and carries a slip_factor - see the
    manifest, where the measured achieved/commanded ratios are recorded, and the port log for why
    the vendor's own number could not simply be adopted.

    The rocker/differential suspension is FIXED in this description. Nothing articulates but the
    four wheels, so a Warthog crossing rough ground here will not articulate its rockers the way the
    real machine does; its chassis is rigid.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="warthog">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Real driven tyres: this base moves because these push on the ground, unlike the
             Ridgeback's near-frictionless mecanum load carriers. -->
        <geom type="cylinder" group="3" rgba="0.05 0.05 0.05 0.4" friction="1.0 0.005 0.0001"/>
      </default>
      <default class="wheel">
        <joint axis="0 1 0" damping="1.0" armature="0.5"/>
      </default>
    </default>
  </default>

  <asset>
{materials}{assets}  </asset>

  <worldbody>
    <body name="base_link" childclass="warthog">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <body name="chassis_link" pos="0 0 {chassis_z}">
        <inertial pos="{chassis_pos}" mass="{chassis_mass}" diaginertia="{chassis_diaginertia}"/>
{chassis_geoms}        <geom class="collision" mesh="chassis-collision" name="chassis_collision"/>
        <!-- The description ships no scanner. This one clears the fenders, which reach 0.533 m
             above base_link and would otherwise be the only thing a deck-height scan could see.
             The height is an assumption - see the port log. -->
        <site name="lidar" pos="0.5 0 0.6" size="0.01" rgba="1 0 0 0.6"/>
{units}      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servos, one per wheel. ctrlrange is diff_4wd.yaml's 5.0 m/s over the 0.3 m wheel
         radius (16.7 rad/s), rounded up; forcerange is sized to accelerate 260 kg at that file's
         published 2.5 m/s^2 with margin for the scrub a skid-steer turn imposes. -->
    <velocity name="front_left_wheel_motor" joint="front_left_wheel_joint" kv="600" ctrlrange="-17 17" forcerange="-900 900"/>
    <velocity name="front_right_wheel_motor" joint="front_right_wheel_joint" kv="600" ctrlrange="-17 17" forcerange="-900 900"/>
    <velocity name="rear_left_wheel_motor" joint="rear_left_wheel_joint" kv="600" ctrlrange="-17 17" forcerange="-900 900"/>
    <velocity name="rear_right_wheel_motor" joint="rear_right_wheel_joint" kv="600" ctrlrange="-17 17" forcerange="-900 900"/>
  </actuator>

  <keyframe>
    <!-- base_link rides at wheel radius minus the wheel-centre height; spawn a hair above so it
         settles onto its tyres rather than starting interpenetrated. -->
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
    description = sources["clearpath_platform_description"]
    target = PKG / "warthog.xml"

    def fresh() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            return build(expand_xacro(sources, description / "urdf/w200/w200.urdf.xacro",
                                      Path(tmp), wrapper=WRAPPER))

    if args.check:
        if not target.exists() or target.read_text() != fresh():
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {CLEARPATH_COMMIT[:12]}")
        return 0

    copy_meshes(description)
    xml = fresh()
    shutil.copy2(description.parent / "LICENSE", PKG / "warthog_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + meshes + warthog_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
