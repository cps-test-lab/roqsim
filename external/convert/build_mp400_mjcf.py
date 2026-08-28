#!/usr/bin/env python3
"""Build roqsim's Neobotix MP-400 MJCF from `neobotix/neo_simulation2`.

A **differential-drive** base: two driven wheels and four passive casters. The third Neobotix port and
the first differential one, so it uses ``diff_drive`` -- and carries **no** ``slip_factor``, because
two driven wheels with casters do not turn by scrubbing. The same line ``turtlebot3_waffle``,
``raspimouse`` and ``oomwoo_one`` draw, and the reason this is the cheapest Neobotix port.

Shares the pinned source, Collada pipeline, palette convention and MuJoCo quirks with its siblings
through :mod:`neobotix`; only the body tree, the drive and the actuators are here.

**Read the mass table before using this model for dynamics.** The description's inertials are not
trustworthy, and it is possible to say why rather than merely suspect it -- see the port log. Each
38 mm caster sphere is declared at **12.7 kg**, which is exactly the mass of the MPO-700's steering
modules in the same repository: 50.8 kg of an 84.4 kg robot sits in four casters. The geometry and
kinematics are sound; the mass distribution is upstream copy-paste.

Usage::

    python external/convert/build_mp400_mjcf.py           # fetch, convert, write
    python external/convert/build_mp400_mjcf.py --check    # rebuild and diff
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neobotix import (  # noqa: E402
    NEO_COMMIT, NEO_URL, asset_block, colours, convert_meshes, subs_for, wrapper,
)
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, mesh_scales, pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/mp_400"

WHEELS = ("left", "right")
CASTERS = ("front_left", "front_right", "back_left", "back_right")
SENSOR_LINKS = ("lidar_1_link",)


def build(urdf: ET.Element, shipped: set[str], scales: dict[str, str]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    palette = colours(PKG, "mp_400")

    def geoms(link: ET.Element, indent: str, tyre_class: str = "wheel_collision") -> str:
        out = ""
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, quat = pose(visual)
            for sub in subs_for(Path(mesh.get("filename")).stem, shipped):
                mat = f' material="{sub}_mat"' if sub in palette else ""
                out += f'{indent}<geom class="visual" mesh="{sub}"{mat} pos="{xyz}"{quat}/>\n'
        for collision in link.findall("collision"):
            shape = collision.find("geometry")[0]
            xyz, quat = pose(collision)
            if shape.tag == "sphere":
                out += (f'{indent}<geom class="{tyre_class}" name="{link.get("name")}_tyre"'
                        f' size="{float(shape.get("radius")):g}" pos="{xyz}"/>\n')
            else:
                sub = subs_for(Path(shape.get("filename")).stem, shipped)[0]
                out += (f'{indent}<geom class="collision" mesh="{sub}"'
                        f' name="{link.get("name")}_collision" pos="{xyz}"{quat}/>\n')
        return out

    base = links["base_link"]
    sensors = ""
    for name in SENSOR_LINKS:
        pos, quat = pose(joints[name])
        sensors += FIXED_BODY.format(
            name=name, body_pos=pos, body_quat=quat,
            geoms=geoms(links[name], "          "), **inertial(links[name]),
        )
    # The casters are FIXED in this description -- passive spheres, not articulated wheels -- so they
    # are emitted as jointless bodies. They still need their own contact class: they slide rather
    # than roll, and at the floor's friction four of them fight every turn.
    casters = ""
    for corner in CASTERS:
        name = f"mp_400_caster_wheel_{corner}_link"
        pos, quat = pose(joints[name])
        casters += FIXED_BODY.format(
            name=name, body_pos=pos, body_quat=quat,
            geoms=geoms(links[name], "          ", tyre_class="caster_collision"),
            **inertial(links[name]),
        )
    wheels = ""
    for side in WHEELS:
        name = f"mp_400_fixed_wheel_{side}_link"
        pos, quat = pose(joints[name])
        wheels += WHEEL_BODY.format(
            name=name, body_pos=pos, body_quat=quat, joint=joints[name].get("name"),
            geoms=geoms(links[name], "            "), **inertial(links[name]),
        )
    wheel_joint = joints["mp_400_fixed_wheel_left_link"]
    wheel_z = float(pose(wheel_joint)[0].split()[2])
    radius = float(links["mp_400_fixed_wheel_left_link"]
                   .find("collision/geometry/sphere").get("radius"))
    lidar, _ = pose(joints["lidar_1_link"])
    excludes = "".join(
        f'    <exclude body1="base_link" body2="mp_400_fixed_wheel_{s}_link"/>\n' for s in WHEELS
    ) + "".join(
        f'    <exclude body1="base_link" body2="mp_400_caster_wheel_{c}_link"/>\n' for c in CASTERS
    )
    return TEMPLATE.format(
        commit=NEO_COMMIT, assets=asset_block(shipped, palette, scales), excludes=excludes,
        base_geoms=geoms(base, "        "), sensors=sensors, casters=casters, wheels=wheels,
        lidar=lidar, rest_height=f"{radius - wheel_z:g}",
        **{f"base_{k}": v for k, v in inertial(base).items()},
    )


FIXED_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}        </body>
"""

WHEEL_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="{joint}" class="wheel"/>
{geoms}        </body>
"""

TEMPLATE = """<mujoco model="mp_400">
  <!--
    Neobotix MP-400 - a DIFFERENTIAL-drive base: two driven wheels, four passive casters.

    GENERATED by external/convert/build_mp400_mjcf.py from neobotix/neo_simulation2 @ {commit}
    (MIT - see mp_400_LICENSE). Do not hand-edit: re-run the generator.

    Drive: a true two-wheel differential drive, so `diff_drive` and NO slip_factor - it does not turn
    by scrubbing. The same line turtlebot3_waffle, raspimouse and oomwoo_one draw, and the opposite of
    the four skid-steers here.

    THE MASSES ARE NOT TRUSTWORTHY, and provably so rather than suspiciously: each 38 mm caster
    sphere is declared at 12.7 kg, which is exactly the mass of the MPO-700's steering modules in the
    same repository. 50.8 kg of this 84.4 kg robot sits in four casters. Geometry and kinematics are
    sound; use this model for navigation, not for dynamics. See the port log.

    The casters are FIXED in the description - passive spheres, not articulated wheels - so they
    slide rather than roll. They carry a low-friction contact class with `priority`, without which
    MuJoCo takes the MAXIMUM of the two contacting geoms' friction, the floor's value wins, and four
    loaded spheres fight every turn.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="mp_400">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Driven tyres: this base moves because these push on the ground. -->
        <geom type="sphere" group="3" rgba="0.05 0.05 0.05 0.4" friction="1.0 0.005 0.0001"/>
      </default>
      <default class="caster_collision">
        <!-- A passive caster swivels; a fixed sphere cannot, so it stands in for one with a low
             friction and `priority` to make that friction actually apply. -->
        <geom type="sphere" group="3" rgba="0.6 0.1 0.1 0.35"
              friction="0.04 0.005 0.0001" priority="2"/>
      </default>
      <default class="wheel">
        <!-- The description's axis is `-1 0 0`, which its joint rpy (0 0 1.57) rotates to -y in the
             base frame. This is `1 0 0` instead, i.e. +y: `diff_drive` writes the commanded wheel
             rate straight to the actuator with no sign derivation, so it REQUIRES the +y convention
             that all five existing diff_drive models happen to use. With the vendor's sign the robot
             drives and turns backwards -- measured, -0.984 of a forward command.

             Flipping the axis is a sign convention, not a geometry change: a revolute axis is
             arbitrary up to sign and the vendor's choice is one of the two equivalent ones. The
             better fix is in the plugin, which should derive the sign off the model the way
             omni_drive does -- see the port log. -->
        <joint axis="1 0 0" damping="0.5" armature="0.02" limited="false"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <contact>
    <!--
      The vendor's base collision IS the full body mesh, and MuJoCo convex-hulls a collision mesh, so
      the hull closes over the wheel cutouts and overlaps the wheels and casters inside them.
    -->
{excludes}  </contact>

  <worldbody>
    <body name="base_link" childclass="mp_400">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <!-- The vendor's own single front scanner mount. -->
      <site name="lidar" pos="{lidar}" size="0.01" rgba="1 0 0 0.6"/>
{base_geoms}{sensors}{casters}{wheels}    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servos, one per driven wheel. ctrlrange is the vendor's own Nav2 profile
         (configs/mp_400/navigation.yaml) 0.8 m/s over the 0.0765 m wheel radius, with headroom;
         forcerange is sized to accelerate the declared 84.4 kg at that file's 0.25 m/s^2. -->
    <velocity name="wheel_left_motor" joint="mp_400_fixed_wheel_left_joint" kv="120" ctrlrange="-14 14" forcerange="-60 60"/>
    <velocity name="wheel_right_motor" joint="mp_400_fixed_wheel_right_joint" kv="120" ctrlrange="-14 14" forcerange="-60 60"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("neo_simulation2", NEO_URL, NEO_COMMIT)
    target = PKG / "mp_400.xml"
    with tempfile.TemporaryDirectory() as tmp:
        urdf = expand_xacro({"neo_simulation2": source}, Path("mp_400.urdf.xacro"),
                            Path(tmp), wrapper=wrapper("mp_400", "continuous"))

    if args.check:
        shipped = {p.stem for p in (PKG / "meshes").glob("*.obj")}
        if not target.exists() or target.read_text() != build(urdf, shipped, mesh_scales(urdf)):
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {NEO_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    scales = convert_meshes(source, urdf, PKG, "mp_400", ROOT)
    shipped = {p.stem for p in (PKG / "meshes").glob("*.obj")}
    shutil.copy2(source / "LICENSE", PKG / "mp_400_LICENSE")
    target.write_text(build(urdf, shipped, scales))
    print(f"wrote {target} + meshes + mp_400_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
