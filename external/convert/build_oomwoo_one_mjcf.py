#!/usr/bin/env python3
"""Build roqsim's OOMWOO One MJCF from `makerspet/oomwoo-one`.

The substrate's first **robot vacuum**, and the cheapest port in the corpus. Three properties make
it so, and all three are the vendor's doing rather than ours:

* **No meshes at all.** The description is 291 lines of primitives -- one body cylinder, twelve
  bumper plates, two wheel cylinders, a caster sphere, a lidar puck and six sensor markers. Nothing
  to convert, so none of the Collada, Blender, axis, scale or winding traps that cost the ROSbot,
  the Doosan and the Raspberry Pi Mouse their iterations can apply here.
* **No `$(find ...)` anywhere.** Its four includes are plain relative filenames inside its own
  package, so none of the ament-index plumbing the Clearpath, Doosan and Husarion ports needed is
  used -- ``expand_xacro`` is called with an empty package map.
* **No sensor assumptions.** Every other mobile port in this batch had to invent a scanner mount
  height, a scan rate or a range. This description mounts its own lidar off the body geometry and
  ``plugins.xacro`` states the scan itself: 360 samples over a full turn, 0.1--10 m, 5 Hz.

The drive limits are the vendor's own Nav2 configuration (``config/navigation.yaml``), and the
acceleration is ``params.xacro``'s ``max_linear_acceleration``.

**No ``slip_factor``.** This is a true two-wheel differential drive with a real modelled caster, so
it does not turn by scrubbing -- the same line ``turtlebot3_waffle`` and ``raspimouse`` draw, and the
opposite of the four skid-steers.

Usage::

    python external/convert/build_oomwoo_one_mjcf.py           # fetch, expand, write
    python external/convert/build_oomwoo_one_mjcf.py --check   # rebuild and diff
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

OOMWOO_URL = "https://github.com/makerspet/oomwoo-one.git"
#: The jazzy branch, which is this package's default and only release branch.
OOMWOO_COMMIT = "5305e6c5bb64208563b2f68c15258b0815c7d7cf"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/oomwoo_one"

#: The description's own palette, from urdf/materials.xacro. Only the ones this robot uses.
PALETTE = {
    "oomwoo_black": "0.0 0.0 0.0 1",
    "oomwoo_dark": "0.3 0.3 0.3 1",
    "oomwoo_grey": "0.5 0.5 0.5 1",
    "oomwoo_orange": "1.0 0.4235 0.0392 1",
    "oomwoo_red": "0.8 0.0 0.0 1",
}
#: URDF material name -> ours. The description's names ("dark", "grey") are too generic to put in a
#: shared MJCF namespace, where they would collide with another model's.
MATERIALS = {"black": "oomwoo_black", "dark": "oomwoo_dark", "grey": "oomwoo_grey",
             "orange": "oomwoo_orange", "red": "oomwoo_red"}

#: Fixed children of base_link that carry mass. Emitted as jointless bodies rather than merged, so
#: MuJoCo welds them and the mass audit reproduces the description's sum exactly.
FIXED_PARTS = ["caster_link", "base_scan", "range_left_link", "range_right_link",
               "tof_front_link", "camera_left_link", "camera_right_link", "imu_link"]
WHEELS = ["left", "right"]


def _geoms(link: ET.Element, tag: str, cls: str, indent: str) -> str:
    """A link's ``<visual>`` or ``<collision>`` primitives as MJCF geoms.

    ``link_visuals`` covers the visual side for every other generator; the collision side is written
    here because only this model has collision *primitives* worth reproducing one-for-one -- the
    twelve bumper plates, which are what a bump sensor would read. Everything else in the corpus
    ships a collision mesh or a single cylinder.
    """
    out = ""
    for index, element in enumerate(link.findall(tag)):
        xyz, quat = pose(element)
        geometry = element.find("geometry")[0]
        named = element.find("material")
        material = MATERIALS.get(named.get("name")) if named is not None else None
        attr = f' material="{material}"' if material and tag == "visual" else ""
        name = f' name="{link.get("name")}_{cls}{index}"' if tag == "collision" else ""
        if geometry.tag == "cylinder":
            size = f'{float(geometry.get("radius")):g} {float(geometry.get("length")) / 2:g}'
        elif geometry.tag == "sphere":
            size = f'{float(geometry.get("radius")):g}'
        else:
            size = " ".join(f"{float(v) / 2:g}"
                            for v in geometry.get("size").replace(",", " ").split())
        out += (f'{indent}<geom class="{cls}" type="{geometry.tag}" size="{size}"'
                f'{name}{attr} pos="{xyz}"{quat}/>\n')
    return out


def build(urdf: ET.Element) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    def joint_to(name: str) -> ET.Element:
        return next(j for j in joints.values() if j.find("child").get("link") == name)

    def frame_of(name: str) -> tuple[str, str]:
        """The child link's frame as MJCF ``(pos, quat)``.

        The **rpy matters** and dropping it is not cosmetic. This is the first description in the
        corpus that puts a rotation on the JOINT rather than on the visual: the wheel joints carry
        ``rpy="-pi/2 0 0"`` with ``axis="0 0 1"``, so the child frame is rotated and the wheel's own
        cylinder -- unrotated in that frame -- ends up axis-along-y as a wheel must be. Reading only
        the xyz leaves the cylinders axis-vertical, flat discs 28 mm clear of the floor, and the
        robot then rests on its bumper with the wheels never touching anything.
        """
        return pose(joint_to(name))

    def offset_of(name: str) -> tuple[float, ...]:
        xyz, _ = frame_of(name)
        return tuple(float(v) for v in xyz.split())

    base = links["base_link"]
    parts = ""
    for name in FIXED_PARTS:
        body_pos, body_quat = frame_of(name)
        parts += PART_BODY.format(
            name=name, body_pos=body_pos, body_quat=body_quat,
            geoms=_geoms(links[name], "visual", "visual", "          ")
            + _geoms(links[name], "collision", "collision", "          "),
            **inertial(links[name]),
        )

    wheels = ""
    for side in WHEELS:
        link = links[f"wheel_{side}_link"]
        body_pos, body_quat = frame_of(f"wheel_{side}_link")
        cylinder = link.find("collision/geometry/cylinder")
        _, quat = pose(link.find("collision"))
        wheels += WHEEL_BODY.format(
            side=side, body_pos=body_pos, body_quat=body_quat, quat=quat,
            radius=f'{float(cylinder.get("radius")):g}',
            half_width=f'{float(cylinder.get("length")) / 2:g}',
            geoms=link_visuals(link, "          ", materials=None,
                               default_material="oomwoo_red"),
            **inertial(link),
        )

    materials = "".join(
        f'    <material name="{n}" rgba="{rgba}"/>\n' for n, rgba in sorted(PALETTE.items())
    )
    wheel_radius = float(links["wheel_left_link"].find("collision/geometry/cylinder").get("radius"))
    return TEMPLATE.format(
        commit=OOMWOO_COMMIT, materials=materials,
        base_geoms=_geoms(base, "visual", "visual", "        ")
        + _geoms(base, "collision", "collision", "        "),
        parts=parts, wheels=wheels,
        lidar_z=f"{offset_of('base_scan')[2]:g}",
        imu_z=f"{offset_of('imu_link')[2]:g}",
        rest_height=f"{wheel_radius - offset_of('wheel_left_link')[2]:g}",
        **{f"base_{k}": v for k, v in inertial(base).items()},
    )


PART_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}        </body>
"""

WHEEL_BODY = """        <body name="wheel_{side}_link" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="wheel_{side}_joint" class="wheel"/>
{geoms}          <geom class="wheel_collision" name="wheel_{side}_tyre"{quat}
                size="{radius} {half_width}"/>
        </body>
"""

TEMPLATE = """<mujoco model="oomwoo_one">
  <!--
    OOMWOO One open-source robot vacuum (2-wheel differential drive + caster) for MuJoCo.

    GENERATED by external/convert/build_oomwoo_one_mjcf.py from makerspet/oomwoo-one @ {commit}
    (Apache-2.0 - see oomwoo_one_LICENSE). Do not hand-edit: re-run the generator. Every mass,
    inertia, offset and primitive below is the description's own value.

    The substrate's first robot vacuum, and the only model here with NO meshes: the description is
    primitives throughout, which is why its collision geometry is reproduced one-for-one rather than
    hulled.

    The twelve `bumper` plates around the front arc are the platform's distinguishing feature - a
    real bump ring, six facets per side, each ~5 mm proud of the body cylinder. They are collision
    geoms here as they are in the description, so a contact-based bump sensor has geometry to read;
    roqsim ships no bumper plugin, so nothing reads them yet (see the port log).

    Drive: a TRUE two-wheel differential drive with a modelled caster, so there is no slip_factor -
    it does not turn by scrubbing. The same line turtlebot3_waffle and raspimouse draw.
  -->
  <compiler angle="radian" autolimits="true"/>

  <default>
    <default class="oomwoo_one">
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Real driven tyres on a 2.93 kg robot: friction is what moves it. -->
        <geom type="cylinder" group="3" rgba="0.05 0.05 0.05 0.4" friction="1.0 0.005 0.0001"/>
      </default>
      <default class="wheel">
        <!-- axis 0 0 1 in the wheel's OWN frame, which the joint's rpy has already rotated to lie
             along the robot's y. This is the description's own axis, not a re-derived one. -->
        <joint axis="0 0 1" damping="0.001" armature="0.0005"/>
      </default>
    </default>
  </default>

  <asset>
{materials}  </asset>

  <worldbody>
    <body name="base_link" childclass="oomwoo_one">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <site name="base_imu" pos="0 0 {imu_z}" size="0.005" rgba="0 0 0 0"/>
      <!-- The scan plane, off the description's own scan_joint - not an assumption, unlike every
           other mobile port in this batch. -->
      <site name="lidar" pos="0 0 {lidar_z}" size="0.005" rgba="1 0 0 0.6"/>
{base_geoms}{parts}{wheels}    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servos, one per wheel. ctrlrange is navigation.yaml's 0.2 m/s over the 0.034 m
         wheel radius (5.9 rad/s) with headroom for the velocity smoother's 0.5; forcerange is the
         description's own motor_stall_torque of 0.5 N.m.

         kv 0.25 keeps kv*dt/I below 1 at this scale (wheel inertia 5.8e-5 plus 5e-4 of armature).
         Raising it buys almost nothing: 0.974 of commanded at 0.25 against 0.982 at 1.0, because
         the residual is caster drag rather than servo lag. See the port log. -->
    <velocity name="wheel_left_motor" joint="wheel_left_joint" kv="0.25" ctrlrange="-15 15" forcerange="-0.5 0.5"/>
    <velocity name="wheel_right_motor" joint="wheel_right_joint" kv="0.25" ctrlrange="-15 15" forcerange="-0.5 0.5"/>
  </actuator>

  <keyframe>
    <!-- base_link rides at wheel radius minus the wheel-centre height, which is the description's
         own floor_clearance. Spawn a hair above so it settles rather than starting interpenetrated. -->
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("oomwoo_one", OOMWOO_URL, OOMWOO_COMMIT)
    target = PKG / "oomwoo_one.xml"

    def fresh() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            # An empty package map: this description resolves no $(find ...), so there is no ament
            # index to satisfy. It is the only port in the corpus that needs none.
            return build(expand_xacro({}, source / "urdf/robot.urdf.xacro", Path(tmp)))

    if args.check:
        if not target.exists() or target.read_text() != fresh():
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {OOMWOO_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    xml = fresh()
    shutil.copy2(source / "LICENSE", PKG / "oomwoo_one_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + oomwoo_one_LICENSE (no meshes: the description is primitives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
