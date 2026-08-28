#!/usr/bin/env python3
"""Build roqsim's Raspberry Pi Mouse MJCF from `rt-net/raspimouse_description`.

The smallest robot in the substrate by an order of magnitude -- 0.74 kg and 117 mm long, against the
Panther's 55 kg -- and the simplest port: a true two-wheel differential drive with no skid-steer
scrub, so no ``slip_factor`` calibration at all. Every mass, inertia, link offset and collision
primitive below is RT Corporation's own value, read out of the expanded xacro.

Worth recording next to its siblings: **RT Corporation ships this description under plain MIT**,
while the same vendor's CRANE-X7 and Sciurus17 model files are under a non-commercial agreement that
puts both out of reach. The platform ledger has all three, and this is why it reads the licence per
repository rather than per vendor.

Two small things the source does that the generator handles:

1. **The wheels are mounted through a rotated link frame** (``rpy`` of +/-1.57 about x, with the
   joint axis then along the link's z). Reproduced faithfully as a body quaternion rather than
   flattened to a y axis, so the visual mesh keeps the orientation the vendor gave it.
2. **The four light sensors are massless marker links.** They carry a 1e-4 inertia and zero mass,
   which is a fixed frame rather than a body; they are emitted as small visual boxes on the base so
   the robot looks right, with no roqsim behaviour behind them.

Usage::

    python external/convert/build_raspimouse_mjcf.py           # fetch, expand, convert, write
    python external/convert/build_raspimouse_mjcf.py --check   # rebuild and diff
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
from urdf_source import expand_xacro, inertial, link_visuals, pose, rpy_to_quat  # noqa: E402

RASPIMOUSE_URL = "https://github.com/rt-net/raspimouse_description.git"
RASPIMOUSE_COMMIT = "ed2c8b7a5039ff0a8711c1d817ea678ddb687beb"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/raspimouse"
TARGET_FACES = 4000

#: Sub-directories of meshes/dae holding the parts the URDF actually references.
MESH_DIRS = ("body", "wheel")
#: STLs copied as-is (MuJoCo loads STL). The vendor's own lidar mount and LDS-01, pulled in by
#: lidar:=lds, so the scanner the manifest declares is geometry rather than a bare site marker.
STL_MESHES = ("RasPiMouse_MultiLiDARMount.stl", "robotis_lds01.stl")
WHEELS = {"left_wheel": "left", "right_wheel": "right"}


def convert_meshes(source: Path) -> dict[str, list]:
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in list((PKG / "meshes").glob("*.obj")) + list((PKG / "meshes").glob("*.stl")):
        stale.unlink()
    palette: dict[str, list] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for folder in MESH_DIRS:
            raw = Path(tmp) / folder
            subprocess.run(
                [sys.executable, str(Path(__file__).parent / "dae2obj.py"),
                 str(source / "meshes/dae" / folder), str(raw)],
                check=True, capture_output=True,
            )
            for stem, parts in json.loads((raw / "materials.json").read_text()).items():
                palette[stem] = parts
                for sub, _rgb in parts:
                    subprocess.run(
                        [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                         "--target-faces", str(TARGET_FACES), "--no-materials",
                         str(raw / f"{sub}.obj"), str(PKG / "meshes" / f"{sub}.obj")],
                        check=True, cwd=ROOT, capture_output=True,
                    )
    for stl in STL_MESHES:
        shutil.copy2(source / "meshes/stl" / stl, PKG / "meshes" / stl)
    return palette


def build(urdf: ET.Element, palette: dict[str, list]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    base = inertial(links["base_link"])
    box = links["base_link"].find("collision/geometry/box").get("size").replace(",", " ")
    box_origin = links["base_link"].find("collision/origin").get("xyz").replace(",", " ")
    half = " ".join(f"{float(v) / 2:g}" for v in box.split())

    materials, assets = [], []
    for parts in palette.values():
        for sub, rgb in parts:
            materials.append(
                f'    <material name="mat_{sub}" rgba="{" ".join(f"{c:g}" for c in rgb)} 1"/>\n'
            )
            assets.append(f'    <mesh file="{sub}.obj"/>\n')
    materials = sorted(set(materials))
    assets = sorted(set(assets))

    base_geoms = "".join(
        f'        <geom class="visual" mesh="{sub}" material="mat_{sub}"/>\n'
        for sub, _ in palette.get("raspimouse_body", [])
    )
    # The light sensors are massless marker links; keep them as visual boxes on the base.
    for name in ("rf", "rs", "ls", "lf"):
        joint = joints.get(f"{name}_joint")
        if joint is None:
            continue
        xyz = joint.find("origin").get("xyz").replace(",", " ")
        size = links[f"{name}_lightsensor_link"].find("visual/geometry/box").get("size")
        half_size = " ".join(f"{float(v) / 2:g}" for v in size.replace(",", " ").split())
        base_geoms += (
            f'        <geom class="visual" type="box" size="{half_size}" pos="{xyz}"'
            f' rgba="0.15 0.15 0.15 1"/>\n'
        )

    # The vendor's own mount/scanner offsets and geometry, read from the expanded tree.
    mount_z = float(joints["lds_multi_mount_joint"].find("origin").get("xyz").split()[2])
    lidar_z = mount_z + float(joints["laser_joint"].find("origin").get("xyz").split()[2])
    mesh_material = {"RasPiMouse_MultiLiDARMount": "mat_mount", "robotis_lds01": "mat_lidar"}
    lidar_geoms = ""
    for link_name, offset in (("lds_multi_mount_link", mount_z), ("laser", lidar_z)):
        block = link_visuals(links[link_name], "        ", materials=mesh_material,
                              default_rgba="0.1 0.1 0.1 1")
        # shift each geom into base_link by the link's own z offset
        for line in block.splitlines():
            if 'pos="' in line:
                head, rest = line.split('pos="', 1)
                vec, tail = rest.split('"', 1)
                x, y, z = (float(v) for v in vec.split())
                line = f'{head}pos="{x:g} {y:g} {z + offset:g}"{tail}'
            lidar_geoms += line + "\n"

    wheels = ""
    for link_name, side in WHEELS.items():
        joint = joints[f"{link_name}_joint"]
        origin = joint.find("origin")
        xyz = (origin.get("xyz") or "0 0 0").replace(",", " ")
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").replace(",", " ").split()]
        axis = (joint.find("axis").get("xyz") or "0 0 1").replace(",", " ")
        cyl = links[link_name].find("collision/geometry/cylinder")
        c_origin = links[link_name].find("collision/origin")
        c_xyz = (c_origin.get("xyz") if c_origin is not None else "0 0 0").replace(",", " ")
        geoms = "".join(
            f'          <geom class="visual" mesh="{sub}" material="mat_{sub}"/>\n'
            for sub, _ in palette.get("raspimouse_wheel", [])
        )
        wheels += WHEEL_BODY.format(
            side=side, xyz=xyz, quat=rpy_to_quat(*rpy), axis=axis, geoms=geoms,
            radius=f"{float(cyl.get('radius')):g}", halflen=f"{float(cyl.get('length')) / 2:g}",
            c_xyz=c_xyz, **inertial(links[link_name]),
        )

    return TEMPLATE.format(
        commit=RASPIMOUSE_COMMIT, materials="".join(materials), assets="".join(assets),
        base_mass=base["mass"], base_pos=base["pos"], base_inertia=base["diaginertia"],
        half=half, box_origin=box_origin, base_geoms=base_geoms, wheels=wheels,
        lidar_geoms=lidar_geoms, lidar_z=f"{lidar_z:g}",
    )


WHEEL_BODY = """        <body name="{side}_wheel" pos="{xyz}" quat="{quat}">
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="{side}_wheel_joint" class="wheel" axis="{axis}"/>
{geoms}          <geom class="wheel_collision" name="{side}_wheel_geom" pos="{c_xyz}"
                size="{radius} {halflen}"/>
        </body>
"""

TEMPLATE = """<mujoco model="raspimouse">
  <!--
    RT Corporation Raspberry Pi Mouse (2-wheel differential drive) for MuJoCo.

    GENERATED by external/convert/build_raspimouse_mjcf.py from rt-net/raspimouse_description
    @ {commit} (MIT - see raspimouse_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia, link offset and collision primitive
    below is RT Corporation's own value, read out of the expanded xacro.

    Drive: a TRUE two-wheel differential drive. There is no lateral scrub, so unlike husky_a200,
    clearpath_jackal, rosbot and panther this model needs NO slip_factor - the same line
    turtlebot3_waffle draws.

    The smallest robot in the substrate: 0.74 kg and 117 mm long. Its scale is the thing to watch
    rather than its kinematics - contact and timestep behaviour at a tenth of a TurtleBot are not
    automatically the bigger bases'.

    The robot rests on two wheels and the underside of its chassis box, which clears the floor by
    1.85 mm; the real platform skids on that edge the same way. Collision is the vendor's own
    primitives (chassis box + two wheel cylinders).
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="raspimouse">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="box" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <geom type="cylinder" group="3" rgba="0.1 0.1 0.1 0.4" friction="1.0 0.005 0.0001"/>
      </default>
      <default class="wheel">
        <joint damping="0.0005" armature="1e-5"/>
      </default>
    </default>
  </default>

  <asset>
    <material name="mat_mount" rgba="0.2 0.2 0.2 1"/>
    <material name="mat_lidar" rgba="0.1 0.1 0.1 1"/>
{materials}{assets}    <mesh file="RasPiMouse_MultiLiDARMount.stl" scale="0.001 0.001 0.001"/>
    <mesh file="robotis_lds01.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <body name="base_link" childclass="raspimouse">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.005" rgba="0 0 0 0"/>
      <body name="body_link" pos="0 0 0">
        <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_inertia}"/>
{base_geoms}        <geom class="collision" name="chassis_geom" size="{half}" pos="{box_origin}"/>
        <!-- The vendor's own multi-lidar mount and LDS-01, with EVERY visual of both links: the
             scanner carries four leg cylinders as well as its mesh, and emitting only the mesh
             leaves it floating 3 cm above the mount. `lidar:=lds` is a supported option of the
             description, so the mount height and the 0.120 m scan plane are the vendor's numbers,
             not assumptions. -->
{lidar_geoms}        <site name="lidar" pos="0 0 {lidar_z}" size="0.004" rgba="1 0 0 0.6"/>
{wheels}      </body>
    </body>
  </worldbody>

  <contact>
    <!-- Wheel/floor friction stated explicitly: MuJoCo max-combines per-geom friction, so a
         high-friction scene floor would otherwise override the wheels. -->
    <pair geom1="left_wheel_geom" geom2="floor" condim="3" solref="0.01 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="right_wheel_geom" geom2="floor" condim="3" solref="0.01 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <!-- The chassis underside IS a bearing surface here, not an accident. With two driven wheels
         and no caster in the description, the base tips ~2 degrees and rests on the box edge - which
         is what the real robot does too, on a smooth plastic skid. At the geom default friction of
         1.0 that edge drags hard enough to cost 38% of commanded yaw; a skid's friction restores it
         without inventing a caster the vendor did not model. -->
    <pair geom1="chassis_geom" geom2="floor" condim="3" solref="0.01 1" friction="0.05 0.05 0.005 0.0001 0.0001"/>
  </contact>

  <actuator>
    <!-- Velocity servos. RT publishes a 0.5 m/s top speed, which over the 0.024 m wheel radius is
         21 rad/s; forcerange is small because the whole robot is 0.74 kg. kv is CALIBRATED: at the
         0.05 first tried, the servo needs a large error to make any torque at this scale and the
         base reached only 0.62 of commanded yaw. 0.5 gives 1.007 of commanded straight-line speed
         and 0.91-0.94 of commanded yaw. -->
    <velocity name="left_wheel_motor" joint="left_wheel_joint" kv="0.5" ctrlrange="-22 22" forcerange="-0.3 0.3"/>
    <velocity name="right_wheel_motor" joint="right_wheel_joint" kv="0.5" ctrlrange="-22 22" forcerange="-0.3 0.3"/>
  </actuator>

  <keyframe>
    <!-- base_link sits 1.85 mm above the ground; spawn a hair higher so the wheels settle. -->
    <key name="home" qpos="0 0 0.003 1 0 0 0  0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("raspimouse", RASPIMOUSE_URL, RASPIMOUSE_COMMIT)
    target = PKG / "raspimouse.xml"

    if args.check:
        palette = json.loads((PKG / "meshes" / "raspimouse.materials.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            xml = build(expand_xacro({"raspimouse_description": source},
                                     source / "urdf/raspimouse.urdf.xacro", Path(tmp),
                                     args=["lidar:=lds", "lidar_frame:=laser"]), palette)
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {RASPIMOUSE_COMMIT[:12]}")
        return 0

    palette = convert_meshes(source)
    (PKG / "meshes" / "raspimouse.materials.json").write_text(
        json.dumps(palette, indent=2, sort_keys=True)
    )
    with tempfile.TemporaryDirectory() as tmp:
        xml = build(expand_xacro({"raspimouse_description": source},
                                     source / "urdf/raspimouse.urdf.xacro", Path(tmp),
                                     args=["lidar:=lds", "lidar_frame:=laser"]), palette)
    shutil.copy2(source / "LICENSE", PKG / "raspimouse_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + meshes + raspimouse_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
