#!/usr/bin/env python3
"""Build roqsim's Husarion Panther MJCF from `husarion/husarion_ugv_ros`'s description.

The third xacro-tree port, and the cheapest of them: the Husarion path was already established by the
ROSbot, and this vendor is unusually generous with what it publishes. Every mass, inertia, link
offset and wheel parameter below is Husarion's own value, read out of the expanded xacro and
``config/WH01.yaml``.

Two things this source gives that the earlier ports had to work for:

* **A real collision mesh.** ``meshes/panther/base_collision.stl`` is 9.7 kB against the visual
  ``base.dae``'s 1.4 MB -- a genuinely simplified hull, not the copy-of-the-visual-CAD that made the
  Doosan port fit its own primitives. It is used as shipped, which is what the porting playbook asks
  for whenever a vendor supplies one.
* **A published skid-steer correction.** ``husarion_ugv_controller``'s ``WH01_controller.yaml``
  carries ``wheel_separation_multiplier: 1.5``, which is the same quantity as our ``slip_factor``:
  the ICR compensation a skid-steer needs because it turns by scrubbing. The husky's 3.0 had to be
  calibrated blind; this one starts from a vendor number and is then re-measured against *this*
  model's friction, mass and timestep (see the manifest).

Like the ROSbot, only the base robot is built. The vendor's top-level ``panther.urdf.xacro`` pulls in
``husarion_components_description``, a separate repository bolting on whichever sensors a unit was
ordered with -- a deployment choice rather than the platform -- so the macro is wrapped directly and
the lidar is attached through the roqsim manifest.

``husarion_ugv_controller`` is fetched only so ``$(find ...)`` resolves inside the macro; nothing
from it is read into the model.

Usage::

    python external/convert/build_panther_mjcf.py           # fetch, expand, convert, write
    python external/convert/build_panther_mjcf.py --check   # rebuild and diff against what is committed
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

HUSARION_URL = "https://github.com/husarion/husarion_ugv_ros.git"
HUSARION_COMMIT = "559e784b84c05c813b6ecfce02751d8b3966528a"
PACKAGES = ("husarion_ugv_description", "husarion_ugv_controller")
#: Wheel variant. The vendor ships WH01/WH02/WH04 for the Panther and WH05 for the Lynx; each is a
#: different tyre, and each has its own controller config. WH01 is the Panther's default.
WHEELS = "WH01"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/panther"

#: Per-part triangle budget. base.dae is 1.4 MB of full-detail CAD.
BASE_FACES = 8000
WHEEL_FACES = 3000

WRAPPER = """<?xml version="1.0"?>
<robot name="panther" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find husarion_ugv_description)/urdf/panther/panther_macro.urdf.xacro"
                 ns="husarion"/>
  <xacro:husarion.panther_robot
    use_sim="false" imu_xyz="0.169 0.025 0.092" imu_rpy="0.0 0.0 -1.57"
    wheel_config_file="$(find husarion_ugv_description)/config/{wheels}.yaml"
    controller_config_file="$(find husarion_ugv_controller)/config/{wheels}_controller.yaml"
    battery_config_file="" namespace=""/>
</robot>
"""

CORNERS = {"fl": ("fl", "l"), "fr": ("fr", "r"), "rl": ("rl", "l"), "rr": ("rr", "r")}


def convert_meshes(description: Path) -> dict[str, list]:
    """Collada -> per-material OBJ (pycollada) -> decimated OBJ; plus the vendor's collision STL."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in list((PKG / "meshes").glob("*.obj")) + list((PKG / "meshes").glob("*.stl")):
        stale.unlink()
    palette: dict[str, list] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for folder, budget in ((f"meshes/panther", BASE_FACES), (f"meshes/{WHEELS}", WHEEL_FACES)):
            raw = Path(tmp) / Path(folder).name
            subprocess.run(
                [sys.executable, str(Path(__file__).parent / "dae2obj.py"),
                 str(description / folder), str(raw)],
                check=True, capture_output=True,
            )
            materials = json.loads((raw / "materials.json").read_text())
            for stem, parts in materials.items():
                palette[stem] = parts
                for sub, _rgb in parts:
                    subprocess.run(
                        [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                         "--target-faces", str(budget), "--no-materials",
                         str(raw / f"{sub}.obj"), str(PKG / "meshes" / f"{sub}.obj")],
                        check=True, cwd=ROOT, capture_output=True,
                    )
    # The vendor's own simplified collision hull, used as shipped.
    shutil.copy2(description / "meshes/panther/base_collision.stl", PKG / "meshes")
    return palette


def build(urdf: ET.Element, wheel_cfg: dict, palette: dict[str, list]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    body = inertial(links["body_link"])
    wheel = inertial(links["fl_wheel_link"])
    radius = float(wheel_cfg["wheel_radius"])
    width = float(wheel_cfg["wheel_width"])

    materials, assets, body_geoms = [], [], []
    for stem, parts in sorted(palette.items()):
        for sub, rgb in parts:
            materials.append(
                f'    <material name="mat_{sub}" rgba="{" ".join(f"{c:g}" for c in rgb)} 1"/>\n'
            )
            assets.append(f'    <mesh file="{sub}.obj"/>\n')
    for sub, _ in palette.get("base", []):
        body_geoms.append(f'        <geom class="visual" mesh="{sub}" material="mat_{sub}"/>\n')

    wheels = []
    for corner in CORNERS:
        base_xyz = joints[f"body_to_{corner}_wheel_base_joint"].find("origin").get("xyz")
        hub_xyz = joints[f"{corner}_wheel_joint"].find("origin").get("xyz")
        xyz = " ".join(
            f"{float(a) + float(b):g}"
            for a, b in zip(base_xyz.split(), hub_xyz.split(), strict=True)
        )
        geoms = "".join(
            f'          <geom class="visual" mesh="{sub}" material="mat_{sub}"/>\n'
            for sub, _ in palette.get(f"{corner}_wheel", [])
        )
        wheels.append(WHEEL_BODY.format(name=corner, xyz=xyz, geoms=geoms,
                                        radius=f"{radius:g}", halfwidth=f"{width / 2:g}", **wheel))

    return TEMPLATE.format(
        commit=HUSARION_COMMIT, wheels_variant=WHEELS,
        rest_height=f"{radius:g}", radius=f"{radius:g}", width=f"{width:g}",
        body_mass=body["mass"], body_pos=body["pos"], body_inertia=body["diaginertia"],
        materials="".join(materials), assets="".join(assets),
        body_geoms="".join(body_geoms), wheels="".join(wheels),
    )


WHEEL_BODY = """        <body name="{name}_wheel_link" pos="{xyz}">
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="{name}_wheel_joint" class="wheel"/>
{geoms}          <geom class="wheel_collision" name="{name}_wheel_geom" size="{radius} {halfwidth}"/>
        </body>
"""

TEMPLATE = """<mujoco model="panther">
  <!--
    Husarion Panther (4-wheel skid-steer UGV) for MuJoCo, {wheels_variant} wheels.

    GENERATED by external/convert/build_panther_mjcf.py from husarion/husarion_ugv_ros's
    husarion_ugv_description @ {commit} (Apache-2.0 - see panther_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia, link offset and wheel parameter
    below is Husarion's own value, from the expanded xacro and config/{wheels_variant}.yaml.

    Drive: TRUE 4-wheel skid-steer - all four wheels driven, no casters. Left (front+rear) and right
    wheels are commanded together, and in-place rotation comes from the wheels scrubbing laterally,
    so wheel/floor friction is set by explicit contact pairs rather than geom defaults. The same
    construction as husky_a200, clearpath_jackal and rosbot, and it needs the same slip_factor - but
    unlike theirs it starts from a published number, since Husarion's own controller config carries
    wheel_separation_multiplier 1.5 for exactly this reason. See the manifest.

    Base robot only. The vendor's top-level xacro also pulls husarion_components_description, a
    separate repository bolting on whichever sensors a unit was ordered with; that is a deployment
    choice rather than the platform, so the lidar is attached through the manifest.

    Body collision is the vendor's OWN base_collision.stl (9.7 kB against the 1.4 MB visual mesh) -
    a genuinely simplified hull, used as shipped. Wheels are cylinders.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="panther">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <geom type="cylinder" group="3" rgba="0.1 0.1 0.1 0.4" quat="0.7071068 0.7071068 0 0"
              friction="1.0 0.005 0.0001"/>
      </default>
      <default class="wheel">
        <joint axis="0 1 0" damping="0.05" armature="0.01"/>
      </default>
    </default>
  </default>

  <asset>
{materials}{assets}    <mesh file="base_collision.stl"/>
  </asset>

  <worldbody>
    <!-- base_link: the URDF root. body_link sits at the wheel radius above the ground, which is
         where base_footprint puts it. freejoint makes the robot mobile; spawn_robot places it. -->
    <body name="base_link" childclass="panther">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <body name="body_link" pos="0 0 0">
        <inertial pos="{body_pos}" mass="{body_mass}" diaginertia="{body_inertia}"/>
{body_geoms}        <geom class="collision" mesh="base_collision"/>
        <!-- The lidar mast. Height is an assumption (see the port log): the base description ships
             no scanner, because Husarion keeps sensors in a separate package. -->
        <site name="lidar" pos="0 0 0.2035" size="0.008" rgba="1 0 0 0.6"/>
{wheels}      </body>
    </body>
  </worldbody>

  <contact>
    <!-- Wheel/floor friction stated explicitly: MuJoCo max-combines per-geom friction, so a
         high-friction scene floor would otherwise override the wheels and change how the skid-steer
         scrubs - which is the behaviour under study. -->
    <pair geom1="fl_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="fr_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="rl_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="rr_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
  </contact>

  <actuator>
    <!-- Velocity servos, one per wheel. ctrlrange is the platform's published 2.0 m/s top speed over
         the {radius} m wheel radius (11 rad/s), rounded up; forcerange is sized to accelerate 55 kg
         at the controller's published 2.7 m/s^2 with margin. -->
    <velocity name="fl_wheel_motor" joint="fl_wheel_joint" kv="120" ctrlrange="-12 12" forcerange="-80 80"/>
    <velocity name="fr_wheel_motor" joint="fr_wheel_joint" kv="120" ctrlrange="-12 12" forcerange="-80 80"/>
    <velocity name="rl_wheel_motor" joint="rl_wheel_joint" kv="120" ctrlrange="-12 12" forcerange="-80 80"/>
    <velocity name="rr_wheel_motor" joint="rr_wheel_joint" kv="120" ctrlrange="-12 12" forcerange="-80 80"/>
  </actuator>

  <keyframe>
    <!-- body_link rides at the wheel radius; spawn a hair above so the wheels settle. -->
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    import yaml

    sources = {
        name: resolve_source("husarion_ugv", HUSARION_URL, HUSARION_COMMIT, sparse=name, subdir=name)
        for name in PACKAGES
    }
    description = sources["husarion_ugv_description"]
    wheel_cfg = yaml.safe_load((description / f"config/{WHEELS}.yaml").read_text())
    target = PKG / "panther.xml"

    if args.check:
        palette = json.loads((PKG / "meshes" / "panther.materials.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            xml = build(expand_xacro(sources, description / "urdf/panther.urdf.xacro", Path(tmp),
                                     wrapper=WRAPPER.format(wheels=WHEELS)), wheel_cfg, palette)
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {HUSARION_COMMIT[:12]}")
        return 0

    palette = convert_meshes(description)
    (PKG / "meshes" / "panther.materials.json").write_text(
        json.dumps(palette, indent=2, sort_keys=True)
    )
    with tempfile.TemporaryDirectory() as tmp:
        xml = build(expand_xacro(sources, description / "urdf/panther.urdf.xacro", Path(tmp),
                                     wrapper=WRAPPER.format(wheels=WHEELS)), wheel_cfg, palette)
    shutil.copy2(description.parent / "LICENSE", PKG / "panther_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + meshes + panther_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
