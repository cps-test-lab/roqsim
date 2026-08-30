#!/usr/bin/env python3
"""Build roqsim's Husarion ROSbot MJCF from `husarion/rosbot_ros`'s `rosbot_description`.

Unlike the Menagerie ports, there is no upstream MJCF here: the source is a xacro tree, so this
script expands it and reads the numbers out. Every mass, inertia tensor, link offset, collision
primitive and mesh transform below is Husarion's own value; nothing is measured off a mesh or
guessed. Re-running against the pinned commit must reproduce the committed file byte for byte, which
``--check`` asserts.

**Only the base robot is built.** The top-level ``rosbot.urdf.xacro`` also pulls in
``husarion_components_description``, a second repository that bolts on the sensors a particular unit
was ordered with. That is a *deployment* choice, not the platform, so the model here is chassis plus
wheels -- and the lidar is attached through the roqsim manifest instead, exactly as ``husky_a200``
does for a description that ships none.

What the source gives, and therefore what this does NOT have to invent:

* **Collision primitives.** Husarion ships a body box (0.197 x 0.150 x 0.080) and four wheel
  cylinders (r 0.0425, l 0.036). No hull fitting, unlike the xArm.
* **Plausible inertials.** 1.728 kg body with a full diagonal tensor, 0.074 kg wheels -- no 1e-6
  placeholder links.
* **Tested drive parameters**, in ``rosbot_controller/config/rosbot/controllers.yaml``: wheel radius
  0.0425, separation 0.186, max 1.1 m/s and 3.14 rad/s.

Two things it must fix, both of which pass a bounds check silently:

1. **The body mesh is in millimetres.** The URDF carries ``scale="0.001 0.001 0.001"`` on
   ``body.glb`` and ``1.0`` on the wheels, so a converter that treats the tree uniformly produces a
   robot 1000x too big -- or, worse, one where only the chassis is wrong. The scale is baked in at
   conversion time (``roqsim assets reduce-mesh --scale``) rather than carried in the MJCF, so the
   committed OBJ is metres like every other mesh in the substrate.
2. **GLB is not loadable by MuJoCo.** The meshes are glTF binary and are converted to OBJ.

Usage::

    python external/convert/build_rosbot_mjcf.py           # fetch, expand, convert, write
    python external/convert/build_rosbot_mjcf.py --check   # rebuild and diff against what is committed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, link_visuals, pose, rpy_to_quat, write_license  # noqa: E402

ROSBOT_URL = "https://github.com/husarion/rosbot_ros.git"
# Pinned. If this moves, the port log moves with it.
ROSBOT_COMMIT = "41fad02196ee200a39e01579c75391385cc9b714"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/rosbot"

#: mesh stem -> (source file, scale to metres, per-part triangle budget). The body's 0.001 is the
#: URDF's own scale attribute, applied to body.glb alone -- see the docstring.
#:
#: Every one of these is SPLIT PER MATERIAL. The vendor's GLBs carry two materials each (a black
#: body with red fenders, dark rims with black tires), and MuJoCo loads one mesh per OBJ and reads no
#: OBJ material -- so joining them, as `reduce-mesh` does by default, delivers a uniformly grey
#: robot. The split writes `<stem>__<material>.obj` plus a `<stem>.materials.json` of base colours,
#: which this script turns into an MJCF `<material>` per sub-geom.
MESHES = {
    "body": ("body.glb", 0.001, 6000),
    "wheel_l": ("wheel_l.glb", 1.0, 3000),
    "wheel_r": ("wheel_r.glb", 1.0, 3000),
}

#: A minimal xacro wrapper around the robot macro, so the components repository is not required.
BASE_XACRO = """<?xml version='1.0'?>
<robot name="rosbot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find rosbot_description)/urdf/rosbot/rosbot_macro.urdf.xacro"
                 ns="husarion"/>
  <xacro:husarion.rosbot controller_config="" mecanum="False" namespace="" use_sim="False"/>
</robot>
"""


def convert_meshes(description: Path) -> dict[str, dict[str, list[float]]]:
    """GLB -> per-material OBJs through `roqsim assets reduce-mesh`, baking the URDF's scale in.

    Returns ``{stem: {material: rgba}}`` so the MJCF can declare the vendor's real colours.
    """
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in list((PKG / "meshes").glob("*.obj")) + list((PKG / "meshes").glob("*.json")):
        stale.unlink()
    palette: dict[str, dict[str, list[float]]] = {}
    for stem, (source, scale, budget) in MESHES.items():
        subprocess.run(
            [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
             "--target-faces", str(budget), "--split-materials", "--scale", str(scale),
             str(description / "meshes/rosbot" / source), str(PKG / "meshes" / f"{stem}.obj")],
            check=True, cwd=ROOT, capture_output=True,
        )
        palette[stem] = json.loads((PKG / "meshes" / f"{stem}.materials.json").read_text())
    return palette


def build(urdf: ET.Element, palette: dict) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    body = inertial(links["body_link"])
    wheel = inertial(links["fl_wheel_link"])
    box = links["body_link"].find("collision/geometry/box").get("size").replace(",", " ")
    half = " ".join(f"{float(v) / 2:g}" for v in box.split())
    cyl = links["fl_wheel_link"].find("collision/geometry/cylinder")
    radius, length = float(cyl.get("radius")), float(cyl.get("length"))
    body_z = joints["base_to_body_joint"].find("origin").get("xyz").split()[2]

    # One <material> and one <mesh> per vendor material, so the robot keeps its real colours.
    materials, assets = [], []
    for stem, colours in sorted(palette.items()):
        for material, rgba in sorted(colours.items()):
            materials.append(
                f'    <material name="{material}" rgba="{" ".join(f"{c:g}" for c in rgba)}"/>\n'
            )
            assets.append(f'    <mesh file="{stem}__{material}.obj"/>\n')
    # A material may be shared across meshes (both wheels use the same rim and tire); declare once.
    materials = sorted(set(materials))

    body_geoms = "".join(
        f'        <geom class="visual" mesh="body__{m}" material="{m}" pos="0 0 -0.0173"/>\n'
        for m in sorted(palette["body"])
    )

    wheels = []
    for name in ("fl", "fr", "rl", "rr"):
        xyz = joints[f"{name}_wheel_joint"].find("origin").get("xyz").replace(",", " ")
        stem = "wheel_l" if name.endswith("l") else "wheel_r"
        geoms = "".join(
            f'        <geom class="visual" mesh="{stem}__{m}" material="{m}"/>\n'
            for m in sorted(palette[stem])
        )
        wheels.append(WHEEL.format(name=name, xyz=xyz, geoms=geoms, radius=f"{radius:g}",
                                   halflen=f"{length / 2:g}", **wheel))

    return TEMPLATE.format(
        commit=ROSBOT_COMMIT, body_z=body_z, box=box, half=half,
        radius=f"{radius:g}", length=f"{length:g}",
        body_mass=body["mass"], body_pos=body["pos"], body_inertia=body["diaginertia"],
        materials="".join(materials), assets="".join(assets),
        body_geoms=body_geoms, wheels="".join(wheels),
    )


WHEEL = """      <body name="{name}_wheel_link" pos="{xyz}">
        <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
        <joint name="{name}_wheel_joint" class="wheel"/>
{geoms}        <geom class="wheel_collision" name="{name}_wheel_geom" size="{radius} {halflen}"/>
      </body>
"""

TEMPLATE = """<mujoco model="rosbot">
  <!--
    Husarion ROSbot (4-wheel skid-steer) for MuJoCo.

    GENERATED by external/convert/build_rosbot_mjcf.py from husarion/rosbot_ros's
    rosbot_description @ {commit} (Apache-2.0 - see rosbot_LICENSE).
    Do not hand-edit: re-run the generator. Every mass, inertia tensor, link offset, collision
    primitive and mesh transform below is Husarion's own value, read out of the expanded xacro.

    Drive: TRUE 4-wheel skid-steer - all four wheels driven, no casters. Left (front+rear) and
    right wheels are commanded together, and in-place rotation comes from the wheels scrubbing
    laterally, so wheel/floor friction is set by explicit contact pairs rather than geom defaults
    (MuJoCo max-combines friction, and a scene floor is usually high-friction). The same
    construction as husky_a200 and clearpath_jackal, and it needs the same slip_factor - see the
    manifest.

    Base robot only. The vendor's top-level xacro also pulls husarion_components_description, which
    bolts on whichever sensors a unit was ordered with; that is a deployment choice rather than the
    platform, so the lidar is attached through the manifest as husky_a200 does.

    Visual meshes are converted GLB->OBJ (group 2, no contacts); all collision geometry is
    Husarion's own primitives (body box + 4 wheel cylinders). The body mesh is authored in
    MILLIMETRES and its 0.001 scale is baked into the committed OBJ.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="rosbot">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="box" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <geom type="cylinder" group="3" rgba="0.1 0.1 0.1 0.4" quat="0.7071068 0.7071068 0 0"
              friction="1.0 0.005 0.0001"/>
      </default>
      <default class="wheel">
        <joint axis="0 1 0" damping="0.01" armature="0.001"/>
      </default>
    </default>
  </default>

  <asset>
{materials}{assets}  </asset>

  <worldbody>
    <!-- base_link: the URDF root, at ground level (the wheels' 0.0425 m radius lifts body_link to
         its own height). freejoint makes the robot mobile; spawn_robot places it. -->
    <body name="base_link" childclass="rosbot">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <body name="body_link" pos="0 0 {body_z}">
        <inertial pos="{body_pos}" mass="{body_mass}" diaginertia="{body_inertia}"/>
{body_geoms}
        <geom class="collision" size="{half}" pos="0 0 0.02"/>
        <!-- The stock scanner sits on the cover. Height is an assumption (see the port log): the
             base description ships no lidar, because Husarion keeps sensors in a separate package. -->
        <site name="lidar" pos="0 0 0.1043" size="0.006" rgba="1 0 0 0.6"/>
{wheels}      </body>
    </body>
  </worldbody>

  <contact>
    <!-- Wheel/floor friction stated explicitly: MuJoCo max-combines per-geom friction, so a
         high-friction scene floor would otherwise override the wheels and change how the skid-steer
         scrubs - which is the whole behaviour under study. -->
    <pair geom1="fl_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="fr_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="rl_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="rr_wheel_geom" geom2="floor" condim="3" solref="0.02 1" friction="1.0 1.0 0.005 0.0001 0.0001"/>
  </contact>

  <actuator>
    <!-- Velocity servos, one per wheel. ctrlrange is the platform's 1.1 m/s top speed over the
         0.0425 m wheel radius (25.9 rad/s), rounded up; forcerange is sized to accelerate the
         2.02 kg platform at the controller's published 1.2 m/s^2 with margin. -->
    <velocity name="fl_wheel_motor" joint="fl_wheel_joint" kv="2.0" ctrlrange="-27 27" forcerange="-3 3"/>
    <velocity name="fr_wheel_motor" joint="fr_wheel_joint" kv="2.0" ctrlrange="-27 27" forcerange="-3 3"/>
    <velocity name="rl_wheel_motor" joint="rl_wheel_joint" kv="2.0" ctrlrange="-27 27" forcerange="-3 3"/>
    <velocity name="rr_wheel_motor" joint="rr_wheel_joint" kv="2.0" ctrlrange="-27 27" forcerange="-3 3"/>
  </actuator>

  <keyframe>
    <!-- base_link is at ground level; spawn a hair above so the wheels settle rather than interpenetrate. -->
    <key name="home" qpos="0 0 0.002 1 0 0 0  0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    description = resolve_source("husarion_rosbot", ROSBOT_URL, ROSBOT_COMMIT,
                                 sparse="rosbot_description", subdir="rosbot_description")
    if args.check:
        # Read the committed palette rather than re-converting: the meshes are the expensive half,
        # and a check that regenerated them would rewrite what it is supposed to be checking.
        palette = {
            stem: json.loads((PKG / "meshes" / f"{stem}.materials.json").read_text())
            for stem in MESHES
        }
        with tempfile.TemporaryDirectory() as tmp:
            xml = build(expand_xacro({"rosbot_description": description},
                                       description / "urdf/rosbot.urdf.xacro", Path(tmp),
                                       wrapper=BASE_XACRO), palette)
        target = PKG / "rosbot.xml"
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {ROSBOT_COMMIT[:12]}")
        return 0

    palette = convert_meshes(description)
    with tempfile.TemporaryDirectory() as tmp:
        xml = build(expand_xacro({"rosbot_description": description},
                                       description / "urdf/rosbot.urdf.xacro", Path(tmp),
                                       wrapper=BASE_XACRO), palette)
    target = PKG / "rosbot.xml"
    write_license(
        description.parent / "LICENSE",
        PKG / "rosbot_LICENSE",
        [
            "Husarion ROSbot -- vendored geometry and description.",
            "",
            f"Upstream:   {ROSBOT_URL.removesuffix('.git')}  (rosbot_description)",
            f"Commit:     {ROSBOT_COMMIT}",
            "Copyright:  Husarion (support@husarion.com)",
            "License:    Apache License 2.0, as declared by rosbot_description/package.xml.",
            "",
            "Regenerate with: external/convert/build_rosbot_mjcf.py",
            "The full text of the grant follows.",
        ],
    )
    target.write_text(xml)
    print(f"wrote {target} + {len(MESHES)} meshes + rosbot_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
