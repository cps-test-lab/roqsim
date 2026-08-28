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
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402

ROSBOT_URL = "https://github.com/husarion/rosbot_ros.git"
# Pinned. If this moves, the port log moves with it.
ROSBOT_COMMIT = "41fad02196ee200a39e01579c75391385cc9b714"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/rosbot"

#: mesh stem -> (source file, scale to metres). The body's 0.001 is the URDF's own scale attribute.
MESHES = {
    "body": ("body.glb", 0.001),
    "wheel_l": ("wheel_l.glb", 1.0),
    "wheel_r": ("wheel_r.glb", 1.0),
}
TARGET_FACES = 8000

#: A minimal xacro wrapper around the robot macro, so the components repository is not required.
BASE_XACRO = """<?xml version='1.0'?>
<robot name="rosbot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find rosbot_description)/urdf/rosbot/rosbot_macro.urdf.xacro"
                 ns="husarion"/>
  <xacro:husarion.rosbot controller_config="" mecanum="False" namespace="" use_sim="False"/>
</robot>
"""


def expand_xacro(description: Path, work: Path) -> ET.Element:
    """Expand the robot macro into a plain URDF, with an ament index pointing at the checkout."""
    xacro = shutil.which("xacro") or "/opt/ros/jazzy/bin/xacro"
    if not Path(xacro).exists():
        raise RuntimeError(
            "xacro is required to expand rosbot_description and was not found.\n"
            "Install ROS 2 (the model ships with ros-jazzy-xacro) or `pip install xacro`, then "
            "re-run. Refusing to guess the expanded tree: the masses and offsets come from it."
        )
    prefix = work / "prefix/share"
    (prefix / "ament_index/resource_index/packages").mkdir(parents=True, exist_ok=True)
    (prefix / "ament_index/resource_index/packages/rosbot_description").write_text("")
    link = prefix / "rosbot_description"
    if not link.exists():
        link.symlink_to(description)

    source = work / "base.urdf.xacro"
    source.write_text(BASE_XACRO)
    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = f"{work / 'prefix'}:{env.get('AMENT_PREFIX_PATH', '')}"
    # The ROS xacro entry point imports its own package, which is only on PYTHONPATH once the ROS
    # setup has been sourced. Sourcing it here would leak the whole ROS environment into this
    # process; adding the one site-packages it needs does not, and keeps the failure legible when
    # ROS is absent entirely.
    ros_site = sorted(Path(xacro).resolve().parents[1].glob("lib/python3*/site-packages"))
    if ros_site:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ros_site[-1]), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    proc = subprocess.run([xacro, str(source)], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"xacro failed:\n{proc.stderr.strip()}")
    return ET.fromstring(proc.stdout)


def convert_meshes(description: Path) -> None:
    """GLB -> OBJ through `roqsim assets reduce-mesh`, baking the URDF's scale in."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stem, (source, scale) in MESHES.items():
        out = PKG / "meshes" / f"{stem}.obj"
        subprocess.run(
            [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
             "--target-faces", str(TARGET_FACES), "--no-materials", "--scale", str(scale),
             str(description / "meshes/rosbot" / source), str(out)],
            check=True, cwd=ROOT, capture_output=True,
        )


def _inertial(link: ET.Element) -> dict[str, str]:
    inertial = link.find("inertial")
    origin = inertial.find("origin")
    inertia = inertial.find("inertia")
    return {
        "pos": origin.get("xyz", "0 0 0").replace(",", " ") if origin is not None else "0 0 0",
        "mass": inertial.find("mass").get("value"),
        "diaginertia": " ".join(inertia.get(k) for k in ("ixx", "iyy", "izz")),
    }


def build(urdf: ET.Element) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    body = _inertial(links["body_link"])
    wheel = _inertial(links["fl_wheel_link"])
    box = links["body_link"].find("collision/geometry/box").get("size").replace(",", " ")
    half = " ".join(f"{float(v) / 2:g}" for v in box.split())
    cyl = links["fl_wheel_link"].find("collision/geometry/cylinder")
    radius, length = float(cyl.get("radius")), float(cyl.get("length"))
    body_z = joints["base_to_body_joint"].find("origin").get("xyz").split()[2]

    wheels = []
    for name in ("fl", "fr", "rl", "rr"):
        xyz = joints[f"{name}_wheel_joint"].find("origin").get("xyz").replace(",", " ")
        mesh = "wheel_l" if name.endswith("l") else "wheel_r"
        wheels.append(WHEEL.format(name=name, xyz=xyz, mesh=mesh, radius=f"{radius:g}",
                                   halflen=f"{length / 2:g}", **wheel))

    return TEMPLATE.format(
        commit=ROSBOT_COMMIT, body_z=body_z, box=box, half=half,
        radius=f"{radius:g}", length=f"{length:g}",
        body_mass=body["mass"], body_pos=body["pos"], body_inertia=body["diaginertia"],
        wheels="".join(wheels),
    )


WHEEL = """      <body name="{name}_wheel_link" pos="{xyz}">
        <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
        <joint name="{name}_wheel_joint" class="wheel"/>
        <geom class="visual" mesh="{mesh}" quat="0.7071068 0.7071068 0 0"/>
        <geom class="wheel_collision" name="{name}_wheel_geom" size="{radius} {halflen}"/>
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
    <mesh file="body.obj"/>
    <mesh file="wheel_l.obj"/>
    <mesh file="wheel_r.obj"/>
  </asset>

  <worldbody>
    <!-- base_link: the URDF root, at ground level (the wheels' 0.0425 m radius lifts body_link to
         its own height). freejoint makes the robot mobile; spawn_robot places it. -->
    <body name="base_link" childclass="rosbot">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <body name="body_link" pos="0 0 {body_z}">
        <inertial pos="{body_pos}" mass="{body_mass}" diaginertia="{body_inertia}"/>
        <geom class="visual" mesh="body" pos="0 0 -0.0173" quat="0.5 0.5 0.5 0.5"/>
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
    with tempfile.TemporaryDirectory() as tmp:
        xml = build(expand_xacro(description, Path(tmp)))
    target = PKG / "rosbot.xml"

    if args.check:
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {ROSBOT_COMMIT[:12]}")
        return 0

    convert_meshes(description)
    shutil.copy2(description.parent / "LICENSE", PKG / "rosbot_LICENSE")
    target.write_text(xml)
    print(f"wrote {target} + {len(MESHES)} meshes + rosbot_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
