#!/usr/bin/env python3
"""Build roqsim's Neobotix MPO-700 MJCF from `neobotix/neo_simulation2`.

The substrate's **first steerable-wheel (swerve) base**: four independently steered wheels, 196.8 kg.
Holonomic like the Ridgeback and the LGDXRobot2, so it uses ``omni_drive`` -- but it is the first
platform whose wheels have to be *aimed*, which is the swerve inverse kinematics added to that plugin
alongside its mecanum table.

Two source facts worth knowing before reading this, both recorded in the platform ledger:

* **The MIT source is `neo_simulation2`, not `neo_mpo_700-2`.** The survey ranked the latter, which
  GitHub reports as having no detected licence. `neo_simulation2` is MIT and carries all four
  Neobotix platforms with meshes, urdf and sim configs together.
* **Only the `humble` branch has the mechanism.** `rolling` and `jazzy-sync` ship a flattened
  ``mpo_700.urdf`` in which *every* joint is ``fixed``; `humble` keeps the xacro macros that build
  the steer/roll chain. The macros are parameterised on joint type and the vendor's top-level passes
  ``fixed`` for both, so this generator supplies its own top-level (see :data:`WRAPPER`) to get the
  articulated tree the macros can already describe.

**That last point is the port's main finding, and it cuts both ways.** The vendor does not model the
steering at all -- their Gazebo setup drives the base directly and the wheels are decoration -- so a
"faithful" port would have a swerve robot whose wheels never turn. Instantiating the joints is
therefore a deliberate *improvement* on upstream, and it is honest about its limits: there are no
vendor steer limits, no steer controller config and no rate figures to calibrate against, so the
steering here is geometrically right and dynamically nominal.

Meshes are millimetre-scale Collada (``scale="0.001 0.001 0.001"``), which is exactly the trap
:func:`urdf_source.mesh_scales` exists to catch -- see its docstring.

Usage::

    python external/convert/build_mpo700_mjcf.py           # fetch, convert, write
    python external/convert/build_mpo700_mjcf.py --check    # rebuild and diff
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
#: The `humble` branch: the only one that still carries the articulated xacro macros.
NEO_COMMIT = "832041452c1a0199afea1e9b65adf37381e96214"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/mpo_700"
TARGET_FACES = 4000

#: Our own top-level, because the vendor's hardcodes `ODM_joint_type` to "fixed" as a
#: <xacro:property> rather than a <xacro:arg> -- so it cannot be overridden from outside, and the
#: articulated tree its own macros support is unreachable through it.
WRAPPER = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="mpo_700">
  <xacro:arg name="use_docking_adapter" default="false"/>
  <xacro:property name="ODM_joint_type" value="continuous"/>
  <xacro:property name="arm" value=""/>
  <xacro:property name="use_arm" value="false"/>
  <xacro:include filename="$(find neo_simulation2)/robots/mpo_700/urdf/mpo_700_body.urdf.xacro"/>
</robot>
"""

#: Corner order matching omni_drive's WHEEL_ORDER (front_left, front_right, rear_left, rear_right).
CORNERS = ("front_left", "front_right", "back_left", "back_right")
#: Massless-ish fixed links whose visual rides on the base but whose mass must be kept.
SENSOR_LINKS = ("lidar_1_link", "lidar_2_link")


def convert_meshes(source: Path, urdf: ET.Element) -> dict[str, str]:
    """Convert only the meshes the expanded tree actually references. Returns ``{stem: scale}``.

    The per-material split is also what carries COLOUR: these Collada files bind ten distinct
    diffuse colours across four meshes -- including the SICK scanners' signature yellow and the
    wheels' orange accent -- and MuJoCo reads no OBJ material, so without one geom per material the
    whole robot renders flat grey. dae2obj's ``materials.json`` is kept beside the meshes so
    ``--check`` can rebuild the MJCF without Blender or pycollada.
    """
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
        (PKG / "meshes").mkdir(parents=True, exist_ok=True)
        (PKG / "meshes/mpo_700.materials.json").write_text(
            json.dumps(palette, indent=2, sort_keys=True) + "\n")
        for parts in palette.values():
            for sub, _rgb in parts:
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(TARGET_FACES), "--no-materials",
                     str(raw / f"{sub}.obj"), str(PKG / "meshes" / f"{sub}.obj")],
                    check=True, cwd=ROOT, capture_output=True,
                )
    return scales


def build(urdf: ET.Element, shipped: set[str], scales: dict[str, str]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    # One material per sub-mesh, named after it. A shared palette keyed on colour would need a
    # hand-maintained table (as the LGDXRobot2 port has); with ten colours over four meshes, 1:1 is
    # both simpler and impossible to get wrong -- and an upstream repaint just changes an rgba.
    colours = {
        sub: " ".join(f"{float(c):g}" for c in (*rgb[:3], 1.0))
        for parts in json.loads((PKG / "meshes/mpo_700.materials.json").read_text()).values()
        for sub, rgb in parts
    }

    def geoms(link: ET.Element, indent: str, cls: str, offset=("0", "0", "0")) -> str:
        """A link's mesh visuals and its collision, in MJCF form.

        Sub-meshes are matched by prefix because dae2obj splits a multi-material Collada into
        ``<stem>__<material>`` and the URDF only ever names ``<stem>``.
        """
        out = ""
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            stem = Path(mesh.get("filename")).stem
            xyz, quat = pose(visual)
            x, y, z = (float(a) + float(b) for a, b in zip(xyz.split(), offset, strict=True))
            for sub in sorted(s for s in shipped if s == stem or s.startswith(f"{stem}__")):
                mat = f' material="{sub}_mat"' if sub in colours else ""
                out += (f'{indent}<geom class="visual" mesh="{sub}"{mat}'
                        f' pos="{x:g} {y:g} {z:g}"{quat}/>\n')
        for collision in link.findall("collision"):
            shape = collision.find("geometry")[0]
            xyz, quat = pose(collision)
            x, y, z = (float(a) + float(b) for a, b in zip(xyz.split(), offset, strict=True))
            if shape.tag == "sphere":
                out += (f'{indent}<geom class="{cls}" name="{link.get("name")}_tyre"'
                        f' size="{float(shape.get("radius")):g}"'
                        f' pos="{x:g} {y:g} {z:g}"{quat}/>\n')
            else:
                stem = Path(shape.get("filename")).stem
                sub = next(iter(sorted(s for s in shipped
                                       if s == stem or s.startswith(f"{stem}__"))))
                out += (f'{indent}<geom class="collision" mesh="{sub}"'
                        f' name="{link.get("name")}_collision"'
                        f' pos="{x:g} {y:g} {z:g}"{quat}/>\n')
        return out

    base = links["base_link"]
    base_geoms = geoms(base, "        ", "collision")

    # The sensor links stay their own bodies rather than being merged, so the mass audit reproduces
    # the description's sum -- lidar_1 is 1.2 kg, not a rounding error on a 196.8 kg robot.
    sensors = ""
    for name in SENSOR_LINKS:
        pos, quat = pose(joints[name])
        sensors += SENSOR_BODY.format(
            name=name, body_pos=pos, body_quat=quat,
            geoms=geoms(links[name], "          ", "collision"),
            **inertial(links[name]),
        )

    corners = ""
    for corner in CORNERS:
        steer_link = f"mpo_700_caster_{corner}_link"
        roll_link = f"mpo_700_wheel_{corner}_link"
        steer_pos, steer_quat = pose(joints[steer_link])
        roll_pos, roll_quat = pose(joints[roll_link])
        corners += CORNER_BODY.format(
            steer_link=steer_link, steer_pos=steer_pos, steer_quat=steer_quat,
            steer_joint=joints[steer_link].get("name"),
            steer_geoms=geoms(links[steer_link], "          ", "collision"),
            roll_link=roll_link, roll_pos=roll_pos, roll_quat=roll_quat,
            roll_joint=joints[roll_link].get("name"),
            roll_geoms=geoms(links[roll_link], "            ", "wheel_collision"),
            steer_inertial=_inertial_line(links[steer_link]),
            roll_inertial=_inertial_line(links[roll_link]),
        )

    wheel_z = (float(pose(joints["mpo_700_caster_front_left_link"])[0].split()[2])
               + float(pose(joints["mpo_700_wheel_front_left_link"])[0].split()[2]))
    radius = float(links["mpo_700_wheel_front_left_link"]
                   .find("collision/geometry/sphere").get("radius"))
    assets = "".join(
        f'    <material name="{sub}_mat" rgba="{rgba}"/>\n'
        for sub, rgba in sorted(colours.items()) if sub in shipped
    ) + "".join(
        f'    <mesh file="{sub}.obj" scale="{scales.get(sub.split("__")[0], "1 1 1")}"'
        f' inertia="shell"/>\n'
        for sub in sorted(shipped)
    )
    return TEMPLATE.format(
        commit=NEO_COMMIT, assets=assets, base_geoms=base_geoms, sensors=sensors, corners=corners,
        rest_height=f"{radius - wheel_z:g}",
        **{f"base_{k}": v for k, v in inertial(base).items()},
    )


def _inertial_line(link: ET.Element) -> str:
    a = inertial(link)
    return f'<inertial pos="{a["pos"]}" mass="{a["mass"]}" diaginertia="{a["diaginertia"]}"/>'


SENSOR_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}        </body>
"""

CORNER_BODY = """        <body name="{steer_link}" pos="{steer_pos}"{steer_quat}>
          {steer_inertial}
          <joint name="{steer_joint}" class="steer"/>
{steer_geoms}          <body name="{roll_link}" pos="{roll_pos}"{roll_quat}>
            {roll_inertial}
            <joint name="{roll_joint}" class="roll"/>
{roll_geoms}          </body>
        </body>
"""

TEMPLATE = """<mujoco model="mpo_700">
  <!--
    Neobotix MPO-700 - a four-wheel SWERVE (independently steered) omnidirectional base, 196.8 kg.

    GENERATED by external/convert/build_mpo700_mjcf.py from neobotix/neo_simulation2 @ {commit}
    (MIT - see mpo_700_LICENSE). Do not hand-edit: re-run the generator.

    The substrate's first steerable-wheel base. Drive is HOLONOMIC, so it uses `omni_drive` like the
    ridgeback and lgdxrobot2 - but it is the first platform whose wheels must be AIMED, which is the
    swerve inverse kinematics in that plugin beside its mecanum table. No slip_factor: a swerve base
    does not turn by scrubbing.

    Built from the vendor's `humble` branch, whose xacro macros still describe the steer/roll chain.
    Their `rolling` and `jazzy-sync` branches ship a flattened urdf in which every joint is FIXED,
    and their own top-level hardcodes the joint type to "fixed" as a property rather than an arg -
    so this model uses a generator-supplied top-level to reach the articulated tree the macros can
    already build. The vendor does not model the steering at all; instantiating it is a deliberate
    improvement on upstream, with the honest consequence that there are no vendor steer limits or
    controller figures to calibrate against.

    One consequence to know: omni_drive integrates odometry from the ACHIEVED twist and models no
    wheel slip, so encoder odometry and ground truth coincide by construction. And the steering here
    is observational - a paper measuring steer rate limits or reorientation delay needs real steer
    actuation, which this is not.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <!--
    Every mesh is `inertia="shell"`. These are CAD surface exports split per material, so several
    sub-meshes are thin shells with no meaningful enclosed volume and MuJoCo refuses to integrate an
    inertia over them ("mesh volume is too small"). It never needed to: every body here carries the
    vendor's own explicit <inertial>, so a mesh-derived inertia would be discarded anyway.
  -->

  <default>
    <default class="mpo_700">
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
      <default class="steer">
        <joint axis="0 0 1" damping="30" armature="0.5"/>
      </default>
      <default class="roll">
        <joint axis="0 -1 0" damping="0.1" armature="0.05"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <contact>
    <!--
      The vendor's base collision IS the full body mesh, and MuJoCo convex-hulls a collision mesh -
      so the hull closes over the wheel arches and overlaps the wheels inside them (measured: the
      front-right tyre penetrates it by 1.4 mm at the reference pose). A wheel is a GRANDCHILD of
      base_link, via its steering link, so MuJoCo's automatic parent-child exclusion does not cover
      the pair and the robot fights itself at rest.

      Excluded explicitly rather than by widening a contact group: these four pairs are the ones that
      cannot be real, and naming them leaves every other self-collision live.
    -->
    <exclude body1="base_link" body2="mpo_700_wheel_front_left_link"/>
    <exclude body1="base_link" body2="mpo_700_wheel_front_right_link"/>
    <exclude body1="base_link" body2="mpo_700_wheel_back_left_link"/>
    <exclude body1="base_link" body2="mpo_700_wheel_back_right_link"/>
  </contact>

  <worldbody>
    <body name="base_link" childclass="mpo_700">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <!-- The vendor's own two scanner mounts, diagonally opposite. Neither sees 360 degrees on its
           own; together they cover the robot. -->
      <site name="lidar_1" pos="0.338 0.288 0.223" size="0.01" rgba="1 0 0 0.6"/>
      <site name="lidar_2" pos="-0.338 -0.288 0.223" size="0.01" rgba="1 0 0 0.6"/>
{base_geoms}{sensors}{corners}    </body>
  </worldbody>

  <actuator>
    <!-- The planar drive. Neobotix publishes 1.0 m/s and 1.0 rad/s for the MPO-700; forcerange is
         sized to move 196.8 kg. -->
    <velocity name="base_vx" joint="base_free" gear="1 0 0 0 0 0" kv="25000" ctrlrange="-1.0 1.0" forcerange="-2500 2500"/>
    <velocity name="base_vy" joint="base_free" gear="0 1 0 0 0 0" kv="25000" ctrlrange="-1.0 1.0" forcerange="-2500 2500"/>
    <velocity name="base_wz" joint="base_free" gear="0 0 0 0 0 1" kv="12500" ctrlrange="-1.0 1.0" forcerange="-1500 1500"/>
    <!-- Steer POSITION servos, one per corner. Unlimited range because the joints are continuous;
         omni_drive resolves each target to the nearer of the two equivalent headings, so the wheels
         never slew half a turn to cross straight-ahead. -->
    <position name="steer_front_left" joint="mpo_700_caster_front_left_joint" kp="800" kv="80" forcerange="-200 200"/>
    <position name="steer_front_right" joint="mpo_700_caster_front_right_joint" kp="800" kv="80" forcerange="-200 200"/>
    <position name="steer_back_left" joint="mpo_700_caster_back_left_joint" kp="800" kv="80" forcerange="-200 200"/>
    <position name="steer_back_right" joint="mpo_700_caster_back_right_joint" kp="800" kv="80" forcerange="-200 200"/>
    <!-- Observational roll servos: they make the wheels turn at the right rate for the commanded
         twist. They are not the motive force. -->
    <velocity name="wheel_front_left_motor" joint="mpo_700_wheel_front_left_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="wheel_front_right_motor" joint="mpo_700_wheel_front_right_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="wheel_back_left_motor" joint="mpo_700_wheel_back_left_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
    <velocity name="wheel_back_right_motor" joint="mpo_700_wheel_back_right_joint" kv="5" ctrlrange="-30 30" forcerange="-10 10"/>
  </actuator>

  <keyframe>
    <!-- base_link rides at wheel radius minus the wheel-centre height, which is NEGATIVE here: the
         vendor's base_link frame sits a centimetre below the tyre contact. -->
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("neo_simulation2", NEO_URL, NEO_COMMIT)
    target = PKG / "mpo_700.xml"
    with tempfile.TemporaryDirectory() as tmp:
        urdf = expand_xacro({"neo_simulation2": source}, Path("mpo_700.urdf.xacro"),
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
    shutil.copy2(source / "LICENSE", PKG / "mpo_700_LICENSE")
    target.write_text(build(urdf, shipped, scales))
    print(f"wrote {target} + meshes + mpo_700_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
