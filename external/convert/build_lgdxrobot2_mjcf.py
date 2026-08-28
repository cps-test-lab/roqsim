#!/usr/bin/env python3
"""Build roqsim's LGDXRobot2 MJCF from `yukaitung/lgdxrobot2-ros2`.

A 0.24 m four-wheel **mecanum** research platform, 2.68 kg. The second holonomic base in
`roqsim_mobile` after the Ridgeback, so it uses ``omni_drive`` and carries **no** ``slip_factor`` --
it does not turn by scrubbing.

The platform ledger recorded "skid-steer or mecanum" as this port's biggest unknown, on the strength
of four *distinct* per-wheel meshes (mirrored roller directions need separate geometry where a
skid-steer reuses two files). That hint was right, and it did not have to be relied on: the vendor's
own ``lgdxrobot2_sim.urdf`` declares ``gz::sim::systems::MecanumDrive`` outright, and the drive
geometry below is that plugin's own configuration.

**Two descriptions, and only one of them is usable.** ``lgdxrobot2.urdf`` is the visualisation
description: nine links, twelve meshes, and **no inertial or collision elements whatsoever** -- it
sums to 0.0 kg. Everything physical comes from ``lgdxrobot2_sim.urdf``, which adds the inertials, a
base collision box and a collision **sphere** per wheel. Building from the plain URDF would have
produced a massless robot, and that is exactly why the ledger ranks ``official_desc_sim`` above
``official_desc``.

The wheel collision is the vendor's sphere, not a cylinder. For a mecanum wheel that is the better
stand-in -- a sphere offers no preferred rolling direction -- and it matches how ``omni_drive`` treats
wheels anyway: as load carriers, since the commanded twist drives the base free joint directly.

Meshes are Collada authored in Blender (the ``.blend`` ships too, which is unusually good
provenance), so they route through ``dae2obj.py`` -- one OBJ per bound material, since MuJoCo reads
no OBJ material -- and then through ``roqsim assets reduce-mesh``. 7.1 MB of DAE becomes ~2.8 MB of
decimated OBJ, which is unremarkable beside the 5.8 MB ``turtlebot3_waffle`` already ships.

Requires ``pycollada`` (``pip install 'pycollada>=0.7'``); ``dae2obj.py`` says so and fails loudly.

Usage::

    python external/convert/build_lgdxrobot2_mjcf.py           # fetch, convert, write
    python external/convert/build_lgdxrobot2_mjcf.py --check    # rebuild and diff (skips meshes)
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
from urdf_source import inertial, pose  # noqa: E402

LGDX_URL = "https://github.com/yukaitung/lgdxrobot2-ros2.git"
LGDX_COMMIT = "b821040094dd2a93a109269a978fc65a4f44c23a"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/lgdxrobot2"
#: Blender's Decimate budget per material split. The parts are CAD detail -- nuts, brackets, a
#: coupling -- so a tight budget costs nothing anyone will see, and the mecanum rollers (the black
#: `__m1` splits) are the one place the silhouette matters.
TARGET_FACES = 3000

#: Distinct colours the description's Collada actually binds, named once so the MJCF can assign a
#: material per sub-geom. Keyed by the rounded rgb dae2obj reports, because the part names are CAD
#: exports ("b_M4_Nut_001_021") and say nothing about colour.
PALETTE = {
    (0.527, 0.527, 0.527): ("lgdx_aluminium", "0.527 0.527 0.527 1"),
    (0.0, 0.0, 0.0): ("lgdx_black", "0.05 0.05 0.05 1"),
    (0.0, 0.262, 0.069): ("lgdx_pcb", "0.0 0.262 0.069 1"),
    (0.223, 0.011, 0.012): ("lgdx_red", "0.223 0.011 0.012 1"),
    (0.447, 0.474, 0.502): ("lgdx_steel", "0.447 0.474 0.502 1"),
}
#: Massless fixed links whose visual rides on the base.
FIXED_VISUAL_LINKS = ("camera_link", "lidar_link")
WHEELS = ("wheel1_link", "wheel2_link", "wheel3_link", "wheel4_link")


def convert_meshes(description: Path) -> dict[str, list]:
    """DAE -> per-material OBJ -> decimated OBJ. Returns dae2obj's ``{stem: [(sub, rgb), ...]}``."""
    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (PKG / "meshes").glob("*.obj"):
        stale.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "obj"
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "dae2obj.py"),
             str(description / "meshes/dae"), str(raw)],
            check=True, capture_output=True,
        )
        palette = json.loads((raw / "materials.json").read_text())
        for parts in palette.values():
            for sub, _rgb in parts:
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(TARGET_FACES), "--no-materials",
                     str(raw / f"{sub}.obj"), str(PKG / "meshes" / f"{sub}.obj")],
                    check=True, cwd=ROOT, capture_output=True,
                )
    return palette


def material_of(rgb: list[float]) -> str:
    """The MJCF material name for a bound Collada colour, failing loudly on an unknown one.

    A colour the palette does not know would otherwise render as MuJoCo's default grey, which looks
    like a lighting problem rather than a missing entry. If upstream repaints a part, this says so.
    """
    key = tuple(round(float(c), 3) for c in rgb[:3])
    if key not in PALETTE:
        raise RuntimeError(
            f"unknown bound colour {key} -- add it to PALETTE with a name, rather than letting the "
            f"part fall back to MuJoCo's default grey."
        )
    return PALETTE[key][0]


def build(urdf: ET.Element, palette: dict[str, list]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}

    def visual_geoms(link: ET.Element, indent: str, offset=("0", "0", "0")) -> str:
        """Every mesh visual of *link*, expanded into one geom per bound material.

        A URDF visual names one Collada file; MuJoCo reads no OBJ material, so dae2obj split each
        file into one OBJ per material and each of those becomes its own geom here. That is what
        keeps the PCB green and the mecanum rollers black instead of everything flat grey.
        """
        out = ""
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            stem = Path(mesh.get("filename")).stem
            xyz, quat = pose(visual)
            x, y, z = (float(a) + float(b) for a, b in zip(xyz.split(), offset, strict=True))
            for sub, rgb in palette[stem]:
                out += (f'{indent}<geom class="visual" mesh="{sub}"'
                        f' material="{material_of(rgb)}" pos="{x:g} {y:g} {z:g}"{quat}/>\n')
        return out

    base = links["base_link"]
    base_visuals = visual_geoms(base, "        ")
    for name in FIXED_VISUAL_LINKS:
        xyz, _ = pose(joints[name])
        base_visuals += visual_geoms(links[name], "        ", tuple(xyz.split()))

    box = base.find("collision/geometry/box").get("size").replace(",", " ")
    box_pos, box_quat = pose(base.find("collision"))
    half = " ".join(f"{float(v) / 2:g}" for v in box.split())

    wheels = ""
    for name in WHEELS:
        link = links[name]
        joint = joints[name]
        pos, quat = pose(joint)
        wheels += WHEEL_BODY.format(
            name=name, body_pos=pos, body_quat=quat, joint=joint.get("name"),
            radius=link.find("collision/geometry/sphere").get("radius"),
            geoms=visual_geoms(link, "          "),
            **inertial(link),
        )

    lidar_xyz, _ = pose(joints["lidar_link"])
    imu_xyz, _ = pose(joints["imu_link"])
    wheel_z = float(pose(joints["wheel1_link"])[0].split()[2])
    radius = float(links["wheel1_link"].find("collision/geometry/sphere").get("radius"))
    assets = "".join(
        f'    <material name="{n}" rgba="{rgba}"/>\n' for n, rgba in sorted(PALETTE.values())
    ) + "".join(
        f'    <mesh file="{sub}.obj"/>\n'
        for sub in sorted(s for parts in palette.values() for s, _ in parts)
    )
    return TEMPLATE.format(
        commit=LGDX_COMMIT, assets=assets, base_visuals=base_visuals,
        box_half=half, box_pos=box_pos, box_quat=box_quat, wheels=wheels,
        lidar_pos=lidar_xyz, imu_pos=imu_xyz,
        rest_height=f"{radius - wheel_z:g}",
        **{f"base_{k}": v for k, v in inertial(base).items()},
    )


WHEEL_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="{joint}" class="wheel"/>
{geoms}          <geom class="wheel_collision" name="{name}_tyre" size="{radius}"/>
        </body>
"""

TEMPLATE = """<mujoco model="lgdxrobot2">
  <!--
    LGDXRobot2 - a 0.24 m four-wheel mecanum research platform, 2.68 kg.

    GENERATED by external/convert/build_lgdxrobot2_mjcf.py from yukaitung/lgdxrobot2-ros2 @ {commit}
    (MIT - see lgdxrobot2_LICENSE). Do not hand-edit: re-run the generator.

    Drive: HOLONOMIC, the second such base here after the ridgeback, so it uses `omni_drive` and
    carries NO slip_factor - a mecanum base does not turn by scrubbing. The commanded twist drives
    three velocity actuators on the base free joint; the wheel servos exist so the visuals and
    joint_states are right, not as the motive force.

    Every mass, inertia and collision below comes from the vendor's lgdxrobot2_SIM.urdf. Its plain
    lgdxrobot2.urdf is visualisation-only - nine links, no inertial and no collision elements at
    all, summing to 0.0 kg - so building from it would have produced a massless robot.

    The wheel collision is the vendor's own SPHERE rather than a cylinder, which for a mecanum wheel
    is the better stand-in: no preferred rolling direction.

    One consequence to know before using this model: omni_drive integrates odometry from the
    ACHIEVED twist and models no wheel slip, so encoder odometry and ground truth coincide by
    construction. This platform cannot be used to study odometry drift.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="lgdxrobot2">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Near-frictionless spheres: an omni wheel IS this in the directions that matter, and the
             base is driven through the planar actuators below rather than through these. `priority`
             is what makes the low friction take effect at all - MuJoCo otherwise takes the MAXIMUM
             of the two contacting geoms' friction and the floor's value wins. -->
        <geom type="sphere" group="3" rgba="0.05 0.05 0.05 0.4"
              friction="0.02 0.005 0.0001" priority="2"/>
      </default>
      <default class="wheel">
        <!-- axis 0 0 1 in the wheel's OWN frame, which the joint's rpy has already rotated to lie
             along the robot's y. The description's own axis, not a re-derived one. -->
        <joint axis="0 0 1" damping="0.001" armature="0.0005"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <worldbody>
    <body name="base_link" childclass="lgdxrobot2">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <site name="base_imu" pos="{imu_pos}" size="0.005" rgba="0 0 0 0"/>
      <!-- The scan plane, off the description's own lidar_link_joint. -->
      <site name="lidar" pos="{lidar_pos}" size="0.005" rgba="1 0 0 0.6"/>
{base_visuals}        <geom class="collision" type="box" size="{box_half}" name="base_collision"
              pos="{box_pos}"{box_quat}/>
{wheels}    </body>
  </worldbody>

  <actuator>
    <!-- The planar drive. Limits are the vendor's own Nav2 profile for the REAL robot
         (lgdxrobot2_bringup/param/loc/nav2.yaml): 0.33 m/s in x and y, 0.3 rad/s yaw. Its Gazebo
         profile sets max_vel_y to 0, treating the platform as if it were differential; the real
         profile is the one that describes the machine. forcerange is sized to move 2.68 kg at the
         MecanumDrive plugin's own 5 m/s^2. -->
    <velocity name="base_vx" joint="base_free" gear="1 0 0 0 0 0" kv="300" ctrlrange="-0.33 0.33" forcerange="-40 40"/>
    <velocity name="base_vy" joint="base_free" gear="0 1 0 0 0 0" kv="300" ctrlrange="-0.33 0.33" forcerange="-40 40"/>
    <velocity name="base_wz" joint="base_free" gear="0 0 0 0 0 1" kv="30" ctrlrange="-0.3 0.3" forcerange="-8 8"/>
    <!-- Observational wheel servos, driven from the mecanum inverse kinematics so the wheels turn,
         and turn differently when strafing. They transmit almost nothing. -->
    <velocity name="wheel1_motor" joint="wheel1_link_joint" kv="0.05" ctrlrange="-20 20" forcerange="-0.5 0.5"/>
    <velocity name="wheel2_motor" joint="wheel2_link_joint" kv="0.05" ctrlrange="-20 20" forcerange="-0.5 0.5"/>
    <velocity name="wheel3_motor" joint="wheel3_link_joint" kv="0.05" ctrlrange="-20 20" forcerange="-0.5 0.5"/>
    <velocity name="wheel4_motor" joint="wheel4_link_joint" kv="0.05" ctrlrange="-20 20" forcerange="-0.5 0.5"/>
  </actuator>

  <keyframe>
    <!-- base_link rides at wheel radius minus the wheel-centre height. -->
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("lgdxrobot2", LGDX_URL, LGDX_COMMIT)
    description = source / "lgdxrobot2_description"
    target = PKG / "lgdxrobot2.xml"
    urdf = ET.parse(description / "description/lgdxrobot2_sim.urdf").getroot()

    if args.check:
        # The palette is read back from the shipped meshes rather than re-derived, so --check does
        # not need Blender or pycollada: it verifies the MJCF, which is what can be hand-edited.
        palette = json.loads((PKG / "meshes/lgdxrobot2.materials.json").read_text())
        if not target.exists() or target.read_text() != build(urdf, palette):
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {LGDX_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    palette = convert_meshes(description)
    (PKG / "meshes/lgdxrobot2.materials.json").write_text(
        json.dumps(palette, indent=2, sort_keys=True) + "\n")
    shutil.copy2(source / "LICENSE", PKG / "lgdxrobot2_LICENSE")
    target.write_text(build(urdf, palette))
    print(f"wrote {target} + meshes + lgdxrobot2_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
