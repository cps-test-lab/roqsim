#!/usr/bin/env python3
"""Build roqsim's PiRacer MJCF from `lalywr2000/PiRacer-Gazebo-Simulation`.

A 0.236 m Ackermann-steered RC car -- the first car-like base in `roqsim_mobile`, and the model
`ackermann_drive` was written for. Its geometry is what the other bases here cannot express: two
front wheels steered through *independent* joints, driven wheels only at the back, and no way to
turn on the spot.

**Why this source.** The platform's own vendor publishes no CAD, and the descriptions that ship with
the better-known cars in this class are either visual-only (fixed front hinges, no steering degree of
freedom at all) or carry no redistribution grant. This one is MIT with a named copyright holder, and
its meshes are original: seven SketchUp exports of 12 to 1294 triangles authored in millimetres,
whose modelling session the author ships as a screenshot. That is what makes it committable.

**The masses here are ours, not the source's.** The upstream description gives the chassis 1000 kg
and each wheel 50 kg with `ixx=iyy=izz=0.1` throughout -- roughly 1203 kg for a 24 cm car. Those are
Gazebo solver hacks, not physical data, and reproducing them would give a toy the yaw inertia of a
van. Everything in MEASURED below replaces them, and MuJoCo derives the tensors from the geoms.

**Two front steering joints, deliberately not coupled.** `ackermann_drive` computes the inner/outer
split itself (`atan(L / (R -/+ track/2))`), so the left and right steer joints must stay independent.
The upstream description also carries a third, virtual `main_steer` joint on the centreline; it is
dropped, because a plugin that computes the split has no use for a shared angle.

Requires no Blender and no mesh conversion: MuJoCo reads STL directly, and at these triangle counts
there is nothing to decimate. The meshes are copied byte-for-byte, so the model ships exactly the
geometry the licence covers.

Usage::

    python external/convert/build_piracer_mjcf.py           # fetch, convert, write
    python external/convert/build_piracer_mjcf.py --check    # rebuild and diff (skips meshes)
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import pose, rpy_to_quat, write_license  # noqa: E402

PIRACER_URL = "https://github.com/lalywr2000/PiRacer-Gazebo-Simulation.git"
PIRACER_COMMIT = "db9f49c85e4417ec3566d0d87702f49febbd3ee1"

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/piracer"

#: Physical quantities the upstream description does not honestly supply, so they come from the
#: hardware instead. Each carries how it was obtained: an ASSUMED value is a placeholder to be
#: replaced by a measurement, and the manifest and port log must say so for as long as it is one.
MEASURED = {
    "total_mass": (1.5, "ASSUMED"),      # kg, car as run: chassis, electronics, battery, wheels
    "wheel_mass": (0.045, "ASSUMED"),    # kg per tyre+rim assembly
    "com_x": (0.118, "ASSUMED"),         # m from the chassis origin; the axle load split fixes it
    "max_steer": (0.5, "ASSUMED"),       # rad at full lock -- the description declares NO limit
}

#: Mass of one steering knuckle -- the small body carrying a front wheel's steer joint. It is not a
#: part anyone weighs separately, but it must come OUT of the chassis budget rather than be added on
#: top, or the car ends up heavier than the total mass it claims.
KNUCKLE_MASS = 0.01

#: Colours the source renders in SketchUp but does not encode anywhere a converter can read: the
#: meshes are bare STL, which carries no material. Taken from the author's own design screenshot so
#: the model looks like the car it is named after rather than uniform MuJoCo grey.
MATERIALS = {
    "chassis": ("mat_piracer_chassis", "0.22 0.70 0.24 1"),
    "board": ("mat_piracer_board", "0.12 0.12 0.12 1"),
    "rpi": ("mat_piracer_pcb", "0.05 0.35 0.12 1"),
    "cover": ("mat_piracer_battery", "0.80 0.92 0.55 1"),
    "motor": ("mat_piracer_motor", "0.62 0.62 0.62 1"),
    "tire": ("mat_piracer_tyre", "0.10 0.10 0.10 1"),
    "disk": ("mat_piracer_rim", "0.10 0.20 0.80 1"),
}

#: Fixed, massless links the description hangs off the chassis purely to be seen. They become geoms
#: on `base_link` rather than bodies: a zero-mass body with no joint is an invitation for MuJoCo to
#: warn, and none of them moves relative to the chassis.
FIXED_VISUALS = ["board", "rpi", "motor1", "motor2", "cover"]

#: (body, steering knuckle or None, wheel joint) for the four wheels, front pair first. Which of them
#: is DRIVEN is not recorded here: it is the actuator block that decides, and naming it twice is how
#: the two drift apart. The front pair has a knuckle and no motor; the rear pair the reverse.
WHEELS = [
    ("front_left_tire", "left_steer", "front_left_tire_joint"),
    ("front_right_tire", "right_steer", "front_right_tire_joint"),
    ("rear_left_tire", None, "rear_left_tire_joint"),
    ("rear_right_tire", None, "rear_right_tire_joint"),
]


def mesh_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) of a binary STL in metres. The source authors in millimetres."""
    blob = path.read_bytes()
    count = struct.unpack("<I", blob[80:84])[0]
    records = np.frombuffer(blob[84 : 84 + count * 50], dtype=np.uint8).reshape(count, 50)
    tris = np.frombuffer(records[:, 12:48].tobytes(), dtype="<f4").reshape(count * 3, 3)
    return tris.min(0) / 1000.0, tris.max(0) / 1000.0


def box_inertia(mass: float, size: np.ndarray) -> str:
    """Diagonal inertia of a uniform box of full extents *size*."""
    x, y, z = size
    return " ".join(
        f"{mass * (a ** 2 + b ** 2) / 12:.6g}" for a, b in ((y, z), (x, z), (x, y))
    )


def cylinder_inertia(mass: float, radius: float, length: float) -> str:
    """Diagonal inertia of a uniform cylinder about its own axes, spin axis last (local z)."""
    transverse = mass * (3 * radius**2 + length**2) / 12
    return f"{transverse:.6g} {transverse:.6g} {mass * radius ** 2 / 2:.6g}"


def build(urdf: ET.Element, meshes: Path) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}

    chassis_lo, chassis_hi = mesh_bbox(meshes / "chassis.stl")
    chassis_size = chassis_hi - chassis_lo
    chassis_mid = (chassis_lo + chassis_hi) / 2

    tyre_lo, tyre_hi = mesh_bbox(meshes / "tire.stl")
    radius = float(tyre_hi[0])
    half_width = float(tyre_hi[2])

    wheel_z = float(joints["rear_left_tire"].find("origin").get("xyz").split()[2])
    # The description places the wheel centres 15 mm above the chassis origin on a 34 mm radius, so
    # as authored the car stands 19 mm below the floor. base_link is lifted by the difference; the
    # keyframe spawns a hair higher again so the tyres settle rather than start interpenetrating.
    rest_height = radius - wheel_z

    total_mass, _ = MEASURED["total_mass"]
    wheel_mass, _ = MEASURED["wheel_mass"]
    com_x, _ = MEASURED["com_x"]
    max_steer, _ = MEASURED["max_steer"]
    chassis_mass = total_mass - 4 * wheel_mass - 2 * KNUCKLE_MASS
    if chassis_mass <= 0:
        raise RuntimeError(
            f"the wheels and knuckles ({4 * wheel_mass + 2 * KNUCKLE_MASS} kg) weigh at least as "
            f"much as the whole car ({total_mass} kg) -- check MEASURED"
        )

    def visual(link_name: str, stem: str, offset=(0.0, 0.0, 0.0), quat: str = "") -> str:
        material, _ = MATERIALS[stem]
        pos = " ".join(f"{v:g}" for v in offset)
        return f'      <geom class="visual" mesh="{stem}" material="{material}" pos="{pos}"{quat}/>\n'

    # The fixed visual links, resolved to chassis-frame offsets. `rpi` hangs off `board` rather than
    # off the chassis, so its offset is the sum of two joints -- a chain the converter must walk
    # rather than assume, since a link parented to another link is invisible in a flat joint list.
    fixed = ""
    for name in FIXED_VISUALS:
        stem = links[name].find("visual/geometry/mesh").get("filename").split("/")[-1][:-4]
        offset = np.zeros(3)
        quat = ""
        node = name
        while node in joints:
            origin = joints[node].find("origin")
            offset = offset + np.array([float(v) for v in origin.get("xyz").split()])
            rpy = [float(v) for v in origin.get("rpy").split()]
            if any(rpy):
                quat = f' quat="{rpy_to_quat(*rpy)}"'
            node = joints[node].find("parent").get("link")
            if node == "chassis":
                break
        fixed += visual(name, stem, tuple(offset), quat)

    wheels = ""
    for body, steer_parent, joint_name in WHEELS:
        joint = joints[body]
        pos, quat = pose(joint)
        axis = joint.find("axis").get("xyz").replace(",", " ")
        indent = "        " if steer_parent else "      "
        disk = joints[f"{body.replace('_tire', '_disk')}"]
        _, disk_quat = pose(disk)
        wheel = (
            f'{indent}<body name="{body}" pos="{pos}"{quat}>\n'
            f'{indent}  <inertial pos="0 0 0" mass="{wheel_mass:g}"'
            f' diaginertia="{cylinder_inertia(wheel_mass, radius, 2 * half_width)}"/>\n'
            f'{indent}  <joint name="{joint_name}" class="wheel" axis="{axis}"/>\n'
            f'{indent}  <geom class="visual" mesh="tire" material="{MATERIALS["tire"][0]}"/>\n'
            f'{indent}  <geom class="visual" mesh="disk" material="{MATERIALS["disk"][0]}"'
            f'{disk_quat}/>\n'
            f'{indent}  <geom class="wheel_collision" name="{body}_geom"'
            f' size="{radius:g} {half_width:g}"/>\n'
            f'{indent}</body>\n'
        )
        if steer_parent:
            steer_joint = joints[steer_parent]
            steer_pos, steer_quat = pose(steer_joint)
            steer_axis = steer_joint.find("axis").get("xyz").replace(",", " ")
            wheels += (
                f'      <body name="{steer_parent}" pos="{steer_pos}"{steer_quat}>\n'
                f'        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>\n'
                f'        <joint name="{steer_joint.get("name")}" class="steer"'
                f' axis="{steer_axis}" range="-{max_steer:g} {max_steer:g}"/>\n'
                f'{wheel}'
                f'      </body>\n'
            )
        else:
            wheels += wheel

    camera_xyz, _ = pose(joints["camera"])
    assets = "".join(
        f'    <material name="{name}" rgba="{rgba}"/>\n'
        for name, rgba in sorted(MATERIALS.values())
    ) + "".join(
        f'    <mesh file="{stem}.stl" scale="0.001 0.001 0.001"/>\n'
        for stem in sorted({"chassis", "board", "rpi", "motor", "cover", "tire", "disk"})
    )
    wheel_pairs = "".join(
        f'    <pair geom1="{body}_geom" geom2="floor" condim="3" solref="0.01 1"'
        f' friction="1.2 1.2 0.005 0.0001 0.0001"/>\n'
        for body, *_ in WHEELS
    )
    return TEMPLATE.format(
        commit=PIRACER_COMMIT,
        assets=assets,
        chassis_mass=f"{chassis_mass:g}",
        chassis_com=f"{com_x:g} 0 {chassis_mid[2]:g}",
        chassis_inertia=box_inertia(chassis_mass, chassis_size),
        chassis_material=MATERIALS["chassis"][0],
        box_half=" ".join(f"{v / 2:g}" for v in chassis_size),
        box_pos=" ".join(f"{v:g}" for v in chassis_mid),
        fixed=fixed,
        wheels=wheels,
        camera_pos=camera_xyz,
        wheel_pairs=wheel_pairs,
        max_steer=f"{max_steer:g}",
        wheel_ctrl=f"{2.0 / radius:.0f}",
        rest_height=f"{rest_height:g}",
        spawn_height=f"{rest_height + 0.002:g}",
    )


TEMPLATE = """<mujoco model="piracer">
  <!--
    Waveshare PiRacer - a 0.236 m Ackermann-steered RC car, and the first car-like base here.

    GENERATED by external/convert/build_piracer_mjcf.py from lalywr2000/PiRacer-Gazebo-Simulation
    @ {commit} (MIT - see piracer_LICENSE). Do not hand-edit: re-run the generator.

    Drive: ACKERMANN, so it uses `ackermann_drive` and neither of the other two base plugins can
    stand in for it. The front wheels steer through independent joints and only the rear pair is
    driven; a command to turn on the spot moves nothing, which is the constraint the plugin exists
    to model. There is no slip_factor and none is wanted - a car turns by steering, not by scrubbing
    its wheels sideways as the skid-steer bases here do.

    Geometry, link offsets and collision primitives are the source's own. The MASSES ARE NOT: the
    description gives the chassis 1000 kg and each wheel 50 kg with a uniform 0.1 inertia tensor,
    about 1203 kg for a 24 cm car. Those are solver hacks rather than measurements, so they are
    replaced here and the tensors derived from the geometry.

    The car rests on four tyres; the chassis underside clears the floor by {rest_height} m and is
    not a bearing surface, unlike the skid-plate bases here.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="piracer">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="box" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <geom type="cylinder" group="3" rgba="0.1 0.1 0.1 0.4" friction="1.2 0.005 0.0001"/>
      </default>
      <default class="wheel">
        <joint damping="0.0005" armature="1e-5"/>
      </default>
      <default class="steer">
        <!-- Damped and given armature so the rack holds its angle when the car stops, which is what
             a real steering linkage does and what ackermann_drive's odometry reads back. -->
        <joint damping="0.01" armature="1e-4"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <worldbody>
    <body name="base_link" childclass="piracer">
      <freejoint name="base_free"/>
      <site name="base_imu" pos="0 0 0" size="0.005" rgba="0 0 0 0"/>
      <inertial pos="{chassis_com}" mass="{chassis_mass}" diaginertia="{chassis_inertia}"/>
      <geom class="visual" mesh="chassis" material="{chassis_material}"/>
{fixed}      <geom class="collision" name="chassis_geom" size="{box_half}" pos="{box_pos}"/>
      <site name="camera" pos="{camera_pos}" size="0.004" rgba="1 0 0 0.6"/>
{wheels}    </body>
  </worldbody>

  <contact>
    <!-- The front tyres sit inside the chassis footprint, as wheels in wheel arches do. MuJoCo
         excludes contacts only between a body and its DIRECT parent, so the rear pair is filtered
         automatically while the front pair - one body further down, through the steering knuckle -
         is not, and each front tyre otherwise grinds 4 mm into the chassis box for the whole run.
         The asymmetry is the collision filter's, not the car's. -->
    <exclude body1="base_link" body2="front_left_tire"/>
    <exclude body1="base_link" body2="front_right_tire"/>
    <!-- Tyre/floor friction stated explicitly: MuJoCo max-combines per-geom friction, so a
         high-friction scene floor would otherwise override the tyres and change how the car
         understeers. -->
{wheel_pairs}  </contact>

  <actuator>
    <!-- Steering: one POSITION servo per front wheel. Not one servo and a linkage, because
         ackermann_drive computes the inner and outer angles separately - on a curve the inner wheel
         follows the tighter radius, and a shared angle scrubs both tyres.

         kp is CALIBRATED against this model, and the quantity it has to get right is the SPLIT, not
         just the average: a servo too soft to hold its angle under tyre scrub lands both wheels
         short, and the inner/outer difference the plugin computes is then simply not what the car
         does. At kp=2 a commanded 0.750 m radius gave 0.199/0.162 rad against a geometric
         0.221/0.189; at 40 it gives 0.220/0.188, within 0.002 rad of both. Re-measure if the tyre
         friction, the steering damping or the car's mass change. -->
    <position name="left_steer_motor" joint="left_steer_joint" kp="40.0"
              ctrlrange="-{max_steer} {max_steer}" forcerange="-5 5"/>
    <position name="right_steer_motor" joint="right_steer_joint" kp="40.0"
              ctrlrange="-{max_steer} {max_steer}" forcerange="-5 5"/>
    <!-- Drive: VELOCITY servos on the REAR pair only. The front wheels spin free, as they do on the
         car: the upstream ros2_control block gives a command interface to the rear joints and the
         steering, and only state interfaces to the front wheels.

         kv is CALIBRATED: 0.05 reaches 93% of a commanded straight-line speed, 0.5 reaches 98%, and
         above that the servo overshoots. Under-tracking is left in rather than tuned away, because
         a driven tyre does slip. -->
    <velocity name="rear_left_wheel_motor" joint="rear_left_tire_joint" kv="0.5"
              ctrlrange="-{wheel_ctrl} {wheel_ctrl}" forcerange="-2 2"/>
    <velocity name="rear_right_wheel_motor" joint="rear_right_tire_joint" kv="0.5"
              ctrlrange="-{wheel_ctrl} {wheel_ctrl}" forcerange="-2 2"/>
  </actuator>

  <keyframe>
    <!-- base_link is the source's chassis origin, {rest_height} m above the ground once the tyres
         are on it; spawn a hair higher so they settle rather than start interpenetrating. -->
    <key name="home" qpos="0 0 {spawn_height} 1 0 0 0  0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="rebuild and diff against the committed MJCF; do not write")
    args = parser.parse_args()

    root = resolve_source("piracer", PIRACER_URL, PIRACER_COMMIT)
    pkg = root / "simulation_ws/src/sim"
    urdf = ET.parse(pkg / "description/piracer.xacro").getroot()
    xml = build(urdf, pkg / "meshes")

    target = PKG / "piracer.xml"
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current == xml:
            print(f"{target.relative_to(ROOT)}: up to date")
            return 0
        print(f"{target.relative_to(ROOT)}: DIFFERS from a fresh build", file=sys.stderr)
        return 1

    (PKG / "meshes").mkdir(parents=True, exist_ok=True)
    for stem in sorted(MATERIALS):
        shutil.copyfile(pkg / f"meshes/{stem}.stl", PKG / f"meshes/{stem}.stl")
    write_license(
        root / "LICENSE",
        PKG / "piracer_LICENSE",
        [
            "The PiRacer meshes and description in this folder come from",
            f"lalywr2000/PiRacer-Gazebo-Simulation @ {PIRACER_COMMIT},",
            "and are used under the MIT licence reproduced below.",
        ],
    )
    target.write_text(xml)
    print(f"wrote {target.relative_to(ROOT)}")
    for key, (value, provenance) in MEASURED.items():
        if provenance == "ASSUMED":
            print(f"  note: {key}={value} is ASSUMED, not measured -- see the port log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
