#!/usr/bin/env python3
"""Build the nine standalone scanner device models in ``roqsim_sensors/models/<device>/``.

    python external/convert/build_scanner_devices.py            # all nine
    python external/convert/build_scanner_devices.py sick_s300  # one

Writes, per device: ``meshes/*.obj`` (converted and, where the source is heavy, decimated), the MJCF
``<device>.xml`` and the licence sidecar ``<device>_LICENSE``. The ``<device>.manifest.yaml`` is
authored by hand, because its numbers come from manufacturer datasheets rather than from a source
tree. Thumbnails come from ``roqsim assets render-thumbnails``.

Each device's geometry is taken from the vendor ROS description a robot mounts it with, so a robot
port can hang the device at the vendor joint origin and get the vendor frame:

    device           mesh source (licence)                                 vendor scan frame
    sick_s300        neo_simulation2 components/meshes/SICK-S300.dae (MIT)  lidar_1_link
    sick_microscan3  neo_simulation2 components/meshes/SICK-MICROSCAN3.dae  lidar_1_link
    sick_tim571      pal_urdf_utils meshes/laser/sick_tim551.stl (Apache)   <name>_link
    rplidar_a1       turtlebot4 turtlebot4_description/meshes/rplidar.dae    rplidar_link
    rplidar_c1       husarion_components_description meshes/rplidar/c1.glb  laser (child of rplidar_link)
    lds01            turtlebot3 turtlebot3_description/meshes/sensors/lds.stl base_scan
    hokuyo_ust       clearpath_common clearpath_sensors_description/        <name>_laser (child of
                     meshes/hokuyo_ust.stl (BSD-3-Clause)                    <name>_link)
    sick_lms1xx      clearpath_common clearpath_sensors_description/        <name>_laser (child of
                     meshes/sick_lms1xx_small.dae (BSD-3-Clause)             <name>_link)
    velodyne_vlp16   velodyne_simulator velodyne_description/meshes/        ${name} (child of
                     VLP16_{base_1,base_2,scan}.stl (BSD-3-Clause)           ${name}_base_link)

The VLP-16 is a 16-plane lidar; its device casts the one horizontal plane a 2D consumer reads, the
planar projection the robot that mounts it documents.

The MJCF ``mount`` body is the link the vendor macro attaches to its parent. Visual geoms carry the
vendor visual origin; the collision geom is the vendor's primitive, except for the two Neobotix
devices whose vendor collision is the full mesh -- there it is the axis-aligned box around the
converted housing -- and the LMS1xx, whose vendor collision mesh it bounds. The ``scan`` site is the
vendor scan frame, never a datasheet optical offset.

Units and axes: every OBJ is written in metres (``--scale`` bakes the vendor mesh scale in), so the
MJCF carries no mesh scale. The Neobotix Collada files declare metres but hold millimetre
coordinates, which is why their vendor URDF scales by 0.001. Blender's glTF importer turns the C1's
Y-up into Z-up, which is the same rotation as the Husarion visual origin rpy (pi/2, 0, 0), so that
geom carries no rotation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neobotix import NEO_COMMIT, NEO_URL  # noqa: E402
from sources import resolve_source  # noqa: E402

from roqsim.pose import rpy_to_quat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "roqsim_sensors/src/roqsim_sensors/models"

PAL_URDF_UTILS_URL = "https://github.com/pal-robotics/pal_urdf_utils.git"
#: `humble-devel`, tag 2.9.2 -- the pin build_tiago_pro_mjcf.py already uses.
PAL_URDF_UTILS_COMMIT = "775cdd6886296e6c00f17dbdfd9bcdd20e0e6622"
TURTLEBOT4_URL = "https://github.com/turtlebot/turtlebot4.git"
#: `jazzy`, tag 2.1.1.
TURTLEBOT4_COMMIT = "7fd29fb420e906f3aca4a904adb54b69b11c7c00"
HUSARION_COMPONENTS_URL = "https://github.com/husarion/husarion_components_description.git"
#: `ros2`, the only branch.
HUSARION_COMPONENTS_COMMIT = "5f783f89961bb16098184f5381b1a76058cec19e"
TURTLEBOT3_URL = "https://github.com/ROBOTIS-GIT/turtlebot3.git"
#: `jazzy`.
TURTLEBOT3_COMMIT = "0c0be84e3f5c3194fb2adea8426a58a96060eab5"
CLEARPATH_COMMON_URL = "https://github.com/clearpathrobotics/clearpath_common.git"
#: `jazzy` -- the pin build_ridgeback_mjcf.py and build_warthog_mjcf.py already use.
CLEARPATH_COMMON_COMMIT = "b0f6d920422ad302372a1c65e31d61648da884ed"
VELODYNE_SIMULATOR_URL = "https://bitbucket.org/DataspeedInc/velodyne_simulator.git"
#: `humble-devel`, the ROS 2 branch.
VELODYNE_SIMULATOR_COMMIT = "03e3ce2e1a92991c31463f8935a98aa344f17da2"


@dataclass(frozen=True)
class Source:
    name: str  # directory under external/sources/
    url: str
    commit: str
    licence: str  # licence file in the source tree
    copyright: str
    spdx: str


NEO = Source("neo_simulation2", NEO_URL, NEO_COMMIT, "LICENSE", "2021 neobotix gmbh", "MIT")
PAL = Source(
    "pal_urdf_utils",
    PAL_URDF_UTILS_URL,
    PAL_URDF_UTILS_COMMIT,
    "LICENSE",
    "2025 PAL Robotics S.L.",
    "Apache-2.0",
)
TB4 = Source(
    "turtlebot4",
    TURTLEBOT4_URL,
    TURTLEBOT4_COMMIT,
    "LICENSE",
    "2021 Clearpath Robotics, Inc.",
    "Apache-2.0",
)
HUSARION = Source(
    "husarion_components_description",
    HUSARION_COMPONENTS_URL,
    HUSARION_COMPONENTS_COMMIT,
    "LICENSE.txt",
    "Husarion sp. z o.o.",
    "Apache-2.0",
)
TB3 = Source(
    "turtlebot3",
    TURTLEBOT3_URL,
    TURTLEBOT3_COMMIT,
    "LICENSE",
    "2019 ROBOTIS CO., LTD.",
    "Apache-2.0",
)
CLEARPATH = Source(
    "clearpath_common",
    CLEARPATH_COMMON_URL,
    CLEARPATH_COMMON_COMMIT,
    "LICENSE",
    "2023, clearpathrobotics",
    "BSD-3-Clause",
)
VELODYNE = Source(
    "velodyne_simulator",
    VELODYNE_SIMULATOR_URL,
    VELODYNE_SIMULATOR_COMMIT,
    "LICENSE",
    "2015-2021, Dataspeed Inc.",
    "BSD-3-Clause",
)


@dataclass(frozen=True)
class Device:
    name: str
    source: Source
    mesh: str  # path in the source tree
    scale: float  # vendor mesh scale, baked into the OBJ
    budget: int  # triangle budget per sub-mesh; at or above the source count it is not decimated
    visual_pos: tuple[float, float, float]
    visual_rpy: tuple[float, float, float]
    #: rgba for a source that carries no colour of its own (an STL); None reads the source's.
    rgba: tuple[float, float, float, float] | None
    #: ``<geom .../>`` attributes of the collision primitive; None boxes the converted housing.
    collision: str | None
    inertial: str
    site_pos: tuple[float, float, float]
    site_rpy: tuple[float, float, float]
    header: str  # MJCF header comment body
    collision_note: str
    site_note: str
    #: Further visual meshes of the same vendor link, placed like ``mesh``; same file type.
    extra_meshes: tuple[str, ...] = ()
    #: For several STL meshes, which carry no colour: one rgba per entry of ``meshes``.
    mesh_rgba: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def meshes(self) -> tuple[str, ...]:
        return (self.mesh, *self.extra_meshes)


DEVICES = {
    d.name: d
    for d in (
        Device(
            name="sick_s300",
            source=NEO,
            mesh="components/meshes/SICK-S300.dae",
            scale=0.001,
            budget=4000,  # 3963 in the source: shipped undecimated, it already is modest
            visual_pos=(0.0, 0.0, -0.12),
            visual_rpy=(-1.57, 0.0, 3.14),  # as the vendor writes it, not pi/2 and pi
            rgba=None,
            collision=None,
            # mpo_700_body.urdf.xacro lidar_1_link; 1.2 kg is also the datasheet weight.
            inertial='<inertial pos="0 0 0" mass="1.2" diaginertia="0.11042056 0.11042056 0.11042056"/>',
            site_pos=(0.0, 0.0, 0.0),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    SICK S300 safety laser scanner: a standalone mount (housing mesh + a `scan` site) for the
    `spawn_sensor` plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Neobotix `neo_simulation2`, whose MP-400 and MPO-700 carry this scanner: the body is
    `lidar_1_link` (mpo_700_body.urdf.xacro), the mesh keeps that link's visual origin
    xyz (0, 0, -0.12), rpy (-1.57, 0, 3.14), and the scan is stamped in the link itself.

    Body-local axes: x = the scan's zero bearing, z = up with the device upright. The optics cover is
    the dark round head at z -0.025 .. +0.032; the black wedge at the bottom (z -0.12 .. -0.047) is
    the system plug, at the rear. The datasheet puts the scan plane 116 mm above the housing bottom and
    36.4 mm below its top, which is z = -0.004 here, inside the cover's window; the site stays on the
    vendor frame.""",
            collision_note="The vendor collides with the full mesh; this box bounds the converted "
            "housing instead.",
            site_note="The vendor scan frame (lidar_1_link) is the mount itself.",
        ),
        Device(
            name="sick_microscan3",
            source=NEO,
            mesh="components/meshes/SICK-MICROSCAN3.dae",
            scale=0.001,
            budget=1500,  # 38652 in the source, 25873 of them in one dark housing part
            visual_pos=(0.0, 0.0, -0.06),
            visual_rpy=(1.57, 0.0, 0.0),
            rgba=None,
            collision=None,
            # mpo_500_body.urdf.xacro lidar_1_link. The datasheet weight is 1.15 kg.
            inertial='<inertial pos="0 0 0" mass="0.0001" diaginertia="0.0001 0.000001 0.0001"/>',
            site_pos=(0.0, 0.0, 0.0),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    SICK microScan3 Core safety laser scanner: a standalone mount (housing mesh + a `scan` site) for
    the `spawn_sensor` plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Neobotix `neo_simulation2`, whose MPO-500 carries two of them: the body is
    `lidar_1_link` (mpo_500_body.urdf.xacro), the mesh keeps that link's visual origin
    xyz (0, 0, -0.06), rpy (1.57, 0, 0), and the scan is stamped in the link itself.

    Body-local axes: x = the scan's zero bearing, z = up. The mesh is 151.8 mm tall against the
    operating instructions' 135.1 mm housing, so the datasheet scan plane (40.1 mm below the top,
    95.0 mm above the bottom) maps to z = -0.0037 or z = -0.0204 depending on which end is taken
    as reference; the site stays on the vendor frame.""",
            collision_note="The vendor collides with the full mesh; this box bounds the converted "
            "housing instead.",
            site_note="The vendor scan frame (lidar_1_link) is the mount itself.",
        ),
        Device(
            name="sick_tim571",
            source=PAL,
            mesh="meshes/laser/sick_tim551.stl",
            scale=1.0,
            budget=2000,  # 1989 in the source
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),
            rgba=(0.1, 0.1, 0.1, 1.0),  # pal_urdf_utils materials.urdf.xacro `DarkGrey`
            collision='type="cylinder" pos="0 0 0" size="0.01 0.005"',
            inertial='<inertial pos="-0.02559 -0.00056 -0.05732" mass="0.28922" '
            'fullinertia="0.00002628919 0.00003374542 0.00005832599 0.00000024298 -0.00000368129 '
            '-0.0000000133"/>',
            site_pos=(0.0, 0.0, 0.0),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    SICK TiM571 2D lidar: a standalone mount (housing mesh + a `scan` site) for the `spawn_sensor`
    plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from PAL Robotics `pal_urdf_utils` (sick_tim571_laser.urdf.xacro), the macro TIAGo Pro's
    base lasers use: the body is `<name>_link` and the scan is stamped in it. The vendor macro draws
    the TiM571 with the TiM551 mesh -- the same TiM5xx housing -- and no TiM571 mesh exists upstream.
    Inertial as the vendor writes it (0.289 kg; the datasheet weight is 250 g).

    Body-local axes: x = the scan's zero bearing, z = up. The dimensional drawing puts the light
    emission level 23.3 mm below the housing top, which is where the mesh places the vendor frame.""",
            collision_note="The vendor's own collision: a 1 cm cylinder at the link origin, not the "
            "housing.",
            site_note="The vendor scan frame (<name>_link) is the mount itself.",
        ),
        Device(
            name="rplidar_a1",
            source=TB4,
            mesh="turtlebot4_description/meshes/rplidar.dae",
            scale=1.0,
            budget=2000,  # 5282 in the source
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),
            rgba=None,
            collision='type="box" pos="0 0.013 -0.019" size="0.0355 0.05 0.03"',
            # rplidar.urdf.xacro: 0.17 kg, `inertial_cuboid` over the 7.1 x 10 x 6 cm collision box.
            inertial='<inertial pos="0 0 0" mass="0.17" '
            'diaginertia="0.00019267 0.00012241 0.00021308"/>',
            site_pos=(0.0, 0.0, 0.0),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    Slamtec RPLIDAR A1 (A1M8) 2D lidar: a standalone mount (housing mesh + a `scan` site) for the
    `spawn_sensor` plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Clearpath's `turtlebot4_description` (urdf/sensors/rplidar.urdf.xacro), the scanner
    on the TurtleBot 4: the body is `rplidar_link`, the mesh has no visual origin, and the scan is
    stamped in the link itself.

    Body-local axes: x = the scan's zero bearing, z = up; the mesh spans z -46.1 .. +8.9 mm, so the
    vendor frame sits 8.9 mm below the top of the rotor housing.""",
            collision_note="The vendor's own collision box (7.1 x 10 x 6 cm).",
            site_note="The vendor scan frame (rplidar_link) is the mount itself.",
        ),
        Device(
            name="rplidar_c1",
            source=HUSARION,
            mesh="meshes/rplidar/c1.glb",
            scale=1.0,
            budget=2000,  # 316 in the source
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),  # the vendor's (pi/2, 0, 0) is applied by the glTF import
            rgba=None,
            collision='type="box" pos="0 0 0.02065" size="0.0278 0.0278 0.02065"',
            inertial='<inertial pos="0 0 0.02065" mass="0.11" diaginertia="0.000044 0.000044 0.0000567"/>',
            site_pos=(0.0, 0.0, 0.032),
            site_rpy=(0.0, 0.0, 3.141592653589793),
            header="""\
    Slamtec RPLIDAR C1 2D lidar: a standalone mount (housing mesh + a `scan` site) for the
    `spawn_sensor` plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Husarion `husarion_components_description` (urdf/slamtec_rplidar.urdf.xacro, model
    `c1`), the scanner on the ROSbot: the body is `rplidar_link`, whose origin is the housing base, and
    the scan is stamped in its child `laser` at xyz (0, 0, 0.032), rpy (0, 0, pi).

    Body-local axes: z = up, the housing spans z 0 .. 41.3 mm. The datasheet gives the laser
    emitting and receiving height as 29.8 mm, 2.2 mm below the vendor `laser` frame; the site
    stays on the vendor frame.""",
            collision_note="The vendor's own collision box over the 55.6 x 55.6 x 41.3 mm housing.",
            site_note="The vendor scan frame `laser`: 32 mm up and turned half a revolution.",
        ),
        Device(
            name="lds01",
            source=TB3,
            mesh="turtlebot3_description/meshes/sensors/lds.stl",
            scale=0.001,
            budget=3000,  # 14566 in the source
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),
            rgba=(0.3, 0.3, 0.3, 1.0),  # turtlebot3_description common_properties.urdf `dark`
            collision='type="cylinder" pos="0.015 0 -0.0065" size="0.055 0.01575"',
            inertial='<inertial pos="0 0 0" mass="0.114" diaginertia="0.001 0.001 0.001"/>',
            site_pos=(0.0, 0.0, 0.0),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    ROBOTIS LDS-01 (HLS-LFCD2) 2D lidar: a standalone mount (housing mesh + a `scan` site) for the
    `spawn_sensor` plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from ROBOTIS `turtlebot3_description` (urdf/turtlebot3_waffle.urdf), the scanner on the
    TurtleBot 3: the body is `base_scan`, the mesh has no visual origin, and the scan is stamped in
    the link itself. Inertial as the vendor writes it (0.114 kg; the datasheet gives under 125 g).

    Body-local axes: x = the scan's zero bearing, z = up; the mesh spans z -30.2 .. +9.0 mm.""",
            collision_note="The vendor's own collision cylinder (r 55 mm, 31.5 mm long), offset "
            "15 mm forward.",
            site_note="The vendor scan frame (base_scan) is the mount itself.",
        ),
        Device(
            name="hokuyo_ust",
            source=CLEARPATH,
            mesh="clearpath_sensors_description/meshes/hokuyo_ust.stl",
            scale=1.0,
            budget=2200,  # 2116 in the source
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),
            rgba=(0.2, 0.2, 0.2, 1.0),  # clearpath_platform_description common.urdf.xacro `clearpath_dark_grey`
            collision='type="box" pos="0 0 0.04" size="0.03 0.03 0.04"',
            # Hokuyo UST-10LX specification C-42-04077: weight 130 g, 50 x 50 x 70 mm; a solid box over
            # that body. Clearpath's macro puts 1.1 kg on `<name>_laser`, an LMS1xx figure.
            inertial='<inertial pos="0 0 0.035" mass="0.13" '
            'diaginertia="0.000080167 0.000080167 0.000054167"/>',
            site_pos=(0.0, 0.0, 0.0474),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    Hokuyo UST-10LX 2D lidar: a standalone mount (housing mesh + a `scan` site) for the `spawn_sensor`
    plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Clearpath's `clearpath_sensors_description` (urdf/hokuyo_ust.urdf.xacro), the
    `hokuyo_ust` accessory of Clearpath's ROS 2 robots: the body is `<name>_link`, whose origin is the
    base of the bracket, and the scan is stamped in its child `<name>_laser`, 47.4 mm up "to the
    LIDAR's focal point".

    Body-local axes: x = the scan's zero bearing, z = up; the mesh spans z 0 .. 70 mm, the
    specification's 70 mm sensor height.""",
            collision_note="The vendor's own collision box (6 x 6 x 8 cm) over the bracket and sensor.",
            site_note="The vendor scan frame `<name>_laser`: 47.4 mm above the bracket base.",
        ),
        Device(
            name="sick_lms1xx",
            source=CLEARPATH,
            mesh="clearpath_sensors_description/meshes/sick_lms1xx_small.dae",
            scale=1.0,
            budget=2000,
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),
            rgba=None,
            # The vendor collides with sick_lms1xx_collision.stl; this box bounds that mesh (x -0.0001
            # .. 0.1056, y +-0.0511, z -0.0891 .. 0.0727), 105.6 x 102.2 x 161.8 mm against the data
            # sheet's 105 x 102 x 162 mm.
            collision='type="box" pos="0.05275 0 -0.0082" size="0.05285 0.0511 0.0809"',
            # SICK data sheet LMS111-10100: weight 1.1 kg, a solid box over the collision bounds. The
            # vendor link carries no inertial.
            inertial='<inertial pos="0.05275 0 -0.0082" mass="1.1" '
            'diaginertia="0.0033572 0.0034239 0.0019816"/>',
            site_pos=(0.0549, 0.0, 0.0367),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    SICK LMS111 2D lidar: a standalone mount (housing mesh + a `scan` site) for the `spawn_sensor`
    plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Clearpath's `clearpath_sensors_description` (urdf/sick_lms1xx.urdf.xacro), the
    `sick_lms1xx` accessory of Clearpath's ROS 2 robots: the body is `<name>_link`, whose origin is
    the rear bottom of the mounting face, and the scan is stamped in its child `<name>_laser` at
    xyz (0.0549, 0, 0.0367).

    Body-local axes: x = the scan's zero bearing, z = up. The operating instructions put the mirror
    axis 55 mm from the rear (the vendor frame's 54.9 mm). The data sheet drawing puts the scan plane
    116 mm above the bottom of the 152 mm housing, which stands on 11 mm of connectors, so 36 mm below
    its top: z = 0.0368 on the collision mesh, the vendor frame within 0.1 mm.""",
            collision_note="The vendor collides with a mesh; this box bounds that collision mesh.",
            site_note="The vendor scan frame `<name>_laser` at xyz (0.0549, 0, 0.0367).",
        ),
        Device(
            name="velodyne_vlp16",
            source=VELODYNE,
            # The macro's three visuals: two on `${name}_base_link`, and `VLP16_scan` on the scan link
            # at xyz (0, 0, -0.0377), which puts it at the base link's origin too. The macro names the
            # .dae files, but those are exports of these STLs that no Collada parser reads (Blender
            # wrote the node id `<STL_BINARY>` unescaped into an attribute), so the vendor STLs are
            # converted and the DAEs' one diffuse colour each is carried here.
            mesh="velodyne_description/meshes/VLP16_base_1.stl",
            extra_meshes=(
                "velodyne_description/meshes/VLP16_base_2.stl",
                "velodyne_description/meshes/VLP16_scan.stl",
            ),
            mesh_rgba=((0.55, 0.55, 0.55, 1.0), (0.55, 0.55, 0.55, 1.0), (0.1, 0.1, 0.1, 1.0)),
            scale=1.0,
            budget=2000,  # 1912, 1328 and 104 in the source
            visual_pos=(0.0, 0.0, 0.0),
            visual_rpy=(0.0, 0.0, 0.0),
            rgba=None,
            collision='type="cylinder" pos="0 0 0.03585" size="0.0516 0.03585"',
            # VLP-16.urdf.xacro `${name}_base_link`: 0.83 kg (the data sheet's ~830 g), a solid
            # cylinder r 0.0516, 0.0717 long, centred 0.03585 up.
            inertial='<inertial pos="0 0 0.03585" mass="0.83" '
            'diaginertia="0.000908059 0.000908059 0.001104962"/>',
            site_pos=(0.0, 0.0, 0.0377),
            site_rpy=(0.0, 0.0, 0.0),
            header="""\
    Velodyne VLP-16 (Puck) lidar: a standalone mount (housing mesh + a `scan` site) for the
    `spawn_sensor` plugin, mounted by a robot manifest at its vendor joint origin.

    Geometry from Dataspeed's `velodyne_description` (urdf/VLP-16.urdf.xacro), the macro that
    jackal_description's `vlp16_mount` calls: the body is `${name}_base_link`, whose origin is the
    housing base, and the scan is stamped in its child `${name}` (default `velodyne`), 37.7 mm up.

    Body-local axes: x = the scan's zero bearing, z = up. The data sheet's dimension drawing puts
    the optical centre 37.8 mm above the base, 0.1 mm above the vendor frame; the site stays on the
    vendor frame. The device's manifest casts one horizontal plane of the 16.""",
            collision_note="The vendor's own collision cylinder (r 51.6 mm, 71.7 mm long).",
            site_note="The vendor scan frame `${name}`: 37.7 mm above the housing base.",
        ),
    )
}


def _fmt(values) -> str:
    # Rounded first, so cos(pi/2) prints as 0 rather than 6.1e-17.
    return " ".join(f"{round(float(v), 9) + 0.0:.10g}" for v in values)


def _reduce(src: Path, dst: Path, faces: int, scale: float, *extra: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "roqsim.commands",
            "assets",
            "reduce-mesh",
            "--target-faces",
            str(faces),
            "--scale",
            str(scale),
            *extra,
            str(src),
            str(dst),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def convert(device: Device, source: Path, meshes: Path) -> dict[str, tuple[float, ...]]:
    """Write the device's OBJs into *meshes*; return ``{mesh stem: rgba}``."""
    sources = [source / mesh for mesh in device.meshes]
    src = sources[0]
    kind = src.suffix.lower()
    if any(each.suffix.lower() != kind for each in sources):
        raise RuntimeError(f"{device.name}: every mesh must be a {kind} like {src.name}")
    if kind == ".glb" and len(sources) > 1:
        raise RuntimeError(f"{device.name}: several meshes are converted only from Collada or STL")
    parts: dict[str, tuple[float, ...]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if kind == ".dae":
            staged = tmp / "dae"
            staged.mkdir()
            for each in sources:
                shutil.copy2(each, staged / each.name)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "dae2obj.py"),
                    str(staged),
                    str(tmp / "obj"),
                ],
                check=True,
                capture_output=True,
            )
            materials = json.loads((tmp / "obj/materials.json").read_text())
            for each in sources:
                for sub, rgb in materials[each.stem]:
                    _reduce(
                        tmp / "obj" / f"{sub}.obj",
                        meshes / f"{sub}.obj",
                        device.budget,
                        device.scale,
                        "--no-materials",
                    )
                    parts[sub] = device.rgba or (*rgb[:3], 1.0)
        elif kind == ".glb":
            _reduce(
                src, meshes / f"{src.stem}.obj", device.budget, device.scale, "--split-materials"
            )
            sidecar = meshes / f"{src.stem}.materials.json"
            for material, rgba in json.loads(sidecar.read_text()).items():
                parts[f"{src.stem}__{material}"] = device.rgba or tuple(rgba)
            sidecar.unlink()  # colours live in the MJCF; MuJoCo reads no OBJ material
            for mtl in meshes.glob("*.mtl"):
                mtl.unlink()
        elif kind == ".stl":
            if device.mesh_rgba:
                if len(device.mesh_rgba) != len(sources):
                    raise RuntimeError(f"{device.name}: `mesh_rgba` needs one rgba per mesh")
                colours = device.mesh_rgba
            elif device.rgba is None:
                raise RuntimeError(f"{device.name}: an STL carries no colour; set `rgba`")
            else:
                colours = (device.rgba,) * len(sources)
            for each, rgba in zip(sources, colours, strict=True):
                _reduce(
                    each, meshes / f"{each.stem}.obj", device.budget, device.scale, "--no-materials"
                )
                parts[each.stem] = rgba
        else:
            raise RuntimeError(f"{device.name}: no converter for {kind} ({src})")
    for stem in parts:
        if not (meshes / f"{stem}.obj").is_file():
            raise RuntimeError(f"{device.name}: conversion did not write {stem}.obj")
    return parts


def _vertices(path: Path) -> np.ndarray:
    return np.array(
        [[float(x) for x in line.split()[1:4]] for line in path.open() if line.startswith("v ")]
    )


def housing_box(device: Device, meshes: Path, parts) -> str:
    """Box geom attributes bounding every visual mesh, in the mount frame."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.array(rpy_to_quat(*device.visual_rpy)))
    verts = np.vstack([_vertices(meshes / f"{stem}.obj") for stem in parts])
    placed = verts @ rot.reshape(3, 3).T + np.array(device.visual_pos)
    lo = np.floor(placed.min(axis=0) * 1e4) / 1e4
    hi = np.ceil(placed.max(axis=0) * 1e4) / 1e4
    return f'type="box" pos="{_fmt((lo + hi) / 2)}" size="{_fmt((hi - lo) / 2)}"'


def mjcf(device: Device, parts: dict, collision: str) -> str:
    materials = "".join(
        f'    <material name="{stem}_mat" rgba="{_fmt(rgba)}"/>\n' for stem, rgba in parts.items()
    )
    meshes = "".join(f'    <mesh name="{stem}" file="{stem}.obj"/>\n' for stem in parts)
    placement = ""
    if any(device.visual_pos):
        placement += f' pos="{_fmt(device.visual_pos)}"'
    if any(device.visual_rpy):
        placement += f' quat="{_fmt(rpy_to_quat(*device.visual_rpy))}"'
    many = len(parts) > 1
    visuals = "".join(
        f'      <geom name="{device.name}_visual{f"_{i}" if many else ""}" type="mesh" '
        f'mesh="{stem}" material="{stem}_mat"{placement}\n'
        f'            contype="0" conaffinity="0"/>\n'
        for i, stem in enumerate(parts)
    )
    site = f'pos="{_fmt(device.site_pos)}"'
    if any(device.site_rpy):
        site += f' quat="{_fmt(rpy_to_quat(*device.site_rpy))}"'
    return f"""<mujoco model="{device.name}">
  <!--
{device.header}

    Built by external/convert/build_scanner_devices.py from {device.source.name} @ {device.source.commit}.
    Scan parameters and the `frames:` entry for this site are in {device.name}.manifest.yaml; see
    {device.name}_LICENSE for the mesh licence.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <asset>
{materials}{meshes}  </asset>

  <worldbody>
    <body name="mount">
      {device.inertial}
{visuals}      <!-- {device.collision_note} -->
      <geom name="{device.name}_collision" {collision} group="3"/>
      <!-- {device.site_note}
           Keep in step with the manifest's `frames:` entry. -->
      <site name="scan" {site} size="0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


def licence(device: Device, source: Path) -> str:
    files = "\n".join(f"    file   {mesh}" for mesh in device.meshes)
    return f"""The visual meshes in meshes/ are converted from

    {device.source.url.removesuffix(".git")}
    commit {device.source.commit}
{files}

Copyright (c) {device.source.copyright}
Licence: {device.source.spdx}, full text below.
Converted (and, where the source is heavy, decimated) to OBJ in metres by
external/convert/build_scanner_devices.py; the MJCF's link frame, visual origin, collision
primitive and inertial are read from the same repository.

--------------------------------------------------------------------------------

{(source / device.source.licence).read_text().strip()}
"""


def build(device: Device) -> None:
    source = resolve_source(device.source.name, device.source.url, device.source.commit)
    folder = MODELS / device.name
    meshes = folder / "meshes"
    if meshes.exists():
        shutil.rmtree(meshes)
    meshes.mkdir(parents=True)
    parts = convert(device, source, meshes)
    collision = device.collision or housing_box(device, meshes, parts)
    (folder / f"{device.name}.xml").write_text(mjcf(device, parts, collision))
    (folder / f"{device.name}_LICENSE").write_text(licence(device, source))
    faces = sum(
        sum(1 for line in (meshes / f"{s}.obj").open() if line.startswith("f ")) for s in parts
    )
    print(
        f"{device.name}: {len(parts)} mesh(es), {faces} triangles, from "
        f"{device.source.name}@{device.source.commit[:12]}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("devices", nargs="*", help=f"any of {', '.join(DEVICES)}; default: all")
    args = parser.parse_args(argv)
    if unknown := sorted(set(args.devices) - set(DEVICES)):
        parser.error(f"unknown device(s): {', '.join(unknown)}")
    for name in args.devices or DEVICES:
        build(DEVICES[name])


if __name__ == "__main__":
    main()
