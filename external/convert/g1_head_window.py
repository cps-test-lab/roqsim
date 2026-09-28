#!/usr/bin/env python3
"""Cut the Unitree G1 head's opening for its Livox Mid-360, from the sensor's own field and housing.

Unitree's ``g1_description`` places ``mid360_link`` (the Mid-360's point-cloud frame) on
``torso_link`` by ``mid360_joint``, upside down, inside ``head_link``. The vendor head mesh is a
closed solid with no opening, so every ray a real Mid-360 would send out of the head starts inside
that solid. The real head passes the laser; this module removes from the head mesh the volume the
sensor occupies and the volume its field sweeps, so the model does the same without excluding any
robot geometry from the scan:

* the sensor's housing, from the Livox Mid-360 User Manual v1.2 (2024), Appendix "Livox Mid-360
  Dimensions" (p. 19): the 65.0 x 65.0 mm body up to the dome's 39.5 mm base and the dome, a sphere
  of radius 22.4 mm centred 37.6 mm above the bottom face (scaled from the same drawing), with the
  point-cloud origin O 47.0 mm above the bottom face -- grown by ``HOUSING_CLEARANCE`` on every side;
* the field, manual "Specifications" (p. 20): 360 deg horizontally, -7 deg .. +52 deg vertically in
  the sensor frame, widened by ``FIELD_MARGIN`` at both edges so a ray on the edge does not graze
  the cut face, out to ``FIELD_REACH`` from O (well past the head's surface).

Both are placed at the vendor's ``mid360_joint`` origin composed with ``head_joint``, read from the
URDF rather than restated, so the cut follows the mount it serves. What remains of the head is its
top above the field, which holds the housing, and a cone beneath O inside the field's 38 deg upper
gap, which no ray reaches. The mass and inertia of the head are not derived from its mesh anywhere
this cut is used: both models carry the vendor's explicit inertials.

The opening is a documented deviation: Unitree publishes no drawing of the head's window, so its
extent is the sensor's field and housing rather than the real part's outline. See the
``unitree_g1_dex1`` port log.

Used by ``build_g1_dex1.py`` on the unitree_ros head mesh. Run directly, it cuts the decimated head
mesh ``unitree_g1`` vendors (``models/meshes/head_link.STL``) into ``head_link_mid360_window.STL``:

    python external/convert/g1_head_window.py [--check]

Needs ``manifold3d`` (a pip wheel) for the boolean and ``trimesh`` for mesh IO.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from sources import resolve_source

UNITREE_ROS_COMMIT = "f3772ce54c56ef2d34c6aee8100bc768896c7d19"
UNITREE_ROS_URL = "https://github.com/unitreerobotics/unitree_ros"
#: The URDF the mount is read from. Every current ``*_rev_1_0`` and ``mode_*`` G1 description gives
#: the same ``mid360_joint`` and ``head_joint`` origins; this is the one ``unitree_g1``'s torso and
#: head meshes belong to.
MOUNT_URDF = "g1_29dof_rev_1_0.urdf"

# Livox Mid-360 User Manual v1.2, in the sensor frame O-XYZ (z out of the dome), metres.
BODY_HALF_WIDTH = 0.0325  # 65.0 mm footprint
BOTTOM_Z = -0.047  # O is 47.0 mm above the bottom face
DOME_BASE_Z = -0.0075  # 39.5 mm above the bottom face
DOME_RADIUS = 0.0224
DOME_CENTRE_Z = -0.0094
FIELD_MIN = math.radians(-7.0)
FIELD_MAX = math.radians(52.0)

#: Gap left between the housing and the head, on every side. An assumption: a cavity that met the
#: housing exactly would leave coincident faces, and Unitree publishes no fit.
HOUSING_CLEARANCE = 0.001
#: Widening of the field at both vertical edges, so an edge ray does not run along the cut face.
FIELD_MARGIN = math.radians(1.0)
#: How far from O the field is cut: past every point of the head (its mesh is 0.21 m tall).
FIELD_REACH = 0.25
#: Facets of the revolved field. At 720 a facet's chord lies 0.5 microradians inside the true cone,
#: far inside FIELD_MARGIN.
FIELD_SEGMENTS = 720


def _origin(urdf_root: ET.Element, joint: str) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz, rpy)`` of a fixed joint's origin; raises if the joint is absent."""
    el = urdf_root.find(f"joint[@name='{joint}']/origin")
    if el is None:
        raise RuntimeError(f"{joint} not in the URDF: the mount this cut is placed at is unknown")
    xyz = np.array([float(v) for v in el.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in el.get("rpy", "0 0 0").split()])
    return xyz, rpy


def _rot(rpy) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def sensor_in_head(urdf_text: str) -> tuple[np.ndarray, np.ndarray]:
    """``(position, rotation)`` of ``mid360_link`` in the ``head_link`` frame, from the URDF.

    Both joints hang from ``torso_link``; the head's own origin is ``head_joint``'s, so the sensor in
    the head frame is ``head_joint``^-1 composed with ``mid360_joint``.
    """
    root = ET.fromstring(urdf_text)
    for joint in ("head_joint", "mid360_joint"):
        parent = root.find(f"joint[@name='{joint}']/parent")
        if parent is None or parent.get("link") != "torso_link":
            raise RuntimeError(f"{joint} no longer hangs from torso_link; re-derive the head cut")
    head_xyz, head_rpy = _origin(root, "head_joint")
    lidar_xyz, lidar_rpy = _origin(root, "mid360_joint")
    head_r = _rot(head_rpy)
    return head_r.T @ (lidar_xyz - head_xyz), head_r.T @ _rot(lidar_rpy)


def _cut_volume(position: np.ndarray, rotation: np.ndarray):
    import manifold3d as mf

    lo, hi = FIELD_MIN - FIELD_MARGIN, FIELD_MAX + FIELD_MARGIN
    wedge = np.array(
        [
            [0.0, 0.0],
            [FIELD_REACH * math.cos(lo), FIELD_REACH * math.sin(lo)],
            [FIELD_REACH * math.cos(hi), FIELD_REACH * math.sin(hi)],
        ]
    )
    field = mf.Manifold.revolve(mf.CrossSection([wedge]), circular_segments=FIELD_SEGMENTS)
    c = HOUSING_CLEARANCE
    body = mf.Manifold.cube(
        [2 * (BODY_HALF_WIDTH + c), 2 * (BODY_HALF_WIDTH + c), DOME_BASE_Z - BOTTOM_Z + c]
    ).translate([-(BODY_HALF_WIDTH + c), -(BODY_HALF_WIDTH + c), BOTTOM_Z - c])
    dome = mf.Manifold.sphere(DOME_RADIUS + c, 96).translate([0.0, 0.0, DOME_CENTRE_Z])
    return (field + body + dome).transform(np.hstack([rotation, position[:, None]]))


def cut_head(head_mesh: Path, urdf_text: str):
    """The head mesh with the Mid-360's housing and field removed, as a ``trimesh.Trimesh``."""
    import manifold3d as mf
    import trimesh

    head = trimesh.load(str(head_mesh), force="mesh")
    if not head.is_watertight:
        raise RuntimeError(f"{head_mesh} is not a closed solid; a boolean cut of it is undefined")
    solid = mf.Manifold(
        mf.Mesh(
            vert_properties=np.asarray(head.vertices, dtype=np.float32),
            tri_verts=np.asarray(head.faces, dtype=np.uint32),
        )
    )
    if solid.status() != mf.Error.NoError:
        raise RuntimeError(f"{head_mesh} is not a valid manifold: {solid.status()}")
    out = (solid - _cut_volume(*sensor_in_head(urdf_text))).to_mesh()
    cut = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts, process=False)
    if not cut.is_watertight or cut.volume >= head.volume:
        raise RuntimeError(f"cutting {head_mesh} did not open it (volume {cut.volume:.3e} m^3)")
    return cut


def stl_bytes(mesh) -> bytes:
    return mesh.export(file_type="stl")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="rebuild and compare with the committed mesh"
    )
    args = ap.parse_args()

    meshes = (
        Path(__file__).resolve().parents[2] / "roqsim_humanoid/src/roqsim_humanoid/models/meshes"
    )
    source = meshes / "head_link.STL"
    target = meshes / "head_link_mid360_window.STL"
    description = resolve_source(
        "unitree_ros",
        UNITREE_ROS_URL,
        UNITREE_ROS_COMMIT,
        subdir="robots/g1_description",
        sparse="robots/g1_description",
    )
    data = stl_bytes(cut_head(source, (description / MOUNT_URDF).read_text()))
    if args.check:
        if not target.is_file() or target.read_bytes() != data:
            print(f"{target} differs from a fresh cut - was it edited by hand?", file=sys.stderr)
            return 1
        print(f"{target.name}: up to date with {UNITREE_ROS_COMMIT[:12]}")
        return 0
    target.write_bytes(data)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
