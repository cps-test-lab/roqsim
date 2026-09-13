#!/usr/bin/env python3
"""Build roqsim's Maker's Pet MJCF models from the `makerspet/makerspet_*` descriptions.

Parameterised by robot, because the vendor ships one design at four sizes (Mini 170 mm, Loki 200 mm,
Fido 250 mm, Snoopy 300 mm) and they differ in dimensions rather than in kind. Only the ones in
:data:`ROBOTS` are built; adding a sibling is a line there plus its pinned commit -- which is how
this generator was repointed from Loki to Mini without touching anything below it.

Unlike ``build_oomwoo_one_mjcf.py``, which walks a flat list of links hanging off ``base_link``,
these have a **nested** tree -- the Mini hangs its lidar motor off the scanner puck, and the Loki
stacks a head and tablet above two decks -- so the body emitter here is recursive. That is the only structural
difference; both vendors share the same self-contained, primitive-heavy, ``$(find)``-free idiom, and
both use :func:`urdf_source.link_primitives`.

**The joint rpy is load-bearing and is why this generator reads the joint's full frame.** The wheel
joints carry ``rpy="-pi/2 0 0"``, the scanner ``rpy="0 -pi 0"`` (an inverted puck between the decks)
and the tablet a 20-degree pitch. Reading only the xyz -- which four earlier generators got away with,
because Clearpath, Husarion and RT all put their rotations on the *visual* -- leaves the wheels as
flat discs clear of the floor and the robot resting on its body. See the OOMWOO port log.

Usage::

    python external/convert/build_makerspet_mjcf.py                   # build all in ROBOTS
    python external/convert/build_makerspet_mjcf.py --check           # rebuild and diff
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import (  # noqa: E402
    expand_xacro,
    inertial,
    link_primitives,
    mesh_scales,
    pose,
    write_license,
)

#: model short name -> (repo, pinned commit on the jazzy branch, human name, body diameter mm)
ROBOTS = {
    "makerspet_mini": (
        "https://github.com/makerspet/makerspet_mini.git",
        "77d196b6749f577cc181c6263eaecef6ef808a6c",
        "Maker's Pet Mini",
        170,
    ),
}

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "roqsim_mobile/src/roqsim_mobile/models"

#: Links whose collision geometry needs a class of its own. A caster is a SWIVELLING wheel, so it
#: offers almost no lateral resistance -- but a plain low `friction` does nothing, because MuJoCo
#: takes the *maximum* of the two contacting geoms' friction unless one of them sets `priority`.
#: Without the priority the floor's 1.0 wins and the caster scrubs, costing ~35% of commanded yaw.
COLLISION_CLASS = {"caster_link": "caster_collision"}

#: The description's own palette (urdf materials), namespaced so it cannot collide with another
#: model's in a shared MJCF asset namespace -- "white" and "dark" are not names to claim globally.
PALETTE = {
    "black": "0.0 0.0 0.0 1",
    "blue": "0.0 0.0 0.8 1",
    "dark": "0.3 0.3 0.3 1",
    "grey": "0.5 0.5 0.5 1",
    "red": "0.8 0.0 0.0 1",
    "white": "1.0 1.0 1.0 1",
}

#: The scan plane's height above the floor (m), from the manufacturer's CAD rather than the description.
#: makerspet/store @ e338516e4e4c19bec75d947f9c32589d32a29387,
#: ``MINI-BDC30P-BODY/v1.0.1/makerspet_mini-bdc30p_v1_0_1.step``, whose z = 0 is the floor: the LD14P
#: stands upright on its skirt (lower shell ``XIAKE_360`` 58.3-69.0 mm) and its rotating optical block
#: (``GJZJ_ASM``) spans 68.0-76.5 mm, so the plane is taken at that block's centre, 72.25 mm. The
#: description's ``scan_joint`` puts it at 85.3 mm.
CAD_SCAN_HEIGHT = 0.07225

#: ``(vendor mesh stem, shipped stem)`` of the head's visual. It ships cut at the scanner gap (see
#: :func:`open_scan_gap`) under its own name, so the file is not mistaken for the vendor's.
HEAD_MESH = ("hemisphere", "hemisphere_scan_gap")


def place_scan_at_cad_height(urdf: ET.Element) -> float:
    """Move ``scan_joint`` to :data:`CAD_SCAN_HEIGHT` above the floor, in *urdf* in place; its z in base_link.

    base_link stands ``wheel radius - wheel joint z`` above the floor (the description's
    ``floor_clearance``). The joint's x, y and rotation stay the description's: the rotation is the frame
    the scan is stamped in, and moving it would mirror every bearing a consumer reads.
    """
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    radius = float(links["wheel_left_link"].find("collision/geometry/cylinder").get("radius"))
    wheel_z = float(joints["wheel_left_link"].find("origin").get("xyz").split()[2])
    origin = joints["base_scan"].find("origin")
    x, y, _ = origin.get("xyz").split()
    z = CAD_SCAN_HEIGHT - (radius - wheel_z)
    origin.set("xyz", f"{x} {y} {z:g}")
    return z


def open_scan_gap(urdf: ET.Element) -> tuple[float, float]:
    """Open the scanner's gap in the head, in *urdf* in place: ``(gap bottom in base_link, in head_link)``.

    The head (``head_link``, 0.032-0.0708 m) encloses the scan plane, at the description's 0.0704 m and
    at the CAD height :func:`place_scan_at_cad_height` sets alike. The real Mini carries its LiDAR in an open
    gap: the makerspet/store ``MINI-BDC30P-BODY`` print files stand it on four 38.8 mm posts
    (``Lidar_Post_LD14P_x4``) under a ring skirt (``Lidar_Skirt_LD14P``), with no body material beside
    it. The head therefore ends at the bottom face of the vendor's own scanner puck -- ``scan_joint``
    height less half ``laser_puck_height``, both from params.xacro -- and nothing of it is left beside
    the puck: the collision cylinder ends there, and :func:`clip_below` cuts the hemisphere visual at
    the same height. The link's mass and inertia are its explicit inertial and do not change.
    """
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    scan_z = float(joints["base_scan"].find("origin").get("xyz").split()[2])
    puck = links["base_scan"].find("collision/geometry/cylinder")
    head_z = float(joints["head_link"].find("origin").get("xyz").split()[2])
    gap = scan_z - float(puck.get("length")) / 2
    cut = gap - head_z
    collision = links["head_link"].find("collision")
    cylinder, origin = collision.find("geometry/cylinder"), collision.find("origin")
    x, y, z = (float(v) for v in origin.get("xyz").split())
    half = float(cylinder.get("length")) / 2
    if not z - half < cut < z + half:
        raise ValueError(
            f"head_link's collision spans {z - half:g}-{z + half:g} m in its frame, which the gap "
            f"bottom {cut:g} m does not cut -- the description changed under the pin"
        )
    cylinder.set("length", f"{cut - (z - half):g}")
    origin.set("xyz", f"{x:g} {y:g} {(z - half + cut) / 2:g}")
    return gap, cut


#: A binary STL record: normal, three vertices, attribute bytes.
_STL = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def clip_below(stl: bytes, z_cut: float) -> bytes:
    """A binary STL cut at ``z = z_cut`` (mesh units): the part below, capped with a flat top.

    Triangles crossing the plane are split, so no face reaches above it. The cap is a fan over the
    cut's outline, which is convex for the head's surface of revolution.
    """
    count = int.from_bytes(stl[80:84], "little")
    if len(stl) != 84 + 50 * count:
        raise ValueError("clip_below: not a binary STL")
    tris = np.frombuffer(stl, dtype=_STL, count=count, offset=84)["v"].astype(np.float64)
    kept, ring = [], []
    for tri in tris:
        below = tri[:, 2] < z_cut
        n = int(below.sum())
        if n == 3:
            kept.append(tri)
            continue
        if n == 0:
            continue
        # Rotate the lone vertex to the front, which keeps the winding.
        odd = int(np.flatnonzero(below if n == 1 else ~below)[0])
        a, b, c = tri[odd], tri[(odd + 1) % 3], tri[(odd + 2) % 3]
        ab = a + (b - a) * ((z_cut - a[2]) / (b[2] - a[2]))
        ac = a + (c - a) * ((z_cut - a[2]) / (c[2] - a[2]))
        kept += [np.array([a, ab, ac])] if n == 1 else [np.array([ab, b, c]), np.array([ab, c, ac])]
        ring += [ab, ac]
    if not ring:
        raise ValueError(f"clip_below: nothing of the mesh crosses z = {z_cut:g}")
    ring = np.unique(np.round(np.array(ring), 6), axis=0)
    centre = ring.mean(axis=0)
    ring = ring[np.argsort(np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0]))]
    kept += [np.array([centre, ring[i], ring[(i + 1) % len(ring)]]) for i in range(len(ring))]
    out = np.array(kept)
    normal = np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    rec = np.zeros(len(out), dtype=_STL)
    rec["n"] = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)
    rec["v"] = out
    header = f"cut at z {z_cut:.4f} by build_makerspet_mjcf.py".encode().ljust(80)
    return header + len(out).to_bytes(4, "little") + rec.tobytes()


def shipped_meshes(source: Path) -> dict[str, str]:
    """``{vendor mesh stem: shipped stem}`` for every STL of the description."""
    stems = {p.stem for p in source.rglob("*.stl") if ".git" not in p.parts}
    if HEAD_MESH[0] not in stems:
        raise ValueError(f"{HEAD_MESH[0]}.stl is not in {source}")
    return {s: HEAD_MESH[1] if s == HEAD_MESH[0] else s for s in stems}


def vendor_mesh(source: Path, stem: str) -> Path:
    (path,) = [p for p in source.rglob(f"{stem}.stl") if ".git" not in p.parts]
    return path



def build(urdf: ET.Element, model: str, commit: str, human: str, meshes: dict[str, str],
          gap: float) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = list(urdf.findall("joint"))
    materials = {name: f"{model}_{name}" for name in PALETTE}

    def children_of(parent: str) -> list[ET.Element]:
        return [j for j in joints if j.find("parent").get("link") == parent]

    def emit(joint: ET.Element, depth: int) -> str:
        """One link as an MJCF body, recursively. Carries the joint's FULL frame, rpy included."""
        name = joint.find("child").get("link")
        link = links[name]
        indent = "  " * (depth + 4)
        pos, quat = pose(joint)
        out = f'{indent}<body name="{name}" pos="{pos}"{quat}>\n'
        if link.find("inertial") is not None:
            attrs = inertial(link)
            out += (f'{indent}  <inertial pos="{attrs["pos"]}" mass="{attrs["mass"]}"'
                    f' diaginertia="{attrs["diaginertia"]}"/>\n')
        if joint.get("type") == "continuous":
            out += f'{indent}  <joint name="{joint.get("name")}" class="wheel"/>\n'
        out += link_primitives(link, "visual", "visual", indent + "  ", materials,
                               mesh_stems=meshes)
        out += link_primitives(link, "collision", COLLISION_CLASS.get(name, "collision"),
                               indent + "  ", mesh_stems=meshes)
        for child in children_of(name):
            out += emit(child, depth + 1)
        return out + f"{indent}</body>\n"

    base = links["base_link"]
    base_attrs = inertial(base)
    # base_link is the ROOT body, so its own geoms are not reached by emit(), which walks joints.
    # Leaving them out silently dropped the robot's main body cylinder -- visual AND collision --
    # and no dynamics check noticed, because the wheels and caster still carried it.
    base_geoms = (link_primitives(base, "visual", "visual", "        ", materials,
                                  mesh_stems=meshes)
                  + link_primitives(base, "collision",
                                    COLLISION_CLASS.get("base_link", "collision"),
                                    "        ", mesh_stems=meshes))
    bodies = "".join(emit(j, 1) for j in children_of("base_link"))
    wheel = links["wheel_left_link"].find("collision/geometry/cylinder")
    wheel_joint = next(j for j in joints if j.find("child").get("link") == "wheel_left_link")
    wheel_z = float((wheel_joint.find("origin").get("xyz")).split()[2])
    scan_joint = next(j for j in joints if j.find("child").get("link") == "base_scan")
    # The lidar site takes the scan joint's rotation as well as its height: the scan is stamped in
    # base_scan, so a ray's bearing must be measured in that (inverted) frame.
    _, lidar_quat = pose(scan_joint)
    scales = mesh_scales(urdf)
    assets = "".join(
        f'    <material name="{model}_{n}" rgba="{rgba}"/>\n' for n, rgba in sorted(PALETTE.items())
    ) + "".join(
        f'    <mesh file="{shipped}.stl" scale="{scales.get(stem, "1 1 1")}"/>\n'
        for stem, shipped in sorted(meshes.items(), key=lambda kv: kv[1])
    )
    return TEMPLATE.format(
        model=model, human=human, commit=commit, assets=assets, gap=f"{gap:g}",
        cad_mm=f"{CAD_SCAN_HEIGHT * 1000:g}",
        head_mesh=meshes[HEAD_MESH[0]],
        base_pos=base_attrs["pos"], base_mass=base_attrs["mass"],
        base_diaginertia=base_attrs["diaginertia"], base_geoms=base_geoms, bodies=bodies,
        lidar_z=f'{float(scan_joint.find("origin").get("xyz").split()[2]):g}', lidar_quat=lidar_quat,
        rest_height=f'{float(wheel.get("radius")) - wheel_z:g}',
        top_speed=TOP_SPEED[model][0], top_yaw=TOP_SPEED[model][1],
        wheel_ctrl=f"{TOP_SPEED[model][0] / float(wheel.get('radius')) * 2:.0f}",
    )


#: From each robot's own config/navigation.yaml: (max_vel_x, max_vel_theta).
TOP_SPEED = {"makerspet_mini": (0.1, 0.5)}

TEMPLATE = """<mujoco model="{model}">
  <!--
    {human} - a 3D-printed differential-drive pet robot with a 2D lidar.

    GENERATED by external/convert/build_makerspet_mjcf.py from makerspet/{model} @ {commit}
    (Apache-2.0 - see {model}_LICENSE). Do not hand-edit: re-run the generator. Every mass, inertia,
    offset and primitive below is the description's own value.

    A TRUE two-wheel differential drive with a modelled caster, so there is no slip_factor - it does
    not turn by scrubbing. The same line turtlebot3_waffle, raspimouse and oomwoo_one draw.

    The scan frame is INVERTED (the description's scan_joint carries rpy="0 -pi 0"), and kept: it is
    the frame the robot stamps its scan in. Its height is not the description's 0.0704 but the
    manufacturer's CAD, the LD14P's optical block {cad_mm} mm above the floor - see
    build_makerspet_mjcf.CAD_SCAN_HEIGHT, the manifest and the port log.

    DEVIATION: the head ends at z {gap} in base_link, the bottom face of the scanner puck. The
    description's head encloses the scan plane; the real robot carries its LiDAR in an open gap. The
    collision cylinder is shortened to that height and the hemisphere visual ships cut there as
    {head_mesh}.stl; the head's mass and inertia are the description's. See
    build_makerspet_mjcf.open_scan_gap and the port log.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="{model}">
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <geom type="cylinder" group="3" rgba="0.05 0.05 0.05 0.4" friction="1.0 0.005 0.0001"/>
      </default>
      <default class="caster_collision">
        <!-- A swivelling caster offers almost no lateral resistance. `priority` is what makes the
             low friction take effect at all: MuJoCo otherwise uses the MAXIMUM of the two geoms'
             friction, so the floor's 1.0 wins and the caster scrubs. Measured: without it, yaw
             tracks 0.77-0.87 of commanded; with it, 0.92-0.93. -->
        <geom group="3" rgba="0.6 0.1 0.1 0.35" friction="0.05 0.005 0.0001" priority="2"/>
      </default>
      <default class="wheel">
        <!-- axis 0 0 1 in the wheel's OWN frame, which the joint's rpy has already rotated to lie
             along the robot's y. The description's own axis, not a re-derived one.

             armature 0.002 is a geared motor's reflected rotor inertia, and it is what lets kv be
             high enough to track: a velocity servo's steady-state error is (required torque)/kv, so
             at kv 0.25 the wheels ran 21% slow while drawing only a quarter of their torque limit.
             The pair is chosen together to keep kv*dt/I just under 1. -->
        <joint axis="0 0 1" damping="0.001" armature="0.002"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <worldbody>
    <body name="base_link" childclass="{model}">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <!-- The scan plane and frame: base_scan's, the description's scan_joint rotation at the CAD's height. -->
      <site name="lidar" pos="0 0 {lidar_z}"{lidar_quat} size="0.005" rgba="1 0 0 0.6"/>
      <site name="base_imu" pos="0 0 0" size="0.005" rgba="0 0 0 0"/>
{base_geoms}{bodies}    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servos, one per wheel. ctrlrange is the robot's own navigation.yaml top speed over
         its wheel radius, doubled for headroom; forcerange is its motor_stall_torque. kv is kept
         paired with the wheel class's armature so kv*dt/I stays just under 1 - see the wheel
         default above, and the port log for the measurement. -->
    <velocity name="wheel_left_motor" joint="wheel_left_joint" kv="1.0" ctrlrange="-{wheel_ctrl} {wheel_ctrl}" forcerange="-0.49 0.49"/>
    <velocity name="wheel_right_motor" joint="wheel_right_joint" kv="1.0" ctrlrange="-{wheel_ctrl} {wheel_ctrl}" forcerange="-0.49 0.49"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0"/>
  </keyframe>
</mujoco>
"""


def copy_meshes(source: Path, package: Path, meshes: dict[str, str],
                derived: dict[str, bytes]) -> None:
    """Ship the description's STL meshes under their ``meshes`` stems; a stem in ``derived`` is
    written from those bytes instead of copied.

    The URDF references ``package://<pkg>/mesh/head.stl``, but the file lives at
    ``sdf/<pkg>/mesh/head.stl`` and ``CMakeLists.txt`` installs ``sdf``, not ``mesh`` -- so that
    reference does not resolve in an installed package. Upstream's, not ours; matched by basename
    and recorded in the port log rather than papered over.
    """
    found = {p.stem: p for p in source.rglob("*.stl") if ".git" not in p.parts}
    (package / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (package / "meshes").glob("*.stl"):
        stale.unlink()
    for stem, path in sorted(found.items()):
        target = package / "meshes" / f"{meshes[stem]}.stl"
        if meshes[stem] in derived:
            target.write_bytes(derived[meshes[stem]])
        else:
            shutil.copy2(path, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    failed = False
    for model, (url, commit, human, _) in ROBOTS.items():
        source = resolve_source(model, url, commit)
        package = MODELS / model
        target = package / f"{model}.xml"
        meshes = shipped_meshes(source)
        head_target = package / "meshes" / f"{HEAD_MESH[1]}.stl"

        def fresh(meshes=meshes, model=model, commit=commit, human=human,
                  source=source) -> tuple[str, bytes]:
            with tempfile.TemporaryDirectory() as tmp:
                urdf = expand_xacro({}, source / "urdf/robot.urdf.xacro", Path(tmp))
            place_scan_at_cad_height(urdf)
            gap, head_cut = open_scan_gap(urdf)
            scale_z = float(mesh_scales(urdf)[HEAD_MESH[0]].split()[2])
            head = clip_below(vendor_mesh(source, HEAD_MESH[0]).read_bytes(), head_cut / scale_z)
            return build(urdf, model, commit, human, meshes, gap), head

        if args.check:
            xml, head = fresh()
            if not target.exists() or target.read_text() != xml:
                print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
                failed = True
            elif not head_target.exists() or head_target.read_bytes() != head:
                print(f"{head_target} differs from a fresh cut of the vendor mesh", file=sys.stderr)
                failed = True
            else:
                print(f"{target}: up to date with {commit[:12]}")
            continue

        package.mkdir(parents=True, exist_ok=True)
        xml, head = fresh()
        copy_meshes(source, package, meshes, {HEAD_MESH[1]: head})
        write_license(
            source / "LICENSE",
            package / f"{model}_LICENSE",
            [
                f"{human} -- vendored geometry and description.",
                "",
                f"Upstream:   {url.removesuffix('.git')}",
                f"Commit:     {commit}",
                "Copyright:  Ilia O. (iliao@makerspet.com)",
                "License:    Apache License 2.0, as declared by the upstream package.xml.",
                "",
                "Regenerate with: external/convert/build_makerspet_mjcf.py",
                "The full text of the grant follows.",
            ],
        )
        target.write_text(xml)
        print(f"wrote {target} + meshes + {model}_LICENSE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
