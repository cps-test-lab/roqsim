#!/usr/bin/env python3
"""Build roqsim's Maker's Pet MJCF models from the `makerspet/makerspet_*` descriptions.

Parameterised by robot, because the vendor ships one design at four sizes (Loki 200 mm, Snoopy
300 mm, Fido 250 mm, Mini 170 mm) and they differ in dimensions rather than in kind. Only the ones
in :data:`ROBOTS` are built; adding a sibling is a line there plus its pinned commit.

Unlike ``build_oomwoo_one_mjcf.py``, which walks a flat list of links hanging off ``base_link``,
these have a genuinely **nested** tree -- ``base_link -> base_upper_link -> head_link ->
tablet_link -> {camera, imu}`` -- so the body emitter here is recursive. That is the only structural
difference; both vendors share the same self-contained, primitive-heavy, ``$(find)``-free idiom, and
both use :func:`urdf_source.link_primitives`.

**The joint rpy is load-bearing and is why this generator reads the joint's full frame.** The wheel
joints carry ``rpy="-pi/2 0 0"``, the scanner ``rpy="-pi 0 0"`` (an inverted puck between the decks)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, link_primitives, pose  # noqa: E402

#: model short name -> (repo, pinned commit on the jazzy branch, human name, body diameter mm)
ROBOTS = {
    "makerspet_loki": (
        "https://github.com/makerspet/makerspet_loki.git",
        "e778ecde98b4f0dc2dcb32d25f43cb4cb4af1158",
        "Maker's Pet Loki",
        200,
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


def mesh_scales(urdf: ET.Element) -> dict[str, str]:
    """``{mesh stem: MJCF scale string}`` for every mesh the URDF references.

    URDF puts the scale on the *reference*, MJCF puts it on the *asset*, so it has to be collected
    before the assets are written. Ignoring it is the classic meters-vs-millimeters trap: Loki's
    head carries ``scale="${0.001*base_diameter} ..."``, roughly 2e-4 and non-uniform, so at 1:1 the
    mesh comes out 1000 units across on a 0.2 m robot. Nothing in a physics battery notices, because
    that link's *collision* is a cylinder and only its visual is the mesh -- it took a render.

    Raises when one mesh is referenced at two different scales, rather than silently picking one.
    """
    scales: dict[str, str] = {}
    for mesh in urdf.iter("mesh"):
        stem = Path(mesh.get("filename")).stem
        raw = (mesh.get("scale") or "1 1 1").replace(",", " ").split()
        scale = " ".join(f"{float(v):g}" for v in raw)
        if scales.setdefault(stem, scale) != scale:
            raise RuntimeError(
                f"mesh {stem!r} is referenced at two scales ({scales[stem]!r} and {scale!r}); MJCF "
                f"puts the scale on the asset, so this needs one asset per scale."
            )
    return scales


def build(urdf: ET.Element, model: str, commit: str, human: str, meshes: dict[str, str]) -> str:
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
    scales = mesh_scales(urdf)
    assets = "".join(
        f'    <material name="{model}_{n}" rgba="{rgba}"/>\n' for n, rgba in sorted(PALETTE.items())
    ) + "".join(
        f'    <mesh file="{stem}.stl" scale="{scales.get(stem, "1 1 1")}"/>\n'
        for stem in sorted(meshes.values())
    )
    return TEMPLATE.format(
        model=model, human=human, commit=commit, assets=assets,
        base_pos=base_attrs["pos"], base_mass=base_attrs["mass"],
        base_diaginertia=base_attrs["diaginertia"], base_geoms=base_geoms, bodies=bodies,
        lidar_z=f'{float(scan_joint.find("origin").get("xyz").split()[2]):g}',
        rest_height=f'{float(wheel.get("radius")) - wheel_z:g}',
        top_speed=TOP_SPEED[model][0], top_yaw=TOP_SPEED[model][1],
        wheel_ctrl=f"{TOP_SPEED[model][0] / float(wheel.get('radius')) * 2:.0f}",
    )


#: From each robot's own config/navigation.yaml: (max_vel_x, max_vel_theta).
TOP_SPEED = {"makerspet_loki": (0.26, 1.0)}

TEMPLATE = """<mujoco model="{model}">
  <!--
    {human} - a 3D-printed differential-drive pet robot with a 2D lidar.

    GENERATED by external/convert/build_makerspet_mjcf.py from makerspet/{model} @ {commit}
    (Apache-2.0 - see {model}_LICENSE). Do not hand-edit: re-run the generator. Every mass, inertia,
    offset and primitive below is the description's own value.

    A TRUE two-wheel differential drive with a modelled caster, so there is no slip_factor - it does
    not turn by scrubbing. The same line turtlebot3_waffle, raspimouse and oomwoo_one draw.

    The scanner sits INVERTED between the two decks (the description's scan_joint carries
    rpy="-pi 0 0"), which is how this design fits a 360 degree puck under a head. Its mount is the
    vendor's; the scan parameters are not - see the manifest and the port log.
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
      <!-- The scan plane, off the description's own scan_joint. -->
      <site name="lidar" pos="0 0 {lidar_z}" size="0.005" rgba="1 0 0 0.6"/>
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


def copy_meshes(source: Path, package: Path) -> dict[str, str]:
    """Copy the description's STL meshes, returning ``{urdf stem: shipped stem}``.

    The URDF references ``package://<pkg>/mesh/head.stl``, but the file lives at
    ``sdf/<pkg>/mesh/head.stl`` and ``CMakeLists.txt`` installs ``sdf``, not ``mesh`` -- so that
    reference does not resolve in an installed package. Upstream's, not ours; matched by basename
    and recorded in the port log rather than papered over.
    """
    found = {p.stem: p for p in source.rglob("*.stl") if ".git" not in p.parts}
    if not found:
        return {}
    (package / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (package / "meshes").glob("*.stl"):
        stale.unlink()
    for stem, path in sorted(found.items()):
        shutil.copy2(path, package / "meshes" / f"{stem}.stl")
    return {stem: stem for stem in found}


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
        meshes = {p.stem: p.stem for p in source.rglob("*.stl") if ".git" not in p.parts}

        def fresh(meshes=meshes, model=model, commit=commit, human=human, source=source) -> str:
            with tempfile.TemporaryDirectory() as tmp:
                urdf = expand_xacro({}, source / "urdf/robot.urdf.xacro", Path(tmp))
            return build(urdf, model, commit, human, meshes)

        if args.check:
            if not target.exists() or target.read_text() != fresh():
                print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
                failed = True
            else:
                print(f"{target}: up to date with {commit[:12]}")
            continue

        package.mkdir(parents=True, exist_ok=True)
        copy_meshes(source, package)
        shutil.copy2(source / "LICENSE", package / f"{model}_LICENSE")
        target.write_text(fresh())
        print(f"wrote {target} + meshes + {model}_LICENSE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
