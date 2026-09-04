#!/usr/bin/env python3
"""Generate ``open_manipulator_x.xml`` (MJCF) from the ROBOTIS OpenMANIPULATOR-X URDF.

Why a generator rather than a hand-authored MJCF (as ``ur5e.xml`` is): the OM-X arm's numbers all
come from one upstream file, the vendor's own URDF, and the arm is a plain 4-joint serial chain. A
script that reads that URDF cannot mistranscribe a link offset or an inertia tensor, and it records
in one place exactly which upstream version the model was built from. Re-run it after a
``ros-jazzy-open-manipulator-description`` upgrade to see whether the vendor moved anything.

Source (default): the installed debian, ``/opt/ros/<distro>/share/open_manipulator_description``.
That package is a *versioned* artifact (``dpkg -l ros-jazzy-open-manipulator-description``) and it is
the pair of the ``open_manipulator_moveit_config`` debian whose SRDF/joint limits the experiment's
MoveIt stack loads -- so sim geometry and planning geometry come from the same release by
construction. Building from a git clone of ROBOTIS-GIT/open_manipulator instead would break that
pairing: its ``main`` branch ships a *redesigned* arm (``omx_f``, different link offsets).

Usage::

    python3 external/convert/build_open_manipulator_x_mjcf.py            # write the model in-tree
    python3 external/convert/build_open_manipulator_x_mjcf.py --check    # fail if in-tree is stale

Meshes are NOT copied by this script -- they are vendored once (Apache-2.0, see
``OPEN_MANIPULATOR_X_LICENSE``) and are stable across patch releases. ``--check`` covers the XML only.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_URDF = "/opt/ros/{distro}/share/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf"
DEFAULT_SRDF = "/opt/ros/{distro}/share/open_manipulator_moveit_config/config/open_manipulator_x/open_manipulator_x.srdf"
# roqsim_manipulation_assets, NOT roqsim_manipulation: the models moved to the asset half of that package
# when it was split (roqsim_manipulation kept the plugins). external/ is a sibling of the family packages,
# so anchor back through parents[2] -- see external/convert/README.md.
OUT = (
    Path(__file__).resolve().parents[2]
    / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/open_manipulator_x/open_manipulator_x.xml"
)

#: The four actuated arm joints, in chain order. The gripper is deliberately absent -- see the
#: `gripper` note in the port log: the fingers are welded at their URDF-zero pose because the arm's
#: only consumer (the palm-harvesting benchmark) states the gripper is out of scope, and an
#: unactuated prismatic joint is a numerical nuisance that buys nothing.
HEADLINE = "ROBOTIS OpenMANIPULATOR-X (4-joint serial arm + parallel gripper) for MuJoCo."

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]

#: Welded finger links: (link name, parent joint name whose URDF origin places it).
FINGERS = [
    ("gripper_left_link", "gripper_left_joint"),
    ("gripper_right_link", "gripper_right_joint"),
]

#: Eye-in-hand RealSense mount on link5. The vendor's own pose for it, from the `camera_joint` that
#: ROBOTIS ships COMMENTED OUT in `open_manipulator.urdf.xacro` (noetic branch, 8cb9c5e). It is the
#: only published OM-X camera mount there is, and it is where the benchmark paper's D435i sits.
CAMERA_POS = (0.070, 0.032, 0.052)

#: XM430-W350-T stall torque at 12 V (ROBOTIS e-Manual). The URDF's `effort="1000"` is a placeholder,
#: so the actuator force range comes from the datasheet instead. Servo gains are a substrate
#: calibration, not a vendor number -- see the port log.
JOINT_TORQUE_NM = 4.1
#: Servo gains: the LARGEST gain at which the actuator never reaches the XM430's stall torque during
#: a 3 s ramp across the workspace. Criterion fixed before the sweep; measured (lag during ramp /
#: settled error / fraction of steps at the torque limit): 50 -> .080/.0114/0.00, 100 -> .063/.0057/
#: 0.00, 150 -> .052/.0038/0.00, 200 -> .047/.0054/0.94, 300 -> .042/.0079/1.00. Past 150 the servo
#: runs saturated -- effectively bang-bang -- and the settled error gets WORSE, not better.
GAIN = 150.0
DAMP = 10.0


def load_urdf(path: Path) -> tuple[dict, dict]:
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        entry = {"meshes": []}
        for tag in ("visual", "collision"):
            el = link.find(tag)
            if el is not None and el.find("geometry/mesh") is not None:
                mesh = el.find("geometry/mesh")
                origin = el.find("origin")
                entry[tag] = {
                    "mesh": Path(mesh.get("filename")).stem,
                    "pos": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                }
        if inertial is not None:
            i = inertial.find("inertia")
            entry["inertial"] = {
                "pos": inertial.find("origin").get("xyz"),
                "mass": inertial.find("mass").get("value"),
                # MuJoCo fullinertia order is ixx iyy izz ixy ixz iyz -- NOT the URDF attribute order.
                "fullinertia": " ".join(
                    i.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ),
            }
        links[link.get("name")] = entry

    joints = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        dyn = joint.find("dynamics")
        joints[joint.get("name")] = {
            # The vendor states joint damping (0.1 N-m-s/rad on every arm joint) and nothing else --
            # no friction, no armature. Carried over verbatim; inventing a frictionloss here would
            # spend actuator torque the real XM430 does not spend, on a joint whose 4.1 N-m stall
            # torque is already the binding constraint.
            "damping": dyn.get("damping") if dyn is not None else None,
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "pos": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            "axis": axis.get("xyz") if axis is not None else None,
            "lower": limit.get("lower") if limit is not None else None,
            "upper": limit.get("upper") if limit is not None else None,
            "velocity": limit.get("velocity") if limit is not None else None,
        }
    return links, joints


def load_srdf_exclusions(path: Path) -> list[tuple[str, str]]:
    """The SRDF's ``disable_collisions`` pairs, as MJCF ``<exclude>`` body pairs.

    Why read the planner's file to build the simulator's model: MuJoCo does not filter contacts
    between neighbouring links of a serial chain, and the OM-X's meshes overlap by ~21 mm at every
    joint. Left alone, link1 and link2 push against each other hard enough to jam joint1 outright
    (measured: 4.1 N-m of constraint force, joint1 moves 0.008 rad of a commanded 0.8).

    Taking the pairs from the SRDF rather than inventing them keeps the two halves of the experiment
    describing ONE robot: a self-collision MoveIt ignores cannot jam the simulated arm, and one
    MoveIt checks is not silently permitted in sim. The welded fingers fold into ``link5`` -- they are
    geoms on that body here, so they inherit its pairs.
    """
    welded = {"gripper_left_link": "link5", "gripper_right_link": "link5"}
    pairs = set()
    for el in ET.parse(path).getroot().findall("disable_collisions"):
        a = welded.get(el.get("link1"), el.get("link1"))
        b = welded.get(el.get("link2"), el.get("link2"))
        if a != b:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def fnum(text: str) -> str:
    """Trim URDF scientific notation to something a human can diff (metres/kg/kg-m^2)."""
    return " ".join(f"{float(v):.9g}" for v in text.split())


def body_xml(
    links: dict, joints: dict, link_name: str, joint_name: str | None, depth: int
) -> list[str]:
    pad = "  " * depth
    link = links[link_name]
    out = []
    pos = f' pos="{fnum(joints[joint_name]["pos"])}"' if joint_name else ""
    out.append(f'{pad}<body name="{link_name}"{pos}>')
    inert = link["inertial"]
    out.append(
        f'{pad}  <inertial pos="{fnum(inert["pos"])}" mass="{fnum(inert["mass"])}" '
        f'fullinertia="{fnum(inert["fullinertia"])}"/>'
    )
    if joint_name:
        j = joints[joint_name]
        damping = f' damping="{fnum(j["damping"])}"' if j["damping"] else ""
        out.append(
            f'{pad}  <joint name="{joint_name}" axis="{j["axis"]}" '
            f'range="{fnum(j["lower"])} {fnum(j["upper"])}"{damping}/>'
        )
    vis = link["visual"]
    out.append(
        f'{pad}  <geom name="{link_name}_vis" mesh="{vis["mesh"]}" pos="{fnum(vis["pos"])}" class="visual"/>'
    )
    col = link.get("collision", vis)
    out.append(
        f'{pad}  <geom name="{link_name}_col" mesh="{col["mesh"]}" pos="{fnum(col["pos"])}" class="collision"/>'
    )
    return out


def build(urdf: Path, srdf: Path) -> str:
    links, joints = load_urdf(urdf)
    exclusions = load_srdf_exclusions(srdf)
    chain = [(joints[j]["child"], j) for j in ARM_JOINTS]
    mesh_names = sorted(
        {links[n]["visual"]["mesh"] for n, _ in chain}
        | {links[joints[ARM_JOINTS[0]]["parent"]]["visual"]["mesh"]}
        | {links[n]["visual"]["mesh"] for n, _ in FINGERS}
    )

    L = []
    L.append('<mujoco model="open_manipulator_x">')
    L.append(f"  <!-- {HEADLINE} -->")
    # roqsim_manipulation_assets is one folder per model, so the model's meshes are its own subdir.
    L.append('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
    L.append("")
    L.append("  <!-- GENERATED by external/convert/build_open_manipulator_x_mjcf.py from")
    L.append(f"       {urdf}")
    L.append("       Do not hand-edit: re-run the generator. Every link offset, mass and inertia")
    L.append("       tensor below is the vendor's own URDF value.")
    L.append("")
    L.append(
        "       No <option> here: timestep, integrator, solver and contact overrides belong to"
    )
    L.append("       the EXPERIMENT (the world YAML's `sim:` block), not to the arm. -->")
    L.append("")
    L.append("  <default>")
    L.append('    <default class="open_manipulator_x">')
    L.append('      <material specular="0.3" shininess="0.2"/>')
    L.append(
        "      <!-- armature is the ONE joint property with no vendor value: the XM430's rotor"
    )
    L.append("           inertia reflected through its 353:1 gearbox. 0.01 kg-m^2 is the standard")
    L.append(
        "           Menagerie order for a geared hobby servo; damping comes from the URDF. -->"
    )
    L.append('      <joint axis="0 1 0" armature="0.01"/>')
    L.append("      <!-- Position servo. Gains are a substrate calibration (port log S1), not a")
    L.append("           vendor number; forcerange is the XM430-W350's 4.1 N-m stall torque. -->")
    L.append(f'      <general gaintype="fixed" biastype="affine" gainprm="{GAIN:g}"')
    L.append(
        f'               biasprm="0 -{GAIN:g} -{DAMP:g}" forcerange="-{JOINT_TORQUE_NM} {JOINT_TORQUE_NM}"/>'
    )
    L.append('      <default class="visual">')
    L.append(
        '        <geom type="mesh" contype="0" conaffinity="0" group="2" material="omx_grey"/>'
    )
    L.append("      </default>")
    L.append('      <default class="collision">')
    L.append("        <!-- MuJoCo collides the mesh's convex hull. For a 380 mm arm against tree")
    L.append(
        "             fronds that is the right fidelity, and it is the same shape MoveIt's FCL"
    )
    L.append("             checker sees from the URDF, so sim and planner agree on the robot. -->")
    L.append('        <geom type="mesh" group="3" contype="1" conaffinity="1"/>')
    L.append("      </default>")
    L.append('      <site size="0.002" rgba="0.9 0.3 0.3 0.4" group="4"/>')
    L.append("    </default>")
    L.append("  </default>")
    L.append("")
    L.append("  <asset>")
    L.append('    <material class="open_manipulator_x" name="omx_grey" rgba="0.25 0.25 0.25 1"/>')
    for m in mesh_names:
        L.append(f'    <mesh file="{m}.stl" scale="0.001 0.001 0.001"/>')
    L.append("  </asset>")
    L.append("")
    L.append("  <worldbody>")

    base = joints[ARM_JOINTS[0]]["parent"]
    L.append(f'    <body name="{base}" childclass="open_manipulator_x">')
    binert = links[base]["inertial"]
    L.append(
        f'      <inertial pos="{fnum(binert["pos"])}" mass="{fnum(binert["mass"])}" '
        f'fullinertia="{fnum(binert["fullinertia"])}"/>'
    )
    bvis = links[base]["visual"]
    L.append(
        f'      <geom name="{base}_vis" mesh="{bvis["mesh"]}" pos="{fnum(bvis["pos"])}" class="visual"/>'
    )
    L.append(
        f'      <geom name="{base}_col" mesh="{bvis["mesh"]}" pos="{fnum(bvis["pos"])}" class="collision"/>'
    )

    depth = 3
    for link_name, joint_name in chain:
        L.extend(body_xml(links, joints, link_name, joint_name, depth))
        depth += 1

    pad = "  " * depth
    # The tool frame. `end_effector_joint` is a fixed URDF joint, so it is a site here rather than a
    # body: spawn_arm's `end_effector:` welds a gripper INTO a site's frame, and MoveIt's `arm` group
    # tip is this same point (SRDF end_effector parent_link `end_effector_link`).
    ee = joints["end_effector_joint"]
    L.append(f'{pad}<site name="attachment_site" pos="{fnum(ee["pos"])}"/>')
    L.append(f'{pad}<site name="end_effector_site" pos="{fnum(ee["pos"])}"/>')
    L.append("")
    L.append(f"{pad}<!-- Eye-in-hand RealSense D435i. MuJoCo cameras look down -z with +y up, so")
    L.append(
        f"{pad}     xyaxes below points the optical axis along link5's +x (the tool direction)"
    )
    L.append(
        f"{pad}     and leaves the image upright -- the pose realsense_d435's docstring gives."
    )
    L.append(f"{pad}     fovy is the D435 depth stream's 58 deg vertical FOV. -->")
    L.append(
        f'{pad}<camera name="d435_color" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}" '
        'xyaxes="0 -1 0 0 0 1" fovy="58" resolution="640 480"/>'
    )
    L.append(
        f'{pad}<site name="d435_mount" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}"/>'
    )
    L.append("")
    L.append(f"{pad}<!-- Gripper fingers, WELDED at their URDF-zero pose (no joint, no actuator).")
    L.append(
        f"{pad}     The paper this arm was ported for states the gripper is out of scope, and a"
    )
    L.append(f"{pad}     zero-pose weld makes the sim's finger geometry identical to what MoveIt")
    L.append(f"{pad}     plans against from the same URDF at joint value 0. -->")
    for link_name, joint_name in FINGERS:
        f_link = links[link_name]
        f_pos = fnum(joints[joint_name]["pos"])
        fv = f_link["visual"]
        L.append(
            f'{pad}<geom name="{link_name}_vis" mesh="{fv["mesh"]}" pos="{f_pos}" class="visual"/>'
        )
        L.append(
            f'{pad}<geom name="{link_name}_col" mesh="{fv["mesh"]}" pos="{f_pos}" class="collision"/>'
        )

    # Close link5 .. link1: the chain bodies sit at indent depths (depth-1) down to 2.
    for d in range(depth - 1, 1, -1):
        L.append(f"{'  ' * d}</body>")
    L.append("  </worldbody>")
    L.append("")
    L.append("  <contact>")
    L.append(
        f"    <!-- Generated from {srdf.name}'s `disable_collisions`, so the simulated arm and"
    )
    L.append("         the planned arm agree on which self-collisions are real. MuJoCo does not")
    L.append("         filter chain neighbours, and these meshes overlap ~21 mm at every joint --")
    L.append("         without this, link1-vs-link2 contact jams joint1 solid. -->")
    for a, b in exclusions:
        L.append(f'    <exclude body1="{a}" body2="{b}"/>')
    L.append("  </contact>")
    L.append("")
    L.append("  <actuator>")
    for j in ARM_JOINTS:
        lo, hi = fnum(joints[j]["lower"]), fnum(joints[j]["upper"])
        L.append(
            f'    <general name="{j}" joint="{j}" ctrlrange="{lo} {hi}" class="open_manipulator_x"/>'
        )
    L.append("  </actuator>")
    L.append("</mujoco>")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", default=None, help="source URDF (default: the installed debian)")
    ap.add_argument("--srdf", default=None, help="source SRDF (default: the installed debian)")
    ap.add_argument("--distro", default="jazzy")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if --out is stale")
    args = ap.parse_args()

    urdf = Path(args.urdf or DEFAULT_URDF.format(distro=args.distro))
    srdf = Path(args.srdf or DEFAULT_SRDF.format(distro=args.distro))
    # Fail loudly: a missing source must not silently leave a stale model in place, and an SRDF-less
    # run would emit an arm with NO contact exclusions -- which loads fine and then jams joint1.
    for path, pkg in ((urdf, "description"), (srdf, "moveit-config")):
        if not path.is_file():
            print(
                f"error: not found: {path}\n"
                f"  install it:  sudo apt install ros-{args.distro}-open-manipulator-{pkg}",
                file=sys.stderr,
            )
            return 2

    xml = build(urdf, srdf)
    out = Path(args.out)
    if args.check:
        if not out.is_file() or out.read_text() != xml:
            print(f"error: {out} is stale -- re-run without --check", file=sys.stderr)
            return 1
        print(f"{out} is up to date with {urdf}")
        return 0
    out.write_text(xml)
    print(f"wrote {out} ({len(xml.splitlines())} lines) from {urdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
