"""Build the roqsim ``tiago_pro`` model from PAL Robotics' official ROS 2 description packages.

PAL splits the TIAGo Pro across six repositories (the robot, the OMNI base, the SEA arm, the head,
the PRO gripper, and shared urdf utils), so the description only exists as an *expanded xacro*. This
script owns the whole chain and is the reproducible record of the port:

    6 pinned repos -> xacro expansion -> URDF with a flat meshdir -> MuJoCo compile -> tiago_pro.xml

Everything a URDF cannot express is added in ``postprocess()``: the holonomic planar drive, the
visual/collision split, the lidar/IMU sites, actuators, the 14 gripper ``<mimic>`` couplings as MJCF
joint equalities, sensors, and the home keyframe. See the package THIRD_PARTY.md for the assumptions
and their sensitivity.

Requires ROS 2 (for ``xacro``) -- PAL's description is xacro-only, so there is no ROS-free path to
the URDF. The script builds a throwaway ament index over the pinned checkouts, so no colcon
workspace or ``rosdep`` install is needed; it never reads an installed ``*_description`` package,
which would silently defeat the commit pinning.

Usage (from anywhere)::

    python external/convert/build_tiago_pro_mjcf.py

Add ``--keep-work`` to inspect the intermediate URDF/base MJCF.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from xml.dom import minidom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_headline import with_headline  # noqa: E402
from sources import resolve_source  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Pinned upstream revisions. All six are Apache-2.0 (c) PAL Robotics S.L. -- the licence text is
# vendored once as models/tiago_pro/tiago_pro_LICENSE. Bumping a pin here must also update the provenance
# table in the package THIRD_PARTY.md; the two must not drift.
# ---------------------------------------------------------------------------------------------
PAL = "https://github.com/pal-robotics"
SOURCES = {
    # name                     (repo,                    commit,     tag,      subdir with the pkg)
    "pal_tiago_pro": ("tiago_pro_robot", "0469e2d15a7f6f14612fd7acb92c2543ce179056", "2.5.0"),
    "pal_omni_base": ("omni_base_robot", "251ecc4cf57ed1870e84dcc135be917676e318f2", "2.18.0"),
    "pal_sea_arm": ("pal_sea_arm", "2949eb4db2c360b7f677a289ca5964716d94d95a", "2.8.3"),
    "pal_tiago_pro_head": ("tiago_pro_head_robot", "c9ea3291f0001fd40f40770d4eed9bacd03d889d", "1.12.0"),
    "pal_pro_gripper": ("pal_pro_gripper", "3588daa87cf4dd7f2310b098c03dd8d2bb27faa6", "1.12.5"),
    "pal_urdf_utils": ("pal_urdf_utils", "775cdd6886296e6c00f17dbdfd9bcdd20e0e6622", "2.9.2"),
}
# ROS package name -> (source key, path within that checkout). `xacro`'s $(find …) is resolved
# against these and nothing else.
PACKAGES = {
    "tiago_pro_description": ("pal_tiago_pro", "tiago_pro_description"),
    "omni_base_description": ("pal_omni_base", "omni_base_description"),
    "pal_sea_arm_description": ("pal_sea_arm", "pal_sea_arm_description"),
    "tiago_pro_head_description": ("pal_tiago_pro_head", "tiago_pro_head_description"),
    "pal_pro_gripper_description": ("pal_pro_gripper", "pal_pro_gripper_description"),
    "pal_urdf_utils": ("pal_urdf_utils", "."),
}

# xacro args. Everything except `camera_model` is PAL's own default (dual tiago-pro arms, spherical
# wrists + tool changers, pal-pro-grippers, sick-571 front+rear lasers, no teleop pilot station).
#
# camera_model: PAL's default is `realsense-d435i`, but at this pin
# tiago_pro_head_description/meshes/ ships only `realsense-d435_cover_link.stl` -- the `d435i` cover
# mesh does not exist upstream, so the default config cannot be compiled. `realsense-d435` is the
# nearest present variant (the d435 and d435i differ only by the IMU the sim does not read).
XACRO_ARGS = ["camera_model:=realsense-d435"]

MODEL = "tiago_pro"
HEADLINE = "PAL Robotics TIAGo Pro (omnidirectional base, SEA arm, head and PRO gripper) for MuJoCo."
# The package's models/ dir, addressed relative to external/ (a sibling of the roqsim_* packages) so
# the script runs from any cwd. The package is laid out one folder per model, so everything this
# script writes goes under `models/<MODEL>/`: the MJCF, and its own `meshes/` beside it. Per-model
# mesh dirs are not cosmetic -- link names like `head_2_link.stl` are generic enough to collide
# across robots in a shared flat meshdir.
#
# `roqsim_mobile_manipulation`, not `roqsim_mobile`: the TIAGo Pro is a base AND an arm, so it belongs to
# neither family package it is made of. See that package's pyproject for why the layering matters.
PKG_MODELS = (
    Path(__file__).resolve().parents[2]
    / "roqsim_mobile_manipulation/src/roqsim_mobile_manipulation/models"
)
MODEL_DIR = PKG_MODELS / MODEL
MESH_SUBDIR = "meshes"

# ---------------------------------------------------------------------------------------------
# Joint groups. Names come straight from the expanded URDF.
# ---------------------------------------------------------------------------------------------
WHEELS = ["wheel_front_left", "wheel_front_right", "wheel_rear_left", "wheel_rear_right"]
WHEEL_JOINTS = [f"{w}_joint" for w in WHEELS]
TORSO = ["torso_lift_joint"]
HEAD = ["head_1_joint", "head_2_joint"]
ARM_L = [f"arm_left_{i}_joint" for i in range(1, 8)]
ARM_R = [f"arm_right_{i}_joint" for i in range(1, 8)]
GRIP_MASTER = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

# The PRO gripper's finger linkage. Each finger is a four-bar:
#
#     base_finger -> inner_finger -> fingertip          (the driven branch)
#     base        -> outer_finger                       (the coupler branch)
#     the two branches are PINNED together at the fingertip/outer_finger pivot
#
# The URDF cannot say that: it is a tree, so PAL breaks the loop and fakes the constraint with
# `<mimic>` multipliers (inner/outer -8.28, fingertip +8.28 off the prismatic master). Those
# multipliers are a LINEARISATION of the mechanism about one configuration, and reproducing them
# literally as MJCF joint equalities inherits the approximation: the pads then splay, measured at
# 19 deg off perpendicular at the grasp opening and up to 49 deg wide open, so a "parallel-jaw"
# gripper never actually presents parallel jaws. It closes on an object and cannot carry it -- the
# contact is an edge, the pinch is 2-point, and the object pitches out of it under the lift.
#
# So the loop is closed properly instead:
#
#   DRIVEN     the master prismatic drives `inner_finger_*` (and the dummy screw leaf) by mimic, as
#              in the URDF -- that branch is a genuine open chain and the multiplier is exact for it.
#   LOOP       `fingertip_* <-> outer_finger_*` get an <equality><connect>, which is how a
#              closed-loop linkage is expressed in MuJoCo's tree. The fingertip's angle then FOLLOWS
#              from the mechanism instead of from a fitted constant, which is what keeps the pad
#              parallel across the travel.
#
# The anchor is the physical pivot, measured rather than guessed: at the reference pose the fingertip
# and outer_finger meshes touch to 0.26 mm, and the contact midpoint is the same point in the
# fingertip's own frame for all four fingers.
MIMIC = {
    f"gripper_{s}_{f}": (f"gripper_{s}_finger_joint", m)
    for s in ("left", "right")
    for f, m in (
        ("inner_finger_left_joint", -8.28),
        ("inner_finger_right_joint", -8.28),
        # Drives only the dummy `screw_*_link` leaf, so it is cosmetic. PAL's own sources disagree
        # about it (this URDF says 0.22, the legacy Gazebo plugin in the same file says 1.0); measured,
        # the fingertip pad gap is identical either way. The URDF value is kept as the faithful one.
        ("finger_right_joint", 0.22),
    )
}

#: Four-bar loop closures: (fingertip body, outer_finger body) per finger, pinned at ANCHOR.
LOOP_FINGERS = [
    (f"gripper_{s}_fingertip_{lr}_link", f"gripper_{s}_outer_finger_{lr}_link")
    for s in ("left", "right")
    for lr in ("left", "right")
]
#: The pivot, in the FINGERTIP's own frame (measured; see the comment above).
LOOP_ANCHOR = "0.017 -0.0057 -0.004"

# Position-servo gains: (kp, forcerange). Every forcerange is the URDF <limit effort>, so the model
# saturates where the real joint does. kp is sized to hold the pose against gravity and is the one
# genuinely tuned number (see port log A4); `dampratio=1` lets MuJoCo derive the matching damping
# from the joint's own effective inertia instead of hand-tuned per-joint damping.
POS_GAINS = {
    "torso_lift_joint": (60000, 2000),  # slide: carries the whole ~28 kg upper body
    "head_1_joint": (100, 5.197),
    "head_2_joint": (100, 2.77),
    **{j: (2000, 43.0) for j in (ARM_L[:2] + ARM_R[:2])},  # shoulder: effort 43
    **{j: (900, 26.0) for j in (ARM_L[2:] + ARM_R[2:])},  # elbow..wrist: effort 26
    **{j: (300, 10.0) for j in GRIP_MASTER},  # slide 0..0.07 m, effort 10 N
}

# PAL's `home` motion, final waypoint (tiago_pro_bringup/config/motions/
# tiago_pro_motions_general_spherical-wrist.yaml). The authoritative tucked stance -- at qpos0 both
# arms stick straight out forward, which is not a usable rest pose. Used for the keyframe and echoed
# in the manifest's `rest:` (the manifest is what actually seats it, since spawn_robot strips
# keyframes -- see arm_controller's `rest` docs).
HOME = {
    "torso_lift_joint": 0.1,
    **dict(zip(ARM_L, [0.36, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0], strict=True)),
    **dict(zip(ARM_R, [-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0], strict=True)),
}

# 2D nav lidar mounts. Positions are the source laser links' exact poses expressed in base_link
# (base_sensors.urdf.xacro: laser_height 0.13244, front/rear at diagonal corners); the yaw is each
# laser's own mounting yaw (-45 deg front, +135 deg rear), so the two 270-deg fans overlap into full
# 360-deg coverage. Own upright sites rather than the source links because the real laser links are
# mounted z-down and roqsim's `lidar` casts its fan in the site's local xy plane -- a flipped site
# scans the same plane with reversed handedness. Position/yaw are the source's, the upright z is
# ours. See port log C1.
LIDAR_SITES = {
    "lidar_front": ("0.2751 -0.183 0.13244", "0 0 -0.7853981634"),
    "lidar_rear": ("-0.2751 0.183 0.13244", "0 0 2.3561944902"),
}


def _sh(cmd: list[str], **kw) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def fetch_sources() -> dict[str, Path]:
    """Resolve every pinned repo into external/sources/ and return ROS package name -> path."""
    roots = {
        key: resolve_source(key, f"{PAL}/{repo}", commit)
        for key, (repo, commit, _tag) in SOURCES.items()
    }
    out = {}
    for pkg, (key, sub) in PACKAGES.items():
        path = roots[key] if sub == "." else roots[key] / sub
        if not path.is_dir():
            raise RuntimeError(f"{pkg}: {path} missing -- the pin may predate this layout")
        out[pkg] = path
    return out


def expand_xacro(pkgs: dict[str, Path], work: Path) -> Path:
    """Expand tiago_pro.urdf.xacro against ONLY the pinned checkouts.

    ``$(find <pkg>)`` goes through ament, so a throwaway index is built over the pinned trees and
    prepended to AMENT_PREFIX_PATH. Anything PAL's xacro needs that is not in ``PACKAGES`` fails
    loudly here rather than silently resolving to a differently-versioned installed package.
    """
    prefix = work / "ament"
    (prefix / "share/ament_index/resource_index/packages").mkdir(parents=True)
    for pkg, path in pkgs.items():
        (prefix / "share" / pkg).symlink_to(path)
        (prefix / "share/ament_index/resource_index/packages" / pkg).touch()

    xacro = shutil.which("xacro")
    if not xacro:
        raise RuntimeError(
            "`xacro` not on PATH. PAL ships the TIAGo Pro as xacro only, so ROS 2 is required to "
            "rebuild this model:\n  source /opt/ros/<distro>/setup.bash\n"
            "(the committed tiago_pro.xml needs no ROS -- this is only for regenerating it)"
        )
    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = f"{prefix}{os.pathsep}{env.get('AMENT_PREFIX_PATH', '')}"
    src = pkgs["tiago_pro_description"] / "robots/tiago_pro.urdf.xacro"
    urdf = work / "tiago_pro_expanded.urdf"
    urdf.write_text(_sh([xacro, str(src), *XACRO_ARGS], env=env))
    return urdf


def flatten_meshes(urdf: Path, pkgs: dict[str, Path], meshdir: Path) -> Path:
    """Copy every referenced mesh into *meshdir* under a flat unique name and rewrite the refs.

    All 39 meshes are STL, which MuJoCo loads directly -- no Blender/pycollada step, and no
    materials to recover: MuJoCo's URDF reader carries each ``<visual><material><color rgba>``
    onto the geom, so the source colours survive the compile. The only Collada in PAL's tree
    (realsense2_description's d435.dae) is not referenced in this configuration.
    """
    meshdir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}

    def repl(match: re.Match) -> str:
        rel = match.group(1)
        pkg, sub = rel.split("/", 1)
        if pkg not in pkgs:
            raise RuntimeError(f"mesh from unpinned package {pkg!r}: {rel}")
        src = pkgs[pkg] / sub
        if not src.is_file():
            raise RuntimeError(f"mesh missing from the pinned source: {src}")
        # Flat unique name from the path within the package, so same-named links across the six
        # packages (head_2_link, base_link, …) cannot collide.
        stem = re.sub(r"[^A-Za-z0-9]+", "_", sub.rsplit(".", 1)[0]).strip("_").lower()
        dst = meshdir / f"{stem}{src.suffix.lower()}"
        if stem in seen and seen[stem] != src:
            raise RuntimeError(f"flat name {stem!r} collides: {seen[stem]} vs {src}")
        seen[stem] = src
        shutil.copy(src, dst)
        return f'filename="{dst.name}"'

    txt = re.sub(r'filename="package://([^"]*)"', repl, urdf.read_text())
    # `strippath=false` keeps our flat names; `balanceinertia` repairs the few links whose source
    # inertia tensor violates the triangle inequality; `fusestatic=false` keeps the fixed-joint
    # links as real bodies so sensor frames (lasers, camera, tool/grasping links) stay addressable.
    inject = (
        f'\n  <mujoco><compiler meshdir="{MESH_SUBDIR}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false" strippath="false" autolimits="true"/></mujoco>\n'
    )
    txt = re.sub(r"(<robot\b[^>]*>)", r"\1" + inject, txt, count=1)
    out = urdf.with_name("tiago_pro_mj.urdf")
    out.write_text(txt)
    print(f"  {len(seen)} meshes -> {meshdir}")
    return out


def compile_base(urdf: Path, work: Path) -> Path:
    """MuJoCo-compile the URDF to a base MJCF that postprocess() then rewrites."""
    import mujoco

    # meshdir in the injected block is relative to the URDF, so compile with the real package
    # meshdir reachable: run from the models/ dir.
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compile()  # fail here, not later, if the source tree is inconsistent
    base = work / "tiago_pro_base.xml"
    base.write_text(spec.to_xml())
    return base


def postprocess(base: Path, out: Path) -> None:
    """Everything URDF cannot express. See the module docstring and the port log."""
    root = ET.parse(base).getroot()
    root.set("model", MODEL)

    comp = root.find("compiler")
    comp.set("meshdir", MESH_SUBDIR)
    comp.set("angle", "radian")
    comp.set("autolimits", "true")

    # ---- defaults: visual / collision classes --------------------------------------------
    default = ET.Element("default")
    dvis = ET.SubElement(default, "default", {"class": f"{MODEL}_visual"})
    # density="0" is KEPT on visual geoms (see below), so it belongs in the class.
    #
    # No type= here: not every URDF visual is a mesh (the grippers' base_finger links are 1 mm
    # spheres), and forcing type="mesh" on those fails the compile with "must have valid meshid".
    # MuJoCo's URDF reader already writes an explicit type on every geom.
    ET.SubElement(dvis, "geom", {"contype": "0", "conaffinity": "0", "group": "2", "density": "0"})
    dcol = ET.SubElement(default, "default", {"class": f"{MODEL}_collision"})
    # contype=2/conaffinity=1: collides with the world (1) but not with itself (2&2 == 0). The
    # source collision primitives of neighbouring links overlap at rest, and self-collision
    # avoidance is out of scope for this port.
    ET.SubElement(
        dcol,
        "geom",
        {"group": "3", "rgba": "0.4 0.5 0.6 0.3", "condim": "3", "contype": "2", "conaffinity": "1"},
    )
    root.insert(list(root).index(comp) + 1, default)

    worldbody = root.find("worldbody")

    # ---- regroup geoms -------------------------------------------------------------------
    # MuJoCo's URDF reader marks visuals with density=0 + contype=0; everything else is the
    # source's collision geometry (convex meshes AND primitives -- boxes, spheres, cylinders).
    # Explicit attrs beat class defaults in MJCF, so pop them first. rgba is KEPT on visuals: it
    # is the real URDF colour, which is why this port needs no material-recovery step.
    #
    # density="0" is deliberately kept on visual geoms rather than stripped. robot-porting warns
    # that a zero-density mesh geom renders as its bounding sphere; that does NOT reproduce here
    # (checked by rendering base_link with and without it under MuJoCo 3.11 -- identical mesh), and
    # stripping it instead corrupts the mass: 14 of the 72 source links carry no <inertial>, so
    # MuJoCo derives their inertia from geometry and the visual meshes then contribute ~20 kg of
    # phantom mass (82.35 kg vs the URDF's 62.45 kg). Re-check the render if MuJoCo is upgraded.
    for g in worldbody.iter("geom"):
        visual = g.get("density") == "0" and g.get("contype") == "0"
        strip = ["contype", "conaffinity", "group", "density"]
        if not visual:
            strip.append("rgba")  # collision geoms take the class's translucent overlay colour
        for a in strip:
            g.attrib.pop(a, None)
        g.set("class", f"{MODEL}_{'visual' if visual else 'collision'}")

    bodies = {b.get("name"): b for b in root.iter("body")}

    # ---- wheels: near-frictionless load carriers -----------------------------------------
    # The real base runs 4 mecanum wheels. Their rollers are NOT modelled (see the port log's
    # drive-model decision): the holonomic motion comes from the planar actuators below, and the
    # wheel cylinders only carry the vertical load. mu=0.02 is the honest idealisation of an omni
    # wheel -- free to slide in every direction -- and keeps the planar drive authoritative.
    for w in WHEELS:
        for g in bodies[f"{w}_link"].findall("geom"):
            if g.get("type") == "cylinder":
                g.set("name", f"{w}_tyre")
                g.set("friction", "0.02 0.005 0.0001")
                g.set("priority", "2")
                g.set("rgba", "0.05 0.05 0.05 1")
        # Replace the source's wheel-joint damping (1.0 N.m.s/rad) + frictionloss (2.0 N.m) with an
        # armature.
        #
        # Those two are Gazebo stabilisation values, not physical ones, and they contradict the
        # source's own limits: at the base's top speed the damping term alone demands 13 N.m from a
        # joint whose <limit effort> is 6 N.m, so the wheel cannot reach the speed the same file
        # permits -- the observational wheel servos under-run and joint_states misreports wheel speed.
        #
        # But they were also what kept the wheel servos stable, because the bare wheel inertia is tiny
        # (~4e-4 kg.m^2): an explicit velocity servo needs kv < I/dt to be stable, i.e. kv < 0.2 here,
        # and the wheels spin chaotically without either. `armature` is the principled fix -- it is
        # the motor's rotor inertia reflected through the gearbox, which a real geared wheel drive
        # genuinely has and the source URDF simply omits. 0.05 kg.m^2 leaves kv=5 an order of
        # magnitude inside the stability bound. Substrate assumption; see port log A5.
        for j in bodies[f"{w}_link"].findall("joint"):
            j.attrib.pop("damping", None)
            j.attrib.pop("frictionloss", None)
            j.set("armature", "0.05")

    # ---- gripper pads: explicit rubber friction ------------------------------------------
    # The finger links that touch a grasped object were inheriting the generic collision class, i.e.
    # MuJoCo's default friction (1.0 sliding, 0.005 torsional, 0 rolling) with no `priority` -- so the
    # effective coefficients depended on whatever object was being gripped rather than on the gripper.
    # Real PRO pads are rubber, and `priority=3` makes the pad's values win outright so the pair is a
    # property of the gripper. Torsional friction matters here: the 0.005 default lets a held object
    # spin about the contact normal under its own weight.
    #
    # This is the right value to state and it is NOT a fix for the carry failure recorded in the port
    # log's Known issues -- measured, it does not change the outcome. Do not read it as a tuning knob
    # that resolved anything.
    #
    # STILL OPEN: these links collide via their vendor mesh CONVEX HULLS, which is what makes the pinch
    # 2-point (see the port log). The fix is a pad collision primitive, and it belongs on the
    # `fingertip_*` links -- they are the geometry that straddles the TCP. An attempt to put pads on
    # `inner_finger_*` instead (chosen because it carries the largest jaw-facing facet, 11 x 43 mm) was
    # WRONG and is recorded here so it is not repeated: that link is proximal, its pad sits ~60 mm
    # behind the TCP along the approach axis, and it can never reach an object at the grasp point.
    # Choose the pad link by what straddles the TCP, not by facet area.
    for side in ("left", "right"):
        for lr in ("left", "right"):
            for link in (f"gripper_{side}_fingertip_{lr}_link",
                         f"gripper_{side}_inner_finger_{lr}_link"):
                for g in bodies[link].findall("geom"):
                    if g.get("class") != f"{MODEL}_collision":
                        continue
                    # 0.7, matching roqsim_manipulation_assets' robotiq_2f85 pads (Menagerie's values,
                    # and the reference for a WORKING grasp in this substrate). 1.6 was tried and is
                    # too high once the world sets `impratio: 10` and an elliptic cone: the tangential
                    # capacity then exceeds what the angled pads can hold in shear and the object is
                    # extruded out sideways instead of gripped.
                    g.set("friction", "0.7 0.02 0.001")
                    g.set("priority", "3")
                    g.set("condim", "4")
                    # solref/solimp MUST be set here too, and this is not optional detail. MuJoCo
                    # resolves a contact pair from the HIGHER-priority geom alone -- friction,
                    # condim AND the solver parameters. So `priority=3` silently replaces whatever
                    # grasp tuning the OBJECT carries with this geom's defaults: measured, a
                    # graspable prop's solref of 0.005 became the collision class's 0.02, a 4x
                    # stiffer contact, and grasps that had been working stopped. Softened values
                    # belong on the pad because the pad is what now owns the pair.
                    g.set("solref", "0.004 1")
                    # solimp stiffer than the graspable props' own 0.9 0.95: a held object CREEPS
                    # downward through this gripper's contact under sustained load, because the pads
                    # meet it on an edge (see the port log) and a soft friction constraint drifts.
                    # Measured with a static arm and both pads in contact, 0.15 kg: 0.9/0.95 creeps at
                    # -8.6 mm/s, 0.95/0.99 at -3.7 mm/s. Stiffer still is worse, not better --
                    # 0.99/0.999 chatters and ejects the object outright.
                    g.set("solimp", "0.95 0.99 0.001")

    # ---- gripper linkage joints: armature + damping ---------------------------------------
    # The source URDF gives the finger linkage joints no armature and almost no damping, and their
    # effort limit is 0.1 N.m. A revolute joint on a small link therefore has near-zero effective
    # inertia -- the same condition that made the wheel servos chatter (A5) -- and under the stiff
    # contact solve a grasp needs (elliptic cone, impratio 10) the linkage rings and the two jaws grab
    # unevenly, so one pad loses contact and the object is pushed out sideways.
    #
    # Values follow roqsim_manipulation_assets' robotiq_2f85, the reference for a grasp that holds in this
    # substrate: armature 0.005 on the driven joint, 0.001 on the linkage followers, with the same
    # solimplimit/solreflimit hardness on their range limits.
    for side in ("left", "right"):
        for jname, arm, damp in (
            (f"gripper_{side}_finger_joint", "0.005", "0.1"),
            (f"gripper_{side}_finger_right_joint", "0.005", "0.1"),
            *[
                (f"gripper_{side}_{part}_{lr}_joint", "0.001", "0.01")
                for part in ("inner_finger", "outer_finger", "fingertip")
                for lr in ("left", "right")
            ],
        ):
            for j in root.iter("joint"):
                if j.get("name") != jname:
                    continue
                j.set("armature", arm)
                j.set("damping", damp)
                j.set("solimplimit", "0.95 0.99 0.001")
                j.set("solreflimit", "0.005 1")

    # ---- base_footprint: free joint + frames ---------------------------------------------
    # base_footprint is the URDF root and PAL's `base_frame_id`; it sits at floor level, so a free
    # joint here puts the wheels exactly on the ground at qpos z=0.
    base_fp = bodies["base_footprint"]
    base_fp.insert(0, ET.Comment(" mobile base: free joint (driven holonomically, see <actuator>) "))
    base_fp.insert(1, ET.Element("freejoint", {"name": "base_free"}))
    base_fp.insert(
        2,
        ET.Element(
            "camera",
            {"name": "track", "mode": "track", "pos": "-2.8 -2.8 2.4",
             "xyaxes": "0.7 -0.7 0 0.3 0.3 0.9"},
        ),
    )

    # IMU site on the source's own base_imu_link (100 Hz in base_sensors.urdf.xacro) -- no invented
    # mount pose. The odom/gyro/accelerometer sensors below all read this site.
    bodies["base_imu_link"].insert(
        0, ET.Element("site", {"name": "base_imu", "size": "0.01", "rgba": "0 0 0 0"})
    )

    base_link = bodies["base_link"]
    for i, (name, (pos, euler)) in enumerate(LIDAR_SITES.items()):
        base_link.insert(
            i,
            ET.Element(
                "site",
                {"name": name, "pos": pos, "euler": euler, "size": "0.006", "rgba": "1 0 0 0.6"},
            ),
        )

    # Head RGB-D camera on the source's camera link. fovy 42.5 deg / 640x480 matches roqsim_sensors'
    # d435 model (the D435 colour stream's vertical FOV), so the two agree on one sensor.
    # xyaxes puts -z (MuJoCo's view direction) along the link's +x, i.e. looking where the head faces.
    bodies["head_front_camera_link"].insert(
        0,
        ET.Element(
            "camera",
            {"name": "head_cam", "xyaxes": "0 -1 0 0 0 1", "fovy": "42.5", "resolution": "640 480"},
        ),
    )

    # ---- actuators -----------------------------------------------------------------------
    act = ET.SubElement(root, "actuator")
    act.append(
        ET.Comment(
            " Holonomic planar drive. The OMNI base is a 4-wheel mecanum platform driven by PAL's "
            "omni_drive_controller; its rollers are not modelled, so vx/vy/wz are applied directly "
            "to the free joint: exactly the base_x/base_y/base_tau actuators PAL left commented "
            "out in omni_base_description/mujoco/mj_tags.xacro. gear on a free joint acts in the "
            "WORLD frame (verified), so the `omni_drive` plugin rotates the body-frame command by "
            "the base yaw before writing ctrl. ctrlrange = PAL's mobile_base_controller limits "
            "(vx/vy +-1.0 m/s, wz +-2.09 rad/s), so the model saturates where the stack does. "
        )
    )
    for name, gear, kv, ctrl, force in (
        ("base_vx", "1 0 0 0 0 0", 4000, 1.0, 800),
        ("base_vy", "0 1 0 0 0 0", 4000, 1.0, 800),
        ("base_wz", "0 0 0 0 0 1", 1200, 2.09, 400),
    ):
        ET.SubElement(
            act,
            "velocity",
            {"name": name, "joint": "base_free", "gear": gear, "kv": str(kv),
             "ctrlrange": f"-{ctrl} {ctrl}", "forcerange": f"-{force} {force}"},
        )

    act.append(
        ET.Comment(
            " Wheel velocity servos. Observational, not motive: `omni_drive` spins them from the "
            "mecanum inverse kinematics so the visual and joint_states are right, but at mu=0.02 "
            "they transmit almost no traction. forcerange is the URDF effort (6 N.m). "
        )
    )
    for j in WHEEL_JOINTS:
        ET.SubElement(
            act,
            "velocity",
            {"name": j.replace("_joint", "_motor"), "joint": j, "kv": "5",
             "ctrlrange": "-60 60", "forcerange": "-6 6"},
        )

    act.append(ET.Comment(" torso / head / arm / gripper position servos "))
    for j in TORSO + HEAD + ARM_L + ARM_R + GRIP_MASTER:
        kp, fr = POS_GAINS[j]
        ET.SubElement(
            act,
            "position",
            {"name": j.replace("_joint", "_pos"), "joint": j, "kp": str(kp), "dampratio": "1",
             "forcerange": f"-{fr} {fr}", "inheritrange": "1"},
        )

    # ---- equalities: the URDF gripper mimics ---------------------------------------------
    eq = ET.SubElement(root, "equality")
    eq.append(
        ET.Comment(
            " Gripper linkage. MuJoCo ignores URDF <mimic>, so the DRIVEN branch is reproduced as joint "
            "equalities, and the four-bar's closing pivot (which a URDF tree cannot express at all) is a "
            "`connect`. See MIMIC / LOOP_FINGERS in the build script for why the URDF's fitted "
            "fingertip multiplier is not reproduced. "
        )
    )
    for follower, (master, mult) in MIMIC.items():
        ET.SubElement(
            eq,
            "joint",
            {"joint1": follower, "joint2": master, "polycoef": f"0 {mult} 0 0 0",
             "solimp": "0.95 0.99 0.001"},
        )
    for fingertip, outer in LOOP_FINGERS:
        ET.SubElement(
            eq,
            "connect",
            {"body1": fingertip, "body2": outer, "anchor": LOOP_ANCHOR,
             "solimp": "0.95 0.99 0.001"},
        )

    # ---- sensors -------------------------------------------------------------------------
    sen = ET.SubElement(root, "sensor")
    for tag, nm in (("framepos", "base_pos"), ("framequat", "base_quat"),
                    ("framelinvel", "base_linvel"), ("frameangvel", "base_angvel")):
        ET.SubElement(sen, tag, {"name": nm, "objtype": "site", "objname": "base_imu"})
    ET.SubElement(sen, "gyro", {"name": "imu_gyro", "site": "base_imu"})
    ET.SubElement(sen, "accelerometer", {"name": "imu_acc", "site": "base_imu"})
    for j in WHEEL_JOINTS:
        ET.SubElement(sen, "jointvel", {"name": j.replace("_joint", "_vel"), "joint": j})
    for j in TORSO + HEAD + ARM_L + ARM_R + GRIP_MASTER:
        ET.SubElement(sen, "jointpos", {"name": j.replace("_joint", "_state"), "joint": j})

    # ---- home keyframe -------------------------------------------------------------------
    # Ordered by the compiled joint order, so it survives a re-generation that reorders joints.
    order = [j.get("name") for j in root.iter("joint") if j.get("name")]
    qpos = ["0 0 0 1 0 0 0"] + [f"{HOME.get(j, 0.0)}" for j in order]
    kf = ET.SubElement(root, "keyframe")
    kf.append(ET.Comment(" PAL's `home` motion, final waypoint (see HOME in the build script) "))
    ET.SubElement(kf, "key", {"name": "home", "qpos": " ".join(qpos)})

    xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    out.write_text("\n".join(line for line in xml.splitlines() if line.strip()) + "\n")
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-work", action="store_true", help="keep the intermediate URDF/base MJCF")
    args = ap.parse_args()

    print("1/4 resolving pinned PAL sources")
    pkgs = fetch_sources()
    work = Path(tempfile.mkdtemp(prefix="tiago_pro_port_"))
    try:
        print("2/4 expanding xacro")
        urdf = expand_xacro(pkgs, work)
        print("3/4 flattening meshes + compiling")
        mj_urdf = flatten_meshes(urdf, pkgs, MODEL_DIR / MESH_SUBDIR)
        # Compile from the model's own folder so the injected relative meshdir resolves to the real
        # package path.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        staged = MODEL_DIR / mj_urdf.name
        shutil.copy(mj_urdf, staged)
        try:
            base = compile_base(staged, work)
        finally:
            staged.unlink(missing_ok=True)
        print("4/4 post-processing")
        out = MODEL_DIR / f"{MODEL}.xml"
        postprocess(base, out)
        add_pad_boxes(out)
    finally:
        if args.keep_work:
            print(f"  work dir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)




# ---------------------------------------------------------------------------------------------
# Pad boxes, derived from the written model (a second pass, because it needs compiled kinematics).
# ---------------------------------------------------------------------------------------------
#: Per-pad box half-extents and friction, copied from roqsim_manipulation_assets' robotiq_2f85 -- the
#: reference for a grasp that actually holds in this substrate (Menagerie's values). TWO stacked pads
#: per finger, not one: two separated contact patches per jaw is what resists the rotation that a
#: single patch cannot, and it is why that gripper carries a load and a convex hull does not.
PAD_HALF = (0.011, 0.004, 0.009375)
PAD_FRICTION = ("0.7", "0.6")          # pad1 (distal), pad2 (proximal), as in the reference
PAD_SOLIMP = "0.95 0.99 0.001"
PAD_SOLREF = "0.004 1"
#: Gripper opening the pad frame is derived at: the working grasp aperture, where the jaws are as
#: close to parallel as this linkage gets.
PAD_REF_TRAVEL = 0.020


def add_pad_boxes(model_path: Path) -> None:
    """Replace each fingertip's convex-hull collision with two flat pad boxes.

    Why this is a separate pass: the pad frame is not expressible in the fingertip's own mesh axes.
    The PAL fingertip's object-facing surface sits ~39 deg off its body axes, so an axis-aligned box
    does not fit it -- the box has to be built from the **pad-separation axis**, which is only known
    once the kinematics are compiled. So this loads the model just written, derives each pad's pose
    from the TCP frame at a reference opening, and rewrites the XML.

    Choosing the LINK matters and is easy to get wrong: pick the one that BRACKETS the TCP, not the one
    with the largest flat facet. Here `fingertip_*` spans -45..+12 mm about the TCP while
    `inner_finger_*` -- which carries a much nicer 11 x 43 mm face -- sits ~60 mm behind it and can
    never touch an object at the grasp point.

    The inward direction is asserted below rather than assumed. This pass was once disabled on the
    belief that the `lr == "left"` sign put that pad on the finger's BACK; measuring the generated
    geoms disproved it -- both jaws come out symmetric, centres at +-13.4 mm in TCP y with the normals
    facing each other and a 18.8 mm face gap at PAD_REF_TRAVEL. The assertion is what keeps that
    question answered: if a future upstream link rename or frame change does flip a side, this raises
    instead of silently shipping a jaw that cannot touch anything.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    qadr = {}
    for jn in ("finger_joint", "finger_right_joint", "inner_finger_left_joint",
               "inner_finger_right_joint"):
        for side in ("left", "right"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_{side}_{jn}")
            qadr[(side, jn)] = int(m.jnt_qposadr[j])
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for side in ("left", "right"):
        d.qpos[qadr[(side, "finger_joint")]] = PAD_REF_TRAVEL
        d.qpos[qadr[(side, "finger_right_joint")]] = 0.22 * PAD_REF_TRAVEL
        for jn in ("inner_finger_left_joint", "inner_finger_right_joint"):
            d.qpos[qadr[(side, jn)]] = -8.28 * PAD_REF_TRAVEL
    mujoco.mj_forward(m, d)

    tree = ET.parse(model_path)
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    mesh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "meshes_fingertip")
    va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
    verts = m.mesh_vert[va : va + vn].astype(float)

    for side in ("left", "right"):
        tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{side}_grasping_link")
        r_tcp = d.xmat[tcp].reshape(3, 3)
        for lr in ("left", "right"):
            name = f"gripper_{side}_fingertip_{lr}_link"
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            r_b = d.xmat[bid].reshape(3, 3)
            # Pad-separation axis, pointing from THIS finger toward the other one, in the body frame.
            # The `left` finger sits on the TCP's -y side, so its inward direction is +y there.
            sign = 1.0 if lr == "left" else -1.0
            n_world = r_tcp @ np.array([0.0, sign, 0.0])
            # Inward must point from this finger TOWARD the TCP. A pad built on the wrong sign sits on
            # the finger's back, where it can never touch an object -- and the jaw still LOOKS right in
            # a render, so nothing downstream would catch it.
            toward_tcp = float(n_world @ (d.xpos[tcp] - d.xpos[bid]))
            if toward_tcp <= 0.0:
                raise SystemExit(
                    f"{name}: inward axis points AWAY from the TCP (projection {toward_tcp:+.5f} m). "
                    "The fingertip is not on the TCP side this code assumes -- re-derive the sign from "
                    "the measured body offset instead of the left/right name."
                )
            n = r_b.T @ n_world
            n /= np.linalg.norm(n)
            # Long axis of the pad = the TCP's z (across the jaws), orthogonalised against n.
            ez = r_b.T @ (r_tcp @ np.array([0.0, 0.0, 1.0]))
            ez -= n * float(ez @ n)
            ez /= np.linalg.norm(ez)
            ex = np.cross(n, ez)
            rot = np.column_stack([ex, n, ez])          # box local y == pad normal
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).reshape(-1))
            # Inner surface of this fingertip along n, and the TCP's own position in these axes, so the
            # pads straddle the grasp point rather than sitting wherever the mesh happens to be centred.
            inner = float((verts @ n).max())
            tcp_local = r_b.T @ (d.xpos[tcp] - d.xpos[bid])
            cx, cz = float(tcp_local @ ex), float(tcp_local @ ez)

            body = bodies[name]
            for g in body.findall("geom"):
                if g.get("class") == f"{MODEL}_collision":
                    g.set("contype", "0")       # the hull stops colliding; the boxes take over
                    g.set("conaffinity", "0")
            for k, (tag, mu) in enumerate(zip(("pad1", "pad2"), PAD_FRICTION, strict=True)):
                # Two pads stacked along the jaw's long axis, offset either side of the grasp point.
                off = (0.5 - k) * 2.0 * PAD_HALF[2]
                c = ex * cx + ez * (cz + off) + n * (inner - PAD_HALF[1])
                body.insert(0, ET.Element("geom", {
                    "name": f"gripper_{side}_{lr}_{tag}",
                    "type": "box",
                    "mass": "0",
                    "size": f"{PAD_HALF[0]} {PAD_HALF[1]} {PAD_HALF[2]}",
                    "pos": f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}",
                    "quat": f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}",
                    "friction": f"{mu} 0.02 0.001",
                    "priority": "1",
                    "condim": "4",
                    "solimp": PAD_SOLIMP,
                    "solref": PAD_SOLREF,
                    "group": "3",
                    "contype": "2",
                    "conaffinity": "1",
                    "rgba": "0.15 0.15 0.16 1",
                }))
    xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    model_path.write_text(
        with_headline("\n".join(l for l in xml.splitlines() if l.strip()), HEADLINE) + "\n"
    )
    print(f"  added 8 pad boxes (2 per finger), fingertip hulls no longer collide")


if __name__ == "__main__":
    main()
