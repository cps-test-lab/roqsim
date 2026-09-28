#!/usr/bin/env python3
"""Build ``models/unitree_g1_dex1.xml`` -- the 29-DoF Unitree G1 with Dex1 parallel grippers.

The existing ``unitree_g1.xml`` is leg-only: its shoulders, elbows and wrists are rigid geoms welded
to ``base_link``, so the platform cannot manipulate anything. This builds the manipulation variant:
the same 12-DoF torque-driven leg chain the pretrained walking policy expects, plus 3 waist + 2x7 arm
joints under position control and a 2-DoF Dex1 parallel gripper per side.

Source (see THIRD_PARTY.md): ``unitree_ros`` @ f3772ce, ``robots/g1_description``, BSD-3-Clause.

Built from ``g1_29dof_mode_15_with_dex1_1.urdf`` -- ONE upstream revision -- rather than grafting the
gripper onto the shipped ``g1_29dof_rev_1_0.xml`` MJCF, because the two disagree: the MJCF places
``left_wrist_yaw_joint`` at x=0.046 and the Dex1 URDF at x=0.051 (different wrist hardware, the
``_5010`` parts). Mixing them silently mismatches the gripper mount by 5 mm.

Five upstream and conversion quirks this handles, each of which is silently wrong if taken at face
value:

  * The URDF carries BOTH ``<side>_rubber_hand`` and the Dex1 gripper on the same wrist mount
    (palm joint at 0.0415 0.003 0, gripper base at 0.0415 0 0). The rubber hand is dropped; keeping it
    would ride 0.17 kg of phantom hand inside each gripper.
  * ``<mujoco><compiler meshdir="meshes"/>`` combines with mesh filenames that are ALREADY prefixed
    ``meshes/``, so MuJoCo looks for ``meshes/meshes/*.STL``. The prefix is stripped.
  * The floating base is commented out ("uncomment when convert to mujoco"). Without it MuJoCo fuses
    ``pelvis`` into the world body at parse time and the root link vanishes.
  * ``MjSpec.to_xml()`` of a URDF import whose fixed links were fused into their parents writes the
    fused geoms without the fixed joint's offset and the parent's inertial without the fused mass:
    ``head_link`` and ``logo_link`` land 44 mm high on ``torso_link`` with the head's 1.036 kg gone,
    and each Dex1 base and its fingers 41.5 mm inside the wrist with 0.191 kg gone. Static links are
    therefore kept as welded bodies (``fusestatic`` off), and :func:`verify_against_urdf` compares
    every mesh geom's pose, the total mass and the centre of mass with MuJoCo's direct compile of
    the same URDF. The links that carry nothing (the IMU, camera and lidar frames) are removed.
  * Both finger joints are prismatic on OPPOSING axes and their origins coincide at q=0 -- from which
    "q=0 is fully closed" follows and is FALSE. The pads sit ~23 mm outboard each, so q=0 already
    stands 45.9 mm open and the whole useful closing range is upstream's NEGATIVE half. Measured, not
    inferred: see the aperture table below. Clamping that half away as a "crossed" state leaves a
    gripper that cannot grip anything.

The head carries the Livox Mid-360 the ``unitree_g1`` manifest mounts at ``mid360_joint``; its mesh is
cut with the sensor's housing and field by ``g1_head_window.py``, so the scan leaves the head.

Run:  python external/convert/build_g1_dex1.py [--src DIR] [--check]
Emits models/unitree_g1_dex1.xml and copies the referenced meshes into models/meshes/unitree_g1_dex1/.
``--check`` rebuilds in memory and fails if the committed model or its cut head mesh differs.
Needs ``manifold3d`` for the head cut (see g1_head_window.py).

A caller planning with this robot can ask for the matching MoveIt-side URDF in the same pass:

    python external/convert/build_g1_dex1.py --moveit-urdf <pkg>/urdf/unitree_g1_dex1.urdf \
                                             --mesh-package <ament_package_name>

Both come from the same prepared source, so the planner and the simulator cannot disagree. Where that
URDF goes is the caller's business -- a MoveIt configuration belongs to the task that plans with it,
not to the substrate, so this script neither knows nor defaults to a location.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from g1_head_window import cut_head, stl_bytes
from model_headline import with_headline
from sources import resolve_source

# Pinned upstream revision -- must match the table in roqsim_humanoid/THIRD_PARTY.md.
HEADLINE = (
    "Unitree G1, 29-DoF manipulation variant: the 12-DoF walking legs plus waist, arms and a "
    "Dex1 parallel gripper per side."
)

UNITREE_ROS_COMMIT = "f3772ce54c56ef2d34c6aee8100bc768896c7d19"
UNITREE_ROS_URL = "https://github.com/unitreerobotics/unitree_ros"
URDF_NAME = "g1_29dof_mode_15_with_dex1_1.urdf"

# The 12 leg joints in policy order. Must stay byte-identical to roqsim_humanoid.plugins.g1_locomotion
# LEG_JOINTS: the pretrained motion.pt indexes its observation and action vectors by this exact
# sequence, and these keep <motor> (torque) actuators so that plugin drives them unchanged.
LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

ARM_JOINTS = tuple(
    f"{side}_{j}"
    for side in ("left", "right")
    for j in (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
    )
)

# Position-actuator gains by joint group. STARTING values, not vendor data: upstream ships no arm PD
# gains (the Menagerie G1 README says outright that its position actuators "need tuning"). Scaled to
# each group's actuatorfrcrange -- the waist carries the whole upper body, the wrists have only
# +-5 Nm. These are a substrate artifact and are recorded as such in the port log; expect to revisit
# them after the drive test.
GAINS = {
    "waist": (300.0, 10.0),
    "shoulder": (120.0, 4.0),
    "elbow": (100.0, 3.0),
    "wrist_roll": (40.0, 1.5),
    "wrist": (20.0, 1.0),
}

# Dex1 gripper. Both finger joints are prismatic on OPPOSING axes, so aperture grows with q on both
# sides: aperture(q) = PAD_GAP_AT_ZERO + 2q. MEASURED from the collision-mesh vertices, because the
# body origins are coincident at q=0 and reasoning from those says the fingers touch there, which is
# wrong -- the pads sit ~23 mm outboard each, so q=0 already stands 45.9 mm open. The whole useful
# closing range is therefore the NEGATIVE half of upstream's limits; clamping it away as a "crossed"
# state leaves a gripper that cannot grip anything.
#
#   q = -0.0200 -> 5.9 mm aperture (closed)
#   q =  0.0000 -> 45.9 mm
#   q = +0.0245 -> 94.9 mm (fully open)
FINGER_OPEN = 0.0245
FINGER_CLOSE = -0.02
PAD_GAP_AT_ZERO = 0.0459  # measured; documented here so the manifest's box sizing is traceable
# Tool centre point between the pads, in the wrist_yaw frame: the pad span of the finger collision
# meshes at q=0, x[0.1185, 0.1848], z[-0.0145, 0.0145], centred on y. This is the frame a grasp is
# planned to and MoveIt's end-effector link, so it belongs in the model rather than being re-derived
# by every caller. :func:`verify_tcp` re-measures it on every build.
TCP_POS = (0.1517, 0.0, 0.0)
#: How far the measured pad centre may sit from TCP_POS before the build refuses.
TCP_TOLERANCE = 0.0005
# The tendon sums both fingers with coef -1, so its length is -(q1+q2). The sign matters:
# roqsim_manipulation.arm_controller maps its configured `gripper_open` onto the actuator's ctrlrange LOW
# end (see its set_gripper docstring, and gen3 where Robotiq ctrl 0 == open). With coef +1 the low end
# would be the closed end and a GripperCommand "open" would clamp shut.
TENDON_COEF = -1.0

# Foot contact: four small spheres per sole, transplanted verbatim from unitree_g1.xml so the leg
# contact model the walking policy currently runs against is unchanged. Upstream models each sole as
# a single sphere, which gives a foot no yaw or roll friction footprint to stand on.
FOOT_SPHERES = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)
FOOT_SPHERE_SIZE = 0.005

# Standing height of the pelvis, matching unitree_g1.xml (and upstream's own 29-DoF MJCF).
BASE_HEIGHT = 0.793

#: The head mesh the Mid-360's opening is cut into, and the file the cut is written to.
HEAD_MESH = "head_link"
HEAD_WINDOW_FILE = "head_link_mid360_window.STL"

#: Welded links with no geometry and no mass that the URDF declares only as frames. Removed from the
#: MJCF, which the build checks: a link listed here that carried anything would be refused.
FRAME_ONLY_LINKS = ("imu_in_pelvis", "imu_in_torso", "d435_link", "mid360_link")

#: Tolerances of the comparison with MuJoCo's direct URDF compile: float round trips through XML.
POSE_TOLERANCE = 1e-5
MASS_TOLERANCE = 1e-6


def gains_for(joint: str) -> tuple[float, float]:
    if joint in WAIST_JOINTS:
        return GAINS["waist"]
    if "shoulder" in joint:
        return GAINS["shoulder"]
    if "elbow" in joint:
        return GAINS["elbow"]
    if "wrist_roll" in joint:
        return GAINS["wrist_roll"]
    return GAINS["wrist"]


def prepare_urdf(text: str) -> str:
    """Apply the three URDF fixups described in the module docstring."""
    for side in ("left", "right"):
        # Drop the rubber hand: its link AND the fixed joint mounting it, so no orphan remains.
        text, n_joint = re.subn(
            rf'<joint name="{side}_hand_palm_joint".*?</joint>\s*', "", text, flags=re.S
        )
        text, n_link = re.subn(
            rf'<link name="{side}_rubber_hand">.*?</link>\s*', "", text, flags=re.S
        )
        if not (n_joint and n_link):
            raise RuntimeError(
                f"{side}_rubber_hand / {side}_hand_palm_joint not found in the URDF "
                f"(joint={n_joint}, link={n_link}). Upstream layout changed -- re-check the pin "
                f"before trusting the mass properties."
            )

    # meshdir="meshes" + filename="meshes/x.STL" would resolve to meshes/meshes/x.STL.
    if 'filename="meshes/' not in text:
        raise RuntimeError(
            "URDF mesh filenames are not 'meshes/'-prefixed; the fixup is now wrong."
        )
    text = text.replace('filename="meshes/', 'filename="')

    # Enable the floating base upstream leaves commented out, else pelvis is fused into the world.
    before = text
    text = text.replace('<!-- <link name="world"></link>', '<link name="world"></link>', 1)
    text = text.replace("  </joint> -->", "  </joint>", 1)
    if text == before:
        raise RuntimeError(
            "Could not enable the commented-out floating_base_joint; without it MuJoCo fuses "
            "pelvis into the world body and there is no root link to rename."
        )
    return text


def urdf_to_mjcf(urdf_path: Path) -> tuple[str, mujoco.MjModel]:
    """``(MJCF text, MuJoCo's direct compile)`` of the URDF, static links kept as welded bodies.

    The direct compile fuses static links and is the reference :func:`verify_against_urdf` checks
    the written model against: its fusion is right, only ``to_xml()`` of a fused spec is not.
    """
    reference = mujoco.MjSpec.from_file(str(urdf_path)).compile()
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.compiler.fusestatic = False
    spec.compile()  # fail here, on the untouched conversion, rather than after our edits
    return spec.to_xml(), reference


def _mesh_geom_poses(model: mujoco.MjModel, root: str) -> list[tuple]:
    """Every mesh geom as ``(mesh, group, position, rotation)`` relative to *root*, sorted."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    rmat = data.xmat[rid].reshape(3, 3)
    out = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
        pos = rmat.T @ (data.geom_xpos[g] - data.xpos[rid])
        rot = rmat.T @ data.geom_xmat[g].reshape(3, 3)
        out.append((name, int(model.geom_group[g]), pos, rot))
    return sorted(out, key=lambda e: (e[0], e[1], tuple(np.round(e[2], 4))))


def verify_against_urdf(model: mujoco.MjModel, reference: mujoco.MjModel, root: str) -> None:
    """Refuse a model whose geometry or mass differs from MuJoCo's direct compile of the URDF.

    Only geoms and mass this build does not change on purpose are compared: the head, whose mesh is
    cut, and the feet, whose contact spheres are replaced, are primitives or named exceptions.
    """
    ours = [e for e in _mesh_geom_poses(model, "base_link") if e[0] != HEAD_MESH]
    ref = [e for e in _mesh_geom_poses(reference, root) if e[0] != HEAD_MESH]
    if [e[:2] for e in ours] != [e[:2] for e in ref]:
        raise RuntimeError("the built model's mesh geoms are not the URDF's")
    for a, b in zip(ours, ref, strict=True):
        if not (
            np.allclose(a[2], b[2], atol=POSE_TOLERANCE)
            and np.allclose(a[3], b[3], atol=POSE_TOLERANCE)
        ):
            raise RuntimeError(
                f"mesh geom {a[0]!r} (group {a[1]}) sits at {np.round(a[2], 5)} relative to "
                f"base_link, the URDF puts it at {np.round(b[2], 5)}"
            )
    if abs(model.body_mass.sum() - reference.body_mass.sum()) > MASS_TOLERANCE:
        raise RuntimeError(
            f"total mass {model.body_mass.sum():.6f} kg, the URDF's is {reference.body_mass.sum():.6f} kg"
        )
    com = []
    for m, r in ((model, "base_link"), (reference, root)):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r)
        com.append(d.xmat[rid].reshape(3, 3).T @ (d.subtree_com[rid] - d.xpos[rid]))
    if not np.allclose(com[0], com[1], atol=POSE_TOLERANCE):
        raise RuntimeError(f"centre of mass {com[0]} relative to base_link, the URDF's is {com[1]}")


def verify_tcp(model: mujoco.MjModel) -> None:
    """Refuse a TCP_POS that is not the centre of the finger pads at q=0, measured on *model*."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        wpos, wmat = data.xpos[wid], data.xmat[wid].reshape(3, 3)
        pads = []
        for g in range(model.ngeom):
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not (
                mesh.startswith("dex1_col_") and body.startswith(f"{side}_dex1_finger")
            ):
                continue
            mid = model.geom_dataid[g]
            vert = model.mesh_vert[
                model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
            ]
            world = vert @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            pads.append((world - wpos) @ wmat)
        if len(pads) != 2:
            raise RuntimeError(f"{side}: expected two finger collision meshes, found {len(pads)}")
        span = np.vstack(pads)
        centre = (span.min(axis=0) + span.max(axis=0)) / 2
        if not np.allclose(centre, TCP_POS, atol=TCP_TOLERANCE):
            raise RuntimeError(
                f"{side}: the pads are centred at {np.round(centre, 4)} in the wrist_yaw frame, "
                f"TCP_POS says {TCP_POS}. Re-measure before changing the pin."
            )


def find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"body {name!r} not in the converted MJCF")
    return body


def remove_frame_only_links(root: ET.Element) -> None:
    """Remove the welded links the URDF declares only as frames; refuse one that carries anything."""
    parents = {child: parent for parent in root.iter() for child in parent}
    for name in FRAME_ONLY_LINKS:
        body = find_body(root, name)
        if len(body):
            raise RuntimeError(f"{name} carries {[c.tag for c in body]}; it is not only a frame")
        parents[body].remove(body)


def apply_roqsim_conventions(xml: str) -> ET.ElementTree:
    root = ET.fromstring(xml)
    root.set("model", "unitree_g1_dex1")

    # -- root body: pelvis -> base_link / base_free, at standing height ---------------------------
    pelvis = find_body(root, "pelvis")
    pelvis.set("name", "base_link")
    pelvis.set("pos", f"0 0 {BASE_HEIGHT}")
    free = pelvis.find("joint[@type='free']")
    if free is None:
        raise RuntimeError("pelvis has no free joint; the floating-base fixup did not take effect")
    free.set("name", "base_free")
    free.set("limited", "false")
    free.set("actuatorfrclimited", "false")

    remove_frame_only_links(root)

    # -- joint defaults, matching unitree_g1.xml ---------------------------------------------------
    default = ET.Element("default")
    ET.SubElement(default, "joint", {"damping": "0.001", "armature": "0.01", "frictionloss": "0.1"})
    root.insert(list(root).index(root.find("compiler")) + 1, default)

    # -- head: the mesh with the Mid-360's opening (g1_head_window.py) -----------------------------
    head = root.find(f"asset/mesh[@name='{HEAD_MESH}']")
    if head is None:
        raise RuntimeError(f"mesh {HEAD_MESH!r} not in the converted MJCF")
    head.set("file", HEAD_WINDOW_FILE)

    # -- feet: replace upstream's single sphere per sole with the roqsim four-sphere footprint --------
    for side in ("left", "right"):
        foot = find_body(root, f"{side}_ankle_roll_link")
        for geom in [g for g in foot.findall("geom") if g.get("contype") != "0"]:
            foot.remove(geom)
        for x, y, z in FOOT_SPHERES:
            ET.SubElement(
                foot,
                "geom",
                {"size": str(FOOT_SPHERE_SIZE), "pos": f"{x} {y} {z}", "rgba": "0.2 0.2 0.2 1"},
            )

    # -- gripper: verify the finger range, add the TCP site, then couple + actuate ------------------
    for side in ("left", "right"):
        for idx in (1, 2):
            jname = f"{side}_dex1_finger_joint_{idx}"
            joint = root.find(f".//joint[@name='{jname}']")
            if joint is None:
                raise RuntimeError(f"gripper joint {jname!r} missing from the converted MJCF")
            lo, hi = (float(v) for v in (joint.get("range") or "0 0").split())
            if (lo, hi) != (FINGER_CLOSE, FINGER_OPEN):
                raise RuntimeError(
                    f"{jname} range is [{lo}, {hi}], expected [{FINGER_CLOSE}, {FINGER_OPEN}]. The "
                    f"gripper aperture mapping and the manifest's gripper_open/gripper_close are "
                    f"derived from that range -- re-measure before changing the pin."
                )
        # Tool centre point between the pads: the frame a grasp is planned to, and MoveIt's
        # end-effector link origin. Sited on wrist_yaw, which the gripper base is welded to.
        wrist = find_body(root, f"{side}_wrist_yaw_link")
        wrist.append(
            ET.Element(
                "site",
                {"name": f"{side}_grasp", "pos": " ".join(map(str, TCP_POS)), "size": "0.005"},
            )
        )

    equality = ET.SubElement(root, "equality")
    tendon = ET.SubElement(root, "tendon")
    actuator = ET.Element("actuator")

    # Legs first and in policy order: torque actuators, exactly as unitree_g1.xml, so g1_locomotion
    # resolves and drives them with no change.
    for joint in LEG_JOINTS:
        ET.SubElement(actuator, "motor", {"name": joint, "joint": joint})

    # Waist + arms: position servos, so arm_controller's position hold applies.
    for joint in (*WAIST_JOINTS, *ARM_JOINTS):
        jel = root.find(f".//joint[@name='{joint}']")
        if jel is None:
            raise RuntimeError(f"joint {joint!r} missing from the converted MJCF")
        kp, kv = gains_for(joint)
        attrs = {"name": joint, "joint": joint, "kp": str(kp), "kv": str(kv)}
        if jrange := jel.get("range"):
            attrs["ctrlrange"] = jrange  # never command outside the mechanical limit
        ET.SubElement(actuator, "position", attrs)

    # Gripper: one tendon summing both fingers, one actuator, one equality keeping them symmetric.
    # Presence of this NON-JOINT (tendon) actuator is exactly what makes arm_controller treat the arm
    # as gripper-equipped and expose a GripperCommand action -- no config flag involved.
    for side in ("left", "right"):
        j1, j2 = f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"
        ET.SubElement(equality, "joint", {"joint1": j1, "joint2": j2, "polycoef": "0 1 0 0 0"})
        fixed = ET.SubElement(tendon, "fixed", {"name": f"{side}_dex1_split"})
        ET.SubElement(fixed, "joint", {"joint": j1, "coef": str(TENDON_COEF)})
        ET.SubElement(fixed, "joint", {"joint": j2, "coef": str(TENDON_COEF)})
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{side}_dex1_gripper",
                "tendon": f"{side}_dex1_split",
                # Tendon length is -(q1+q2), so the OPEN extreme is the most negative: low end ==
                # fully open, which is the end arm_controller maps `gripper_open` to.
                "ctrlrange": f"{2 * FINGER_OPEN * TENDON_COEF} {2 * FINGER_CLOSE * TENDON_COEF}",
                # Stiff and force-limited, which is how a real gripper grasps: the servo saturates
                # against `forcerange` (the URDF's 20 N finger effort limit) rather than being told a
                # gentle position. Both numbers are set by how lower ones fail:
                #   * kp=200 gives only kp*err = 2 N at a 10 mm over-closure, so friction (~4.8 N) barely
                #     matches the 0.5 kg box's weight (4.9 N) and the parcel slides out of the jaws. At
                #     kp=2000 the same command saturates at 20 N -> ~48 N of friction.
                #   * kv=5 is underdamped: the servo overshoots the commanded aperture by ~9 mm, which
                #     squeezes a 40 mm box to 28 mm and extrudes it sideways before settling on target.
                #     kv=40 is near-critical for the ~0.17 kg of moving finger (2*sqrt(kp*m) ~= 37).
                # The failure in both cases looks like insufficient friction and is not.
                "kp": "2000",
                "kv": "40",
                "forcerange": "-20 20",
            },
        )

    root.append(actuator)
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def urdf_for_moveit(prepared_urdf: str, mesh_package: str) -> ET.ElementTree:
    """The MoveIt-side URDF, from the SAME prepared source as the MJCF.

    Emitted by this script rather than hand-maintained because MoveIt plans against the URDF and the
    trajectory executes against the MJCF: if the two disagree by even a link offset, MoveIt plans
    collision-free paths that collide in the sim, and the failure is attributed to the controller. One
    source, one revision, one run.

    Differences from the MJCF, all of them necessary rather than incidental:

      * ``pelvis`` -> ``base_link`` (the same rename), but NO floating joint -- MoveIt attaches the
        robot to the world with an SRDF ``virtual_joint``, and a URDF floating joint would double it.
      * Mesh references become ``package://<mesh_package>/meshes/...``. The meshes themselves are
        not copied: the ament package installs them from ``roqsim_humanoid``'s vendored set at build time,
        so there is exactly one copy of 19 MB of STLs in the tree. The head keeps upstream's uncut
        mesh: the planner's collision model is not what a lidar sees through.
      * The MJCF's roqsim-only additions (the four-sphere foot contacts, actuators, tendons) have no
        URDF equivalent and are simply absent -- MoveIt needs kinematics, limits and collision
        geometry, none of which they affect.
    """
    root = ET.fromstring(prepared_urdf)
    root.set("name", "unitree_g1_dex1")

    for link in root.iter("link"):
        if link.get("name") == "pelvis":
            link.set("name", "base_link")
    for joint in root.iter("joint"):
        for end in ("parent", "child"):
            node = joint.find(end)
            if node is not None and node.get("link") == "pelvis":
                node.set("link", "base_link")

    # The floating base is enabled in the prepared URDF for MuJoCo's benefit; strip it and the world
    # link so MoveIt's virtual_joint is the single authority on how the robot attaches to the world.
    for joint in list(root.findall("joint")):
        if joint.get("type") == "floating":
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") == "world":
            root.remove(link)

    for mesh in root.iter("mesh"):
        fname = Path(mesh.get("filename")).name
        mesh.set("filename", f"package://{mesh_package}/meshes/{fname}")

    # Tool frames. The MJCF carries these as <site>s, which URDF has no equivalent for, so MoveIt would
    # otherwise have no frame to plan a grasp to -- and the obvious substitute, the wrist link, is
    # 152 mm short of where the fingers actually meet. Massless fixed links at the identical offset, so
    # the sim's `<side>_grasp` site and MoveIt's `<side>_grasp` link are the same point by construction.
    for side in ("left", "right"):
        ET.SubElement(root, "link", {"name": f"{side}_grasp"})
        joint = ET.SubElement(root, "joint", {"name": f"{side}_grasp_joint", "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": " ".join(map(str, TCP_POS)), "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": f"{side}_wrist_yaw_link"})
        ET.SubElement(joint, "child", {"link": f"{side}_grasp"})

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def build(
    src: Path, prepared_text: str, mesh_dst: Path
) -> tuple[str, dict[str, bytes], mujoco.MjModel]:
    """``(MJCF text, {mesh file: bytes}, compiled model)``, with the meshes staged in *mesh_dst*.

    The compiled model is loaded from *mesh_dst*, so what is verified is what would be written.
    """
    prepared = src / "_roqsim_build_g1_dex1.urdf"
    try:
        prepared.write_text(prepared_text)
        mjcf, reference = urdf_to_mjcf(prepared)
    finally:
        prepared.unlink(missing_ok=True)
    tree = apply_roqsim_conventions(mjcf)
    root = tree.getroot()
    root.find("compiler").set("meshdir", "meshes/unitree_g1_dex1/")

    meshes: dict[str, bytes] = {}
    for mesh in root.findall(".//mesh[@file]"):
        fname = mesh.get("file")
        if fname == HEAD_WINDOW_FILE:
            meshes[fname] = stl_bytes(cut_head(src / "meshes" / f"{HEAD_MESH}.STL", prepared_text))
            continue
        source = src / "meshes" / fname
        if not source.exists():
            raise RuntimeError(f"mesh {fname} referenced but not found in {src / 'meshes'}")
        meshes[fname] = source.read_bytes()
    mesh_dst.mkdir(parents=True, exist_ok=True)
    for fname, data in meshes.items():
        (mesh_dst / fname).write_bytes(data)

    xml = ET.tostring(root, encoding="unicode")
    xml = with_headline(xml, HEADLINE) + "\n"
    staged = mesh_dst.parent.parent / "_roqsim_build_g1_dex1.xml"
    try:
        staged.write_text(xml)
        model = mujoco.MjModel.from_xml_path(str(staged))
    finally:
        staged.unlink(missing_ok=True)
    verify_against_urdf(model, reference, "pelvis")
    verify_tcp(model)
    return xml, meshes, model


def main() -> int:
    here = Path(__file__).resolve().parent  # <repo>/external/convert/
    pkg = here.parents[1] / "roqsim_humanoid"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="g1_description dir (default: pinned)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="rebuild in a scratch directory and fail if the committed model or head mesh differs",
    )
    # The MoveIt-side URDF is OPTIONAL and its destination is the caller's, because the MoveIt config
    # is not the substrate's -- it belongs to whichever task plans with this robot. This script emits
    # the MJCF; a task that also wants the matching URDF asks for it by path, and owns keeping the two
    # in step (see that package's own regeneration entry point).
    ap.add_argument(
        "--moveit-urdf",
        type=Path,
        default=None,
        help="also emit the MoveIt-side URDF here (requires --mesh-package)",
    )
    ap.add_argument(
        "--mesh-package",
        default=None,
        help="ament package name for the URDF's package:// mesh URIs",
    )
    args = ap.parse_args()
    if bool(args.moveit_urdf) != bool(args.mesh_package):
        ap.error("--moveit-urdf and --mesh-package must be given together")

    src = (
        args.src
        or resolve_source(
            "unitree_ros",
            UNITREE_ROS_URL,
            UNITREE_ROS_COMMIT,
            subdir="robots/g1_description",
            sparse="robots/g1_description",
        )
    ).resolve()

    models = pkg / "src/roqsim_humanoid/models"
    out_xml = models / "unitree_g1_dex1.xml"
    mesh_dst = models / "meshes/unitree_g1_dex1"
    prepared_text = prepare_urdf((src / URDF_NAME).read_text())

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            xml, meshes, _ = build(src, prepared_text, Path(tmp) / "meshes" / "unitree_g1_dex1")
        stale = [] if out_xml.is_file() and out_xml.read_text() == xml else [out_xml.name]
        stale += [f for f, data in meshes.items() if (mesh_dst / f).read_bytes() != data]
        if stale:
            print(f"differs from a fresh build - was it hand-edited? {stale}", file=sys.stderr)
            return 1
        print(f"{out_xml.name}: up to date with {UNITREE_ROS_COMMIT[:12]}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "meshes" / "unitree_g1_dex1"
        xml, meshes, model = build(src, prepared_text, staging)
        mesh_dst.mkdir(parents=True, exist_ok=True)
        for fname in meshes:
            shutil.copy2(staging / fname, mesh_dst / fname)
    out_xml.write_text(xml)
    print(f"wrote {out_xml.relative_to(pkg.parent)} ({len(meshes)} meshes -> {mesh_dst.name}/)")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"  total mass {sum(model.body_mass):.3f} kg")

    # The MoveIt-side URDF, from the same prepared source, so the planner and the sim cannot disagree.
    # Only when asked for: see --moveit-urdf. A missing destination is an error rather than a skip --
    # a silent skip leaves a stale URDF beside a rebuilt MJCF, which is the exact disagreement this
    # whole path exists to prevent.
    if args.moveit_urdf:
        urdf_dst = args.moveit_urdf.resolve()
        if not urdf_dst.parent.is_dir():
            raise SystemExit(
                f"error: {urdf_dst.parent} does not exist -- nothing to write the URDF to"
            )
        urdf_tree = urdf_for_moveit(prepared_text, args.mesh_package)
        urdf_tree.write(urdf_dst, encoding="unicode")
        urdf_dst.write_text(urdf_dst.read_text() + "\n")
        joints = [
            j.get("name") for j in urdf_tree.getroot().iter("joint") if j.get("type") != "fixed"
        ]
        print(f"wrote {urdf_dst} ({len(joints)} movable joints)")
        # The two artifacts must name the same joints, or MoveIt plans for joints the controller does
        # not own. Cheap to check here, expensive to discover at execution time.
        mjcf_joints = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
        } - {"base_free"}
        if missing := sorted(set(joints) - mjcf_joints):
            raise RuntimeError(f"URDF joints absent from the MJCF: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
