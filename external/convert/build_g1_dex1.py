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

Four upstream quirks this handles, each of which is silently wrong if taken at face value:

  * The URDF carries BOTH ``<side>_rubber_hand`` and the Dex1 gripper on the same wrist mount
    (palm joint at 0.0415 0.003 0, gripper base at 0.0415 0 0). The rubber hand is dropped; keeping it
    would ride 0.17 kg of phantom hand inside each gripper.
  * ``<mujoco><compiler meshdir="meshes"/>`` combines with mesh filenames that are ALREADY prefixed
    ``meshes/``, so MuJoCo looks for ``meshes/meshes/*.STL``. The prefix is stripped.
  * The floating base is commented out ("uncomment when convert to mujoco"). Without it MuJoCo fuses
    ``pelvis`` into the world body at parse time and the root link vanishes.
  * Both finger joints are prismatic on OPPOSING axes and their origins coincide at q=0 -- from which
    "q=0 is fully closed" follows and is FALSE. The pads sit ~23 mm outboard each, so q=0 already
    stands 45.9 mm open and the whole useful closing range is upstream's NEGATIVE half. Measured, not
    inferred: see the aperture table below. An earlier version of this script clamped that half away as
    a "crossed" state and produced a gripper that could not grip anything.

Run:  python external/convert/build_g1_dex1.py [--src DIR]
Emits models/unitree_g1_dex1.xml and copies the referenced meshes into models/meshes/unitree_g1_dex1/.

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
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
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
# closing range is therefore the NEGATIVE half of upstream's limits, which an earlier version of this
# script clamped away as a "crossed" state, leaving a gripper that could not grip anything.
#
#   q = -0.0200 -> 5.9 mm aperture (closed)
#   q =  0.0000 -> 45.9 mm
#   q = +0.0245 -> 94.9 mm (fully open)
FINGER_OPEN = 0.0245
FINGER_CLOSE = -0.02
PAD_GAP_AT_ZERO = 0.0459  # measured; documented here so the manifest's box sizing is traceable
# Tool centre point between the pads, in the wrist_yaw frame: measured pad span x[0.077, 0.143],
# z[-0.0145, 0.0145], centred on y. This is the frame a grasp is planned to and MoveIt's
# end-effector link, so it belongs in the model rather than being re-derived by every caller.
TCP_POS = (0.1112, 0.0, 0.0)
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
# Lidar mount, kept on the pelvis rather than the torso: the torso now hangs off three waist joints,
# so a torso-mounted scan would rotate with the waist and break nav2's scan matching. Same offset as
# unitree_g1.xml -> ~1.09 m off the ground when standing.
LIDAR_SITE_POS = (0.0, 0.0, 0.30)


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
    """Apply the four upstream fixups described in the module docstring."""
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


def urdf_to_mjcf(urdf_path: Path) -> str:
    """Convert via MjSpec, which resolves inertias, joint limits and mesh references for us."""
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.compile()  # fail here, on the untouched conversion, rather than after our edits
    return spec.to_xml()


def find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"body {name!r} not in the converted MJCF")
    return body


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

    # The nav2 scan mount. Inserted after the free joint so the body still reads joint-then-geoms.
    site = ET.Element(
        "site", {"name": "lidar", "pos": " ".join(map(str, LIDAR_SITE_POS)), "size": "0.01"}
    )
    pelvis.insert(list(pelvis).index(free) + 1, site)

    # -- joint defaults, matching unitree_g1.xml ---------------------------------------------------
    default = ET.Element("default")
    ET.SubElement(default, "joint", {"damping": "0.001", "armature": "0.01", "frictionloss": "0.1"})
    root.insert(list(root).index(root.find("compiler")) + 1, default)

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
        # end-effector link origin. Sited on wrist_yaw, which the gripper base is welded into.
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
                # gentle position. Both numbers were arrived at by failing:
                #   * kp=200 gave only kp*err = 2 N at a 10 mm over-closure, so friction (~4.8 N) barely
                #     matched the 0.5 kg box's weight (4.9 N) and the parcel slid out of the jaws. At
                #     kp=2000 the same command saturates at 20 N -> ~48 N of friction.
                #   * kv=5 was underdamped: the servo overshot the commanded aperture by ~9 mm, which
                #     squeezed a 40 mm box to 28 mm and extruded it sideways before settling on target.
                #     kv is now near-critical for the ~0.17 kg of moving finger (2*sqrt(kp*m) ~= 37).
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
        so there is exactly one copy of 19 MB of STLs in the tree.
      * The MJCF's roqsim-only additions (the ``lidar`` site, the four-sphere foot contacts, actuators,
        tendons) have no URDF equivalent and are simply absent -- MoveIt needs kinematics, limits and
        collision geometry, none of which they affect.
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
    # 111 mm short of where the fingers actually meet. Massless fixed links at the identical offset, so
    # the sim's `<side>_grasp` site and MoveIt's `<side>_grasp` link are the same point by construction.
    for side in ("left", "right"):
        ET.SubElement(root, "link", {"name": f"{side}_grasp"})
        joint = ET.SubElement(root, "joint", {"name": f"{side}_grasp_joint", "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": " ".join(map(str, TCP_POS)), "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": f"{side}_wrist_yaw_link"})
        ET.SubElement(joint, "child", {"link": f"{side}_grasp"})

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def copy_meshes(xml_root: ET.Element, src_meshes: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for mesh in xml_root.findall(".//mesh[@file]"):
        fname = mesh.get("file")
        src = src_meshes / fname
        if not src.exists():
            raise RuntimeError(f"mesh {fname} referenced but not found in {src_meshes}")
        shutil.copy2(src, dst / fname)
        n += 1
    return n


def main() -> None:
    here = Path(__file__).resolve().parent  # <repo>/external/convert/
    pkg = here.parents[1] / "roqsim_humanoid"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="g1_description dir (default: pinned)")
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

    src = args.src or resolve_source(
        "unitree_ros",
        UNITREE_ROS_URL,
        UNITREE_ROS_COMMIT,
        subdir="robots/g1_description",
        sparse="robots/g1_description",
    )

    models = pkg / "src/roqsim_humanoid/models"
    out_xml = models / "unitree_g1_dex1.xml"
    mesh_dst = models / "meshes/unitree_g1_dex1"

    prepared = src / "_roqsim_build_g1_dex1.urdf"
    prepared_text = prepare_urdf((src / URDF_NAME).read_text())
    try:
        prepared.write_text(prepared_text)
        tree = apply_roqsim_conventions(urdf_to_mjcf(prepared))
    finally:
        prepared.unlink(missing_ok=True)

    root = tree.getroot()
    # Meshes live beside the model in their own subdir, so set meshdir accordingly.
    root.find("compiler").set("meshdir", "meshes/unitree_g1_dex1/")
    n_meshes = copy_meshes(root, src / "meshes", mesh_dst)
    tree.write(out_xml, encoding="unicode")
    out_xml.write_text(with_headline(out_xml.read_text(), HEADLINE) + "\n")

    # Compile the emitted model: an MJCF that does not load is worse than no MJCF at all.
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    print(f"wrote {out_xml.relative_to(pkg.parent)} ({n_meshes} meshes -> {mesh_dst.name}/)")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"  total mass {sum(model.body_mass):.3f} kg")

    # The MoveIt-side URDF, from the same prepared source, so the planner and the sim cannot disagree.
    # Only when asked for: see --moveit-urdf. A missing destination is an error rather than a skip --
    # a silent skip leaves a stale URDF beside a rebuilt MJCF, which is the exact disagreement this
    # whole path exists to prevent.
    if args.moveit_urdf:
        urdf_dst = args.moveit_urdf.resolve()
        if not urdf_dst.parent.is_dir():
            raise SystemExit(f"error: {urdf_dst.parent} does not exist -- nothing to write the URDF to")
        urdf_tree = urdf_for_moveit(prepared_text, args.mesh_package)
        urdf_tree.write(urdf_dst, encoding="unicode")
        urdf_dst.write_text(urdf_dst.read_text() + "\n")
        joints = [j.get("name") for j in urdf_tree.getroot().iter("joint") if j.get("type") != "fixed"]
        print(f"wrote {urdf_dst} ({len(joints)} movable joints)")
        # The two artifacts must name the same joints, or MoveIt plans for joints the controller does
        # not own. Cheap to check here, expensive to discover at execution time.
        mjcf_joints = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
        } - {"base_free"}
        if missing := sorted(set(joints) - mjcf_joints):
            raise RuntimeError(f"URDF joints absent from the MJCF: {missing}")


if __name__ == "__main__":
    main()
