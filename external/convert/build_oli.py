#!/usr/bin/env python3
"""Build ``models/oli.xml`` (the serial-PR LimX Oli / HU_D04_01) from the vendor sources.

The port is deliberately reproducible from upstream rather than hand-authored, because the Oli's
parallel ankle/waist linkages make a hand transcription error-prone. Sources (see THIRD_PARTY.md):

  * ``humanoid-description`` @ a90f734  -- HU_D04_01 URDF (SERIAL / PR-space tree: ankle_pitch/roll,
    waist_roll/pitch as real joints), SRDF (rotor mass + gear ratio -> armature), and the vendor
    MJCF (HU_D04_01.xml, the parallel AB-space model) which we mine ONLY for its hand-authored
    PRIMITIVE collision geoms (foot boxes, leg capsules, torso/arm cylinders).
  * ``humanoid-rl-deploy-python`` @ 6d8771c -- walk_param.yaml (user_torque_limit -> actuator
    ctrlrange; default_angle -> home keyframe stance), in the 31-joint PR "policy order".

Why build from the URDF and not the vendor MJCF: the pretrained walk policy speaks PR space
(ankle_pitch/roll, waist_roll/pitch). The vendor MJCF actuates the parallel A/B motors and closes
6 <connect> loops; driving it needs the SDK's PR<->AB projection. The URDF already encodes the
serial PR tree the policy expects, so we take the tree from there and only borrow the MJCF's
collision primitives. This realises the "serial PR-space approximation" chosen for the port: the
linkage's exact coupling/inertia and the visible rods are dropped; joint kinematics + contact match
what the policy trained on.

Run:  python external/convert/build_oli.py [--desc-dir DIR] [--deploy-dir DIR]
Emits models/oli.xml and copies the referenced meshes into models/meshes/oli/.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml
from model_headline import with_headline
from sources import resolve_source

# Pinned upstream revisions -- must match the table in roqsim_humanoid/THIRD_PARTY.md.
HEADLINE = (
    "LimX Oli (HU_D04_01) humanoid for MuJoCo, in the serial PR-space form the vendor walk "
    "policy expects."
)

OLI_DESC_COMMIT = "a90f734c153aa3ecffc8b674af1e0a323cb55d1a"  # humanoid-description 1.0.0.20260706
OLI_DEPLOY_COMMIT = "6d8771cd2b5599e90e7598cfad3623dce66d1218"  # humanoid-rl-deploy 1.0.0.20260330

# 31 joints in the policy's PR order -- the exact index order of every array in walk_param.yaml
# (kp/kd/default_angle/action_scale/user_torque_limit) and of the policy's observation/action
# vectors. See humanoid-rl-deploy-python/doc/parallel_joint_mapping_en.md. NOTE head is
# pitch-then-yaw here, opposite the URDF/vendor-MJCF body order -- the plugin resolves by name.
POLICY_JOINTS = (
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

_GEOM_TYPE = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}


def euler_xyz_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """MuJoCo default eulerseq='xyz', intrinsic. Returns (w, x, y, z)."""

    def q(axis, a):
        h = a / 2.0
        s = np.sin(h)
        v = [0.0, 0.0, 0.0]
        v[axis] = s
        return np.array([np.cos(h), *v])

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    return mul(mul(q(0, rx), q(1, ry)), q(2, rz))


def parse_srdf_armature(srdf_path: Path) -> dict[str, float]:
    """armature = rotor_mass * gear_ratio**2 -- the vendor's own reflected-inertia formula.

    Verified: ankle rotor mass 6.44e-5 * 36**2 = 0.08346 reproduces the vendor MJCF's
    armature='0.083508' on the achilles joints. So this is the source of the vendor's armatures.
    """
    root = ET.parse(srdf_path).getroot()
    out = {}
    for j in root.iter("joint"):
        name, mass, gear = j.get("name"), j.get("mass"), j.get("gear_ratio")
        if name and mass and gear:
            out[name] = float(mass) * float(gear) ** 2
    return out


def parse_collision_geoms(mjcf_path: Path) -> dict[str, list[dict]]:
    """From the vendor (parallel/AB) MJCF, collect PRIMITIVE collision geoms per body.

    Collision geoms are those with neither class='visual' (contype 0) nor a mesh (visual only in
    this model); i.e. the hand-authored box/capsule/cylinder/sphere primitives that inherit the
    model default contype=1. We key them by their owning body name and later re-attach to the
    matching body in the serial tree.
    """
    tree = ET.parse(mjcf_path)
    out: dict[str, list[dict]] = {}

    def walk(elem, body_name):
        for child in elem:
            if child.tag == "body":
                walk(child, child.get("name"))
            elif child.tag == "geom":
                cls = child.get("class")
                gtype = child.get("type")
                if cls == "visual" or gtype in (None, "mesh", "plane"):
                    continue  # visual mesh or the floor -- not a collision primitive
                out.setdefault(body_name, []).append(
                    {
                        "type": gtype,
                        "pos": child.get("pos", "0 0 0"),
                        "size": child.get("size", "0.01"),
                        "euler": child.get("euler", "0 0 0"),
                        "name": child.get("name"),
                    }
                )

    world = tree.getroot().find("worldbody")
    walk(world, None)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent  # roqsim/external/convert/
    pkg = here.parents[1] / "roqsim_humanoid"  # sibling of external/
    ap = argparse.ArgumentParser()
    # Default to the pinned upstream fetched into external/sources/ (see sources.resolve_source).
    # These were absolute paths into an agent session scratchpad that no longer exists, which made
    # the documented rebuild path unrunnable on every machine including the one that wrote it.
    ap.add_argument("--desc-dir", type=Path, default=None)
    ap.add_argument("--deploy-dir", type=Path, default=None)
    args = ap.parse_args()

    desc = args.desc_dir or resolve_source(
        "limx_humanoid_description",
        "https://github.com/limxdynamics/humanoid-description",
        OLI_DESC_COMMIT,
        subdir="HU_D04_description",
    )
    deploy = args.deploy_dir or resolve_source(
        "limx_humanoid_rl_deploy",
        "https://github.com/limxdynamics/humanoid-rl-deploy-python",
        OLI_DEPLOY_COMMIT,
        subdir="controllers/HU_D04_01",
    )
    urdf_path = desc / "urdf/HU_D04_01.urdf"
    srdf_path = desc / "urdf/HU_D04_01.srdf"
    vendor_mjcf = desc / "xml/HU_D04_01.xml"
    mesh_src = desc / "meshes/HU_D04_01"
    walk_param = yaml.safe_load(open(deploy / "walk_controller/walk_param.yaml"))[
        "HumanoidRobotCfg"
    ]

    torque_limit = dict(zip(POLICY_JOINTS, walk_param["control"]["user_torque_limit"], strict=True))
    default_angle = dict(zip(POLICY_JOINTS, walk_param["control"]["default_angle"], strict=True))
    armature = parse_srdf_armature(srdf_path)
    collision = parse_collision_geoms(vendor_mjcf)

    models = pkg / "src/roqsim_humanoid/models"
    mesh_dst = models / "meshes/oli"
    mesh_dst.mkdir(parents=True, exist_ok=True)

    # -- load the serial URDF: strip package:// so MuJoCo resolves meshes from mesh_src ----------
    urdf = open(urdf_path).read()
    urdf = urdf.replace("package://HU_D04_description/meshes/HU_D04_01/", "")
    urdf = urdf.replace(
        "</robot>",
        f'<mujoco><compiler meshdir="{mesh_src}" discardvisual="false"/></mujoco>\n</robot>',
    )
    tmp = mesh_src / "_oli_build.urdf"
    tmp.write_text(urdf)
    try:
        spec = mujoco.MjSpec.from_file(str(tmp))
    finally:
        tmp.unlink()

    spec.modelname = "oli"
    # Copy the referenced meshes into the package, and compile against the ABSOLUTE path (the
    # relative 'meshes/oli/' written into the final XML is only valid once files are in place).
    copied = 0
    for mesh in spec.meshes:
        fname = mesh.file or f"{mesh.name}.STL"
        src = mesh_src / fname
        if src.exists():
            shutil.copy2(src, mesh_dst / fname)
            copied += 1
    spec.meshdir = str(mesh_dst.resolve())

    # -- floating base: base_link + free joint 'base_free' (spawn/odom convention, cf. g1) -------
    base = spec.body("base_link")
    fj = base.add_freejoint()
    fj.name = "base_free"

    # -- per-joint armature (SRDF) + light damping/frictionloss ----------------------------------
    for j in spec.joints:
        if j.name in armature:
            j.armature = armature[j.name]
            j.damping = np.array([0.2, 0.0, 0.0])  # MjsJoint.damping is a 3-vec; [0] used for hinge
            j.frictionloss = 0.1

    # -- make every imported geom visual-only; collision comes only from the transplanted
    #    primitives below (matches the vendor's design: meshes are visual, primitives collide) ---
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
        g.group = 1

    # -- transplant the vendor's primitive collision geoms onto the matching serial bodies -------
    serial_bodies = {b.name for b in spec.bodies}
    transplanted = []
    for body_name, geoms in collision.items():
        if body_name not in serial_bodies:
            continue  # achilles/waist-rod links etc. -- absent in the serial tree
        b = spec.body(body_name)
        for gd in geoms:
            g = b.add_geom()
            g.type = _GEOM_TYPE[gd["type"]]
            g.pos = np.array([float(x) for x in gd["pos"].split()])
            size = [float(x) for x in gd["size"].split()]
            g.size = np.array((size + [0, 0, 0])[:3])
            g.quat = euler_xyz_to_quat(*[float(x) for x in gd["euler"].split()])
            g.contype, g.conaffinity, g.condim = 1, 1, 3
            g.friction = np.array([1.0, 0.3, 0.3])
            g.group = 3
            g.rgba = np.array([0.6, 0.3, 0.3, 0.3])
            if gd["name"]:
                g.name = gd["name"]
        transplanted.append(f"{body_name}({len(geoms)})")

    # -- sites: lidar + imu on base_link ---------------------------------------------------------
    base.add_site(name="lidar", pos=[0.0, 0.0, 0.30], size=[0.01])
    base.add_site(name="imu", pos=[0.0, 0.0, 0.0], size=[0.01])

    # -- depth cameras: head + chest RealSense D435 mount frames from the URDF -------------------
    # Positions are the URDF camera-joint origins (head_camera_joint on head_pitch_link;
    # waist_camera_joint, commented in the URDF but frame given, on waist_pitch_link). The MJCF
    # camera looks down its -z with +y up; xyaxes="0 -1 0 0 0 1" (below, as a quat) points -z along
    # body +x (forward) with +y up -- the standard forward optical axis (cf. the realsense_d435
    # docstring). The exact vendor optical extrinsics (small pitch/roll of the real module) are a
    # documented follow-up for the C1 mount-pose check.
    R_fwd = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q_fwd = np.zeros(4)
    mujoco.mju_mat2Quat(q_fwd, R_fwd.flatten())
    hc = spec.body("head_pitch_link").add_camera()
    hc.name, hc.pos, hc.quat, hc.fovy = (
        "head_camera",
        np.array([0.07453, 0.0175, 0.065]),
        q_fwd,
        58.0,
    )
    cc = spec.body("waist_pitch_link").add_camera()
    cc.name, cc.pos, cc.quat, cc.fovy = (
        "chest_camera",
        np.array([0.092, 0.0175, 0.2751]),
        q_fwd,
        58.0,
    )

    # -- IMU sensors (noise-free) for realism; the policy reads base state from qpos/qvel like g1 -
    fq = spec.add_sensor()
    fq.name, fq.type = "imu_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT
    fq.objtype, fq.objname = mujoco.mjtObj.mjOBJ_SITE, "imu"
    gy = spec.add_sensor()
    gy.name, gy.type = "imu_gyro", mujoco.mjtSensor.mjSENS_GYRO
    gy.objtype, gy.objname = mujoco.mjtObj.mjOBJ_SITE, "imu"
    ac = spec.add_sensor()
    ac.name, ac.type = "imu_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER
    ac.objtype, ac.objname = mujoco.mjtObj.mjOBJ_SITE, "imu"

    # -- 31 torque actuators in POLICY order; ctrl == joint torque, clamped to user_torque_limit -
    for jn in POLICY_JOINTS:
        a = spec.add_actuator()
        a.name = jn
        a.target = jn
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        lim = float(torque_limit[jn])
        a.ctrlrange = np.array([-lim, lim])

    # -- compile, then compute the home keyframe: default_angle stance dropped so feet touch z=0 -
    model = spec.compile()
    data = mujoco.MjData(model)
    qadr = {
        jn: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        for jn in POLICY_JOINTS
    }
    for jn, adr in qadr.items():
        data.qpos[adr] = default_angle[jn]
    base_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
    data.qpos[base_qadr + 2] = 1.2  # provisional lift
    mujoco.mj_forward(model, data)

    def lowest_z(g: int) -> float:
        """World-z of the lowest point of collision geom g (type-aware, not bounding-sphere)."""
        c = data.geom_xpos[g][2]
        R = data.geom_xmat[g].reshape(3, 3)  # columns = local axes in world
        s = model.geom_size[g]
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            return c - abs(R[2, 0]) * s[0] - abs(R[2, 1]) * s[1] - abs(R[2, 2]) * s[2]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            return c - s[0]
        if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            return c - abs(R[2, 2]) * s[1] - s[0]  # local z is the axis; s=[radius, half-length]
        return c - model.geom_rbound[g]

    zmin = min(
        lowest_z(g)
        for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    )
    home_z = 1.2 - zmin + 0.002
    home_qpos = data.qpos.copy()
    home_qpos[base_qadr + 2] = home_z

    # Give base_link a default height so qpos0 already stands on the floor (cf. g1's base pos) --
    # otherwise tools that reset to qpos0 (e.g. the thumbnail renderer) bury the legs in the ground.
    base.pos = np.array([0.0, 0.0, home_z])

    key = spec.add_key()
    key.name = "home"
    key.qpos = home_qpos
    model = spec.compile()  # recompile with base height + keyframe

    # Serialize with the absolute meshdir (to_xml validates mesh files), then rewrite that one
    # attribute to the package-relative path the substrate resolves from the XML's own directory.
    abs_meshdir = str(mesh_dst.resolve())
    xml = spec.to_xml().replace(f'meshdir="{abs_meshdir}"', 'meshdir="meshes/oli/"')
    xml = xml.replace(f'meshdir="{abs_meshdir}/"', 'meshdir="meshes/oli/"')
    out_xml = models / "oli.xml"
    out_xml.write_text(with_headline(xml, HEADLINE))

    total_mass = float(model.body_mass.sum())
    print(f"wrote {out_xml}")
    print(
        f"  bodies={model.nbody} joints={model.njnt} actuators={model.nu} "
        f"geoms={model.ngeom} meshes={model.nmesh} (copied {copied})"
    )
    print(f"  total mass = {total_mass:.2f} kg")
    print(f"  home base height = {home_z:.3f} m")
    print(f"  collision transplanted onto {len(transplanted)} bodies: {', '.join(transplanted)}")
    missing = [b for b in collision if b not in serial_bodies]
    print(f"  vendor collision bodies absent in serial tree (skipped): {missing}")


if __name__ == "__main__":
    main()
