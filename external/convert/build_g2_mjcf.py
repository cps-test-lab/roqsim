"""Turn the MuJoCo-emitted base MJCF (g2_base.xml) into the roqsim agibot_g2 model.

Deterministic post-process: regroup visual/collision geoms, mount base_free + sites + a chassis
collision box on base_link, add position/velocity actuators, reproduce the URDF gripper mimics as
joint equalities, add odom/imu/encoder sensors and a home keyframe. Everything the URDF cannot
express lives here. Run after preprocess_urdf.py + a MuJoCo save of g2_base.xml.

Reads materials_flat.json (from dae2obj) to recover per-material colours: MuJoCo ignores OBJ/MTL
materials, so each visual geom is given an MJCF <material>, and a multi-material mesh is split into
one sub-geom per material.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from model_headline import with_headline

SRC = "g2_base.xml"
OUT = "agibot_g2.xml"
HEADLINE = "AgiBot G2 (wheeled dual-arm mobile manipulator) for MuJoCo."
MATERIALS = "materials_flat.json"

WHEEL_ROLL = ["idx112_chassis_lwheel_front_joint2", "idx132_chassis_rwheel_front_joint2",
              "idx142_chassis_rwheel_rear_joint2", "idx122_chassis_lwheel_rear_joint2"]
LEFT_ROLL = ["idx112_chassis_lwheel_front_joint2", "idx122_chassis_lwheel_rear_joint2"]
RIGHT_ROLL = ["idx132_chassis_rwheel_front_joint2", "idx142_chassis_rwheel_rear_joint2"]

BODY = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
HEAD = [f"idx1{i}_head_joint{i - 0}" for i in (1, 2, 3)]
HEAD = ["idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3"]
ARM_L = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
ARM_R = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
GRIP_MASTER = ["idx31_gripper_l_inner_joint1", "idx71_gripper_r_inner_joint1"]

# follower -> (master, multiplier)   (from the URDF <mimic> tags)
MIMIC = {
    "idx32_gripper_l_inner_joint3": ("idx31_gripper_l_inner_joint1", 0.1),
    "idx33_gripper_l_inner_joint4": ("idx31_gripper_l_inner_joint1", 0.25),
    "idx39_gripper_l_inner_joint0": ("idx31_gripper_l_inner_joint1", -0.7),
    "idx41_gripper_l_outer_joint1": ("idx31_gripper_l_inner_joint1", -1.0),
    "idx42_gripper_l_outer_joint3": ("idx31_gripper_l_inner_joint1", 0.1),
    "idx43_gripper_l_outer_joint4": ("idx31_gripper_l_inner_joint1", -0.25),
    "idx49_gripper_l_outer_joint0": ("idx31_gripper_l_inner_joint1", 0.7),
    "idx72_gripper_r_inner_joint3": ("idx71_gripper_r_inner_joint1", 0.1),
    "idx73_gripper_r_inner_joint4": ("idx71_gripper_r_inner_joint1", 0.25),
    "idx79_gripper_r_inner_joint0": ("idx71_gripper_r_inner_joint1", -0.7),
    "idx81_gripper_r_outer_joint1": ("idx71_gripper_r_inner_joint1", -1.0),
    "idx82_gripper_r_outer_joint3": ("idx71_gripper_r_inner_joint1", 0.1),
    "idx83_gripper_r_outer_joint4": ("idx71_gripper_r_inner_joint1", -0.25),
    "idx89_gripper_r_outer_joint0": ("idx71_gripper_r_inner_joint1", 0.7),
}

# per-joint actuator gains: (kp, forcerange)   effort from URDF; kp sized to hold against gravity
# (kp, forcerange). Torso-lift joints carry the whole upper body on a lever, so they need a stiff
# servo and a large force ceiling (the source effort of 50 N.m cannot hold the torso; raised to hold
# pose -- a documented substrate assumption, see port log). Arms/head/grippers use their source effort.
POS_GAINS = {}
for j in BODY:
    POS_GAINS[j] = (8000, 300)
for j in HEAD:
    POS_GAINS[j] = (200, 50)
for j in ARM_L + ARM_R:
    POS_GAINS[j] = (800, 108 if j.endswith("1") or j.endswith("2") else 35)
for j in GRIP_MASTER:
    POS_GAINS[j] = (50, 10)

DAMP = {}
for j in BODY:
    DAMP[j] = 300.0
for j in HEAD:
    DAMP[j] = 3.0
for j in ARM_L + ARM_R:
    DAMP[j] = 10.0
for j in list(MIMIC) + GRIP_MASTER:
    DAMP[j] = 0.5


def main():
    # Compile the preprocessed URDF to the base MJCF if not already present (folds the
    # g2_mj.urdf -> g2_base.xml step in, so `preprocess_urdf.py && build_g2_mjcf.py` reproduces).
    if not Path(SRC).exists():
        import mujoco

        Path(SRC).write_text(mujoco.MjSpec.from_file("g2_mj.urdf").to_xml())

    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("model", "agibot_g2")

    # ---- compiler + option + defaults --------------------------------------------------
    comp = root.find("compiler")
    comp.set("meshdir", "meshes/agibot_g2")
    comp.set("angle", "radian")
    comp.set("autolimits", "true")

    # default classes for actuated-joint damping + collision geoms
    default = ET.Element("default")
    dvis = ET.SubElement(default, "default", {"class": "g2_visual"})
    # NB: no density="0" -- a zero-density mesh geom renders as its bounding sphere in MuJoCo (grey
    # balls). Every body carries an explicit <inertial> from the URDF, so geom-derived mass is ignored
    # anyway; leaving density default keeps the visual meshes rendering as meshes.
    ET.SubElement(dvis, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0",
                                 "group": "2", "material": "g2_light"})
    dcol = ET.SubElement(default, "default", {"class": "g2_collision"})
    # contype=2/conaffinity=1 disables robot self-collision (the source convex hulls of
    # non-parent-child links overlap at rest) while still colliding with the world floor/objects
    # (contype/conaffinity=1). Self-collision avoidance is out of scope for this port.
    ET.SubElement(dcol, "geom", {"group": "3", "rgba": "0.4 0.5 0.6 0.3", "condim": "3",
                                 "contype": "2", "conaffinity": "1"})
    root.insert(list(root).index(comp) + 1, default)

    # ---- materials ---------------------------------------------------------------------
    asset = root.find("asset")
    for name, rgba in [("g2_light", "0.85 0.86 0.88 1"), ("g2_dark", "0.15 0.16 0.18 1"),
                       ("g2_black", "0.05 0.05 0.05 1")]:
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    # ---- regroup existing geoms --------------------------------------------------------
    # visual geoms (density=0, contype=0) -> class g2_visual. Everything else is collision geometry
    # imported from the URDF -- convex-hull meshes AND primitive spheres (the arm links' collision is
    # a set of spheres). All go to class g2_collision (group 3, transparent, no self-collision). Pop
    # their explicit attrs first, or the class defaults won't take (explicit attrs win in MJCF).
    worldbody = root.find("worldbody")
    for g in worldbody.iter("geom"):
        is_visual = g.get("density") == "0" and g.get("contype") == "0"
        if is_visual:
            for a in ("contype", "conaffinity", "group", "density", "rgba"):
                g.attrib.pop(a, None)
            g.set("class", "g2_visual")
        else:  # collision geometry (convex hull, sphere, cylinder, box)
            for a in ("contype", "conaffinity", "group", "rgba"):
                g.attrib.pop(a, None)
            g.set("class", "g2_collision")

    # ---- per-material colours ----------------------------------------------------------
    # MuJoCo ignores OBJ/MTL colours; recover them as MJCF materials. dae2obj split each source mesh
    # by material (materials_flat.json: mesh -> [(submesh, rgb), ...]). A single-material mesh just
    # gets its material; a multi-material mesh is replaced by one visual sub-geom per material.
    mats = json.loads(Path(MATERIALS).read_text())
    color_name = {}

    def material_for(rgb):
        key = tuple(rgb)
        if key not in color_name:
            rr, gg, bb = (min(255, max(0, round(c * 255))) for c in rgb)
            name = f"mat_{rr:02x}{gg:02x}{bb:02x}"
            color_name[key] = name
            ET.SubElement(asset, "material",
                          {"name": name, "rgba": f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} 1"})
        return color_name[key]

    mesh_names = {mesh.get("name") for mesh in asset.findall("mesh")}

    def ensure_mesh(name):
        if name not in mesh_names:
            ET.SubElement(asset, "mesh", {"name": name, "file": name + ".obj"})
            mesh_names.add(name)

    parent = {c: p for p in root.iter() for c in p}
    for g in list(worldbody.iter("geom")):
        if g.get("class") != "g2_visual" or not g.get("mesh"):
            continue
        subs = mats.get(g.get("mesh"))
        if not subs:
            continue
        if len(subs) == 1:
            g.set("material", material_for(subs[0][1]))
            continue
        body = parent[g]
        at = list(body).index(g)
        for i, (subname, rgb) in enumerate(subs):
            ensure_mesh(subname)
            sub = ET.Element("geom", dict(g.attrib))
            sub.set("mesh", subname)
            sub.set("material", material_for(rgb))
            body.insert(at + 1 + i, sub)
        body.remove(g)

    # drop mesh assets no longer referenced (the combined multi-material meshes)
    used = {g.get("mesh") for g in worldbody.iter("geom") if g.get("mesh")}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used:
            asset.remove(mesh)

    # ---- find bodies -------------------------------------------------------------------
    base = next(b for b in root.iter("body") if b.get("name") == "base_link")
    head3 = next(b for b in root.iter("body") if b.get("name") == "head_link3")

    # wheel roll cylinder collision geoms: give them a name + rolling friction
    for b in root.iter("body"):
        if b.get("name", "").startswith("chassis_") and b.get("name", "").endswith("link2"):
            for g in b.findall("geom"):
                if g.get("type") == "cylinder":
                    g.set("name", b.get("name") + "_tyre")
                    g.set("class", "g2_collision")
                    g.set("condim", "3")
                    # slide friction 0.5: enough traction to drive/hold a slope, low enough that the
                    # 4 fixed (welded-swerve) wheels can scrub during a turn. In-place yaw is still
                    # limited (~0.2-0.3 of commanded) -- inherent to a rigid 4-wheel skid-steer; see
                    # the port log. Straight-line drive is unaffected.
                    g.set("friction", "0.5 0.01 0.001")
                    g.set("priority", "2")
                    g.set("contype", "2")
                    g.set("conaffinity", "1")
                    g.set("rgba", "0.05 0.05 0.05 1")

    # ---- augment base_link -------------------------------------------------------------
    inserts = []
    inserts.append(ET.Comment(" mobile base: free joint + odom/imu frame "))
    inserts.append(ET.Element("freejoint", {"name": "base_free"}))
    inserts.append(ET.Element("site", {"name": "base_imu", "pos": "0 0 0.1",
                                        "size": "0.01", "rgba": "0 0 0 0"}))
    # lidar: no lidar in the source; assumed 2D scan mount on the front of the chassis top.
    inserts.append(ET.Comment(" ASSUMPTION: 2D nav lidar; source has none. Mount = chassis front, 0.30 m. "))
    inserts.append(ET.Element("site", {"name": "lidar", "pos": "0.24 0 0.30",
                                        "size": "0.006", "rgba": "1 0 0 0.6"}))
    inserts.append(ET.Element("camera", {"name": "track", "mode": "track",
                                         "pos": "-2.4 -2.4 2.0",
                                         "xyaxes": "0.7 -0.7 0 0.3 0.3 0.9"}))
    # chassis footprint collision box (source base_link has visual only)
    inserts.append(ET.Element("geom", {"name": "chassis_box", "type": "box",
                                       "size": "0.28 0.20 0.08", "pos": "0 0 0.10",
                                       "class": "g2_collision", "friction": "0.4 0.01 0.001"}))
    # Anti-tip casters (the "wheels welded as caster" approximation). The real base carries a low CoM;
    # the source inertials put the robot CoM at ~0.52 m over a 0.46x0.44 m wheel footprint, so the 4
    # thin drive wheels alone let it wheelie/tip. Four low-friction support spheres at the base corners
    # sit co-planar with the wheel contact (bottom at floor), keeping the base level while the wheels
    # still provide traction. Documented substrate assumption; see port log.
    for cx in (0.30, -0.30):
        for cy in (0.22, -0.22):
            inserts.append(ET.Element("geom", {
                "name": f"caster_{'f' if cx>0 else 'r'}{'l' if cy>0 else 'r'}",
                "type": "sphere", "size": "0.03", "pos": f"{cx} {cy} -0.005",
                "class": "g2_collision", "friction": "0.02 0.005 0.0001",
                "rgba": "0.2 0.2 0.2 1"}))
    for i, el in enumerate(inserts):
        base.insert(i, el)

    # head camera
    head3.insert(0, ET.Element("camera", {"name": "head_cam", "pos": "0.06 0 0.05",
                                          "xyaxes": "0 -1 0 0 0 1", "fovy": "58"}))

    # ---- joint damping -----------------------------------------------------------------
    for j in root.iter("joint"):
        n = j.get("name")
        if n in DAMP:
            j.set("damping", str(DAMP[n]))

    # ---- actuators ---------------------------------------------------------------------
    act = ET.SubElement(root, "actuator")
    act.append(ET.Comment(" wheel velocity servos (diff-drive: left pair / right pair together) "))
    for j in WHEEL_ROLL:
        # kv=30, forcerange +-60 N.m: the wheels must apply enough torque to scrub the 165 kg base
        # during a skid-steer turn (8 N.m stalled it). ctrlrange covers max_vel + slip-inflated yaw.
        ET.SubElement(act, "velocity", {"name": j.replace("idx", "m_"), "joint": j,
                                        "kv": "30", "ctrlrange": "-40 40", "forcerange": "-60 60"})
    act.append(ET.Comment(" torso / head / arms / gripper position servos "))
    for j in BODY + HEAD + ARM_L + ARM_R + GRIP_MASTER:
        kp, fr = POS_GAINS[j]
        ET.SubElement(act, "position", {"name": j.replace("idx", "p_"), "joint": j,
                                        "kp": str(kp), "forcerange": f"-{fr} {fr}",
                                        "inheritrange": "1"})

    # ---- equalities (gripper mimics) ---------------------------------------------------
    eq = ET.SubElement(root, "equality")
    for follower, (master, mult) in MIMIC.items():
        ET.SubElement(eq, "joint", {"joint1": follower, "joint2": master,
                                    "polycoef": f"0 {mult} 0 0 0", "solimp": "0.95 0.99 0.001"})

    # ---- sensors -----------------------------------------------------------------------
    sen = ET.SubElement(root, "sensor")
    ET.SubElement(sen, "framepos", {"name": "base_pos", "objtype": "site", "objname": "base_imu"})
    ET.SubElement(sen, "framequat", {"name": "base_quat", "objtype": "site", "objname": "base_imu"})
    ET.SubElement(sen, "framelinvel", {"name": "base_linvel", "objtype": "site", "objname": "base_imu"})
    ET.SubElement(sen, "frameangvel", {"name": "base_angvel", "objtype": "site", "objname": "base_imu"})
    ET.SubElement(sen, "gyro", {"name": "imu_gyro", "site": "base_imu"})
    ET.SubElement(sen, "accelerometer", {"name": "imu_acc", "site": "base_imu"})
    for j in WHEEL_ROLL:
        ET.SubElement(sen, "jointvel", {"name": j.replace("idx", "vel_"), "joint": j})
    for j in BODY + HEAD + ARM_L + ARM_R + GRIP_MASTER:
        ET.SubElement(sen, "jointpos", {"name": j.replace("idx", "pos_"), "joint": j})

    # ---- home keyframe -----------------------------------------------------------------
    # base_free(7) + 42 hinge joints, all zero; base at wheel rest height 0.04 m
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "home",
                              "qpos": "0 0 0.041 1 0 0 0 " + " ".join(["0"] * 42)})

    # ---- write pretty ------------------------------------------------------------------
    xml = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(xml).toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    open(OUT, "w").write(with_headline(pretty, HEADLINE))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
