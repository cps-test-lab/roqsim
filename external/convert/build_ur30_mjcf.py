#!/usr/bin/env python3
"""Build roqsim's Universal Robots UR30 MJCF from UR's description data and the ur10e's meshes.

Menagerie has no UR30. Universal Robots' ``Universal_Robots_ROS2_Description`` has one, in two
halves with different licences:

* ``config/ur30/*.yaml`` -- link lengths, link masses, centres of mass, inertia tensors, joint
  ranges and effort limits -- is under the repository's BSD-3-Clause, and every one of those numbers
  is passed through here.
* ``meshes/ur30/`` and the ``meshes/ur20/`` files it reuses are under UR's "Terms and Conditions for
  Use of Graphical Documentation", which give UR the rights in any derivative and let UR revoke
  them. That is not a licence roqsim ships, so those meshes are not used.

The arm's looks are instead Menagerie's ur10e meshes (BSD-3-Clause, already vendored beside the
ur10e), reshaped to the UR30: the joint housings scaled about their own joint axes by the ratio of
the two arms' joint offsets, the link tubes stretched or shortened between them so every housing
sits on the UR30's joint. What results looks like a large UR and has the UR30's kinematics exactly;
it is not UR's own artwork for the UR30, and the MJCF says so.

How the frames map. The ur10e MJCF puts the shoulder offset, the elbow offset and the wrist offset
into the bodies' lateral positions; UR's description puts all of it into ``d4``. The lateral
offsets here are the ur10e's, scaled with the housing beside them, and the wrist offset takes up
the rest so that their sum is the UR30's ``d4``. The centres of mass and inertia tensors are
transformed from the description's link frames (with the inertial rotation it applies) into these
body frames -- the axis maps are in :data:`FRAMES`.

Deltas against UR's data, each a roqsim convention:

1. No ``<option>``: timestep, integrator and contact settings belong to the world.
2. Base at the origin; ``spawn_arm`` places the arm.
3. Visual meshes are visual only (group 2); collision is capsules and cylinders sized to the UR30.
4. A ``tool0`` body at the flange carries ``fts_site`` and ``tool_site``. The UR30 measures force
   and torque in its tool flange, so the measurement cut is the flange itself, with no adapter
   stack as on the ur5e.

Usage::

    python external/convert/build_ur30_mjcf.py           # fetch, build, write into the package
    python external/convert/build_ur30_mjcf.py --check   # rebuild the MJCF and diff it
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402

UR_URL = "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git"
# Pinned. ur30.xml and UR30_LICENSE beside it record this revision, and the generator writes both.
UR_COMMIT = "970bf8aaed1ffa160ca86ab4bb6c04181f7b592d"
UR_SUBDIR = "config/ur30"

MODELS = (
    Path(__file__).resolve().parents[2]
    / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models"
)
UR10E_MESHES = MODELS / "ur10e" / "meshes"
PKG = MODELS / "ur30"

# ur10e (Menagerie) offsets the meshes are drawn around.
UR10E = {
    "d1": 0.181,
    "a2": 0.613,
    "a3": 0.571,
    "lat_upper": 0.176,
    "lat_fore": 0.137,
    "lat_wrist": 0.135,
    "d5": 0.12,
    "flange_y": 0.117,
}

#: Housing scale per joint size: base and shoulder by the ratio of the two arms' d1, the elbow by
#: the ratio of their forearm housings (UR's datasheet joint sizes), the wrists by d5 and d6.
S_SHOULDER = 0.2363 / UR10E["d1"]
S_ELBOW = 1.25
S_WRIST = 1.32


def _degrees_loader():
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(
        "!degrees", lambda loader, node: math.radians(float(loader.construct_scalar(node)))
    )
    return Loader


def load_ur30(source: Path) -> dict:
    loader = _degrees_loader()
    out = {}
    for name in ("default_kinematics", "joint_limits", "physical_parameters"):
        with open(source / f"{name}.yaml") as fh:
            out[name] = yaml.load(fh, Loader=loader)
    return out


def dimensions(ur: dict) -> dict:
    k = ur["default_kinematics"]["kinematics"]
    d1 = k["shoulder"]["z"]
    a2 = -k["forearm"]["x"]
    a3 = -k["wrist_1"]["x"]
    d4 = k["wrist_1"]["z"]
    d5 = -k["wrist_2"]["y"]
    d6 = k["wrist_3"]["y"]
    lat_upper = UR10E["lat_upper"] * S_SHOULDER
    lat_fore = UR10E["lat_fore"] * S_ELBOW
    lat_wrist = d4 - lat_upper + lat_fore
    return {
        "d1": d1,
        "a2": a2,
        "a3": a3,
        "d4": d4,
        "d5": d5,
        "d6": d6,
        "lat_upper": lat_upper,
        "lat_fore": lat_fore,
        "lat_wrist": lat_wrist,
    }


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


#: Per link: the MJCF body frame's axes written in the description's link frame (rows), and where
#: the body origin lies in that link frame (a function of the dimensions). Derived from the
#: description's joint origins (shoulder_lift rpy (pi/2,0,0); wrist_2 rpy (pi/2,0,0); wrist_3
#: rpy (pi/2,pi,pi)) against the ur10e MJCF's body quats.
FRAMES = {
    "shoulder": (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), lambda d: np.zeros(3)),
    "upper_arm": (
        np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
        lambda d: np.array([0, 0, d["lat_upper"]]),
    ),
    "forearm": (
        np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
        lambda d: np.array([0, 0, d["lat_upper"] - d["lat_fore"]]),
    ),
    "wrist_1": (
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        lambda d: np.array([0, 0, -d["lat_wrist"]]),
    ),
    "wrist_2": (np.eye(3), lambda d: np.array([0, 0, -d["d5"]])),
    "wrist_3": (np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), lambda d: np.array([0, 0, -d["d6"]])),
}


def inertials(ur: dict, dims: dict) -> dict:
    """Mass, centre of mass and full inertia of each link, in the MJCF body frames."""
    p = ur["physical_parameters"]["inertia_parameters"]
    out = {}
    for link, (axes, origin) in FRAMES.items():
        cog = p["center_of_mass"][f"{link}_cog"]
        com_link = np.array([cog["x"], cog["y"], cog["z"]])
        t = p["tensor"][link]
        tensor = np.array(
            [
                [t["ixx"], t["ixy"], t["ixz"]],
                [t["ixy"], t["iyy"], t["iyz"]],
                [t["ixz"], t["iyz"], t["izz"]],
            ]
        )
        rot = p["rotation"][link]
        r = _rpy(rot["roll"], rot["pitch"], rot["yaw"])
        tensor_link = r @ tensor @ r.T
        com = axes @ (com_link - origin(dims))
        full = axes @ tensor_link @ axes.T
        out[link] = {"mass": p[f"{link}_mass"], "pos": com, "full": full}
    out["base"] = {"mass": p["base_mass"]}
    return out


# --- meshes -------------------------------------------------------------------------------------


def _housing(v: np.ndarray, centre, new_centre, s: float) -> np.ndarray:
    return (v - np.asarray(centre)) * s + np.asarray(new_centre)


def _link(
    v: np.ndarray, length: float, new_length: float, s_p: float, s_d: float, zp: float, zd: float
) -> np.ndarray:
    """A long link along z: housings at z=0 and z=length scaled about their joints, the tube between
    them stretched to meet them and its radius blended from one housing's scale to the other's."""
    out = np.empty_like(v)
    z = v[:, 2]
    prox = z < zp
    dist = z > length - zd
    mid = ~(prox | dist)
    out[prox] = v[prox] * s_p
    out[dist] = _housing(v[dist], (0, 0, length), (0, 0, new_length), s_d)
    lo, hi = zp, length - zd
    nlo, nhi = s_p * zp, new_length - s_d * zd
    f = (z[mid] - lo) / (hi - lo)
    s = s_p + (s_d - s_p) * f
    out[mid, 0] = v[mid, 0] * s
    out[mid, 1] = v[mid, 1] * s
    out[mid, 2] = nlo + f * (nhi - nlo)
    return out


def mesh_transforms(d: dict) -> dict:
    u = UR10E
    return {
        "base": lambda v: v * S_SHOULDER,
        "shoulder": lambda v: v * S_SHOULDER,
        "upperarm": lambda v: _link(v, u["a2"], d["a2"], S_SHOULDER, S_ELBOW, 0.10, 0.09),
        "forearm": lambda v: _link(v, u["a3"], d["a3"], S_ELBOW, S_WRIST, 0.08, 0.075),
        "wrist1": lambda v: _housing(v, (0, u["lat_wrist"], 0), (0, d["lat_wrist"], 0), S_WRIST),
        "wrist2": lambda v: _housing(v, (0, 0, u["d5"]), (0, 0, d["d5"]), S_WRIST),
        "wrist3": lambda v: _housing(v, (0, u["flange_y"], 0), (0, d["d6"], 0), S_WRIST),
    }


MESH_PARTS = {
    "base": ["base_0", "base_1"],
    "shoulder": ["shoulder_0", "shoulder_1", "shoulder_2"],
    "upperarm": ["upperarm_0", "upperarm_1", "upperarm_2", "upperarm_3"],
    "forearm": ["forearm_0", "forearm_1", "forearm_2", "forearm_3"],
    "wrist1": ["wrist1_0", "wrist1_1", "wrist1_2"],
    "wrist2": ["wrist2_0", "wrist2_1", "wrist2_2"],
    "wrist3": ["wrist3"],
}


def write_meshes(d: dict, out_dir: Path) -> None:
    """Reshape each ur10e mesh part and write it with its own vertex normals.

    The source OBJ repeats a vertex per face corner; corners that share a position and a normal are
    merged, which halves the files without touching how they shade. The normals are kept rather than
    left to MuJoCo, which would smooth them across every sharp edge of a housing. Each region is moved
    by a uniform scale, so a normal keeps its direction, and the tubes' slight axial stretch tilts
    theirs by a degree or two at most.
    """
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    for link, transform in mesh_transforms(d).items():
        for part in MESH_PARTS[link]:
            mesh = trimesh.load(UR10E_MESHES / f"{part}.obj", force="mesh", process=False)
            normals = np.asarray(mesh.vertex_normals)
            key = np.round(np.hstack([mesh.vertices, normals]), 7)
            _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
            verts = transform(np.asarray(mesh.vertices)[first])
            faces = inverse.reshape(-1)[np.asarray(mesh.faces)]
            lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
            lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in normals[first]]
            lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in faces + 1]
            (out_dir / f"{part}.obj").write_text("\n".join(lines) + "\n")


# --- MJCF ---------------------------------------------------------------------------------------


def _f(*values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _inertial(i: dict) -> str:
    f = i["full"]
    return (
        f'<inertial mass="{i["mass"]:.6g}" pos="{_f(*i["pos"])}" '
        f'fullinertia="{_f(f[0, 0], f[1, 1], f[2, 2], f[0, 1], f[0, 2], f[1, 2])}"/>'
    )


def build_xml(ur: dict) -> str:
    d = dimensions(ur)
    inr = inertials(ur, d)
    lim = ur["joint_limits"]["joint_limits"]
    eff = {j: lim[j]["max_effort"] for j in lim}
    rng = {j: lim[j]["max_position"] for j in lim}
    s, e, w = S_SHOULDER, S_ELBOW, S_WRIST
    base_r = ur["physical_parameters"]["inertia_parameters"]["links"]["base"]["radius"]
    base_l = ur["physical_parameters"]["inertia_parameters"]["links"]["base"]["length"]
    base_ixx = inr["base"]["mass"] * (3 * base_r**2 + base_l**2) / 12
    base_izz = inr["base"]["mass"] * base_r**2 / 2

    def visual(link: str, materials: list[str]) -> str:
        return "\n".join(
            f'{{ind}}<geom mesh="{part}" material="{mat}" class="visual"/>'
            for part, mat in zip(MESH_PARTS[link], materials, strict=True)
        )

    meshes = "\n".join(
        f'    <mesh file="{p}.obj"/>' for parts in MESH_PARTS.values() for p in parts
    )

    def ind(text: str, n: int) -> str:
        return text.replace("{ind}", " " * n)

    return f"""<mujoco model="ur30">
  <!--
    Universal Robots UR30 (6-DoF collaborative arm, 30 kg payload, 1300 mm reach) for MuJoCo.

    GENERATED by external/convert/build_ur30_mjcf.py -- do not hand-edit, re-run the generator.

    Kinematics, link masses, centres of mass, inertia tensors, joint ranges and effort limits are
    UR's own, from Universal_Robots_ROS2_Description config/ur30 @ {UR_COMMIT[:12]}
    (BSD-3-Clause). The joint effort limits there cite UR's UR20 manual; UR states the UR30's
    joints are stronger, so these limits are on the safe side.

    The meshes are NOT UR's UR30 artwork (that is under UR's graphical-documentation terms, which
    roqsim does not ship). They are MuJoCo Menagerie's ur10e meshes (BSD-3-Clause, see
    UR30_LICENSE), reshaped by the generator: housings scaled about their joints, tubes stretched
    between them, so each housing sits on the UR30's joint.

    d1 {d["d1"]:.4f}  a2 {d["a2"]:.4f}  a3 {d["a3"]:.4f}  d4 {d["d4"]:.4f}  d5 {d["d5"]:.4f}  d6 {d["d6"]:.4f}
    lateral offsets (sum = d4): upper arm {d["lat_upper"]:.4f}, elbow -{d["lat_fore"]:.4f}, wrist {d["lat_wrist"]:.4f}

    Actuator force ranges are the effort limits, so the simulated arm cannot hold what the real
    one is not rated for. The armature is the gearboxes' reflected inertia (5, 2 and 1 kg m^2 for
    the three joint sizes, the order a ~100:1 harmonic drive gives), which is what lets the servos
    be stiff at a 2 ms step: a 25 kg payload not compensated by the controller (a part held in the
    gripper, which is no part of the arm) moves the flange by under a millimetre.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="ur30">
      <material specular="0.5" shininess="0.25"/>
      <joint axis="0 1 0" range="{_f(-rng["shoulder_pan_joint"], rng["shoulder_pan_joint"])}"/>
      <general biastype="affine" ctrlrange="-6.2831 6.2831"/>
      <default class="size5">
        <joint damping="20" armature="5"/>
        <general forcerange="{_f(-eff["shoulder_lift_joint"], eff["shoulder_lift_joint"])}" gainprm="600000" biasprm="0 -600000 -20000"/>
      </default>
      <default class="size4">
        <joint damping="10" armature="2" range="{_f(-rng["elbow_joint"], rng["elbow_joint"])}"/>
        <general forcerange="{_f(-eff["elbow_joint"], eff["elbow_joint"])}" ctrlrange="{_f(-rng["elbow_joint"], rng["elbow_joint"])}"
                 gainprm="300000" biasprm="0 -300000 -10000"/>
      </default>
      <default class="size3">
        <joint damping="3" armature="1"/>
        <general forcerange="{_f(-eff["wrist_1_joint"], eff["wrist_1_joint"])}" gainprm="100000" biasprm="0 -100000 -3000"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="capsule" group="3"/>
        <default class="eef_collision">
          <geom type="cylinder"/>
        </default>
      </default>
      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
  </default>

  <asset>
    <material class="ur30" name="black" rgba="0.033 0.033 0.033 1"/>
    <material class="ur30" name="jointgray" rgba="0.278 0.278 0.278 1"/>
    <material class="ur30" name="linkgray" rgba="0.82 0.82 0.82 1"/>
    <material class="ur30" name="urblue" rgba="0.49 0.678 0.8 1"/>

{meshes}
  </asset>

  <worldbody>
    <body name="base" pos="0 0 0" childclass="ur30">
      <inertial mass="{inr["base"]["mass"]:.6g}" pos="0 0 0" diaginertia="{_f(base_ixx, base_ixx, base_izz)}"/>
{ind(visual("base", ["black", "jointgray"]), 6)}
      <body name="shoulder_link" pos="0 0 {d["d1"]:.6g}">
        {_inertial(inr["shoulder"])}
        <joint name="shoulder_pan_joint" class="size5" axis="0 0 1"/>
{ind(visual("shoulder", ["urblue", "black", "jointgray"]), 8)}
        <!-- Shorter than the housing scale alone gives: scaled, the ur10e's capsule reaches 34 mm
             under the base plate and into the floor the arm stands on. -->
        <geom class="collision" size="{_f(0.078 * s, 0.06)}" pos="0 0 -0.03"/>
        <body name="upper_arm_link" pos="0 {d["lat_upper"]:.6g} 0" quat="1 0 1 0">
          {_inertial(inr["upper_arm"])}
          <joint name="shoulder_lift_joint" class="size5"/>
{ind(visual("upperarm", ["black", "jointgray", "urblue", "linkgray"]), 10)}
          <geom class="collision" pos="0 {-0.05 * s:.6g} 0" quat="1 1 0 0" size="{_f(0.078 * s, 0.08 * s)}"/>
          <geom class="collision" size="{_f(0.06 * s, d["a2"] / 2 - 0.02)}" pos="0 0 {d["a2"] / 2:.6g}"/>
          <body name="forearm_link" pos="0 {-d["lat_fore"]:.6g} {d["a2"]:.6g}">
            {_inertial(inr["forearm"])}
            <joint name="elbow_joint" class="size4"/>
{ind(visual("forearm", ["urblue", "black", "jointgray", "linkgray"]), 12)}
            <geom class="collision" pos="0 {0.08 * e:.6g} 0" quat="1 1 0 0" size="{_f(0.058 * e, 0.065 * e)}"/>
            <geom class="collision" size="{_f(0.043 * e, d["a3"] / 2 - 0.03)}" pos="0 0 {d["a3"] / 2:.6g}"/>
            <body name="wrist_1_link" pos="0 0 {d["a3"]:.6g}" quat="1 0 1 0">
              {_inertial(inr["wrist_1"])}
              <joint name="wrist_1_joint" class="size3"/>
{ind(visual("wrist1", ["black", "urblue", "jointgray"]), 14)}
              <geom class="collision" pos="0 {d["lat_wrist"] + w * (0.06 - UR10E["lat_wrist"]):.6g} 0" quat="1 1 0 0" size="{_f(0.05 * w, 0.07 * w)}"/>
              <body name="wrist_2_link" pos="0 {d["lat_wrist"]:.6g} 0">
                {_inertial(inr["wrist_2"])}
                <joint name="wrist_2_joint" axis="0 0 1" class="size3"/>
{ind(visual("wrist2", ["black", "urblue", "jointgray"]), 16)}
                <geom class="collision" size="{_f(0.046 * w, 0.065 * w)}" pos="0 0 {d["d5"] + w * (0.05 - UR10E["d5"]):.6g}"/>
                <geom class="collision" pos="0 {0.028 * w:.6g} {d["d5"]:.6g}" quat="1 1 0 0" size="{_f(0.046 * w, 0.043 * w)}"/>
                <body name="wrist_3_link" pos="0 0 {d["d5"]:.6g}">
                  {_inertial(inr["wrist_3"])}
                  <joint name="wrist_3_joint" class="size3"/>
                  <geom material="linkgray" mesh="wrist3" class="visual"/>
                  <geom class="eef_collision" pos="0 {d["d6"] + w * (0.097 - UR10E["flange_y"]):.6g} 0" quat="1 1 0 0" size="{_f(0.046 * w, 0.02 * w)}"/>
                  <!-- The flange face, d6 out from the wrist 3 axis. -->
                  <site name="attachment_site" pos="0 {d["d6"]:.6g} 0" quat="-1 1 0 0"/>
                  <!-- Tool body: everything distal of the flange. The UR30 measures force and torque
                       in its flange, so `fts_site` is the flange face: a site sensor there reports
                       the wrench transmitted from this body's children, and a tool welded at
                       `tool_site` hangs below it. -->
                  <body name="tool0" pos="0 {d["d6"]:.6g} 0" quat="-1 1 0 0">
                    <site name="tool0_site" pos="0 0 0" group="1"/>
                    <site name="fts_site" size="0.001 0.001 0.001" pos="0 0 0" rgba="0 1 0 1" group="4"/>
                    <site name="tool_site" pos="0 0 0" group="1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <general class="size5" name="shoulder_pan" joint="shoulder_pan_joint"/>
    <general class="size5" name="shoulder_lift" joint="shoulder_lift_joint"/>
    <general class="size4" name="elbow" joint="elbow_joint"/>
    <general class="size3" name="wrist_1" joint="wrist_1_joint"/>
    <general class="size3" name="wrist_2" joint="wrist_2_joint"/>
    <general class="size3" name="wrist_3" joint="wrist_3_joint"/>
  </actuator>

  <contact>
    <!-- A tool welded at tool_site sits on the flange face, inside wrist_3_link's collision
         cylinder. Without this exclude the F/T sensor reads a constant self-contact wrench. -->
    <exclude name="flange_tool" body1="wrist_3_link" body2="tool0"/>
  </contact>

  <!-- The F/T sensor pair is added by the `force_torque` plugin (roqsim_sensors) at `fts_site`, not
       declared here: whether a run measures the wrench is an experiment's choice. The home pose is
       applied by spawn_arm, not a keyframe. -->
</mujoco>
"""


LICENSE_HEADER = """The UR30 model in this folder is built by external/convert/build_ur30_mjcf.py from two sources,
both under the BSD 3-Clause licence reproduced below:

* Kinematic and dynamic parameters: Universal Robots' Universal_Robots_ROS2_Description,
  config/ur30 @ {commit}, Copyright (c) 2021, Universal Robots A/S, BSD-3-Clause.
* Meshes: MuJoCo Menagerie's universal_robots_ur10e meshes, Copyright 2018 ROS Industrial
  Consortium, BSD-3-Clause, reshaped to the UR30's dimensions. They are not Universal Robots'
  UR30 meshes.

"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="rebuild the MJCF and diff against the committed one"
    )
    args = parser.parse_args()

    source = resolve_source("ur_description", UR_URL, UR_COMMIT, sparse=UR_SUBDIR, subdir=UR_SUBDIR)
    ur = load_ur30(source)
    xml = build_xml(ur)
    target = PKG / "ur30.xml"
    if args.check:
        if not target.exists() or target.read_text() != xml:
            print(f"{target} differs from a fresh build -- was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {UR_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    write_meshes(dimensions(ur), PKG / "meshes")
    bsd = (MODELS / "ur10e" / "UR10e_LICENSE").read_text()
    (PKG / "UR30_LICENSE").write_text(LICENSE_HEADER.format(commit=UR_COMMIT) + bsd)
    target.write_text(xml)
    print(f"wrote {target}, meshes and UR30_LICENSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
