"""Build the roqsim `frankie` model: a Franka Emika Panda on an Omron LD-60 differential-drive base.

Frankie is the QUT Centre for Robotics mobile manipulator (Burgess-Limerick et al.). This script
composes it deterministically instead of hand-authoring 250 lines of duplicated Panda:

  1. author the Omron base (visual mesh + chassis collision + 2 drive wheels + 2 casters + actuators),
  2. `MjSpec.attach` the substrate's existing `panda` model (resolved, not path-hardcoded) at the URDF mount frame,
  3. post-process what neither source expresses (palm camera site, home keyframe),
  4. compile and print the cross-checks, then write `frankie.xml`.

Reusing the substrate's own `panda` model is deliberate: its Menagerie-derived meshes are split per material and
its Franka Hand already carries fingertip-pad collision geoms, both better than the source DAEs. Only
`omron.dae` is converted (via dae2obj.py). The model therefore borrows the panda meshes through its
manifest `assets: [roqsim_manipulation_assets]` key rather than copying them -- see architecture.rst §4.

Provenance
----------
Base geometry + mount transform: `qut_frankie_description`, vendored in
petercorke/robotics-toolbox-python @ 0bb96454 under `rtb-data/rtbdata/xacro/qut_frankie_description`
(MIT, (c) 2020 jhavl). The URDF fixes `panda_base_arm_joint` at xyz="0.15 0 0.38" from the base link to
`panda_link0`, and declares a base collision box of 0.68 x 0.47 x 0.38 m centred at z=0.19.

Run:
    python external/convert/dae2obj.py <rtb>/.../qut_frankie_description/meshes/visual  /tmp/omron_out
    python external/convert/build_frankie_mjcf.py --omron-objs /tmp/omron_out
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from roqsim.models import resolve_model

HERE = Path(__file__).resolve()
# external/ is a sibling of the family packages, so anchor back through parents[2] (see robot-porting
# Step 7: build scripts never live inside the package they build into).
ROOT = HERE.parents[2]
# Output package: `roqsim_mobile_manipulation`, the sibling family for robots that are a base AND an arm.
# Frankie first landed in `roqsim_mobile`, which forced that package to depend on `roqsim_manipulation` --
# inverting a base package's contract, pulling every arm and gripper into a plain TurtleBot install, and
# breaking a container image build. See CLAUDE.md, "Family packages are siblings, not a chain".
FAMILY = ROOT / "roqsim_mobile_manipulation/src/roqsim_mobile_manipulation/models"
# One folder per model: `models/frankie/frankie.xml` with its own `meshes/` beside it, which is what
# the package's MESHES_DIR (= the models root) and its package-data globs expect. The model's own
# meshes are therefore named BARE against `meshdir="meshes"`; only the borrowed panda meshes carry a
# `panda/meshes/` prefix, because they resolve against the *provider's* mesh root.
MODEL_DIR = FAMILY / "frankie"

# --- Geometry ---------------------------------------------------------------------------------------
# From the source URDF (authoritative):
MOUNT_XYZ = (0.15, 0.0, 0.38)  # base link -> panda_link0, fixed joint, no rotation
BOX_L, BOX_W, BOX_H = 0.68, 0.47, 0.38  # declared base collision box

# Substrate assumptions (the URDF models the base as a virtual yaw+prismatic joint pair with NO wheels,
# no inertials and no ground clearance, so all of this is ours and is recorded in
# the port log). The LD-60 carries its drive wheels inboard, under the shell.
WHEEL_R = 0.0625  # drive wheel radius
WHEEL_HW = 0.025  # wheel half-width
WHEEL_SEP = 0.36  # track: axle y = +/- 0.18, comfortably inside the 0.47 m body width
CASTER_R = WHEEL_R
# TRIPOD, not four legs. Getting this wrong cost two iterations, both diagnosed by measuring rather
# than by looking at a settled pose:
#   * Four coplanar contacts (both casters at lift 0) are statically indeterminate -- the base rests on
#     whichever geom is marginally penetrating this tick.
#   * Two casters BOTH near the wheel plane (lift 3 mm) turn the single-axle base into a see-saw: it
#     rocked +/-1.8 deg fore-aft and BOTH DRIVE WHEELS LEFT THE GROUND periodically (measured normal
#     force exactly 0 N on each wheel while they spun at the full commanded 4.775 rad/s). The robot
#     crept forward at 0.10 m/s against a 0.3 m/s command -- the G2 port log's "casters too low -> they
#     carry the load and the wheels lose traction" failure, arrived at from the opposite direction.
# The stable arrangement is a tripod: the two drive wheels plus the FRONT caster carry the load, and
# the rear caster is lifted clear as a pure anti-tip stop. It is stable because the CoM (x = +0.058 m,
# pulled forward by the arm) lies BETWEEN the wheel axle (x = 0) and the front caster (x = +0.26), so
# the front caster can never unload -- no rocking is possible.
CASTER_FRONT_LIFT = 0.0  # load-bearing: the third leg of the tripod
CASTER_REAR_LIFT = 0.015  # anti-tip only; never a support at any normal attitude
CASTER_X = 0.26  # fore/aft caster offset, inside the 0.34 m half-length
CHASSIS_MASS = 60.0  # LD-60-class AGV; the URDF links are inertia-less
WHEEL_MASS = 1.5
# The chassis collision box starts ABOVE the wheels so the box does not swallow them and rest on the
# floor itself. Consequence (documented): obstacles below 0.125 m are not collided by the base.
CHASSIS_Z0 = 0.125
SPAWN_Z = 0.010  # spawn just above rest height, like husky_a200

# The timestep this model is verified at, and the reason it is not the substrate default.
#
# At MuJoCo's default dt=2 ms the STOCK roqsim panda -- bare, standing still at its rest pose, before any
# of this port existed -- produces end-effector acceleration noise of mean 0.94 and max 4.87 m/s^2.
# A robot doing nothing should register no acceleration, and that floor is large enough to swamp any
# smoothness or effort measurement taken on this arm. It is a PD-servo/timestep interaction (kp 4500 /
# kd 450 with armature 0.1), not a Frankie defect, and it disappears with a smaller step: mean 0.007 at
# 1 ms and exactly 0.000 at 0.5 ms. The demo world therefore runs at 0.5 ms and this script verifies
# quiescence there. A substrate artifact: it belongs in the port log beside the model.
VERIFIED_TIMESTEP = 0.0005

# Arm rest pose -- `qr`, the ready pose shipped by the Robotics Toolbox Frankie model
# (rtb models/URDF/Frankie.py).
ARM_REST = (0.0, -0.3, 0.0, -2.2, 0.0, 2.0, np.pi / 4)


def _repair_defaults(xml: str) -> str:
    """Unwrap anonymous nested <default> blocks that MjSpec.attach emits and MuJoCo cannot re-read.

    Attaching a child spec nests the child's whole default tree under a class-less <default>, and
    MuJoCo's XML parser then rejects its own output with "empty class name". Only the single top-level
    <default> may be anonymous, so splice any nested anonymous block's children into its parent. This
    is a lossless structural move: the nested block carries no attributes of its own.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)

    def unwrap(parent: ET.Element) -> None:
        changed = True
        while changed:
            changed = False
            for child in list(parent):
                if child.tag == "default" and "class" not in child.attrib:
                    idx = list(parent).index(child)
                    parent.remove(child)
                    for k, sub in enumerate(list(child)):
                        parent.insert(idx + k, sub)
                    changed = True
            for child in parent:
                if child.tag == "default":
                    unwrap(child)

    for top in root.findall("default"):
        unwrap(top)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _strip_floor(xml: str) -> str:
    """Remove the verification ground plane, leaving the model's own bodies untouched."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    if wb is not None:
        for g in list(wb.findall("geom")):
            if g.get("name") == "floor":
                wb.remove(g)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _rp(quat) -> tuple[float, float]:
    """Roll/pitch (rad) from a wxyz quaternion -- a settled wheeled base must show ~0 for both."""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return round(float(np.degrees(roll)), 3), round(float(np.degrees(pitch)), 3)


def base_xml(materials: list[tuple[str, list[float]]]) -> str:
    """The Omron LD-60 base as standalone MJCF, ready to receive the panda at MOUNT_XYZ."""
    mats = "\n".join(
        f'    <material class="frankie" name="{n}" rgba="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} 1"/>'
        for n, c in materials
    )
    meshes = "\n".join(f'    <mesh file="{n}.obj"/>' for n, _ in materials)
    visuals = "\n".join(
        f'        <geom mesh="{n}" material="{n}" class="f_visual"/>' for n, _ in materials
    )
    hy = WHEEL_SEP / 2.0
    box_hz = (BOX_H - CHASSIS_Z0) / 2.0
    box_cz = CHASSIS_Z0 + box_hz
    return (
        f"""<mujoco model="frankie_base">
  <compiler angle="radian" autolimits="true" meshdir="meshes"/>

  <default>
    <default class="frankie">
      <material specular="0.3" shininess="0.2"/>
      <default class="f_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="f_collision">
        <geom group="3" rgba="0.4 0.5 0.9 0.3"/>
      </default>
      <default class="f_wheel">
        <!-- `fromto` rather than size+quat: a `quat` on a geom DEFAULT is silently dropped by
             MjSpec's XML round-trip, which left the wheel cylinders unrotated -- flat discs lying on
             the floor instead of upright wheels, with the bbox and mass unchanged. fromto states the
             axis in the geom itself and survives. -->
        <geom type="cylinder" fromto="0 {-WHEEL_HW} 0 0 {WHEEL_HW} 0" size="{WHEEL_R}"
              friction="1.2 0.005 0.0001" condim="4" group="3" rgba="0.15 0.15 0.15 1"/>
        <!-- `armature` is the drivetrain's reflected rotor inertia (rotor inertia x gear ratio^2), as
             on the jackal; `frictionloss` is the Coulomb term that makes zero velocity an ATTRACTOR.
             Both are needed for stability, not realism: a bare wheel (I~0.003) under a kv=40 velocity
             servo at dt=2 ms has kv*dt/I >> 1, and the first build of this model sat in a permanent
             +/-10 rad/s limit cycle -- wheels chattering while the base stood still, which also shook
             the arm at ~1 rad/s. armature raises the effective inertia to make the servo loop stable
             (kv*dt/I ~ 1) and frictionloss stops the residual hunting. The husky uses frictionloss
             2.0, the jackal armature 0.02 + frictionloss 0.15. -->
        <joint type="hinge" axis="0 1 0" armature="0.08" frictionloss="1.0"/>
      </default>
      <default class="f_caster">
        <!-- A passive ball caster must not resist yaw, and a LOW `friction` ON THE GEOM DOES NOT
             ACHIEVE THAT: MuJoCo combines the two geoms' friction as the element-wise MAX, so a
             0.02-friction sphere on a 1.0-friction floor still drags at 1.0. The first build made
             exactly that mistake. The turtlebot4's pattern is the reliable one -- disable automatic
             collision (contype/conaffinity 0) and give the caster a single explicit <pair> with
             condim="1" (normal force only, genuinely frictionless). Modelled as a sphere rather than
             an articulated swivel: the LD-60's casters are small and their swivel dynamics do not
             matter at 0.2-0.5 m/s. -->
        <geom type="sphere" size="{CASTER_R}" contype="0" conaffinity="0" group="3"
              rgba="0.2 0.2 0.2 1"/>
      </default>
    </default>
  </default>

  <asset>
{mats}
{meshes}
  </asset>

  <worldbody>
    <body name="base_link" pos="0 0 {SPAWN_Z}" childclass="frankie">
      <freejoint name="base_free"/>
      <inertial pos="0 0 0.19" mass="{CHASSIS_MASS}"
                diaginertia="{CHASSIS_MASS * (BOX_W**2 + BOX_H**2) / 12:.4f} """
        + f"""{CHASSIS_MASS * (BOX_L**2 + BOX_H**2) / 12:.4f} """
        + f"""{CHASSIS_MASS * (BOX_L**2 + BOX_W**2) / 12:.4f}"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <site name="base_mount" pos="{MOUNT_XYZ[0]} {MOUNT_XYZ[1]} {MOUNT_XYZ[2]}" size="0.008"
            rgba="0 1 0 0"/>
{visuals}
      <!-- Chassis collision: the URDF's declared footprint in x/y, raised clear of the wheels in z. -->
      <geom name="chassis" type="box" size="{BOX_L / 2} {BOX_W / 2} {box_hz:.4f}"
            pos="0 0 {box_cz:.4f}" class="f_collision"/>
      <body name="left_wheel" pos="0 {hy} {WHEEL_R}">
        <joint name="left_wheel_joint" class="f_wheel"/>
        <geom name="left_wheel" class="f_wheel" mass="{WHEEL_MASS}"/>
      </body>
      <body name="right_wheel" pos="0 -{hy} {WHEEL_R}">
        <joint name="right_wheel_joint" class="f_wheel"/>
        <geom name="right_wheel" class="f_wheel" mass="{WHEEL_MASS}"/>
      </body>
      <geom name="caster_front" class="f_caster" pos="{CASTER_X} 0 {CASTER_R + CASTER_FRONT_LIFT}"
            mass="0.5"/>
      <geom name="caster_rear" class="f_caster" pos="-{CASTER_X} 0 {CASTER_R + CASTER_REAR_LIFT}"
            mass="0.5"/>
    </body>
  </worldbody>

  <contact>
    <!-- Explicit wheel and caster contacts against the world's ground geom, which by substrate
         convention is named `floor` (every baked scene and roqsim_mobile/models/floor/floor.xml use that name;
         husky_a200 and turtlebot4 do the same). CONSEQUENCE: this model does not compile standalone --
         it must be spawned into a world that provides `floor`, which is also what robot-porting
         Step 6.5 requires anyway.
         Wheels: condim=3 isotropic friction 1.0, softened solref for stable stepping (husky pattern).
         Casters: condim=1 -- normal force only, a true frictionless ball caster (turtlebot4 pattern). -->
    <pair geom1="left_wheel" geom2="floor" condim="3" solref="0.02 1"
          friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="right_wheel" geom2="floor" condim="3" solref="0.02 1"
          friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair geom1="caster_front" geom2="floor" condim="1" solref="0.02 1"/>
    <pair geom1="caster_rear" geom2="floor" condim="1" solref="0.02 1"/>
  </contact>

  <actuator>
    <!-- Velocity servos, as diff_drive expects. kv/forcerange sized to move a ~63 kg base without
         stalling against caster friction; see port log. -->
    <velocity name="left_wheel_motor" joint="left_wheel_joint" kv="40" ctrlrange="-20 20"
              forcerange="-80 80"/>
    <velocity name="right_wheel_motor" joint="right_wheel_joint" kv="40" ctrlrange="-20 20"
              forcerange="-80 80"/>
  </actuator>

  <sensor>
    <framepos name="base_pos" objtype="site" objname="base_imu"/>
    <framequat name="base_quat" objtype="site" objname="base_imu"/>
    <velocimeter name="imu_vel" site="base_imu"/>
    <gyro name="imu_gyro" site="base_imu"/>
    <jointpos name="left_wheel_pos" joint="left_wheel_joint"/>
    <jointpos name="right_wheel_pos" joint="right_wheel_joint"/>
    <jointvel name="left_wheel_vel" joint="left_wheel_joint"/>
    <jointvel name="right_wheel_vel" joint="right_wheel_joint"/>
  </sensor>
</mujoco>
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--omron-objs",
        type=Path,
        required=True,
        help="dae2obj.py output dir containing omron__m*.obj + materials.json",
    )
    ap.add_argument("--out", type=Path, default=MODEL_DIR / "frankie.xml")
    args = ap.parse_args()

    mats_path = args.omron_objs / "materials.json"
    if not mats_path.exists():
        raise SystemExit(
            f"error: {mats_path} missing -- run dae2obj.py on the frankie visual meshes"
        )
    materials = [(n, c) for n, c in json.loads(mats_path.read_text())["omron"]]
    print(f"omron: {len(materials)} material sub-meshes")

    # Install the converted base meshes into the model folder's own mesh dir.
    dest = MODEL_DIR / "meshes"
    dest.mkdir(parents=True, exist_ok=True)
    for name, _ in materials:
        shutil.copy2(args.omron_objs / f"{name}.obj", dest / f"{name}.obj")
    print(f"installed {len(materials)} meshes -> {dest.relative_to(ROOT)}")

    # --- compose ------------------------------------------------------------------------------------
    # A staging meshdir with symlinks to BOTH providers, so the composed model compiles here exactly as
    # `apply_assets` will resolve it at spawn time: the model's own omron meshes bare at the top level,
    # the borrowed `panda/meshes/*` one level down. Without this the verification compile cannot find
    # the borrowed panda meshes.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "meshes"
        stage.mkdir()
        # Resolve the borrowed arm through `resolve_model`, not a hardcoded package path: `panda.xml`
        # and its meshes have already moved once (roqsim_manipulation -> roqsim_manipulation_assets), and a
        # literal path turns that into a silent break in a *build* script, which nothing tests.
        arm = resolve_model("panda")
        # roqsim_manipulation_assets is laid out one folder per model, so the borrowed provider's mesh
        # root is its MODELS dir and the arm's own meshes sit one level down in `panda/meshes/` --
        # which is exactly how frankie.xml names them (`panda/meshes/<file>`).
        assets_root = next(d for d in arm.meshdirs if (d / "panda").is_dir())
        (stage / "panda").symlink_to(assets_root / "panda")
        for name, _ in materials:
            (stage / f"{name}.obj").symlink_to(dest / f"{name}.obj")

        base = base_xml(materials)
        (Path(tmp) / "frankie_base.xml").write_text(base)
        # The model's <contact> pairs name `floor`, so even the composition compile needs one. It is
        # added here and removed from the written model: the WORLD owns the ground plane.
        base_c = base.replace('meshdir="meshes"', f'meshdir="{stage}"').replace(
            "<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="30 30 0.1"/>', 1
        )
        spec = mujoco.MjSpec.from_string(base_c)
        panda = mujoco.MjSpec.from_file(str(arm.path))
        # panda.xml names its meshes bare and resolves them against its own folder's meshes/.
        panda.meshdir = str(arm.path.parent / "meshes")

        # Keyframes cannot merge across an attach (spawn_robot strips them for the same reason).
        for s in (panda,):
            while s.keys:
                s.keys[0].delete()

        frame = spec.body("base_link").add_frame(pos=list(MOUNT_XYZ), quat=[1.0, 0.0, 0.0, 0.0])
        spec.attach(panda, prefix="", frame=frame)

        model = spec.compile()
        print(f"compiled: nq={model.nq} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")

        # --- cross-checks (rule 6: verify numbers, not exit codes) ---------------------------------
        data = mujoco.MjData(model)
        arm_joints = [f"joint{i}" for i in range(1, 8)]
        for name, val in zip(arm_joints, ARM_REST, strict=True):
            data.qpos[model.joint(name).qposadr[0]] = val
        mujoco.mj_forward(model, data)

        mount = data.site_xpos[model.site("base_mount").id]
        link0 = data.xpos[model.body("link0").id]
        print(f"  mount site      : {np.round(mount, 4)}")
        print(f"  panda_link0 body: {np.round(link0, 4)}   (must coincide with the mount site)")
        assert np.allclose(mount, link0, atol=1e-6), "panda_link0 is not at the URDF mount frame"

        hand = data.xpos[model.body("hand").id]
        print(
            f"  hand at rest    : {np.round(hand, 4)}  -> reach {np.linalg.norm(hand[:2] - mount[:2]):.3f} m, z {hand[2]:.3f} m"
        )

        print(f"  total mass      : {model.body_subtreemass[model.body('base_link').id]:.2f} kg")

        # Settle test. The composition spec already carries the `floor` plane its contact pairs
        # reference, so it is reused as-is here. (A bare model MJCF has no floor at all -- before that
        # plane existed the settle test dropped the robot to z = -78 m, which read as a physics bug and
        # was only a missing ground.)
        repaired = _repair_defaults(spec.to_xml())
        gspec = mujoco.MjSpec.from_string(repaired)
        gspec.meshdir = str(stage)
        gmodel = gspec.compile()
        gmodel.opt.timestep = VERIFIED_TIMESTEP  # see the VERIFIED_TIMESTEP note
        gdata = mujoco.MjData(gmodel)
        for name, val in zip(arm_joints, ARM_REST, strict=True):
            gdata.qpos[gmodel.joint(name).qposadr[0]] = val
            # HOLD the arm at the rest pose, as arm_controller does every pre_step. Without this the
            # position servos drive toward ctrl=0, i.e. all-zeros -- a pose where the stock Panda's
            # link5 and hand collision geoms genuinely overlap (-0.030 m). Leaving ctrl at 0 made the
            # settle test report a "self-collision" that is an artifact of not commanding the arm.
            gdata.ctrl[gmodel.actuator(f"actuator{arm_joints.index(name) + 1}").id] = val
        for _ in range(3000):  # 6 s at the default 2 ms timestep
            mujoco.mj_step(gmodel, gdata)
        self_con = [
            (
                gmodel.body(gmodel.geom_bodyid[c.geom1]).name,
                gmodel.body(gmodel.geom_bodyid[c.geom2]).name,
            )
            for c in gdata.contact[: gdata.ncon]
            if gmodel.geom(c.geom1).name != "floor" and gmodel.geom(c.geom2).name != "floor"
        ]
        print(
            f"  arm hold error  : {np.max(np.abs([gdata.qpos[gmodel.joint(n).qposadr[0]] - v for n, v in zip(arm_joints, ARM_REST, strict=True)])):.5f} rad"
        )
        print(f"  self-collisions : {self_con or 'none'}")
        print(
            f"  settled base z  : {gdata.qpos[2]:.5f} m   (spawned at {SPAWN_Z}, wheel r {WHEEL_R})"
        )
        print(f"  settled tilt    : roll/pitch {_rp(gdata.qpos[3:7])} deg (must be ~0)")
        carriers = sorted(
            {gmodel.geom(c.geom1).name or f"g{c.geom1}" for c in gdata.contact[: gdata.ncon]}
            | {gmodel.geom(c.geom2).name or f"g{c.geom2}" for c in gdata.contact[: gdata.ncon]}
        )
        print(f"  contacts at rest: {gdata.ncon} on {[c for c in carriers if c != 'floor']}")
        # Load-bearing check, time-AVERAGED. A single-frame contact snapshot is worthless here: with
        # four near-coplanar contacts it reports whichever geom happens to be penetrating this tick,
        # and it once showed the robot balanced on one caster with both drive wheels off the ground.
        fl = gmodel.geom("floor").id
        force = {}
        for _ in range(500):
            mujoco.mj_step(gmodel, gdata)
            for i, c in enumerate(gdata.contact[: gdata.ncon]):
                other = c.geom2 if c.geom1 == fl else (c.geom1 if c.geom2 == fl else None)
                if other is None:
                    continue
                ft = np.zeros(6)
                mujoco.mj_contactForce(gmodel, gdata, i, ft)
                nm = gmodel.geom(other).name
                force[nm] = force.get(nm, 0.0) + abs(float(ft[0])) / 500.0
        weight = gmodel.body_subtreemass[gmodel.body("base_link").id] * 9.81
        print("  support (mean N): " + "  ".join(f"{k}={v:.0f}" for k, v in sorted(force.items())))
        print(f"  total {sum(force.values()):.0f} N vs weight {weight:.0f} N")
        wheel_share = sum(force.get(w, 0.0) for w in ("left_wheel", "right_wheel")) / max(
            sum(force.values()), 1e-9
        )
        assert wheel_share > 0.30, (
            f"drive wheels carry only {wheel_share:.1%} of the load -- traction will be unreliable"
        )

        # Quiescence: the model must actually come to REST. The first build chattered forever (wheel
        # velocity servo unstable at this timestep), which no compile or contact check would catch.
        wheel_qvel = max(
            abs(gdata.qvel[gmodel.joint(j).dofadr[0]])
            for j in ("left_wheel_joint", "right_wheel_joint")
        )
        arm_qvel = float(
            np.linalg.norm([gdata.qvel[gmodel.joint(f"joint{i}").dofadr[0]] for i in range(1, 8)])
        )
        print(
            f"  quiescence      : wheel |qvel| {wheel_qvel:.4f} rad/s, arm |qvel| {arm_qvel:.4f} rad/s"
        )
        assert wheel_qvel < 0.05, f"wheels never settle ({wheel_qvel:.3f} rad/s): servo limit cycle"
        assert arm_qvel < 0.10, f"arm never settles ({arm_qvel:.3f} rad/s)"

        # END-EFFECTOR ACCELERATION NOISE FLOOR -- the number that decides whether an arm-smoothness
        # or effort measurement on this robot means anything. Measured at REST, where the true value
        # is zero, so whatever this reads is pure substrate noise that any real signal must exceed.
        # 0.015 m/s^2 is two orders below the accelerations a reaching motion produces.
        hid = gmodel.body("hand").id
        track = []
        for _ in range(int(2.0 / gmodel.opt.timestep)):
            mujoco.mj_step(gmodel, gdata)
            track.append(gdata.xpos[hid].copy())
        P = np.array(track)
        acc = np.linalg.norm(
            np.gradient(np.gradient(P, gmodel.opt.timestep, axis=0), gmodel.opt.timestep, axis=0),
            axis=1,
        )
        print(
            f"  EE accel noise  : mean {acc.mean():.4f}  max {acc.max():.4f} m/s^2 "
            f"at dt={gmodel.opt.timestep} (at rest, so the true value is 0)"
        )
        assert acc.mean() < 0.015, (
            f"EE acceleration noise floor {acc.mean():.4f} m/s^2 at rest is high enough to contaminate "
            f"an acceleration measurement -- lower VERIFIED_TIMESTEP"
        )

        # TRACTION UNDER DRIVE. The static support check passes even when the base rocks the wheels off
        # the ground the moment it moves, so the wheels must also be shown loaded WHILE driving: spin
        # them open-loop and require the hull to actually travel at the wheels' surface speed. This is
        # the check that caught the see-saw (wheels at full commanded speed, robot creeping at a third
        # of it, normal force periodically 0 N on both).
        lw = gmodel.actuator("left_wheel_motor").id
        rw = gmodel.actuator("right_wheel_motor").id
        wheel_cmd = 0.3 / WHEEL_R  # 0.3 m/s of wheel surface speed
        x0 = float(gdata.qpos[0])
        lifted = 0
        steps = int(4.0 / gmodel.opt.timestep)
        for _ in range(steps):
            for name, val in zip(arm_joints, ARM_REST, strict=True):
                gdata.ctrl[gmodel.actuator(f"actuator{arm_joints.index(name) + 1}").id] = val
            gdata.ctrl[lw] = gdata.ctrl[rw] = wheel_cmd
            mujoco.mj_step(gmodel, gdata)
            on = {
                gmodel.geom(c.geom2 if c.geom1 == fl else c.geom1).name
                for c in gdata.contact[: gdata.ncon]
                if fl in (c.geom1, c.geom2)
            }
            if not ({"left_wheel", "right_wheel"} & on):
                lifted += 1
        travelled = float(gdata.qpos[0]) - x0
        ideal = 0.3 * 4.0
        print(
            f"  drive traction  : travelled {travelled:.3f} m of {ideal:.3f} m ideal "
            f"({travelled / ideal:.1%}), wheels airborne {100 * lifted / steps:.1f}% of steps"
        )
        assert lifted / steps < 0.02, (
            f"drive wheels leave the ground in {100 * lifted / steps:.0f}% of steps -- the base is "
            f"rocking and traction is intermittent"
        )
        assert travelled / ideal > 0.85, (
            f"travelled only {travelled / ideal:.0%} of the wheels' surface distance -- slipping"
        )

        # Strip the verification floor: the WORLD owns the ground plane, and a model shipping its own
        # would collide with (or duplicate the name of) the scene's. Everything above was measured with
        # it present; only the written file loses it.
        xml = _strip_floor(repaired)

        # Write with the RELATIVE meshdir the package uses; apply_assets rewrites it to absolute paths
        # across both providers at spawn time (architecture.rst §4).
        xml = xml.replace(f'meshdir="{stage}/"', 'meshdir="meshes"')
        xml = xml.replace(f'meshdir="{stage}"', 'meshdir="meshes"')

        # MuJoCo's own writer emits XML its own parser rejects (see _repair_defaults), so the file is
        # only trustworthy if it READS BACK. Verify before writing, against the staging meshdir -- with
        # a floor re-added, since the model's contact pairs reference one (as a real world provides).
        check_xml = xml.replace('meshdir="meshes"', f'meshdir="{stage}"').replace(
            "<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="30 30 0.1"/>', 1
        )
        check = mujoco.MjSpec.from_string(check_xml)
        cmodel = check.compile()
        assert (cmodel.nq, cmodel.nu, cmodel.ngeom) == (model.nq, model.nu, model.ngeom), (
            f"round-trip changed the model: {(cmodel.nq, cmodel.nu, cmodel.ngeom)} != "
            f"{(model.nq, model.nu, model.ngeom)}"
        )
        # And the wheels must still be upright cylinders about y, not flat discs (the quat trap).
        for w in ("left_wheel", "right_wheel"):
            gid = cmodel.geom(w).id
            axis = cmodel.geom_quat[gid]
            half = cmodel.geom_size[gid]
            print(f"  {w}: r={half[0]:.4f} halfwidth={half[1]:.4f} quat={np.round(axis, 4)}")
            assert abs(half[0] - WHEEL_R) < 1e-9, f"{w} radius wrong"
        print("  round-trip: XML reads back with an identical model")

    args.out.write_text(xml)
    print(f"\nwrote {args.out.relative_to(ROOT)}  ({len(xml.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
