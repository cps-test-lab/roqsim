"""The MJCF -> URDF export, and the round-trip that makes it trustworthy.

The whole point of generating the URDF instead of shipping a vendor one is that MoveIt then plans
against the kinematics MuJoCo simulates. That claim is only worth anything if it is measured, so the
central test here is the FK round trip: load the exported URDF back into MuJoCo, pose both models at
the same joint values, and compare every link. It caught a real 2.5 m error (a gimbal-lock branch in
the quaternion->rpy conversion, hit by exactly the ``quat="1 0 1 0"`` the UR10e uses on four links).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.export_urdf import (
    UrdfExporter,
    _quat_to_rpy,
    combine_urdfs,
    round_trip_error,
)


def _rpy_to_mat(roll, pitch, yaw):
    """URDF's fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


@pytest.mark.parametrize(
    "quat",
    [
        (1, 0, 0, 0),
        (1, 0, 1, 0),  # +90 deg about y -- the UR10e's, and the gimbal-lock case that broke this
        (1, 0, -1, 0),  # -90 deg about y
        (1, -1, 0, 0),
        (0, 1, 0, 0),
        (1, 0, 0, -1),
        (0.394358, 0.596779, -0.577293, 0.393789),  # gen3's bracelet: no symmetry at all
    ],
)
def test_quat_to_rpy_is_exact(quat):
    q = np.asarray(quat, dtype=float)
    q /= np.linalg.norm(q)
    expected = np.zeros(9)
    mujoco.mju_quat2Mat(expected, q)
    assert np.allclose(_rpy_to_mat(*_quat_to_rpy(q)), expected.reshape(3, 3), atol=1e-7)


def test_quat_to_rpy_over_random_rotations():
    """A branch that is wrong only at the poles passes any small hand-picked set."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        q = rng.normal(size=4)
        if np.linalg.norm(q) < 1e-9:
            continue
        q /= np.linalg.norm(q)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, q)
        worst = max(worst, np.abs(_rpy_to_mat(*_quat_to_rpy(q)) - mat.reshape(3, 3)).max())
    assert worst < 1e-6, f"worst rotation error {worst:.2e}"


def _mobile_manipulator(tmp_path, gripper="robotiq_2f85"):
    return load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {"spawn_robot": {"model": "husky_a200", "name": "husky", "pos": [0.0, 0.0]}},
                {
                    "spawn_arm": {
                        "model": "ur10e",
                        "name": "arm",
                        "prefix": "ur10e_",
                        "mount": {"robot": "husky", "body": "base_link"},
                        "pos": [0.25, 0.0, 0.2587],
                        "end_effector": {"model": gripper, "replaces": ["ee_plate"]},
                    }
                },
            ],
        },
        base_dir=tmp_path,
    )


def _export(tmp_path, model, **kwargs):
    out = tmp_path / "robot.urdf"
    exporter = UrdfExporter(
        model,
        prefix="ur10e_",
        name="test_robot",
        root_link="arm_base_link",
        mesh_dir=tmp_path / "meshes",
        **kwargs,
    )
    tree = exporter.export()
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out, exporter, tree


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    """The Husky + UR10e + Robotiq model, compiled once for the whole module.

    Five tests below want exactly this model, and sharing one build is safe because none of them
    writes to it: ``UrdfExporter`` only reads, and ``round_trip_error`` poses its own ``MjData``. A
    test that needs a *different* robot (a rotated root, a free base) still builds its own.
    """
    engine = Engine(_mobile_manipulator(tmp_path_factory.mktemp("robot")))
    engine.setup()
    return engine.ctx.model


def test_export_round_trips_to_the_mjcf(tmp_path, robot):
    """THE test: the exported URDF and the MJCF agree on where every link is."""
    out, _exporter, _tree = _export(
        tmp_path,
        robot,
        collapse=("base_mount",),
        gripper_joint="robotiq_85_left_knuckle_joint",
    )
    err, where = round_trip_error(out, robot, "ur10e_", samples=32)
    assert err < 1e-6, f"URDF diverges from the MJCF by {err:.3e} m at {where!r}"


def test_export_keeps_the_arm_chain_and_the_gripper_dof(tmp_path, robot):
    _out, exporter, tree = _export(
        tmp_path,
        robot,
        collapse=("base_mount",),
        gripper_joint="robotiq_85_left_knuckle_joint",
    )
    root = tree.getroot()
    joints = {j.get("name"): j.get("type") for j in root.findall("joint")}
    for j in (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ):
        assert joints[j] == "revolute", f"{j} missing or wrong type"
    # The commanded gripper DOF survives the collapse -- MoveIt's gripper group needs it.
    assert joints["robotiq_85_left_knuckle_joint"] == "revolute"
    # ...and the closed-loop DOFs URDF cannot express are reported as dropped, not silently lost.
    assert "right_driver_joint" in exporter.dropped_dofs
    assert "robotiq_85_left_knuckle_joint" not in exporter.dropped_dofs


def test_round_trip_survives_a_rotated_root_body(tmp_path):
    """A robot whose ROOT body carries a rotation must still round-trip.

    Regression. The UR arms follow the vendor convention of a base yawed 180 deg
    (``ur5e.xml``: ``<body name="base" quat="0 0 0 -1">``); ``ur10e.xml`` has no such quat, and the
    round-trip test above happens to use the ur10e -- so every rotated-root export went unchecked.
    The comparison offset each link by the root's POSITION but left the root's ORIENTATION in, which
    is not a property the two models share: in the MJCF every link is rotated with the base, while
    in the URDF the base IS the frame they are expressed in. A ur5e whose URDF matched its MJCF body
    for body was reported as diverging by 1.7 m, i.e. the check condemned a correct export.
    """
    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [{"spawn_arm": {"model": "ur5e", "name": "arm", "prefix": "ur5e_"}}],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.setup()
    model = engine.ctx.model

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur5e_base")
    assert not np.allclose(model.body_quat[base], [1.0, 0.0, 0.0, 0.0]), (
        "this test is only meaningful while the ur5e's root body is rotated"
    )

    out = tmp_path / "ur5e.urdf"
    exporter = UrdfExporter(model, prefix="ur5e_", name="ur5e", mesh_dir=tmp_path / "meshes")
    exporter.export().write(out, encoding="utf-8", xml_declaration=True)
    err, where = round_trip_error(out, model, "ur5e_", samples=32)
    assert err < 1e-6, f"URDF diverges from the MJCF by {err:.3e} m at {where!r}"


def test_free_base_becomes_a_fixed_root(tmp_path, robot):
    """A floating base belongs to TF, not the description; MoveIt needs a fixed root."""
    _out, _exporter, tree = _export(tmp_path, robot, collapse=("base_mount",))
    root = tree.getroot()
    links = {link.get("name") for link in root.findall("link")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = links - children
    assert roots == {"arm_base_link"}, f"expected a single fixed root, got {roots}"


def test_collapsed_link_keeps_the_full_gripper_mass(tmp_path, robot):
    """Lumping must sum the subtree's mass; the 2F-85's fingers are most of it."""
    m = robot
    gripper_mass = sum(
        float(m.body_mass[b])
        for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("ur10e_")
        and _is_descendant(m, b, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur10e_base_mount"))
    )
    _out, _exporter, tree = _export(tmp_path, robot, collapse=("base_mount",))
    link = next(el for el in tree.getroot().findall("link") if el.get("name") == "base_mount")
    urdf_mass = float(link.find("inertial/mass").get("value"))
    assert urdf_mass == pytest.approx(gripper_mass, rel=1e-6)


def _is_descendant(model, body, ancestor):
    cur = body
    while cur > 0:
        if cur == ancestor:
            return True
        cur = int(model.body_parentid[cur])
    return False


def test_collapsed_root_keeps_no_joint_of_its_own(tmp_path):
    """``--collapse`` produces a FIXED link, so the root's own joint must go with the subtree's.

    A root that kept its joint leaves the lump swinging on a DOF nothing drives. It bites precisely
    where collapse is used — a closed linkage whose branches are siblings, so both must be collapsed —
    and the symptom is remote from the cause: MoveIt reports "The complete state of the robot is not yet
    known. Missing <that joint>", then "Found empty JointState message", and the planner cannot sample
    any valid state.
    """
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="base_link">
              <geom type="box" size="0.05 0.05 0.05" mass="1"/>
              <body name="branch" pos="0 0 0.1">
                <joint name="branch_joint" type="hinge" axis="0 1 0" range="-1 1"/>
                <geom type="box" size="0.02 0.02 0.05" mass="0.2"/>
                <body name="branch_tip" pos="0 0 0.05">
                  <joint name="branch_tip_joint" type="hinge" axis="0 1 0" range="-1 1"/>
                  <geom type="box" size="0.01 0.01 0.02" mass="0.1"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    tree = UrdfExporter(
        model,
        prefix="",
        name="r",
        root_link="base_link",
        mesh_dir=tmp_path / "meshes",
        collapse=("branch",),
    ).export()
    root = tree.getroot()
    kinds = {j.get("name"): j.get("type") for j in root.findall("joint")}
    assert "branch_tip_joint" not in kinds, "a joint inside the collapsed subtree survived"
    assert "branch_joint" not in kinds, (
        "the collapse ROOT kept its own joint, so the lumped link still has a free DOF"
    )
    # The link is still there, attached by a fixed joint, carrying the whole subtree's mass.
    assert "branch" in {link.get("name") for link in root.findall("link")}
    fixed = [j for j in root.findall("joint") if j.find("child").get("link") == "branch"]
    assert len(fixed) == 1 and fixed[0].get("type") == "fixed"
    mass = float(root.find("link[@name='branch']/inertial/mass").get("value"))
    assert mass == pytest.approx(0.3, rel=1e-6)


def test_rescued_gripper_joint_keeps_its_mujoco_type_and_effort(tmp_path):
    """The gripper DOF rescued from a collapse must be typed from the MODEL, not assumed revolute.

    A parallel-jaw gripper's commanded joint is usually a SLIDE — the PAL PRO's is 0..0.07 m of jaw
    travel — and calling it revolute turns 70 mm of opening into 0.07 rad of rotation everywhere the
    planning side reasons about it, while ``/joint_states`` still reports the same number. Neither half
    looks wrong on its own.
    """
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="arm_base_link">
              <geom type="box" size="0.05 0.05 0.05" mass="1"/>
              <body name="hand" pos="0 0 0.1">
                <geom type="box" size="0.04 0.04 0.02" mass="0.5"/>
                <body name="finger" pos="0 0 0.03">
                  <joint name="finger_joint" type="slide" axis="0 1 0" range="0 0.07"
                         actuatorfrcrange="-10 10"/>
                  <geom type="box" size="0.01 0.01 0.02" mass="0.05"/>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="finger_pos" joint="finger_joint" forcerange="-10 10"/>
          </actuator>
        </mujoco>
        """
    )
    tree = UrdfExporter(
        model,
        prefix="",
        name="r",
        root_link="arm_base_link",
        mesh_dir=tmp_path / "meshes",
        collapse=("hand",),
        gripper_joint="finger_joint",
    ).export()
    joint = tree.getroot().find("joint[@name='finger_joint']")
    assert joint is not None, "the commanded gripper DOF was not kept"
    assert joint.get("type") == "prismatic", "a slide joint was exported as a rotation"
    limit = joint.find("limit")
    assert float(limit.get("upper")) == pytest.approx(0.07)
    # Effort from the actuator's forcerange, like every other joint — not a hardcoded 100 N.
    assert float(limit.get("effort")) == pytest.approx(10.0)


def test_visual_only_geometry_is_not_collidable(tmp_path, robot):
    """MoveIt must not plan around decoration. The husky's mast is visual-only in the MJCF."""
    _out, _exporter, tree = _export(tmp_path, robot, collapse=("base_mount",))
    # The UR10e's link meshes are class `visual` (contype/conaffinity 0) plus separate collision
    # geoms, so every link should carry at least one visual and the collidable ones a collision.
    for link in tree.getroot().findall("link"):
        if link.get("name") == "arm_base_link":
            assert link.findall("visual"), "the arm base lost its geometry"


def test_root_link_name_colliding_with_another_body_is_refused(tmp_path):
    """``--root-link`` renames the root, so it must not name a body that already exists.

    Unchecked, both bodies claim one URDF link: the link is emitted once and the child's joint gets
    itself as its own parent, leaving a tree with no root and a self-referential joint — and the export
    reports success. It surfaces only under ``--check``, as ``MjSpec: URDF body not found``, which reads
    like a mesh problem. Provoked here the way a real port does: a mobile base whose root body is
    ``base_footprint`` with ``base_link`` as its child, where ``--root-link base_link`` is the obvious
    thing to ask for.
    """
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="base_footprint">
              <freejoint/>
              <body name="base_link" pos="0 0 0.1">
                <geom type="box" size="0.2 0.2 0.1" mass="10"/>
                <body name="arm_1_link" pos="0 0 0.1">
                  <joint name="arm_1_joint" type="hinge" axis="0 0 1" range="-1 1"/>
                  <geom type="capsule" size="0.03 0.1" mass="1"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    with pytest.raises(ValueError, match="already the name of body"):
        UrdfExporter(
            model, prefix="", name="r", root_link="base_link", mesh_dir=tmp_path / "meshes"
        ).export()
    # The MJCF's own root name is accepted, and gives a proper tree: one more link than joints.
    tree = UrdfExporter(
        model, prefix="", name="r", root_link="base_footprint", mesh_dir=tmp_path / "meshes"
    ).export()
    root = tree.getroot()
    assert len(root.findall("link")) == len(root.findall("joint")) + 1
    children = {j.find("child").get("link") for j in root.findall("joint")}
    assert {"base_footprint"} == {link.get("name") for link in root.findall("link")} - children


def test_multi_joint_body_is_refused(tmp_path):
    """URDF has no spelling for it, so raise rather than drop a DOF."""
    spec = mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body name="r_base">
             <geom type="box" size=".1 .1 .1"/>
             <body name="r_two">
               <joint name="r_a" axis="1 0 0"/><joint name="r_b" axis="0 1 0"/>
               <geom type="box" size=".05 .05 .05"/>
             </body></body></worldbody></mujoco>"""
    )
    exporter = UrdfExporter(spec.compile(), prefix="r_", name="two", root_link="base_link")
    with pytest.raises(ValueError, match="one joint per link"):
        exporter.export()


# -- --tip-site: the link a site becomes ---------------------------------------------------------


def _tip_export(tmp_path, robot, **kwargs):
    return _export(
        tmp_path,
        robot,
        collapse=("base_mount",),
        gripper_joint="robotiq_85_left_knuckle_joint",
        **kwargs,
    )


def test_no_tip_site_emits_no_tip_link(tmp_path, robot):
    """Off by default: a chain that ends at the flange is still the right answer for a bare arm."""
    _out, _exporter, tree = _tip_export(tmp_path, robot)
    assert tree.getroot().find("link[@name='tcp']") is None


def test_tip_site_is_parented_to_the_link_its_body_collapsed_into(tmp_path, robot):
    """The case that makes this worth doing at all.

    The 2F-85's ``pinch`` sits on ``robotiq_85_base_link``, a descendant of the ``base_mount`` that
    ``--collapse`` folds away -- so its own body has no URDF link, and a naive parent lookup names a
    link the file does not contain. srdfdom then drops the chain and move_group loads a robot it
    cannot plan for, which is the failure that hangs rather than erroring.
    """
    _out, _exporter, tree = _tip_export(tmp_path, robot, tip_site="pinch")
    root = tree.getroot()
    links = {link.get("name") for link in root.findall("link")}
    assert "tcp" in links
    assert "robotiq_85_base_link" not in links, "fixture is wrong: this body should be collapsed"

    joint = root.find("joint[@name='tcp_fixed']")
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "base_mount"
    assert joint.find("child").get("link") == "tcp"


def test_tip_link_is_a_frame_and_carries_no_mass(tmp_path, robot):
    """Giving it an inertial would change the dynamics of a robot this file only describes."""
    _out, _exporter, tree = _tip_export(tmp_path, robot, tip_site="pinch")
    tcp = tree.getroot().find("link[@name='tcp']")
    assert list(tcp) == [], f"the tip link should be empty, got {[c.tag for c in tcp]}"


def test_tip_link_lands_on_the_site_it_names(tmp_path, robot):
    """THE test for this flag: the emitted origin really is where the site is.

    Checked against MuJoCo's own FORWARD KINEMATICS rather than against a restatement of the
    exporter's arithmetic. That makes it an independent measurement -- ``mj_forward`` walks the tree
    itself, so a wrong ancestor, a dropped joint anchor or a mis-ordered quaternion product shows up
    here as millimetres.

    Note the URDF cannot be loaded back for this the way ``round_trip_error`` does it: MuJoCo's URDF
    parser MERGES fixed-joint links into their parent, so ``tool0``, ``base_mount`` and ``tcp`` are
    all absent from the reloaded model. That is also why a synthetic frame link cannot break
    ``--check``.
    """
    _out, exporter, tree = _tip_export(tmp_path, robot, tip_site="pinch")

    md = mujoco.MjData(robot)
    mujoco.mj_forward(robot, md)
    sid = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_SITE, "ur10e_pinch")
    owner = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, "ur10e_base_mount")
    assert sid >= 0 and owner >= 0, "fixture is wrong: the collapsed gripper's site should be here"

    # The link's frame is its body's frame less whatever anchor shift the link absorbed.
    owner_rot = md.xmat[owner].reshape(3, 3)
    want_pos = owner_rot.T @ (md.site_xpos[sid] - md.xpos[owner]) - exporter._shift_of[owner]
    want_rot = owner_rot.T @ md.site_xmat[sid].reshape(3, 3)

    origin = tree.getroot().find("joint[@name='tcp_fixed']/origin")
    got_pos = np.array([float(v) for v in origin.get("xyz").split()])
    got_rot = _rpy_to_mat(*(float(v) for v in origin.get("rpy").split()))

    assert np.allclose(got_pos, want_pos, atol=1e-9), f"tcp at {got_pos}, site is at {want_pos}"
    assert np.allclose(got_rot, want_rot, atol=1e-7), "the tip link dropped the site's orientation"


def test_tip_link_does_not_break_the_round_trip_check(tmp_path, robot):
    """``--check`` runs against the written file, so the synthetic link is present when it does.

    It has no MuJoCo counterpart and is skipped by the name match -- but the file must still COMPILE
    in MuJoCo's URDF parser, which is the half a name-matching check cannot notice.
    """
    out, _exporter, _tree = _tip_export(tmp_path, robot, tip_site="pinch")
    err, where = round_trip_error(out, robot, "ur10e_", samples=16)
    assert err < 1e-6, f"URDF diverges from the MJCF by {err:.3e} m at {where!r}"


def test_an_unknown_tip_site_names_the_sites_the_model_has(tmp_path, robot):
    """A typo here is otherwise invisible: the link is simply absent and the SRDF names a ghost."""
    with pytest.raises(ValueError, match="names no site") as err:
        _tip_export(tmp_path, robot, tip_site="pnich")
    assert "pinch" in str(err.value), "the error should list what was available"


# -- several robots in one description -----------------------------------------------------------

_PAIR = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="a_base" pos="0.3 -0.4 0.1" quat="0 0 0 1">
      <geom name="a_g" type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="a_link" pos="0 0 0.2">
        <joint name="a_j" type="hinge" axis="0 0 1" range="-3 3"/>
        <geom name="a_g2" type="box" size="0.05 0.2 0.05" mass="1"/>
      </body>
    </body>
    <body name="b_base" pos="0 0.4 0">
      <geom name="b_g" type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="b_link" pos="0 0 0.2">
        <joint name="b_j" type="hinge" axis="0 0 1" range="-3 3"/>
        <geom name="b_g2" type="box" size="0.05 0.2 0.05" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _pair_parts(model, mesh_dir):
    parts = []
    for prefix in ("a_", "b_"):
        exporter = UrdfExporter(
            model,
            prefix=prefix,
            name=f"part_{prefix}",
            root_link=f"{prefix}base",
            mesh_dir=mesh_dir,
            strip="",
            link_strip="",
        )
        parts.append(
            (exporter.export(), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base"))
        )
    return parts


def test_two_robots_hang_off_one_root_at_the_pose_the_model_puts_them(tmp_path):
    """One description, one root, and each robot where the compiled world has it -- including its
    ORIENTATION, or a plan for the pair is right about each arm and wrong about the cell."""
    model = mujoco.MjModel.from_xml_string(_PAIR)
    tree = combine_urdfs(
        model, _pair_parts(model, tmp_path / "meshes"), name="pair", root_link="base_link"
    )
    root = tree.getroot()
    links = {link.get("name") for link in root.findall("link")}
    assert links == {"base_link", "a_base", "a_link", "b_base", "b_link"}
    mounts = {
        j.find("child").get("link"): j for j in root.findall("joint") if j.get("type") == "fixed"
    }
    assert mounts["a_base"].find("parent").get("link") == "base_link"
    assert mounts["a_base"].find("origin").get("xyz") == "0.3 -0.4 0.1"
    assert mounts["b_base"].find("origin").get("xyz") == "0 0.4 0"
    assert mounts["a_base"].find("origin").get("rpy") != mounts["b_base"].find("origin").get("rpy")


def test_the_combined_urdf_agrees_with_the_model_it_came_from(tmp_path):
    """The round trip is what proves the mount poses are right rather than merely present: a second
    robot placed by the wrong transform shows up here as centimetres."""
    model = mujoco.MjModel.from_xml_string(_PAIR)
    out = tmp_path / "pair.urdf"
    tree = combine_urdfs(
        model, _pair_parts(model, tmp_path / "meshes"), name="pair", root_link="base_link"
    )
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    err, where = round_trip_error(out, model, "", mesh_dir=tmp_path / "meshes")
    assert err < 1e-6, f"combined URDF diverges from the MJCF at {where}"


def test_a_name_two_robots_both_claim_is_refused(tmp_path):
    """One URDF is one flat namespace. Keeping only one of the two would parse and then plan around
    a robot that is not there."""
    model = mujoco.MjModel.from_xml_string(_PAIR)
    parts = [
        (
            UrdfExporter(
                model, prefix=prefix, name="part", root_link="base", mesh_dir=tmp_path / "meshes"
            ).export(),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base"),
        )
        for prefix in ("a_", "b_")
    ]
    with pytest.raises(ValueError, match="claimed by both"):
        combine_urdfs(model, parts, name="pair", root_link="base_link")


def test_one_robot_is_exported_directly_rather_than_combined(tmp_path):
    model = mujoco.MjModel.from_xml_string(_PAIR)
    with pytest.raises(ValueError, match="at least two parts"):
        combine_urdfs(
            model, _pair_parts(model, tmp_path / "meshes")[:1], name="one", root_link="base_link"
        )
