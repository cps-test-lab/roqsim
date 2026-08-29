"""Export one robot's kinematic tree from a compiled roqsim world to URDF, for MoveIt 2.

Why generate rather than ship a vendor URDF
===========================================
MoveIt plans against a URDF; MuJoCo simulates an MJCF. When those two describe the *same* robot from
different sources they agree only approximately -- MuJoCo Menagerie's UR10e, for instance, carries
wrist offsets ~2 mm from `ur_description`'s (0.176 vs 0.17415 m). Two millimetres is roughly the
margin a parallel-jaw grasp has, so the divergence shows up as a grasp that misses for no visible
reason, on a stack where both halves look correct in isolation. Deriving the URDF from the MJCF the
simulator actually loaded makes plan-space and sim-space the same space by construction, and makes the
agreement *testable* (see ``round_trip_error``).

It also removes a dependency: a campaign image needs no vendor description packages to plan.

What it does and does not cover
==============================
This exports a **tree**, because URDF is a tree. That is the one real limitation and it is not
cosmetic:

* One joint per body. MuJoCo allows several; URDF's joint *is* the parent link, so a multi-joint body
  has no URDF spelling. Raises rather than silently dropping a DOF.
* No closed loops. A Robotiq 2F-85's four-bar linkage is closed by ``equality/connect`` and simply
  cannot be written in URDF -- the official `robotiq_description` approximates it with ``mimic``
  joints for the same reason. Use ``collapse`` to lump such a subtree into one fixed link: its
  geometry is preserved (so MoveIt still plans around the real gripper volume) while its internal
  DOFs are dropped. What is lost is the *fingers'* articulation in the planning model, which MoveIt
  does not plan through anyway -- a gripper is driven by a GripperCommand controller, not by IK.
* Free joints become a fixed root. A mobile base's pose belongs to TF (``odom -> base_link``), not to
  the robot description, and MoveIt expects a fixed-root URDF.
* A **hinge or slide** root joint is kept, not dropped: the robot gets a synthetic fixed ``world`` link
  (``--world-link``) above it. That is the standard URDF spelling for a rail-mounted or gantry arm, and
  it still gives MoveIt the fixed root it needs -- only the axis moves below it. Without this the DOF
  would exist in the simulation and in ``/joint_states`` but not in the description, and MoveIt would
  reject the trajectory the controller reports.

Joint anchors are handled properly: a MuJoCo joint may sit at an offset inside its body
(``<joint pos="...">``, as the 2F-85's ``follower`` class does), whereas a URDF joint frame *is* the
child link frame. Where the anchor is non-zero the child frame is shifted onto it and everything
expressed in that frame -- inertial, geoms, child joints -- is shifted back by the same amount, so the
result is geometrically identical rather than merely close.

Usage::

    roqsim export urdf --world w.yaml --prefix ur10e_ --name husky_ur10e --out urdf/robot.urdf \\
        --root-link base_link --collapse base_mount --manifest urdf/.generated.json

``--prefix`` selects the robot: every body whose MJCF name starts with it. The prefix is stripped from
the emitted names, so the URDF carries the model's own names (``shoulder_pan_joint``) and matches the
joint names ``arm_controller`` publishes in ``/joint_states``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from . import logging_setup
from .export_mesh import _quat_to_mat

logger = logging.getLogger(__name__)

# MuJoCo geom type -> how URDF spells it. Capsules have no URDF equivalent; see _geom_geometry.
_PRIMITIVES = {
    mujoco.mjtGeom.mjGEOM_BOX,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_MESH,
}


def _name(model, objtype, idx: int) -> str:
    return mujoco.mj_id2name(model, objtype, idx) or ""


def _quat_to_rpy(quat) -> tuple[float, float, float]:
    """(w,x,y,z) -> fixed-axis XYZ roll/pitch/yaw, the convention URDF's ``rpy`` uses.

    Derived from the rotation MATRIX rather than from the quaternion components directly. That is not
    a style choice: the direct form needs a ``1 - 2(y^2 + z^2)`` term which, for exactly the quaternions
    this substrate's arms are full of (``quat="1 0 1 0"``, a 90 deg turn about y, on four of the UR10e's
    six links), evaluates to -1e-17 instead of 0. ``atan2(0, -1e-17)`` is -pi, not 0, so the link came
    out rotated by 180 deg -- and because it happened at the shoulder it moved the whole arm by metres.
    Going through the matrix keeps the degenerate case in one explicit branch.

    URDF's rpy is R = Rz(yaw) @ Ry(pitch) @ Rx(roll), so pitch = asin(-R[2,0]). At pitch = +-90 deg
    roll and yaw are the same rotation; the convention here puts all of it in roll and leaves yaw at 0.
    """
    r = _quat_to_mat(quat)
    sp = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    if abs(sp) > 1.0 - 1e-9:
        sign = 1.0 if sp > 0 else -1.0
        roll = float(np.arctan2(sign * r[0, 1], sign * r[0, 2]))
        return (roll, pitch, 0.0)
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return (roll, pitch, yaw)


def _origin(parent: ET.Element, pos, quat=None) -> None:
    rpy = _quat_to_rpy(quat) if quat is not None else (0.0, 0.0, 0.0)
    ET.SubElement(
        parent,
        "origin",
        xyz=" ".join(f"{float(v):.9g}" for v in pos),
        rpy=" ".join(f"{v:.9g}" for v in rpy),
    )


class UrdfExporter:
    """Builds the URDF for the bodies of one robot in a compiled model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        prefix: str,
        name: str,
        root_link: str = "base_link",
        collapse: tuple[str, ...] = (),
        mesh_dir: Path | None = None,
        world_link: str = "world",
        gripper_joint: str = "",
        strip: str | None = None,
        mesh_package: str = "",
        tip_site: str = "",
        tip_link: str = "tcp",
    ):
        self.m = model
        self.prefix = prefix
        # What is removed from emitted names, which is NOT always what selects the bodies. A mobile
        # manipulator is exported whole (prefix "") so its root link is the mobile base -- the frame
        # `odom -> base_link` and nav2's costmaps refer to -- but its ARM bodies carry the arm's MJCF
        # prefix, while `arm_controller` publishes those joints to /joint_states with that prefix
        # already stripped. robot_state_publisher matches joint states to URDF joints BY NAME, so an
        # unstripped URDF leaves every arm link frozen at zero and MoveIt plans from a pose the arm is
        # not in. Defaults to `prefix`, which is the single-robot case.
        self.strip_prefix = prefix if strip is None else strip
        self.name = name
        self.root_link = root_link
        # Parent for a root body that is itself jointed (a rail carriage, a gantry). Only emitted
        # when there is such a joint; a welded-down robot keeps root_link as the URDF root.
        self.world_link = world_link
        # Collapse targets are given in stripped (model-local) names, as a user reads them.
        self.collapse = set(collapse)
        self.mesh_dir = mesh_dir
        # Non-empty -> meshes are referenced as package://<pkg>/... instead of file://<abs path>.
        self.mesh_package = mesh_package
        self.dropped_dofs: list[str] = []
        self.mesh_files: dict[int, Path] = {}
        # body id -> the frame shift its URDF link absorbed (non-zero only for a joint anchor).
        self._shift_of: dict[int, np.ndarray] = {}
        self.gripper_joint = gripper_joint
        # (parent link, joint id, rotation, translation) for a gripper DOF rescued from a collapse.
        self._kept_gripper: tuple | None = None
        # A MuJoCo SITE the arm chain should end at, and the name of the link emitted for it. Empty
        # site -> no such link, and the chain ends at whatever body is last.
        self.tip_site = tip_site
        self.tip_link = tip_link

    # -- naming -------------------------------------------------------------------------------
    def _mesh_uri(self, path: Path) -> str:
        """How a mesh is referenced from the URDF.

        ``file://<abs path>`` by default, which is right for a URDF generated and consumed in the same
        tree. It is wrong for one that SHIPS: an ament package installs to a different prefix, and a
        container to a different path again, so a baked absolute path resolves to nothing there.
        ``--mesh-package`` emits ``package://<pkg>/<mesh-dir name>/<file>`` instead, which resolves
        wherever the package is installed.
        """
        if self.mesh_package:
            return f"package://{self.mesh_package}/{Path(path).parent.name}/{Path(path).name}"
        return f"file://{path}"

    def _strip(self, s: str) -> str:
        """Strip the SELECTION prefix. Used for link names, which only have to be unique."""
        return s[len(self.prefix) :] if self.prefix and s.startswith(self.prefix) else s

    def _strip_joint(self, s: str) -> str:
        """Strip the naming prefix. Used for JOINT names, which have to match /joint_states.

        Deliberately different from :meth:`_strip`. Only joint names are name-matched against
        ``/joint_states`` by robot_state_publisher, so only they need the arm's prefix removed; link
        names merely need to be unique, and stripping THEM breaks a real robot -- the Kinova Gen3's
        root body is called ``base_link``, exactly like the Husky's, so a stripped ``gen3_base_link``
        collides with the mobile base's link and the URDF will not parse at all.
        """
        sp = self.strip_prefix
        return s[len(sp) :] if sp and s.startswith(sp) else s

    def _bodies(self) -> list[int]:
        """The robot's bodies, in model order (parents before children)."""
        return [
            b
            for b in range(1, self.m.nbody)
            if _name(self.m, mujoco.mjtObj.mjOBJ_BODY, b).startswith(self.prefix)
        ]

    def _collapsed_into(self, body: int) -> int | None:
        """The nearest ancestor of ``body`` that is a collapse root, if any."""
        cur = body
        while cur > 0:
            if self._strip(_name(self.m, mujoco.mjtObj.mjOBJ_BODY, cur)) in self.collapse:
                return cur
            cur = int(self.m.body_parentid[cur])
        return None

    # -- geometry -----------------------------------------------------------------------------
    def _geom_geometry(self, parent: ET.Element, g: int) -> bool:
        """Write the <geometry> for geom ``g``. False if it has no URDF spelling."""
        gtype = self.m.geom_type[g]
        size = self.m.geom_size[g]
        geo = ET.SubElement(parent, "geometry")
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            ET.SubElement(geo, "box", size=" ".join(f"{2 * float(s):.9g}" for s in size[:3]))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            ET.SubElement(geo, "sphere", radius=f"{float(size[0]):.9g}")
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ET.SubElement(
                geo, "cylinder", radius=f"{float(size[0]):.9g}", length=f"{2 * float(size[1]):.9g}"
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # URDF has no capsule. A cylinder of the same half-length is the inscribed
            # approximation: it under-covers by the two hemispherical caps (radius `r` each), so
            # planning is slightly optimistic at the ends of the segment rather than pessimistic.
            ET.SubElement(
                geo, "cylinder", radius=f"{float(size[0]):.9g}", length=f"{2 * float(size[1]):.9g}"
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.m.geom_dataid[g])
            path = self.mesh_files.get(mesh_id)
            if path is None:
                parent.remove(geo)
                return False
            ET.SubElement(geo, "mesh", filename=self._mesh_uri(path))
        else:
            parent.remove(geo)
            return False
        return True

    def _write_geoms(self, link: ET.Element, body: int, shift: np.ndarray, xform=None) -> None:
        """Emit visual+collision for every geom of ``body``, in the link frame.

        ``shift`` moves geoms out of the MuJoCo body frame into the URDF link frame (non-zero only
        when the body's joint had an anchor). ``xform`` is set for a collapsed descendant: the rigid
        transform from that body's frame into the collapse root's link frame.
        """
        for g in range(self.m.ngeom):
            # int(): mjt* enums compare False against a numpy scalar (set/tuple membership puts the
            # enum on the left of `==`), which would drop every geom and export empty links.
            if self.m.geom_bodyid[g] != body or int(self.m.geom_type[g]) not in _PRIMITIVES:
                continue
            pos = np.asarray(self.m.geom_pos[g], dtype=float) - shift
            quat = np.asarray(self.m.geom_quat[g], dtype=float)
            if xform is not None:
                rot, trans = xform
                pos = rot @ pos + trans
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, self._mat_quat(rot), quat)
                quat = q
            # A MuJoCo geom is collidable unless contype and conaffinity are both zero; that is the
            # substrate's convention for visual-only decoration, and MoveIt must not plan around it.
            collidable = bool(self.m.geom_contype[g] or self.m.geom_conaffinity[g])
            for tag in ("visual",) + (("collision",) if collidable else ()):
                el = ET.SubElement(link, tag)
                _origin(el, pos, quat)
                if not self._geom_geometry(el, g):
                    link.remove(el)

    @staticmethod
    def _mat_quat(rot: np.ndarray) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).reshape(9))
        return q

    def _write_inertial(
        self, link: ET.Element, bodies: list[tuple[int, np.ndarray, object]]
    ) -> None:
        """Lumped inertial for one URDF link, over the MuJoCo bodies it represents.

        A collapsed subtree becomes one rigid link, so its bodies' masses and inertias are summed
        about the combined centre of mass (parallel axis). Doing this instead of keeping only the root
        body's mass matters: the 2F-85's fingers are two thirds of its mass, and an arm planning with a
        third of its real wrist load has the wrong dynamic limits.
        """
        total = sum(float(self.m.body_mass[b]) for b, _, _ in bodies)
        inertial = ET.SubElement(link, "inertial")
        if total <= 0.0:
            _origin(inertial, (0.0, 0.0, 0.0))
            ET.SubElement(inertial, "mass", value="0")
            ET.SubElement(inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")
            return

        coms, mats, masses = [], [], []
        for b, shift, xform in bodies:
            mass = float(self.m.body_mass[b])
            com = np.asarray(self.m.body_ipos[b], dtype=float) - shift
            rot = _quat_to_mat(self.m.body_iquat[b])
            if xform is not None:
                xrot, xtrans = xform
                com = xrot @ com + xtrans
                rot = xrot @ rot
            # Diagonal inertia in the body's principal frame -> full tensor in the link frame.
            diag = np.diag(np.asarray(self.m.body_inertia[b], dtype=float))
            coms.append(com)
            mats.append(rot @ diag @ rot.T)
            masses.append(mass)

        com = sum(mss * c for mss, c in zip(masses, coms, strict=True)) / total
        tensor = np.zeros((3, 3))
        for mss, c, tsr in zip(masses, coms, mats, strict=True):
            d = c - com
            tensor += tsr + mss * (float(d @ d) * np.eye(3) - np.outer(d, d))

        _origin(inertial, com)
        ET.SubElement(inertial, "mass", value=f"{total:.9g}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{tensor[0, 0]:.9g}",
            ixy=f"{tensor[0, 1]:.9g}",
            ixz=f"{tensor[0, 2]:.9g}",
            iyy=f"{tensor[1, 1]:.9g}",
            iyz=f"{tensor[1, 2]:.9g}",
            izz=f"{tensor[2, 2]:.9g}",
        )

    # -- joints -------------------------------------------------------------------------------
    def _body_joint(self, body: int) -> int | None:
        n = int(self.m.body_jntnum[body])
        if n == 0:
            return None
        if n > 1:
            raise ValueError(
                f"body {_name(self.m, mujoco.mjtObj.mjOBJ_BODY, body)!r} has {n} joints; URDF allows "
                f"one joint per link, so there is no faithful spelling. Collapse this subtree "
                f"(--collapse) or split the body in the MJCF."
            )
        return int(self.m.body_jntadr[body])

    def export(self) -> ET.ElementTree:
        robot = ET.Element("robot", name=self.name)
        # MuJoCo's own URDF loader needs this to find relative mesh paths; harmless to ROS. We emit
        # absolute file:// mesh URIs anyway, but keeping it makes the round-trip check work unchanged.
        ET.SubElement(ET.SubElement(robot, "mujoco"), "compiler", discardvisual="false")

        bodies = self._bodies()
        if not bodies:
            raise ValueError(f"no bodies match prefix {self.prefix!r}")

        # `--root-link` RENAMES the root body, so it must not be a name another exported body already
        # holds. Unchecked, both bodies claim one link: `members` keeps only the last, and the child's
        # joint is emitted with itself as its own parent. The URDF then has one link too few, a
        # self-referential joint, and no root -- and none of that raises here. It surfaces only if
        # --check happens to be on, as `MjSpec: URDF body not found`, which reads like a mesh problem.
        # A TIAGo Pro hits it exactly: its root body is `base_footprint` and `base_link` is its child,
        # so `--root-link base_link` looks like the obvious thing to ask for.
        root = bodies[0]
        clash = next(
            (
                b
                for b in bodies[1:]
                if self._strip(_name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)) == self.root_link
            ),
            None,
        )
        if clash is not None:
            raise ValueError(
                f"--root-link {self.root_link!r} is already the name of body "
                f"{_name(self.m, mujoco.mjtObj.mjOBJ_BODY, clash)!r}, which is not the root "
                f"({_name(self.m, mujoco.mjtObj.mjOBJ_BODY, root)!r}). Renaming the root onto it would "
                f"merge two links into one. Pass --root-link "
                f"{self._strip(_name(self.m, mujoco.mjtObj.mjOBJ_BODY, root))!r} to keep the tree as the "
                f"MJCF has it."
            )

        # URDF link name -> the MuJoCo bodies it lumps, each with its frame shift/transform.
        members: dict[str, list[tuple[int, np.ndarray, object]]] = {}
        link_of: dict[int, str] = {}
        shift_of = self._shift_of
        joints: list[tuple[str, int, int, np.ndarray]] = []  # (child link, body, joint, shift)

        for body in bodies:
            bname = self._strip(_name(self.m, mujoco.mjtObj.mjOBJ_BODY, body))
            collapse_root = self._collapsed_into(body)
            if collapse_root is not None and collapse_root != body:
                # A descendant of a collapse root: fold its geometry into that link.
                target = link_of[collapse_root]
                rot, trans = self._rigid_to(body, collapse_root, shift_of[collapse_root])
                members[target].append((body, np.zeros(3), (rot, trans)))
                jid = self._body_joint(body)
                if jid is not None:
                    jname = self._strip_joint(_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid))
                    if jname == self.gripper_joint:
                        # Keep the ONE joint the gripper is commanded through. Everything else in the
                        # loop is genuinely unrepresentable, but this joint is the name MoveIt's
                        # GripperCommand controller and the SRDF's gripper group refer to -- collapse
                        # it away and MoveIt has a gripper it cannot open.
                        self._kept_gripper = (target, jid, rot, trans)
                    else:
                        self.dropped_dofs.append(jname)
                link_of[body] = target
                shift_of[body] = np.zeros(3)
                continue

            link_name = self.root_link if body == root else bname
            jid = self._body_joint(body)
            if collapse_root == body and jid is not None:
                # The collapse ROOT's own joint goes too: `--collapse` lumps a subtree into one FIXED
                # link, and a root that kept its joint would leave the lump swinging on a DOF nothing
                # drives. That matters exactly where collapse is used -- a closed linkage. Each PAL PRO
                # jaw is a four-bar whose two branches are siblings, so the loop is only removed by
                # collapsing BOTH; keeping the second branch's root joint left four revolute DOFs in the
                # URDF that no controller publishes, and MoveIt then never has a complete robot state:
                # "The complete state of the robot is not yet known. Missing
                # gripper_*_outer_finger_*_joint", followed by "Found empty JointState message" and a
                # planner that cannot sample a single valid state.
                self.dropped_dofs.append(
                    self._strip_joint(_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid))
                )
                jid = None
            # A URDF joint frame IS the child link frame, so a non-zero MuJoCo anchor shifts it.
            shift = (
                np.asarray(self.m.jnt_pos[jid], dtype=float)
                if jid is not None and self.m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE
                else np.zeros(3)
            )
            link_of[body] = link_name
            shift_of[body] = shift
            members[link_name] = [(body, shift, None)]
            if body != root:
                joints.append((link_name, body, jid if jid is not None else -1, shift))

        # A jointed root -- a rail carriage, a gantry bridge -- carries a real DOF, and the loop above
        # emits a joint only for non-root bodies, so without this the DOF would vanish from the URDF
        # while still existing in the simulation and in /joint_states. That is the silent DOF drop the
        # module docstring promises not to do, and it fails late and confusingly: MoveIt plans for a
        # 6-DOF arm, then rejects the 7-name trajectory the controller reports.
        #
        # The fix is the standard URDF spelling for a rail: a fixed `world` link parenting the moving
        # root. MoveIt still gets the fixed root it requires; the axis just moves below it. A FREE root
        # is deliberately not handled this way -- a floating base belongs in TF (odom -> base_link),
        # which is why it stays a fixed root.
        root_jid = self._body_joint(root)
        # int(): see the geom_type note above -- the raw numpy value never matches an mjt* enum here.
        if root_jid is not None and int(self.m.jnt_type[root_jid]) in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            ET.SubElement(robot, "link", name=self.world_link)
            # body_parentid[root] is worldbody; mapping it to the synthetic link is what makes
            # _write_joint resolve the parent to `world` instead of falling back to root_link (which
            # would make the root its own parent).
            link_of[int(self.m.body_parentid[root])] = self.world_link
            joints.append((self.root_link, root, root_jid, shift_of[root]))

        self._collect_meshes(members)

        for link_name, mem in members.items():
            link = ET.SubElement(robot, "link", name=link_name)
            self._write_inertial(link, mem)
            for body, shift, xform in mem:
                self._write_geoms(link, body, shift, xform)

        for link_name, body, jid, shift in joints:
            self._write_joint(robot, link_name, body, jid, shift, link_of)

        self._write_gripper_drive(robot)
        self._write_tip_site(robot, link_of, shift_of)

        return ET.ElementTree(robot)

    def _write_tip_site(self, robot: ET.Element, link_of: dict, shift_of: dict) -> None:
        """Emit a frame link at ``--tip-site``, so the arm chain can end where the tool acts.

        A gripper's grasp point is a SITE in the MJCF (the 2F-85's ``pinch``, 0.145 m out along the
        approach axis), and a site is not a body -- so ``export()`` above, which walks bodies, has no
        way to emit it and MoveIt cannot plan for it. Without this the chain ends at the tool flange
        and a goal for the fingertips has to be written as "the flange, plus N mm down the tool axis".

        That indirection is not merely inconvenient, it is wrong in a way that looks right: every
        orientation tolerance is multiplied by the lever arm, so 0.15 rad of permitted tilt becomes
        +-33 mm at the fingers. A cell measured 61 mm of lateral error against 12.2 mm of jaw
        clearance -- MoveIt had satisfied the goal exactly, and the goal was about the wrong point.
        With the chain ending here, a 3 mm position tolerance means 3 mm at the pads.

        Emitted as a pure FRAME: no <visual>, no <collision>, no <inertial>. An inertial-less link is
        legal in URDF and MoveIt treats it as a frame; giving it mass would change the dynamics of a
        robot this file only describes. MuJoCo's own URDF parser accepts it because the joint is
        FIXED -- its mjMINVAL floor applies to bodies that move.

        The site's full pose is carried, orientation included, because that is what the model says:
        the site's frame IS the tool's axis convention, and dropping it would silently redefine what
        an orientation goal means.
        """
        if not self.tip_site:
            return
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, self.prefix + self.tip_site)
        if sid < 0:  # unprefixed, for a model exported whole (see strip_prefix)
            sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, self.tip_site)
        if sid < 0:
            sites = sorted(
                filter(
                    None, (_name(self.m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.m.nsite))
                )
            )
            raise ValueError(
                f"--tip-site {self.tip_site!r} names no site in this model. Sites present: "
                f"{', '.join(sites) or '(none)'}"
            )

        body = int(self.m.site_bodyid[sid])
        parent = link_of.get(body)
        if parent is None:
            raise ValueError(
                f"--tip-site {self.tip_site!r} sits on a body outside the exported robot "
                f"(--prefix {self.prefix!r} selects the robot)."
            )

        # Where the site's frame is, expressed in its LINK's frame. Two cases, one formula: a body
        # that owns its link contributes only that link's own anchor shift, while a body `--collapse`
        # folded away contributes the rigid transform up to the surviving link -- which is the case
        # that matters here, since a 2F-85's `pinch` hangs off `robotiq_85_base_link`, a descendant of
        # the `base_mount` a caller collapses.
        owner = self._collapsed_into(body)
        pos = np.asarray(self.m.site_pos[sid], dtype=float)
        quat = np.asarray(self.m.site_quat[sid], dtype=float)
        if owner is not None and owner != body:
            rot, trans = self._rigid_to(body, owner, shift_of[owner])
            pos = rot @ pos + trans
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self._mat_quat(rot), quat)
            quat = q
        else:
            pos = pos - shift_of[body]

        # The REFERENCE configuration (body_pos/body_quat, via _rigid_to), not FK: a URDF joint origin
        # is the pose at zero, so this needs no MjData and cannot drift with the simulator's state.
        ET.SubElement(robot, "link", name=self.tip_link)
        joint = ET.SubElement(robot, "joint", name=f"{self.tip_link}_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=self.tip_link)
        _origin(joint, pos, quat)
        logger.info(
            "tip site %r -> link %r, %.1f mm from %s",
            self.tip_site,
            self.tip_link,
            float(np.linalg.norm(pos)) * 1000.0,
            parent,
        )

    def _write_gripper_drive(self, robot: ET.Element) -> None:
        """Re-attach the gripper's commanded DOF to the link its linkage was collapsed into.

        The joint is real (same axis, same limits, same name as in ``/joint_states``); the link it
        drives is an empty marker, because the *moving* geometry stayed with the lumped collision
        volume. So MoveIt gets a gripper group it can command and a joint whose reported position is
        meaningful, while the volume it plans around is the whole gripper at its rest opening.

        This is the same trade the official `robotiq_description` makes with ``mimic`` joints: the
        planning model of a closed-loop hand is always an approximation. What matters is that the
        approximation is in the *fingers'* pose, not in the arm's kinematics -- which the round-trip
        check pins to nanometres.
        """
        if self._kept_gripper is None:
            return
        parent_link, jid, rot, trans = self._kept_gripper
        jname = self._strip_joint(_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        child = f"{jname}_link"
        # A frame-marker link with no geometry. It carries a NEGLIGIBLE but non-zero inertia (1 ug):
        # ROS/MoveIt accept an inertia-free link, but MuJoCo -- which parses this URDF back for the
        # round-trip check, and which other tools may too -- refuses a moving body below mjMINVAL. Six
        # orders of magnitude under the gripper's 0.9 kg, so it cannot perturb the arm's dynamics, and
        # it deliberately does NOT carry the fingers' mass: that is already in the lumped link, and
        # counting it twice here would give the wrist double the real payload.
        marker = ET.SubElement(robot, "link", name=child)
        inertial = ET.SubElement(marker, "inertial")
        _origin(inertial, (0.0, 0.0, 0.0))
        ET.SubElement(inertial, "mass", value="1e-9")
        ET.SubElement(
            inertial, "inertia", ixx="1e-9", ixy="0", ixz="0", iyy="1e-9", iyz="0", izz="1e-9"
        )

        # Type and effort come from the MODEL, exactly as `_write_joint` derives them for every other
        # joint. Hardcoding `revolute` here was wrong for the common case: a parallel-jaw gripper's
        # commanded DOF is usually a SLIDE (the PAL PRO's `gripper_*_finger_joint` is a 0..0.07 m
        # travel), and calling it revolute silently turns 70 mm of jaw opening into 0.07 rad of
        # rotation in every planning-side computation, while the number in /joint_states stays the
        # same -- so the two disagree without either looking wrong.
        kind = "prismatic" if self.m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE else "revolute"
        joint = ET.SubElement(robot, "joint", name=jname, type=kind)
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=child)
        anchor = rot @ np.asarray(self.m.jnt_pos[jid], dtype=float) + trans
        _origin(joint, anchor, self._mat_quat(rot))
        ET.SubElement(joint, "axis", xyz=" ".join(f"{v:.9g}" for v in rot @ self.m.jnt_axis[jid]))
        lo, hi = (float(v) for v in self.m.jnt_range[jid])
        ET.SubElement(
            joint,
            "limit",
            lower=f"{lo:.9g}",
            upper=f"{hi:.9g}",
            effort=f"{self._actuator_effort(jid):.9g}",
            velocity="3.14",
        )

    def _rigid_to(self, body: int, ancestor: int, anc_shift: np.ndarray):
        """Rigid transform taking coordinates in ``body``'s frame into ``ancestor``'s link frame."""
        rot = np.eye(3)
        trans = np.zeros(3)
        cur = body
        while cur != ancestor:
            parent = int(self.m.body_parentid[cur])
            r = _quat_to_mat(self.m.body_quat[cur])
            p = np.asarray(self.m.body_pos[cur], dtype=float)
            rot = r @ rot
            trans = r @ trans + p
            cur = parent
        return rot, trans - anc_shift

    def _write_joint(self, robot, link_name, body, jid, shift, link_of) -> None:
        parent_body = int(self.m.body_parentid[body])
        parent_link = link_of.get(parent_body, self.root_link)
        jtype = self.m.jnt_type[jid] if jid >= 0 else None

        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            # A floating base is TF's business (odom -> base_link), not the description's.
            jname, kind = f"{link_name}_fixed", "fixed"
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
            jname = self._strip_joint(_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid))
            kind = "revolute" if self.m.jnt_limited[jid] else "continuous"
        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            jname = self._strip_joint(_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid))
            kind = "prismatic"
        elif jtype is None:
            jname, kind = f"{link_name}_fixed", "fixed"
        else:
            raise ValueError(
                f"joint {_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid)!r} is a ball joint, which "
                f"URDF cannot express; collapse the subtree or replace it with three hinges."
            )

        joint = ET.SubElement(robot, "joint", name=jname, type=kind)
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=link_name)
        # Child link origin: the body's own pose in the parent, plus the joint anchor, minus whatever
        # shift the PARENT link already absorbed.
        parent_shift = self._shift_of.get(parent_body, np.zeros(3))
        pos = (
            np.asarray(self.m.body_pos[body], dtype=float)
            + _quat_to_mat(self.m.body_quat[body]) @ shift
            - parent_shift
        )
        _origin(joint, pos, self.m.body_quat[body])

        if kind == "fixed":
            return
        axis = np.asarray(self.m.jnt_axis[jid], dtype=float)
        ET.SubElement(joint, "axis", xyz=" ".join(f"{v:.9g}" for v in axis))
        lo, hi = (float(v) for v in self.m.jnt_range[jid])
        # Effort/velocity are required by URDF but are not MJCF concepts in the same units; take
        # effort from the driving actuator's forcerange where there is one. MoveIt's own
        # joint_limits.yaml is the place velocity/acceleration limits are meant to be set.
        effort = self._actuator_effort(jid)
        attrs = {"effort": f"{effort:.9g}", "velocity": "3.14"}
        if kind in ("revolute", "prismatic"):
            attrs |= {"lower": f"{lo:.9g}", "upper": f"{hi:.9g}"}
        ET.SubElement(joint, "limit", **attrs)

    def _actuator_effort(self, jid: int) -> float:
        for a in range(self.m.nu):
            if (
                self.m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                and int(self.m.actuator_trnid[a, 0]) == jid
            ):
                lo, hi = (float(v) for v in self.m.actuator_forcerange[a])
                if hi > 0.0:
                    return hi
        return 1000.0

    def _collect_meshes(self, members) -> None:
        """Write each referenced mesh to ``mesh_dir`` as an STL the URDF can point at."""
        if self.mesh_dir is None:
            return
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        wanted = set()
        for mem in members.values():
            for body, _, _ in mem:
                for g in range(self.m.ngeom):
                    if (
                        self.m.geom_bodyid[g] == body
                        and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                    ):
                        wanted.add(int(self.m.geom_dataid[g]))
        for mesh_id in sorted(wanted):
            mname = _name(self.m, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or f"mesh_{mesh_id}"
            out = self.mesh_dir / f"{self._strip(mname)}.stl"
            _write_stl(self.m, mesh_id, out)
            self.mesh_files[mesh_id] = out.resolve()


def _write_stl(model: mujoco.MjModel, mesh_id: int, out: Path) -> None:
    """Dump a compiled mesh as binary STL.

    The compiled mesh is used rather than the source file because it is what MuJoCo actually collides:
    already scaled, and already in the frame the geom refers to. Copying the source OBJ/STL instead
    would reintroduce the scale/frame question this exporter exists to remove.
    """
    vadr, vnum = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    fadr, fnum = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    verts = model.mesh_vert[vadr : vadr + vnum].reshape(-1, 3)
    faces = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3)
    # Whole-mesh at a time, not triangle at a time. A per-face Python loop calling np.cross was
    # ~200k calls for one mobile manipulator and made a single export take four seconds -- most of
    # the cost of `roqsim export-urdf`, and of every test that exercises it.
    #
    # Every VERTEX this writes is bit-identical to what the loop wrote (checked over all 205k faces
    # of a husky+ur10e+robotiq). The face NORMAL can differ by one float32 ulp on a minority of
    # faces, because numpy's axis-wise norm does not reduce in the same order as its scalar path.
    # That field is redundant -- it is derivable from the winding, and STL consumers (MuJoCo's own
    # loader included) recompute it rather than trust it -- so the geometry is unchanged.
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    # A degenerate face (zero area) has no normal; STL's convention is a zero vector, and dividing
    # would put a NaN in the file instead.
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)

    # Binary STL is a packed 50-byte record per face: 12 floats then a 2-byte attribute count. Build
    # it as one structured array so the whole mesh is a single write.
    rec = np.zeros(len(faces), dtype=np.dtype([("v", "<f4", 12), ("attr", "<u2")]))
    rec["v"] = np.hstack([n, a, b, c]).astype("<f4")
    with open(out, "wb") as fh:
        fh.write(b"roqsim-export-urdf".ljust(80, b"\0"))
        fh.write(int(len(faces)).to_bytes(4, "little"))
        fh.write(rec.tobytes())


def round_trip_error(
    urdf: Path,
    model: mujoco.MjModel,
    prefix: str,
    samples: int = 32,
    seed: int = 0,
    mesh_dir: Path | None = None,
) -> tuple[float, str]:
    """Max FK position error (m) between the exported URDF and the MJCF it came from.

    This is the assertion that the export is *correct* rather than merely well-formed, and it needs no
    third-party URDF library: MuJoCo parses URDF too, so the URDF is loaded back and both models are
    posed at the same joint values and compared link by link. An exporter bug in a frame, an axis, a
    joint anchor or a quaternion convention shows up here as centimetres.
    """
    rng = np.random.default_rng(seed)
    # ROS needs `file://` or `package://` mesh URIs; MuJoCo's URDF parser wants a bare path. Check a
    # rewritten copy rather than emitting bare paths, so the URDF that ships is the one ROS can
    # actually load. `package://<pkg>/<sub>/<file>` resolves against the URDF's own package root
    # (<urdf>/../), which is the source-tree layout `--mesh-package` is emitted for -- without this,
    # --mesh-package would silently disable the only check that proves the export is correct.
    import re
    import tempfile

    text = Path(urdf).read_text(encoding="utf-8").replace('filename="file://', 'filename="')
    # A `package://` URI is resolved against the directory the meshes were WRITTEN to, not by guessing
    # a package root from the URDF's location. Guessing assumed `--mesh-package` was a bare package
    # name with the mesh dir one level under the URDF; a value carrying a subpath (needed when the
    # installed layout is share/<pkg>/config/<platform>/meshes) then produced `config/config/...` and
    # the check failed on a file that was never missing. Only the basename is taken from the URI, so
    # any `--mesh-package` value works and the check stays honest about the geometry it loads.
    if mesh_dir is not None:
        # Absolute: the rewritten copy is compiled from a temporary directory, so a relative
        # --mesh-dir (the CLI's default is the bare `meshes`) would resolve against the temp dir and
        # the check would fail on files that are present.
        abs_mesh_dir = Path(mesh_dir).resolve()
        text = re.sub(
            r'filename="package://[^"]*/([^/"]+)"',
            lambda m: f'filename="{abs_mesh_dir}/{m.group(1)}"',
            text,
        )
    else:
        text = re.sub(
            r'filename="package://[^/"]+/', f'filename="{Path(urdf).resolve().parent.parent}/', text
        )
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.urdf"
        probe.write_text(text, encoding="utf-8")
        um = mujoco.MjSpec.from_file(str(probe)).compile()
    ud, md = mujoco.MjData(um), mujoco.MjData(model)

    # Match by name, trying the bare name first and then `prefix + name`. Both are needed because one
    # robot can mix the two: a mobile manipulator's base joints are unprefixed while its arm joints
    # carry the arm's MJCF prefix (which the URDF strips). Resolving only one form would silently drop
    # half the robot from the comparison -- and a check that skips links cannot fail.
    def _resolve(objtype, nm):
        got = mujoco.mj_name2id(model, objtype, nm)
        return got if got >= 0 else mujoco.mj_name2id(model, objtype, prefix + nm)

    pairs = []
    for uj in range(um.njnt):
        nm = _name(um, mujoco.mjtObj.mjOBJ_JOINT, uj)
        mj = _resolve(mujoco.mjtObj.mjOBJ_JOINT, nm)
        if mj >= 0 and um.jnt_type[uj] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            pairs.append((nm, uj, mj))

    body_pairs = []
    for ub in range(1, um.nbody):
        nm = _name(um, mujoco.mjtObj.mjOBJ_BODY, ub)
        mb = _resolve(mujoco.mjtObj.mjOBJ_BODY, nm)
        if mb >= 0:
            body_pairs.append((nm, ub, mb))

    worst, where = 0.0, "none"
    for _ in range(samples):
        for _nm, uj, mj in pairs:
            lo, hi = (float(v) for v in model.jnt_range[mj])
            q = rng.uniform(lo, hi) if hi > lo else rng.uniform(-np.pi, np.pi)
            ud.qpos[um.jnt_qposadr[uj]] = q
            md.qpos[model.jnt_qposadr[mj]] = q
        mujoco.mj_forward(um, ud)
        mujoco.mj_forward(model, md)
        # Compare each link's pose RELATIVE to the robot's root, so the base's placement in the
        # world (which the URDF deliberately does not carry) cannot mask or fake an error.
        #
        # Expressed IN THE ROOT'S OWN FRAME, not merely offset by the root's position. Subtracting
        # the root translation alone leaves the root's *orientation* in the comparison, and the two
        # models do not share it: a model whose root body carries a rotation (the UR arms' base is
        # `quat="0 0 0 -1"`, the standard UR convention) has every link rotated with it in world
        # coordinates, while the URDF's root -- correctly -- is the frame those links are expressed
        # in. That showed up as a ~1.7 m "error" on a UR5e whose URDF matched the MJCF exactly, body
        # for body: a false alarm that condemns a correct export. Rotating into the root frame makes
        # the check what its comment always claimed it was, a frame-independent shape comparison.
        uroot, mroot = ud.xpos[body_pairs[0][1]], md.xpos[body_pairs[0][2]]
        urot = ud.xmat[body_pairs[0][1]].reshape(3, 3)
        mrot = md.xmat[body_pairs[0][2]].reshape(3, 3)
        for nm, ub, mb in body_pairs:
            err = float(
                np.linalg.norm(urot.T @ (ud.xpos[ub] - uroot) - mrot.T @ (md.xpos[mb] - mroot))
            )
            if err > worst:
                worst, where = err, nm
    return worst, where


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export urdf",
        description="Export one robot's kinematic tree from a compiled roqsim world to URDF for MoveIt.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--world", help="path to the world YAML (compiled via the plugin pipeline)")
    source.add_argument("--mjcf", help="path to a bare MJCF file (compiled directly)")
    parser.add_argument("--out", required=True, help="output .urdf path")
    parser.add_argument(
        "--prefix", default="", help="MJCF name prefix selecting the robot (stripped in the output)"
    )
    parser.add_argument("--name", default="robot", help="robot name in the URDF")
    parser.add_argument(
        "--root-link",
        default="base_link",
        help="name for the robot's root link (default base_link, what nav2/MoveIt expect)",
    )
    parser.add_argument(
        "--world-link",
        default="world",
        help="name of the fixed link parenting a JOINTED root (a rail carriage / gantry). Emitted "
        "only when the root body actually has a hinge or slide joint; ignored otherwise",
    )
    parser.add_argument(
        "--collapse",
        default="",
        help="comma-separated bodies whose subtree is lumped into one fixed link -- for a closed-loop "
        "gripper linkage URDF cannot express (e.g. base_mount for a Robotiq 2F-85)",
    )
    parser.add_argument(
        "--strip",
        default=None,
        help="prefix removed from emitted link/joint names (default: --prefix). Set it when --prefix "
        "selects a whole robot but the joint names published to /joint_states are prefix-stripped, "
        "as for an arm mounted on a mobile base",
    )
    parser.add_argument(
        "--gripper-joint",
        default="",
        help="joint (inside a --collapse subtree) that the gripper is COMMANDED through; it is kept "
        "as a URDF joint so MoveIt's GripperCommand controller and SRDF gripper group have one",
    )
    parser.add_argument(
        "--tip-site",
        default="",
        help="MuJoCo site the arm chain should END at -- a gripper's grasp point (e.g. `pinch` on a "
        "Robotiq 2F-85). A site is not a body, so it has no link of its own; this emits one as a "
        "fixed child of whatever link carries it, `--collapse` included. Without it a goal for the "
        "fingertips must be written as an offset from the flange, and every orientation tolerance "
        "is then multiplied by that lever arm",
    )
    parser.add_argument(
        "--tip-link",
        default="tcp",
        help="name of the link --tip-site emits (ignored without it)",
    )
    parser.add_argument(
        "--mesh-package",
        default="",
        help="emit meshes as package://<PKG>/<mesh-dir>/<file> instead of file://<abs path>. Use "
        "when the URDF ships inside an ament package, where a baked absolute path resolves to "
        "nothing after install",
    )
    parser.add_argument(
        "--mesh-dir",
        help="directory to write the referenced meshes into (default: <out>/../meshes)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the export by comparing FK against the MJCF and fail if it diverges",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="--check: max allowed FK error in metres"
    )
    parser.add_argument(
        "--skip-plugins",
        default="",
        help="extra plugin names/refs to drop before compiling; transport/bridge plugins are always "
        "dropped (they contribute no geometry)",
    )
    parser.add_argument(
        "--manifest",
        help="also write {'inputs': [...]} so a caller can tell when this output is stale",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_urdf")

    from .export_web import _compile_from_mjcf, _compile_from_world

    skip = {s.strip() for s in args.skip_plugins.split(",") if s.strip()}
    if args.mjcf:
        model, _data, _view = _compile_from_mjcf(Path(args.mjcf))
    else:
        model, _data, _view = _compile_from_world(args.world, skip, {}, log)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else out.parent / "meshes"
    exporter = UrdfExporter(
        model,
        prefix=args.prefix,
        name=args.name,
        root_link=args.root_link,
        collapse=tuple(s.strip() for s in args.collapse.split(",") if s.strip()),
        mesh_dir=mesh_dir,
        world_link=args.world_link,
        gripper_joint=args.gripper_joint,
        strip=args.strip,
        mesh_package=args.mesh_package,
        tip_site=args.tip_site,
        tip_link=args.tip_link,
    )
    tree = exporter.export()
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    log.info(
        "wrote %s (%d links, %d meshes)%s",
        out,
        len(tree.getroot().findall("link")),
        len(exporter.mesh_files),
        f"; dropped {len(exporter.dropped_dofs)} collapsed DOF(s): "
        + ", ".join(exporter.dropped_dofs)
        if exporter.dropped_dofs
        else "",
    )

    if args.check:
        err, where = round_trip_error(out, model, exporter.strip_prefix, mesh_dir=mesh_dir)
        log.info("round-trip FK error: %.3e m (worst link: %s)", err, where)
        if err > args.tolerance:
            log.error(
                "URDF disagrees with the MJCF by %.3e m at %r (tolerance %.1e). MoveIt would plan "
                "against different kinematics than MuJoCo simulates.",
                err,
                where,
                args.tolerance,
            )
            return 1

    if args.manifest:
        from .config import world_sources

        sources = (
            [str(p) for p in world_sources(args.world)]
            if args.world
            else [str(Path(args.mjcf).resolve())]
        )
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump({"inputs": sources}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
