"""Controller plugin: whole-upper-body joint-position hold for the AgiBot G2.

G2's mobile base is driven by the generic ``diff_drive`` plugin (its manifest wires the four wheel
roll actuators); this plugin owns everything *above* the base -- the 5-DoF torso lift, 3-DoF head,
two 7-DoF arms and the two single-DoF omnipicker grippers. It is the manipulation analogue of
``diff_drive``: it resolves the robot's *position* actuators (named ``<prefix>p_*`` by the port,
which excludes the ``<prefix>m_*`` wheel velocity servos ``diff_drive`` owns), holds a target joint
vector every ``pre_step``, and declares backend-neutral :class:`~roqsim.context.Endpoint`s a
bridge serves: a ``joint_states`` output and a ``joint_command`` input (a single-point
``JointTrajectory`` that sets the held targets -- the same interface a ros2_control
JointTrajectoryController exposes, so one config drives sim and hardware alike).

It also registers a handle on the blackboard under ``robot_body:<name>`` exposing ``joint_names``,
``set_targets(names, positions)`` and ``read_state()`` for in-process consumers (tests, teleop).

Config -- a component of the entry that spawns the robot, since ownership is where the entry
sits rather than a config key::

    agibot_g2_controller:
      namespace: ""           # transport scope (default: inherited from spawn_robot)
      rest: {idx21_arm_l_joint1: 1.6, ...}   # {joint: angle} spawn+hold stance (default: manifest)
      test_target: {idx31_gripper_l_inner_joint1: -0.6}   # optional {joint: pos} held every tick
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin


class AgibotG2ControllerPlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True
    #: actuator-name prefix (after the spawn prefix) that marks a body position servo
    POS_TAG = "p_"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self._acts: list[int] = []  # actuator ids, in model order
        self._jids: list[int] = []  # driven joint ids (aligned with _acts)
        self._jnames: list[str] = []  # unprefixed joint names (aligned)
        self._target = np.zeros(0)  # held ctrl targets (aligned with _acts)

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        tag = prefix + self.POS_TAG

        for aid in range(m.nu):
            aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            if not aname.startswith(tag):
                continue
            if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = int(m.actuator_trnid[aid, 0])
            self._acts.append(aid)
            self._jids.append(jid)
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            self._jnames.append(jn[len(prefix) :] if jn.startswith(prefix) else jn)
        if not self._acts:
            raise RuntimeError(
                f"agibot_g2_controller: no '{tag}*' position actuators for {self.robot!r}"
            )

        # Initial stance. spawn_robot strips the model's <keyframe> on attach, so a spawned G2 resets
        # to the bare qpos0 (all joints at their geometric zero -- for G2 that is a T-pose, arms
        # straight out). The `rest` config restores a rest pose by joint name (attach-safe, unlike the
        # keyframe): it seeds both the reset qpos (so the robot *spawns* in the pose) and the held
        # target (so the controller keeps it there). Absent joints keep qpos0.
        self._apply_stance(ctx)

        ctx.blackboard.set(f"robot_body:{self.robot}", _G2BodyHandle(self))

        ctx.interface.add(
            Endpoint(
                name="joint_states",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_joint_states,
                rate_hz=50.0,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.JointState",
                        "topic": self.topic_override("joint_states") or "joint_states",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="joint_command",
                direction="in",
                owner=self.robot,
                namespace=ns,
                write=self._on_command,
                backend={
                    "ros2": {
                        "type": "trajectory_msgs.msg.JointTrajectory",
                        "topic": self.topic_override("joint_command") or "joint_command",
                    }
                },
            )
        )

    def _apply_stance(self, ctx: SimContext) -> None:
        """Seed reset qpos + held target from the model's qpos0, then overlay the `rest` stance.

        Writes ``data.qpos`` for each named joint (so the robot spawns in the pose) and the matching
        ``_target`` entry (so it is held). Runs under configure/on_reset, both on the physics thread,
        so touching ``data`` obeys the single-writer rule.
        """
        m = ctx.model
        self._target = np.array([ctx.data.qpos[m.jnt_qposadr[j]] for j in self._jids])
        rest = self.config.get("rest") or {}
        idx = {n: i for i, n in enumerate(self._jnames)}
        for jn, ang in rest.items():
            if jn not in idx:
                raise RuntimeError(
                    f"agibot_g2_controller: rest stance names joint {jn!r}, which is not a "
                    f"'{self.POS_TAG}*' position joint of {self.robot!r}"
                )
            i = idx[jn]
            ctx.data.qpos[m.jnt_qposadr[self._jids[i]]] = float(ang)
            self._target[i] = float(ang)

    # ---- command paths ---------------------------------------------------------------------
    def set_targets(self, names, positions) -> None:
        idx = {n: i for i, n in enumerate(self._jnames)}
        for n, p in zip(names, positions, strict=False):
            if n in idx:
                self._target[idx[n]] = float(p)

    def _on_command(self, traj) -> None:
        """Single-point JointTrajectory (names, points[-1].positions) -> held targets."""
        names, positions = traj
        self.set_targets(names, positions)

    def read_joint_states(self):
        m, d = self._ctx.model, self._ctx.data
        pos = np.array([d.qpos[m.jnt_qposadr[j]] for j in self._jids])
        vel = np.array([d.qvel[m.jnt_dofadr[j]] for j in self._jids])
        return (self._jnames, pos, vel)

    # ---- lifecycle -------------------------------------------------------------------------
    def on_reset(self, ctx: SimContext) -> None:
        self._apply_stance(ctx)

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            return  # the viewer's sliders own the actuators this run
        if "test_target" in self.config:
            self.set_targets(
                list(self.config["test_target"]), list(self.config["test_target"].values())
            )
        ctx.data.ctrl[self._acts] = self._target


class _G2BodyHandle:
    """In-process handle (blackboard ``robot_body:<name>``) for tests/teleop."""

    def __init__(self, plugin: AgibotG2ControllerPlugin):
        self._p = plugin

    @property
    def joint_names(self):
        return list(self._p._jnames)

    def set_targets(self, names, positions):
        self._p.set_targets(names, positions)

    def read_state(self):
        return self._p.read_joint_states()
