# SPDX-License-Identifier: Apache-2.0
"""A humanoid robot as a body of the motion core: its own joints, without its controller software.

The ``humanoid_robot`` plugin is nested under the ``spawn_model`` that places a humanoid robot
(``spawn_model`` reads no manifest, so no locomotion policy, arm controller or sensor of the robot
runs). With its retargeting table (:mod:`roqsim_motion.retarget`, ``<model>.retarget.yaml``) it
publishes the robot as a body (``humanoid_body:<entity>``) the moves of a nested ``humanoid`` drive
as they drive a walker, and the ``humanoid`` navigator output moves it.

Two forms (``drive``):

* ``kinematic``: every step the joints are set to their targets and the base to the nav root, with
  their velocities zeroed. The robot pushes what it touches as a walker does and is never pushed.
* ``physical``: every mapped joint is tracked by a plain position servo -- the robot's own where the
  table says ``servos: own`` and the model has one, else one this plugin adds at build with the
  table's gains -- and the base is held by a soft weld to an anchor the navigator moves. The body
  pushes and is pushed; the anchor carries it.

**No locomotion or balance controller is written, and balance is not simulated.** The base hangs on
its anchor (or is set), the legs play a gait over it, the feet may slide or press into the floor,
and the contact forces under them are not a standing robot's.

Its hands are solved on the robot's own arm and waist joints: a few damped-least-squares iterations
per update with MuJoCo's site Jacobian on a scratch copy of the state, closed on the hand's actual
pose, so a servo's sag under a load is taken up. A hand's grip frame is a site the plugin adds on the
hand at the table's frame, turned by its ``grip`` rotation into the grip-frame convention of
:mod:`roqsim_motion.hands`; a gripper closes through its actuator.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.plugin import Plugin
from roqsim.schema import Field
from roqsim_nav.outputs import NavOutput, OutputUnavailable

from . import quat as Q
from .body import PARK, SIDES, HumanoidBody
from .retarget import Table, TableError, find_table, load_table

#: How often the body solves and sets its targets (Hz), and the solver's damping and iterations.
UPDATE_HZ = 100.0
DAMPING = 0.05
ITERATIONS = 3
#: How strongly the solve draws joints back to their rest where the goals leave them free, and how
#: much the waist moves against an arm's joints.
REST_PULL = 0.1
WAIST_WEIGHT = 0.4
#: How much a rotation's error (rad) counts against a position's (m) in the solve.
ROT_WEIGHT = 0.1
#: How fast a hand's aim is trimmed by its actual error (1/s), and how far at most (m).
TRIM_RATE = 4.0
TRIM_MAX = 0.15
#: How far a joint target may move in one update (rad), so a far goal is walked to, not jumped to.
MAX_STEP = 0.08
#: How fast the gait's phase advances per metre travelled, and its swing (rad).
STRIDE = 0.5
HIP_SWING = 0.35
KNEE_LIFT = 0.6
#: The base anchor's weld softness (s, damping ratio).
ANCHOR_SOLREF = (0.03, 1.0)


class RobotBody(HumanoidBody):
    """The robot's side of the motion core: hands solved on its joints, a gait on its legs."""

    has_proxies = False
    two_hands = "both"

    @property
    def hands_track_goals(self) -> bool:  # servo-driven hands lag their goals
        return self.p.config["drive"] == "kinematic"

    def __init__(self, plugin: HumanoidRobotPlugin):
        self.p = plugin
        super().__init__(plugin.entity)

    def reset(self) -> None:
        super().reset()
        self.facing = None
        self.turn_cap = None
        self.keep = None

    # -- answers ------------------------------------------------------------------------------
    def grip(self, data, side):
        sid = self.p.grip_site[side]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[sid])
        return data.site_xpos[sid].copy(), q

    def grip_site(self, side: str) -> str:
        return f"{self.name}/grip_{side}"

    def shoulder(self, data, side):
        return data.xpos[self.p.shoulder_body[side]].copy()

    def arm_reach(self, side: str) -> float:
        return self.p.reach[side]

    def root(self):
        x, y, yaw = self.p.root
        return float(x), float(y), float(yaw)

    def face(self, yaw):
        self.facing = None if yaw is None else float(yaw)

    def keep_heading(self, yaw):
        self.keep = None if yaw is None else float(yaw)

    def limit_turn(self, rate):
        self.turn_cap = None if rate is None else float(rate)

    def settled(self, side: str) -> bool:
        return True

    def carry_height(self) -> float:
        return self.p.table.base_height * 0.95

    def place_reach(self) -> float:
        return 0.4

    def gripper(self, data, side: str, closed: bool) -> None:
        grip = self.p.table.hands[side].get("grip") or {}
        if grip.get("actuator"):
            self.p.gripper_ctrl[side] = float(grip["closed" if closed else "open"])

    def hand_done(self, side: str) -> bool:
        aid = self.p.gripper_act.get(side, -1)
        if aid < 0:
            return True
        jnt = int(self.p.ctx.model.actuator_trnid[aid][0])
        if self.p.ctx.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            return self.p.ctx.sim_time - self.p.gripper_since[side] > 0.5
        dof = int(self.p.ctx.model.jnt_dofadr[jnt])
        return (
            abs(float(self.p.ctx.data.qvel[dof])) < 0.01
            and self.p.ctx.sim_time - self.p.gripper_since[side] > 0.3
        )

    def owns(self, model, body: int) -> bool:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        return name.startswith(self.p.prefix) or name.startswith(f"{self.name}/")


class HumanoidRobotPlugin(Plugin):
    """A humanoid robot as a body of the motion core (see the module docstring)."""

    requires_owner = True
    CONFIG_SCHEMA = {
        "table": Field(
            str,
            default="",
            doc="the retargeting table: a model's name (default: the spawned model's) or a YAML path",
        ),
        "drive": Field(
            str,
            default="physical",
            choices=["kinematic", "physical"],
            doc="joints set, or tracked by servos",
        ),
    }

    @classmethod
    def expand(cls, spec, world, base_dir):
        owner = next((w for w in world if w.address == spec.entity), None)
        if owner is None or owner.ref != "spawn_model":
            raise ValueError(
                f"humanoid_robot {spec.address}: nest it under the spawn_model that places the robot "
                f"(spawn_model reads no manifest, so none of the robot's own controllers runs)"
            )
        spec.config.setdefault("prefix", str((owner.config or {}).get("prefix", "")))
        if not spec.config.get("table"):
            spec.config["table"] = str((owner.config or {}).get("model", ""))
        return []

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.prefix = str(self.config.get("prefix", ""))
        self.table: Table | None = None
        self.body: RobotBody | None = None
        self.root = np.zeros(3)
        self.start = np.zeros(3)

    def _name(self, n: str) -> str:
        return self.prefix + n

    # -- build --------------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        ref = str(self.config["table"])
        try:
            self.table = load_table(ref) if ref.endswith((".yaml", ".yml")) else find_table(ref)
        except TableError as exc:
            raise TableError(f"humanoid_robot {self.address}: {exc}") from exc
        t = self.table
        names = {s.name: s for s in spec.sites}
        for side, hand in t.hands.items():
            rot = Q.from_rpy(*hand["rpy"]) if "rpy" in hand else Q.IDENTITY
            if "frame" in hand:
                frame = names.get(self._name(hand["frame"]))
                if frame is None:
                    raise TableError(
                        f"humanoid_robot {self.address}: the table's hands.{side}.frame "
                        f"{hand['frame']!r} is no site of the model (prefix {self.prefix!r})"
                    )
                parent, pos, q = (
                    frame.parent,
                    list(frame.pos),
                    Q.mul(np.array(frame.quat, float), rot),
                )
            else:
                try:
                    parent = spec.body(self._name(hand["body"]))
                except Exception:
                    parent = None
                if parent is None:
                    raise TableError(
                        f"humanoid_robot {self.address}: the table's hands.{side}.body "
                        f"{hand['body']!r} is no body of the model (prefix {self.prefix!r})"
                    )
                pos, q = list(hand["pos"]), rot
            parent.add_site(name=f"{self.entity}/grip_{side}", pos=pos, quat=list(q))
        joints = {j.name: j for j in spec.joints}
        actuated = {a.target for a in spec.actuators}
        # `own`: the model's servos where it has them, servos added for the rest; `add`: a servo
        # added on every mapped joint, beside the model's own -- both track one target, so their
        # stiffness adds (an arm whose own wrist servos are too soft to hold a load).
        own = (t.physical or {}).get("servos", "own") == "own"
        self._added = []
        if self.config["drive"] == "physical":
            kp = float((t.physical or {}).get("kp", 300.0))
            kv = float((t.physical or {}).get("kv", 10.0))
            for j in t.model_joints():
                full = self._name(j)
                if own and full in actuated and self._is_servo(spec, full):
                    continue
                a = spec.add_actuator()
                a.name = f"{self.entity}/servo/{j}"
                a.target = full
                a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                a.set_to_position(kp=kp, kv=kv)
                if full in joints and joints[full].range[1] > joints[full].range[0]:
                    a.ctrlrange = list(joints[full].range)
                    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                self._added.append(a.name)
            # the base hangs on an anchor the navigator moves
            anchor = spec.worldbody.add_body(
                name=f"{self.entity}/anchor", pos=list(PARK), mocap=True
            )
            anchor.add_site(name=f"{self.entity}/anchor")
            base = spec.body(self._name(t.base_body))
            base.add_site(name=f"{self.entity}/base")
            eq = spec.add_equality()
            eq.name = f"{self.entity}/anchor"
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE
            eq.name1 = f"{self.entity}/base"
            eq.name2 = f"{self.entity}/anchor"
            eq.solref = list(ANCHOR_SOLREF)

    @staticmethod
    def _is_servo(spec, joint: str) -> bool:
        for a in spec.actuators:
            if a.target == joint:
                return (
                    a.biastype == mujoco.mjtBias.mjBIAS_AFFINE
                    and a.gaintype == mujoco.mjtGain.mjGAIN_FIXED
                )
        return False

    # -- configure ------------------------------------------------------------------------------
    def configure(self, ctx) -> None:
        self.ctx = ctx
        m = ctx.model
        t = self.table
        t.check(m, self.prefix)
        self.joint_ids = {
            j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self._name(j))
            for j in t.model_joints()
        }
        self.qadr = {j: int(m.jnt_qposadr[i]) for j, i in self.joint_ids.items()}
        self.dadr = {j: int(m.jnt_dofadr[i]) for j, i in self.joint_ids.items()}
        self.range = {
            j: (float(m.jnt_range[i][0]), float(m.jnt_range[i][1]))
            if m.jnt_limited[i]
            else (-math.pi, math.pi)
            for j, i in self.joint_ids.items()
        }
        self.servo = {}  # joint -> every position servo on it (the model's and the added one)
        names = {self._name(j): j for j in t.model_joints()}
        for i in range(m.nu):
            if (
                m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                and m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
            ):
                jname = (
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i][0]))
                    or ""
                )
                if jname in names:
                    self.servo.setdefault(names[jname], []).append(i)
        self.grip_site = {
            s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{self.entity}/grip_{s}")
            for s in t.hands
        }
        self.gripper_act = {
            s: mujoco.mj_name2id(
                m,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                self._name((h.get("grip") or {}).get("actuator", "")),
            )
            if (h.get("grip") or {}).get("actuator")
            else -1
            for s, h in t.hands.items()
        }
        self.gripper_ctrl = {
            s: float((h.get("grip") or {}).get("open", 0.0)) for s, h in t.hands.items()
        }
        self.gripper_since = {s: 0.0 for s in t.hands}
        self.arms = {
            s: [
                d.joint
                for d in t.joints.get(f"shoulder_{s}", [])
                + t.joints.get(f"elbow_{s}", [])
                + t.joints.get(f"wrist_{s}", [])
            ]
            for s in SIDES
        }
        self.waist = [d.joint for d in t.joints.get("spine", [])]
        self.legs = {
            s: {
                k: [d.joint for d in t.joints.get(f"{k}_{s}", [])] for k in ("hip", "knee", "ankle")
            }
            for s in SIDES
        }
        self.shoulder_body = {s: int(m.jnt_bodyid[self.joint_ids[self.arms[s][0]]]) for s in SIDES}
        self.reach = {}
        base_jnt = None
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self._name(t.base_body))
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                base_jnt = j
        if base_jnt is None:
            raise TableError(
                f"humanoid_robot {self.address}: base {t.base_body!r} has no free joint"
            )
        self.base_q = int(m.jnt_qposadr[base_jnt])
        self.base_d = int(m.jnt_dofadr[base_jnt])
        self.anchor = (
            int(
                m.body_mocapid[
                    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.entity}/anchor")
                ]
            )
            if self.config["drive"] == "physical"
            else -1
        )
        self.scratch = mujoco.MjData(m)
        q0 = m.qpos0
        self.start = np.array(
            [
                float(q0[self.base_q]),
                float(q0[self.base_q + 1]),
                Q.yaw_of(q0[self.base_q + 3 : self.base_q + 7]),
            ]
        )
        self.body = RobotBody(self)
        self._reset_targets()
        self.reach = self._arm_reach(m)
        ctx.blackboard.set(f"humanoid_body:{self.entity}", self.body)
        ctx.blackboard.set(f"humanoid_robot:{self.entity}", self)

    def _arm_reach(self, m) -> dict:
        """How far each grip frame reaches from its shoulder with the arm straightened (m): the
        distances from each arm joint's axis to the next, shoulder to wrist, and from the last to
        the grip site, measured on the model at rest -- the longest the chain can be."""
        d = self.scratch
        mujoco.mj_forward(m, d)
        out = {}
        for s in SIDES:
            points = [d.xanchor[self.joint_ids[j]].copy() for j in self.arms[s]]
            points.append(d.site_xpos[self.grip_site[s]].copy())
            out[s] = float(
                sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:], strict=False))
            )
        return out

    def _reset_targets(self) -> None:
        self.rest = {j: float(self.table.rest.get(j, 0.0)) for j in self.qadr}
        self.target = dict(self.rest)
        self.root = self.start.copy()
        self.phase = 0.0
        self.accum = 0.0
        self.weight = {s: 0.0 for s in SIDES}
        self.trim = {s: np.zeros(3) for s in SIDES}
        # whether the navigator is moving the root (the gait plays): set at the navigator's rate and
        # held until it says otherwise, not cleared at the body's own faster update
        self._moving = False

    def on_reset(self, ctx) -> None:
        self.body.reset()
        self._reset_targets()
        self.gripper_ctrl = {
            s: float((h.get("grip") or {}).get("open", 0.0)) for s, h in self.table.hands.items()
        }
        self._place_base(ctx.data)
        self._apply(ctx.data)

    # -- each step ------------------------------------------------------------------------------
    def pre_step(self, ctx) -> None:
        self.accum += ctx.dt
        if self.accum >= 1.0 / UPDATE_HZ - 1e-9:
            dt, self.accum = self.accum, 0.0
            self._solve(ctx.data, dt)
        self._place_base(ctx.data)
        self._apply(ctx.data)

    def walk(self, pref_vel, dt: float) -> None:
        """The navigator's motion (``humanoid`` output): the root moves, the gait phase follows the
        distance, and the root turns toward where it goes, or a move's facing when it stands."""
        v = np.asarray(pref_vel, float)
        speed = float(np.hypot(v[0], v[1]))
        self.root[:2] += v * dt
        self.phase += speed * dt / STRIDE
        if self.body.keep is not None:
            want = self.body.keep  # a move keeps the heading while it steps
        elif speed > 1e-3:
            want = math.atan2(v[1], v[0])
        elif self.body.facing is not None:
            want = self.body.facing
        else:
            return
        cap = (self.body.turn_cap or 4.0) * max(dt, 0.0)
        err = math.atan2(math.sin(want - self.root[2]), math.cos(want - self.root[2]))
        self.root[2] += max(-cap, min(cap, err))

    def _place_base(self, data) -> None:
        x, y, yaw = self.root
        pos = [x, y, self.table.base_height]
        quat = Q.axis_angle("z", yaw)
        if self.anchor >= 0:
            data.mocap_pos[self.anchor] = pos
            data.mocap_quat[self.anchor] = quat
        else:
            data.qpos[self.base_q : self.base_q + 3] = pos
            data.qpos[self.base_q + 3 : self.base_q + 7] = quat
            data.qvel[self.base_d : self.base_d + 6] = 0.0

    def _apply(self, data) -> None:
        for j, value in self.target.items():
            if self.anchor >= 0:
                for aid in self.servo.get(j, ()):
                    data.ctrl[aid] = value
            else:
                data.qpos[self.qadr[j]] = value
                data.qvel[self.dadr[j]] = 0.0
        for s, aid in self.gripper_act.items():
            if aid >= 0:
                if data.ctrl[aid] != self.gripper_ctrl[s]:
                    self.gripper_since[s] = self.ctx.sim_time
                data.ctrl[aid] = self.gripper_ctrl[s]

    def _solve(self, data, dt: float) -> None:
        m = self.ctx.model
        # legs: a gait while it walks, standing when it does not
        swing = math.sin(2 * math.pi * self.phase)
        moving = abs(swing) > 1e-6 and self._moving
        for k, side in enumerate(SIDES):
            s = swing if k == 0 else -swing
            hips, knees = self.legs[side]["hip"], self.legs[side]["knee"]
            if hips:
                self.target[hips[0]] = -HIP_SWING * s if moving else 0.0
            if knees:
                self.target[knees[0]] = KNEE_LIFT * max(0.0, s) if moving else 0.0
        # arms and waist: each hand with a goal, solved on the actual pose
        goals = {
            s: self.body.goal[s]
            for s in SIDES
            if self.body.goal[s] is not None and s in self.grip_site
        }
        for s in SIDES:
            aim = 1.0 if s in goals else 0.0
            self.weight[s] += max(-2.0 * dt, min(2.0 * dt, aim - self.weight[s]))
            if s not in goals:
                self.trim[s] = np.zeros(3)
                for j in self.arms[s]:
                    self.target[j] += max(
                        -MAX_STEP,
                        min(
                            MAX_STEP,
                            self.weight[s] * (self.target[j] - self.rest[j])
                            + self.rest[j]
                            - self.target[j],
                        ),
                    )
        if not goals:
            return
        sd = self.scratch
        cols = sorted({j for s in goals for j in self.arms[s]} | set(self.waist))
        dofs = [self.dadr[j] for j in cols]
        weights = np.array([WAIST_WEIGHT if j in self.waist else 1.0 for j in cols])
        # A servo sags under a load: each hand's aim is trimmed, slowly, by how far it actually is
        # from its goal (an integral, bounded), so it ends on the goal without chasing the arm's sway.
        sd.qpos[:] = data.qpos
        for j in cols:
            sd.qpos[self.qadr[j]] = self.target[j]
        mujoco.mj_kinematics(m, sd)
        aims = {}
        for s, (gp, gq) in goals.items():
            if self.anchor >= 0:
                err = np.asarray(gp, float) - data.site_xpos[self.grip_site[s]]
                trim = self.trim[s] + TRIM_RATE * dt * err
                n = float(np.linalg.norm(trim))
                self.trim[s] = trim if n <= TRIM_MAX else trim * TRIM_MAX / n
            aims[s] = (np.asarray(gp, float) + self.trim[s], gq)
        for _ in range(ITERATIONS):
            mujoco.mj_comPos(m, sd)
            rows, errs = [], []
            for s, (gp, gq) in aims.items():
                sid = self.grip_site[s]
                jp = np.zeros((3, m.nv))
                jr = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, sd, jp, jr, sid)
                rows.append(jp[:, dofs])
                errs.append(gp - sd.site_xpos[sid])
                if gq is not None:
                    cur = np.zeros(4)
                    mujoco.mju_mat2Quat(cur, sd.site_xmat[sid])
                    rows.append(ROT_WEIGHT * jr[:, dofs])
                    errs.append(ROT_WEIGHT * Q.to_rotvec(Q.mul(gq, Q.inv(cur))))
            # The waist moves less than the arms (its columns weighted down), and in what the goals
            # leave free every joint is drawn back toward its rest, so the redundant ones do not
            # wander from goal to goal into a joint's limit.
            A = np.vstack(rows) * weights
            b = np.concatenate(errs)
            pinv = A.T @ np.linalg.inv(A @ A.T + DAMPING**2 * np.eye(len(b)))
            q_now = np.array([sd.qpos[self.qadr[j]] for j in cols])
            rest = np.array([self.rest[j] for j in cols])
            dq = pinv @ b + (np.eye(len(cols)) - pinv @ A) @ (REST_PULL * (rest - q_now))
            dq = np.clip(dq * weights, -MAX_STEP, MAX_STEP)
            for j, delta in zip(cols, dq, strict=True):
                lo, hi = self.range[j]
                sd.qpos[self.qadr[j]] = min(max(sd.qpos[self.qadr[j]] + delta, lo), hi)
            mujoco.mj_kinematics(m, sd)
        for j in cols:
            self.target[j] = float(sd.qpos[self.qadr[j]])
        for s in goals:
            self.body.residual = getattr(self.body, "residual", {})
            self.body.residual[s] = float(
                np.linalg.norm(aims[s][0] - sd.site_xpos[self.grip_site[s]])
            )


class HumanoidOutput(NavOutput):
    """The ``humanoid`` embodiment: a navigator moving a humanoid robot's root (``humanoid_robot``)."""

    kinematics = "holonomic"
    update_hz = 50.0

    def attach(self, ctx, entity) -> None:
        self.robot = ctx.blackboard.get(f"humanoid_robot:{entity.name}")
        if self.robot is None:
            raise OutputUnavailable(
                f"entity {entity.name!r} has no humanoid_robot: this output moves the robot a "
                f"`humanoid_robot` nested under its spawn_model publishes"
            )

    def emit(self, ctx, pref_vel, yaw, dt) -> None:
        self.robot.walk(pref_vel, dt)
        self.robot._moving = float(np.hypot(*np.asarray(pref_vel, float)[:2])) > 1e-3

    def hold(self, ctx, dt, facing) -> None:
        self.robot.walk(np.zeros(2), dt)
        self.robot._moving = False

    def stop(self, ctx) -> None:
        self.robot._moving = False

    def pose(self, ctx):
        x, y, yaw = self.robot.root
        return float(x), float(y), float(yaw)
