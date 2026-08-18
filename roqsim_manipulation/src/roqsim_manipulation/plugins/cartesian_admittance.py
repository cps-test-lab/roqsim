"""Controller plugin: Cartesian end-effector control for an arm, with or without force feedback.

``arm_controller`` holds a joint vector. That is enough to replay a trajectory and not enough to do
anything in contact: a contact task is specified in the *task* frame ("push down at 10 N until the
peg seats"), not in joint space, and the interesting laws all close a loop around a measured wrench.
This is the substrate's Cartesian layer, and its force-controlled half is the arm-family equivalent
of what a local planner is for a mobile base.

Two laws, one plugin, because they differ only in how the next twist is computed and share
everything downstream:

``position``
    A Cartesian P controller toward a goal pose. Blind to contact by construction -- it commands the
    same descent whether the peg is in free space or crushing itself against a misaligned bore, which
    is exactly what makes it the honest baseline for a force-control comparison.

``admittance``
    ``M xddot = (w_d - w_a) - D xdot - C (x - x_0)``, integrated to a twist. With ``C = 0`` this is a
    pure admittance (the arm follows force); with ``C > 0`` it is the mass-spring-damper that pulls
    back toward ``x_0``. The stiffness term is included but defaults OFF, because a stiffness with no
    stated equilibrium is a spring anchored to wherever the trial happened to start.

Everything after the twist is shared: clamp, resolve to joint velocities through a damped
least-squares inverse of the site Jacobian, integrate to joint position targets, and hand those to
``arm_controller``. The DLS solve is deliberate: near a singularity a plain pseudo-inverse produces
enormous joint velocities from a small Cartesian command, and in a contact task that reads as a
sudden force spike -- a physics artefact indistinguishable, in the metrics, from a real jam.

**Single-writer.** This plugin never touches ``data.ctrl``. It writes joint *targets* through the
``ArmHandle`` that ``arm_controller`` publishes, and ``arm_controller`` remains the only writer of
that arm's actuators. Declare it AFTER the arm in the world YAML.

Config::

    cartesian_admittance:
      arm: ur5e                # entity name (the ArmHandle at `arm:<name>` is required)
      site: tool_site          # site whose pose is controlled (prefixed with the arm's prefix)
      ft: ft                   # blackboard key suffix of the force_torque sensor (`ft:<key>`);
                               #   required for law: admittance, ignored for law: position
      law: admittance          # admittance | position
      rate_hz: 100.0           # control rate; the loop runs at this, not at the physics rate
      # -- admittance law ------------------------------------------------------------------------
      target_wrench: [0, 0, -10, 0, 0, 0]    # w_d, in the wrench's own frame
      mass: [1, 1, 1, 0.6, 0.6, 0.6]         # M, diagonal
      damping: [80, 80, 80, 160, 160, 160]   # D, diagonal
      stiffness: [0, 0, 0, 0, 0, 0]          # C, diagonal (0 -> pure admittance)
      axes: [1, 1, 1, 1, 1, 1]               # per-axis enable mask on the resulting twist
      # -- position law --------------------------------------------------------------------------
      kp: [2, 2, 2, 2, 2, 2]                 # proportional gain on the pose error
      # -- shared --------------------------------------------------------------------------------
      max_linear_vel: 0.1      # m/s, clamp on the commanded twist
      max_angular_vel: 1.0     # rad/s
      ik_damping: 0.01         # damped-least-squares lambda

Publishes a ``CartesianHandle`` on the blackboard under ``cartesian:<arm>`` with ``set_goal(pos,
quat)``, ``read_pose()`` and ``set_law(law)``, so a task plugin can steer it without importing it.

**Frames.** The commanded twist is applied in the WORLD frame, and the measured wrench is used as
given. Configure ``force_torque``'s ``frame:`` to match how the task defines its insertion axis; a
wrench reported in the sensor frame and integrated as if it were world-frame produces a controller
that drifts sideways under load and looks like a friction problem.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin

_LAWS = ("admittance", "position")


@dataclass
class CartesianHandle:
    """Blackboard handle under ``cartesian:<arm>``; all callables run on the physics thread."""

    arm: str
    set_goal: Callable[[np.ndarray, np.ndarray], None]
    read_pose: Callable[[], tuple[np.ndarray, np.ndarray]]
    set_law: Callable[[str], None]
    set_active: Callable[[bool], None]


class CartesianAdmittancePlugin(Plugin):
    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.arm = self.config.get("arm") or self.config.get("robot") or "arm"
        self.site = self.config.get("site", "tool_site")
        self.ft_key = self.config.get("ft", "ft")
        self.law = self.config.get("law", "admittance")
        self.rate_hz = float(self.config.get("rate_hz", 100.0))
        self.w_d = np.array(self.config.get("target_wrench", [0, 0, -10, 0, 0, 0]), dtype=float)
        self.M = np.array(self.config.get("mass", [1, 1, 1, 0.6, 0.6, 0.6]), dtype=float)
        self.D = np.array(self.config.get("damping", [80, 80, 80, 160, 160, 160]), dtype=float)
        self.C = np.array(self.config.get("stiffness", [0, 0, 0, 0, 0, 0]), dtype=float)
        self.axes = np.array(self.config.get("axes", [1, 1, 1, 1, 1, 1]), dtype=float)
        self.kp = np.array(self.config.get("kp", [2, 2, 2, 2, 2, 2]), dtype=float)
        self.v_lin = float(self.config.get("max_linear_vel", 0.1))
        self.v_ang = float(self.config.get("max_angular_vel", 1.0))
        self.ik_damping = float(self.config.get("ik_damping", 0.01))

        self._ctx: SimContext | None = None
        self._arm_handle = None
        self._ft = None
        self._site_id = -1
        self._dofs: np.ndarray = np.zeros(0, dtype=int)
        self._joint_names: list[str] = []
        self._twist = np.zeros(6)
        self._goal_pos: np.ndarray | None = None
        self._goal_mat: np.ndarray | None = None
        self._anchor_pos: np.ndarray | None = None  # x_0 for the stiffness term
        self._active = True
        self._next_t = 0.0
        self._q_target: np.ndarray | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if config.get("law", "admittance") not in _LAWS:
            errors.append(f"'law' must be one of {', '.join(_LAWS)}")
        if float(config.get("rate_hz", 100.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        for key, width in (
            ("target_wrench", 6),
            ("mass", 6),
            ("damping", 6),
            ("stiffness", 6),
            ("axes", 6),
            ("kp", 6),
        ):
            if key in config and len(config[key]) != width:
                errors.append(f"'{key}' must have {width} entries (one per Cartesian axis)")
        if "mass" in config and any(float(v) <= 0 for v in config["mass"]):
            errors.append("'mass' entries must be > 0 (M is inverted in the admittance law)")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.arm)
        prefix = entity.meta.get("prefix", "") if entity else ""

        self._arm_handle = ctx.blackboard.get(f"arm:{self.arm}")
        if self._arm_handle is None:
            raise RuntimeError(
                f"cartesian_admittance: no ArmHandle at 'arm:{self.arm}'. This plugin commands the "
                f"arm through arm_controller rather than writing ctrl itself, so arm_controller must "
                f"be configured first -- list it (or the arm's spawn) BEFORE this plugin."
            )
        if self.law == "admittance":
            self._ft = ctx.blackboard.get(f"ft:{self.ft_key}")
            if self._ft is None:
                raise RuntimeError(
                    f"cartesian_admittance: law 'admittance' needs a wrench, but no force_torque "
                    f"sensor is registered at 'ft:{self.ft_key}'. Add a `force_torque` plugin (its "
                    f"`name` is the key) before this one, or use law 'position'."
                )

        site_name = f"{prefix}{self.site}"
        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise RuntimeError(f"cartesian_admittance: site {site_name!r} not found")

        # The DOFs this controller may move: the arm's own joints, in the order arm_controller
        # commands them. Anything else in the model (a second arm, a conveyor) stays untouched --
        # a Jacobian solve over every DOF in the world would happily move all of them.
        self._joint_names = list(self._arm_handle.joint_names)
        dofs = []
        for jname in self._joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{jname}")
            if jid < 0:
                raise RuntimeError(f"cartesian_admittance: joint {prefix}{jname!r} not found")
            dofs.append(int(m.jnt_dofadr[jid]))
        self._dofs = np.array(dofs, dtype=int)

        ctx.blackboard.set(
            f"cartesian:{self.arm}",
            CartesianHandle(
                arm=self.arm,
                set_goal=self.set_goal,
                read_pose=self.read_pose,
                set_law=self.set_law,
                set_active=self.set_active,
            ),
        )

    # -- handle API ------------------------------------------------------------------------------

    def set_goal(self, pos, quat_or_mat=None) -> None:
        self._goal_pos = np.array(pos, dtype=float)
        if quat_or_mat is not None:
            mat = np.array(quat_or_mat, dtype=float)
            if mat.size == 4:
                out = np.zeros(9)
                mujoco.mju_quat2Mat(out, mat)
                mat = out
            self._goal_mat = mat.reshape(3, 3)

    def read_pose(self) -> tuple[np.ndarray, np.ndarray]:
        d = self._ctx.data
        return (
            np.array(d.site_xpos[self._site_id], dtype=float),
            np.array(d.site_xmat[self._site_id], dtype=float).reshape(3, 3),
        )

    def set_law(self, law: str) -> None:
        if law not in _LAWS:
            raise ValueError(f"cartesian_admittance: unknown law {law!r}")
        self.law = law
        self._twist = np.zeros(6)

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not active:
            self._twist = np.zeros(6)

    # -- lifecycle -------------------------------------------------------------------------------

    def on_reset(self, ctx: SimContext) -> None:
        self._twist = np.zeros(6)
        self._goal_pos = None
        self._goal_mat = None
        self._next_t = 0.0
        self._q_target = None
        pos, _ = self.read_pose()
        self._anchor_pos = pos.copy()

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control or not self._active:
            return
        # Fixed-rate control loop over a finer physics loop: a controller tuned at 100 Hz behaves
        # differently when run at the 1 kHz physics rate (the integrated twist grows ten times as
        # fast per unit time), so the rate is honoured rather than being whatever the world's
        # timestep happens to be.
        if ctx.sim_time < self._next_t:
            return
        dt = 1.0 / self.rate_hz
        self._next_t = ctx.sim_time + dt

        twist = self._admittance_twist(dt) if self.law == "admittance" else self._position_twist()
        twist = self._clamp(twist * self.axes)
        self._apply(ctx, twist, dt)

    # -- laws ------------------------------------------------------------------------------------

    def _admittance_twist(self, dt: float) -> np.ndarray:
        force, torque = self._ft.read()
        w_a = np.concatenate([force, torque])
        pos, _ = self.read_pose()
        anchor = self._anchor_pos if self._anchor_pos is not None else pos
        deflection = np.concatenate([pos - anchor, np.zeros(3)])
        accel = (self.w_d - w_a - self.D * self._twist - self.C * deflection) / self.M
        self._twist = self._clamp(self._twist + accel * dt)
        return self._twist

    def _position_twist(self) -> np.ndarray:
        pos, mat = self.read_pose()
        if self._goal_pos is None:
            return np.zeros(6)
        twist = np.zeros(6)
        twist[:3] = self.kp[:3] * (self._goal_pos - pos)
        if self._goal_mat is not None:
            # Orientation error as a rotation vector: R_err = R_goal @ R^T, taken to axis-angle.
            r_err = self._goal_mat @ mat.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r_err.reshape(9))
            axis = np.zeros(3)
            angle = mujoco.mju_quat2Vel(axis, quat, 1.0)
            twist[3:] = self.kp[3:] * (axis if angle is None else axis)
        return twist

    # -- shared downstream -----------------------------------------------------------------------

    def _clamp(self, twist: np.ndarray) -> np.ndarray:
        out = np.array(twist, dtype=float)
        out[:3] = np.clip(out[:3], -self.v_lin, self.v_lin)
        out[3:] = np.clip(out[3:], -self.v_ang, self.v_ang)
        return out

    def _apply(self, ctx: SimContext, twist: np.ndarray, dt: float) -> None:
        """Resolve a world-frame twist to joint position targets via damped least squares."""
        m, d = ctx.model, ctx.data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self._site_id)
        jac = np.vstack([jacp, jacr])[:, self._dofs]

        # dq = J^T (J J^T + lambda^2 I)^-1 v. Damping trades exactness for boundedness near a
        # singularity, which is the trade a contact task wants: an unbounded joint velocity there
        # shows up in the wrench as a spike that is pure numerics.
        lam2 = self.ik_damping**2
        jjt = jac @ jac.T + lam2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, twist)

        if self._q_target is None:
            _, positions, _, _ = self._arm_handle.read_state()
            by_name = dict(zip(self._arm_handle.joint_names, positions, strict=False))
            self._q_target = np.array([by_name[n] for n in self._joint_names], dtype=float)
        self._q_target = self._q_target + dq * dt
        self._arm_handle.set_targets(self._joint_names, self._q_target.tolist())
