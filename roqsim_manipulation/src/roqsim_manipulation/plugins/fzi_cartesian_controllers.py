"""Plugin: FZI's Cartesian controllers -- motion, force and compliance -- with their law and parameters.

FZI's ``cartesian_controllers`` (BSD-3-Clause, FZI Research Center for Information Technology) are
ros2_control controllers a real arm runs: a stack drives them by streaming ``target_frame`` and
``target_wrench`` and switching them with the controller manager. This plugin runs the same law
under the same names and parameters, so the stack and its gain file are the ones the robot uses.
``cartesian_admittance`` shares the names but not the law; this is the port.

**The law** (``cartesian_controller_base``, ``ForwardDynamicsSolver``). Every control period the
internal model is set to the measured joint positions and simulated forward ``solver.iterations``
times with a fixed internal period of 0.02 s:

* the error ``e`` is the controller's: for motion, target minus current pose (position, and the
  rotation vector of ``R_target R_current^T``, each clamped to 1); for force, the measured wrench
  plus ``target_wrench``; for compliance, ``K e_motion`` plus the force error, ``K`` the diagonal
  ``stiffness`` turned into the base by the compliance reference;
* a per-axis PD (``pd_gains``, its derivative over the internal period) scaled by
  ``solver.error_scale`` gives a Cartesian force ``u`` on a virtual model: every moving link
  ``solver.forward_dynamics.link_mass`` at its frame (inertia 1e-6), the end effector 1 kg with unit
  inertia;
* ``q_ddot = H(q)^-1 J(q)^T u``, integrated by forward Euler from the previous internal state, the
  velocity damped by 10 % per step, positions clamped to the joint ranges;
* the final internal positions are the joint targets for the period.

The end-effector pose the first error reads is the internal model's from the previous period, as in
FZI's solver. Errors, forces and Jacobians are taken in the world frame; targets and wrenches arrive
in ``robot_base_link`` and are turned into it, which leaves ``H^-1 J^T u`` unchanged.

**Frames.** FZI names URDF links; a MuJoCo arm has bodies and sites. ``robot_base_link`` is a body
of the arm and a turn about its z (ur_description's ``base_link`` is the UR5e MJCF's ``base`` body
itself, so the turn is 0); ``end_effector_site`` the controlled frame (``tool0`` is the ``tool0_site``);
``compliance_ref`` either ``robot_base_link`` or ``end_effector``. The force-torque reading is
expected in the end effector's axes (``force_torque`` with ``frame: sensor`` on a site of the flange
body), environment on tool, which is what ``ft_sensor_ref_link: tool0`` means there.

**The servo decides how well the force is held.** The law re-reads the measured joints every
cycle and commands one cycle's step from there, so it presses with the commanded force only on an
arm that realises each commanded position -- which a UR's own servo does, and a simulated position
servo does only as far as its stiffness lets it: what the servo yields under the contact is lost at
the next re-read, and the settled force falls short of the target by that much
(``tests/test_fzi_cartesian_controllers.py``: 1.9 N of a commanded 10 N on the UR5e model's servo
stiffened to p 50000, 6.6 N at p 200000, at gains a stiff contact tolerates). A world that measures
a force hold through these controllers states a servo stiff enough for its contact, and reports the
held force rather than assuming the commanded one.

**What differs from the robot, on purpose.** The robot's controllers read the wrist wrench from a
topic, one control cycle late; ``ft_delay_cycles`` (default 1) delays the reading by as many cycles.
A target arriving while a controller is inactive is dropped, as FZI's callbacks drop it; an
activation starts from the current pose with zero internal velocity.

Config -- a component of the arm, after its ``arm_controller`` and the ``force_torque`` it reads::

    fzi_cartesian_controllers:
      ft: force_torque_sensor_broadcaster    # force_torque label, for force and compliance
      rate_hz: 500.0                         # the controller manager's update rate
      robot_base_link: {body: base, yaw: 0.0}
      end_effector_site: tool0_site
      ft_delay_cycles: 1
      controllers:
        cartesian_motion_controller:
          type: motion                       # motion | force | compliance
          solver: {error_scale: 1.0, iterations: 10, forward_dynamics: {link_mass: 0.1}}
          pd_gains: {trans_x: {p: 10.0}, trans_y: {p: 10.0}, trans_z: {p: 10.0},
                     rot_x: {p: 1.5}, rot_y: {p: 1.5}, rot_z: {p: 1.5}}
        cartesian_compliance_controller:
          type: compliance
          compliance_ref: robot_base_link    # or end_effector
          stiffness: {trans_x: 2000, trans_y: 2000, trans_z: 0, rot_x: 50, rot_y: 50, rot_z: 50}
          solver: {...}
          pd_gains: {...}

Every controller starts inactive; a stack activates one with ``switch_controller``. They claim the
arm's position interfaces, so one of them, or the trajectory controller, holds the arm at a time.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.controllers import INACTIVE, Controller, registry_for
from roqsim.plugin import Plugin
from roqsim.types import Force3, Point3, Pose, Quaternion, Torque3, Wrench

_AXES = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
_TYPES = {
    "motion": ("cartesian_motion_controller", "CartesianMotionController"),
    "force": ("cartesian_force_controller", "CartesianForceController"),
    "compliance": ("cartesian_compliance_controller", "CartesianComplianceController"),
}
#: FZI's internal simulation period, "deliberately independent of the outer control cycle".
_INTERNAL_PERIOD = 0.02
#: FZI's clamps on the motion error.
_MAX_DISTANCE, _MAX_ANGLE = 1.0, 1.0
#: FZI's minimal link inertia and global velocity damping.
_IP_MIN, _DAMPING = 1e-6, 0.9


def _quat_to_mat(wxyz) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(wxyz, dtype=float))
    return m.reshape(3, 3)


def _rotation_vector(R: np.ndarray) -> np.ndarray:
    """Axis times angle of R, the angle in [0, pi] (KDL's GetRotAngle)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(s, q[0])
    return q[1:] / s * angle


class _Law:
    """One FZI controller's parameters and PD state."""

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self.kind = cfg.get("type", "motion")
        solver = cfg.get("solver") or {}
        self.error_scale = float(solver.get("error_scale", 1.0))
        self.iterations = int(solver.get("iterations", 1))
        self.link_mass = float((solver.get("forward_dynamics") or {}).get("link_mass", 0.1))
        gains = cfg.get("pd_gains") or {}
        self.p = np.array([float((gains.get(a) or {}).get("p", 0.0)) for a in _AXES])
        self.d = np.array([float((gains.get(a) or {}).get("d", 0.0)) for a in _AXES])
        stiff = cfg.get("stiffness") or {}
        self.K = np.array([float(stiff.get(a, 0.0)) for a in _AXES])
        self.compliance_ref = cfg.get("compliance_ref", "robot_base_link")
        self.last_error = np.zeros(6)

    def pd(self, error: np.ndarray) -> np.ndarray:
        out = self.p * error + self.d * (error - self.last_error) / _INTERNAL_PERIOD
        self.last_error = error.copy()
        return out


class FZICartesianControllersPlugin(Plugin):
    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        c = self.config
        self.arm = self.entity
        self.ft_key = c.get("ft", "")
        self.rate_hz = float(c.get("rate_hz", 500.0))
        base = c.get("robot_base_link") or {}
        self.base_body = base.get("body", "base")
        self.base_yaw = float(base.get("yaw", 0.0))
        self.ee_site = c.get("end_effector_site", "tool0_site")
        self.ft_delay = int(c.get("ft_delay_cycles", 1))
        self.laws = {n: _Law(n, cfg or {}) for n, cfg in (c.get("controllers") or {}).items()}
        self._active: str | None = None
        self._registered: dict[str, Controller] = {}
        self._target_pos = np.zeros(3)
        self._target_mat = np.eye(3)
        self._target_wrench = np.zeros(6)
        self._ft_queue: list[np.ndarray] = []
        self._q = self._qd = None
        self._ee_pos = np.zeros(3)
        self._ee_mat = np.eye(3)
        self._next_t = 0.0
        self._ee_mat_measured = np.eye(3)

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        laws = config.get("controllers") or {}
        if not laws:
            errors.append("'controllers' names at least one FZI controller")
        for name, cfg in laws.items():
            cfg = cfg or {}
            if cfg.get("type", "motion") not in _TYPES:
                errors.append(f"controllers.{name}.type is one of {', '.join(_TYPES)}")
            if cfg.get("type") in ("force", "compliance") and not config.get("ft"):
                errors.append(
                    f"controllers.{name} reads a wrench: 'ft' must name a force_torque entry"
                )
            if cfg.get("compliance_ref", "robot_base_link") not in (
                "robot_base_link",
                "end_effector",
            ):
                errors.append(
                    f"controllers.{name}.compliance_ref is robot_base_link or end_effector"
                )
            if int((cfg.get("solver") or {}).get("iterations", 1)) < 1:
                errors.append(f"controllers.{name}.solver.iterations must be >= 1")
        if float(config.get("rate_hz", 500.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if int(config.get("ft_delay_cycles", 1)) < 0:
            errors.append("'ft_delay_cycles' must be >= 0")
        return errors

    # -- setup -------------------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.arm)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = entity.meta.get("namespace", "") if entity else ""
        self._arm = ctx.blackboard.get(f"arm:{self.arm}")
        if self._arm is None:
            raise RuntimeError(
                "fzi_cartesian_controllers: no ArmHandle for this arm; declare arm_controller first"
            )
        self._ft = None
        if any(law.kind != "motion" for law in self.laws.values()):
            self._ft = ctx.blackboard.get(f"ft:{self.ft_key}")
            if self._ft is None:
                raise RuntimeError(
                    f"fzi_cartesian_controllers: no force_torque at 'ft:{self.ft_key}'"
                )
            if self._ft.measures != "environment_on_tool":
                raise RuntimeError(
                    "fzi_cartesian_controllers: the force_torque must report the wrench "
                    "the environment applies to the tool (invert: true)"
                )
        self._site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{self.ee_site}")
        self._base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{self.base_body}")
        if self._site < 0 or self._base < 0:
            raise RuntimeError(
                f"fzi_cartesian_controllers: no site {prefix}{self.ee_site!r} or body "
                f"{prefix}{self.base_body!r}"
            )
        names = list(self._arm.joint_names)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{n}") for n in names]
        if min(jids) < 0:
            raise RuntimeError(f"fzi_cartesian_controllers: joints {names} not all found")
        self._names = names
        self._dofs = np.array([int(m.jnt_dofadr[j]) for j in jids])
        self._qadr = np.array([int(m.jnt_qposadr[j]) for j in jids])
        self._range = np.array(
            [m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf) for j in jids]
        )
        # The virtual model's moving links: the bodies each of the arm's joints moves.
        self._link_bodies = [int(m.jnt_bodyid[j]) for j in jids]
        self._kin = mujoco.MjData(m)
        registry = registry_for(ctx)
        for name, law in self.laws.items():
            pkg, cls = _TYPES[law.kind]
            self._registered[name] = registry.register(
                Controller(
                    name=name,
                    type=f"{pkg}/{cls}",
                    claims=tuple(f"{j}/position" for j in names),
                    state=INACTIVE,
                    namespace=ns,
                    owner=self.arm,
                    apply=lambda active, n=name: self._set_active(n, active),
                )
            )
        if getattr(self._arm, "set_command_source", None) is not None:
            self._arm.set_command_source(self.update, self.label or "fzi_cartesian_controllers")

    # -- the base frame -----------------------------------------------------------------------------

    def _base_frame(self, d) -> tuple[np.ndarray, np.ndarray]:
        """robot_base_link's (position, rotation) in the world: the base body turned by base_yaw."""
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return np.array(d.xpos[self._base]), np.array(d.xmat[self._base]).reshape(3, 3) @ turn

    # -- endpoints (FZI's topics, under each controller's name) -------------------------------------

    def _stream_target(self, name, position, orientation) -> None:
        if self._active != name:
            return  # FZI drops a target that arrives while its controller is inactive
        p, R = self._base_frame(self._ctx.data)
        self._target_pos = p + R @ np.asarray(position, dtype=float)
        self._target_mat = R @ _quat_to_mat(orientation)

    def _stream_wrench(self, name, force, torque) -> None:
        if self._active != name:
            return
        _, R = self._base_frame(self._ctx.data)
        self._target_wrench = np.concatenate(
            [R @ np.asarray(force, float), R @ np.asarray(torque, float)]
        )

    @endpoint.stream(
        Pose,
        name="{item}/target_frame",
        each="laws",
        ros2=lambda self, n: {"topic": f"{n}/target_frame"},
    )
    def target_frame(self, n, position: Point3, orientation: Quaternion) -> None:
        """The end effector's target pose in robot_base_link (w, x, y, z)."""
        self._stream_target(n, position, orientation)

    @endpoint.stream(
        Wrench,
        name="{item}/target_wrench",
        each="laws",
        ros2=lambda self, n: {"topic": f"{n}/target_wrench"},
    )
    def target_wrench(self, n, force: Force3, torque: Torque3) -> None:
        """The wrench to apply to the environment, in robot_base_link."""
        self._stream_wrench(n, force, torque)

    # -- activation ---------------------------------------------------------------------------------

    def _set_active(self, name: str, active: bool) -> None:
        if active:
            d = self._ctx.data
            self._active = name
            self._q = np.array(d.qpos[self._qadr])
            self._qd = np.zeros(len(self._names))
            self._ee_pos, self._ee_mat = self._ee(self._q)
            self._target_pos, self._target_mat = self._ee_pos.copy(), self._ee_mat.copy()
            self.laws[name].last_error = np.zeros(6)
            self._ft_queue.clear()
            self._next_t = self._ctx.sim_time
        elif self._active == name:
            self._active = None

    def on_reset(self, ctx: SimContext) -> None:
        """Every controller inactive and the internal model forgotten: a trial starts clean."""
        self._active = None
        for reg in self._registered.values():
            registry_for(ctx).restore(reg, INACTIVE, ctx.sim_time)
        self._target_pos, self._target_mat = np.zeros(3), np.eye(3)
        self._target_wrench = np.zeros(6)
        self._ft_queue.clear()
        self._q = self._qd = None
        self._ee_pos, self._ee_mat, self._ee_mat_measured = np.zeros(3), np.eye(3), np.eye(3)
        for law in self.laws.values():
            law.last_error = np.zeros(6)
        self._next_t = 0.0

    # -- the law ------------------------------------------------------------------------------------

    def _ee(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self._kin
        k.qpos[:] = self._ctx.data.qpos
        k.qpos[self._qadr] = q
        mujoco.mj_kinematics(self._ctx.model, k)
        return np.array(k.site_xpos[self._site]), np.array(k.site_xmat[self._site]).reshape(3, 3)

    def _model(self, q: np.ndarray, link_mass: float) -> tuple[np.ndarray, np.ndarray]:
        """(H, J) of the virtual model at q: link_mass on each moving link, 1 kg at the end effector."""
        m, k = self._ctx.model, self._kin
        k.qpos[:] = self._ctx.data.qpos
        k.qpos[self._qadr] = q
        mujoco.mj_kinematics(m, k)
        mujoco.mj_comPos(m, k)
        n = len(self._dofs)
        H = np.zeros((n, n))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for body in self._link_bodies:
            mujoco.mj_jac(m, k, jp, jr, k.xpos[body], body)
            Jp, Jr = jp[:, self._dofs], jr[:, self._dofs]
            H += link_mass * Jp.T @ Jp + _IP_MIN * Jr.T @ Jr
        mujoco.mj_jacSite(m, k, jp, jr, self._site)
        J = np.vstack([jp[:, self._dofs], jr[:, self._dofs]])
        H += J[:3].T @ J[:3] + J[3:].T @ J[3:]
        return H, J

    def _motion_error(self) -> np.ndarray:
        dp = self._target_pos - self._ee_pos
        dist = float(np.linalg.norm(dp))
        if dist > _MAX_DISTANCE:
            dp *= _MAX_DISTANCE / dist
        rot = _rotation_vector(self._target_mat @ self._ee_mat.T)
        angle = float(np.linalg.norm(rot))
        if angle > _MAX_ANGLE:
            rot *= _MAX_ANGLE / angle
        return np.concatenate([dp, rot])

    def _measured_wrench(self) -> np.ndarray:
        """The reading in world axes, delayed by ft_delay_cycles control cycles."""
        force, torque = self._ft.read()
        R = self._ee_mat_measured
        self._ft_queue.append(np.concatenate([R @ force, R @ torque]))
        while len(self._ft_queue) > self.ft_delay + 1:
            self._ft_queue.pop(0)
        return self._ft_queue[0]

    def _error(self, law: _Law, wrench: np.ndarray | None) -> np.ndarray:
        if law.kind == "motion":
            return self._motion_error()
        force_error = wrench + self._target_wrench
        if law.kind == "force":
            return force_error
        if law.compliance_ref == "end_effector":
            R = self._ee_mat
        else:
            R = self._base_frame(self._ctx.data)[1]
        Rb = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
        K = Rb @ np.diag(law.K) @ Rb.T
        return K @ self._motion_error() + force_error

    def pre_step(self, ctx: SimContext) -> None:
        if self._arm is not None and getattr(self._arm, "ensure_updated", None) is not None:
            self._arm.ensure_updated(ctx)
        else:
            self.update(ctx)

    def update(self, ctx: SimContext) -> None:
        if self._active is None or ctx.manual_control:
            return
        half = 0.5 * ctx.model.opt.timestep
        if ctx.sim_time < self._next_t - half:
            return
        self._next_t += 1.0 / self.rate_hz
        if self._next_t < ctx.sim_time + half:
            self._next_t = ctx.sim_time + 1.0 / self.rate_hz
        law = self.laws[self._active]
        # Synchronise the internal model with the measured joints (positions; velocities carry on).
        q_meas = np.array(ctx.data.qpos[self._qadr])
        self._q = q_meas.copy()
        self._ee_mat_measured = np.array(ctx.data.site_xmat[self._site]).reshape(3, 3)
        wrench = self._measured_wrench() if law.kind != "motion" else None
        for _ in range(law.iterations):
            u = law.error_scale * law.pd(self._error(law, wrench))
            H, J = self._model(self._q, law.link_mass)
            ddq = np.linalg.solve(H, J.T @ u)
            q_new = self._q + self._qd * _INTERNAL_PERIOD
            self._qd = (self._qd + ddq * _INTERNAL_PERIOD) * _DAMPING
            self._q = np.clip(q_new, self._range[:, 0], self._range[:, 1])
            self._ee_pos, self._ee_mat = self._ee(self._q)
        self._arm.set_targets(self._names, self._q.tolist())
