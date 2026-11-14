"""What a Franka Research 3's controller does with a command: the core of the simulated robot.

One :class:`FrankaCore` per arm, owned by the ``franka_fci`` plugin on the physics thread. Once per
millisecond of simulated time (a *tick*) it measures the arm, builds the robot state libfranka
receives, and takes the next command; on every physics step in between it computes the joint
torques. It knows no socket: the FCI server (:mod:`.server`) and an in-process law
(``franka_fci``'s :class:`~roqsim_manipulation.plugins.franka_fci.InProcessMotion`) hand it commands the same way.

**What the robot does, and where it comes from.**

- *Gravity.* A torque command is "without gravity" (``robot_state.h``, ``tau_J_d``); the core adds
  the gravity torque of the robot's model -- the nominal arm and the end effector and load it was
  told (:mod:`.robot_model`). A wrong configured load therefore sags the arm, as on the robot.
- *Motion-generator checks.* Each command is differentiated and checked against
  :mod:`.limits` (``rate_limiting.h``, ``joint_limits.yaml``): a position outside the range, a
  velocity above the position-dependent limit, an acceleration above the maximum (libfranka calls
  it a velocity discontinuity), a jerk above it (an acceleration discontinuity), a torque rate above
  1000 N m/s. A violation ends the motion with that flag, as on the robot. The check is at the
  published limit itself: libfranka's rate limiter stays ``kLimitEps`` inside it on purpose, and
  Franka's own tolerances are not published. The torque rate allows 0.5 N m/s more, the rounding of
  a float32 ``tau_J_d`` that a rate-limited client steps from.
- *Internal controllers.* Joint impedance ``tau = K (q_d - q) - D dq`` with ``setJointImpedance``'s
  K, and Cartesian impedance at the K frame with ``setCartesianImpedance``'s K and a weak nullspace
  hold; both critically damped. Franka publishes neither law: these are stated assumptions.
- *The external-torque estimate* (``tau_ext_hat_filtered``, ``O_F_ext_hat_K``, ``K_F_ext_hat_K``):
  a generalized-momentum observer on the link-side joint torques ``tau_J`` and the robot's model,
  with gain ``observer_gain`` (a first-order lag of ``1 / observer_gain``). Franka documents what the
  signal is (the error between ``tau_J`` and the model's expected torques, low-pass filtered,
  excluding the configured end effector and load) but not how it is filtered.
- *Sign convention.* ``tau_ext_hat_filtered`` and the wrenches are what the **robot applies to the
  environment** ("forces applied by the robot to the environment are positive", ``robot_state.h``):
  pressing a tool down on a table reads a negative ``O_F_ext_hat_K[2]``.
- *Reflexes.* ``joint_reflex`` when an estimated external torque exceeds its upper threshold,
  ``cartesian_reflex`` when a component of ``O_F_ext_hat_K`` does; the "acceleration" thresholds
  apply while the desired motion accelerates (``|ddq_d| > accel_switch``), the "nominal" ones
  otherwise -- Franka does not publish the switch, so it is a stated assumption. The motion stops,
  the arm holds where it is, the flag is latched until ``automaticErrorRecovery``.

Not served, and refused or absent: guiding mode, the robot's own path planner (``PTPMotion``),
power and self-collision limits, packet loss (no late answer can happen in lock-step).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import limits
from .protocol import (
    ERROR_INDEX,
    ERRORS,
    ControllerMode,
    MotionGeneratorMode,
    MoveControllerMode,
    MoveMotionGeneratorMode,
    MoveStatus,
    RobotMode,
)
from .robot_model import RobotModel, combine_loads

#: libfranka ``examples/examples_common.cpp`` ``setDefaultBehavior``: the impedances its examples
#: set. The robot's own defaults are unpublished; these are the documented example values.
EXAMPLE_JOINT_IMPEDANCE = (3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0)
EXAMPLE_CARTESIAN_IMPEDANCE = (3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0)

_GEN_MAP = {
    MoveMotionGeneratorMode.JOINT_POSITION: MotionGeneratorMode.JOINT_POSITION,
    MoveMotionGeneratorMode.JOINT_VELOCITY: MotionGeneratorMode.JOINT_VELOCITY,
    MoveMotionGeneratorMode.CARTESIAN_POSITION: MotionGeneratorMode.CARTESIAN_POSITION,
    MoveMotionGeneratorMode.CARTESIAN_VELOCITY: MotionGeneratorMode.CARTESIAN_VELOCITY,
    MoveMotionGeneratorMode.NONE: MotionGeneratorMode.NONE,
}
_CTRL_MAP = {
    MoveControllerMode.JOINT_IMPEDANCE: ControllerMode.JOINT_IMPEDANCE,
    MoveControllerMode.CARTESIAN_IMPEDANCE: ControllerMode.CARTESIAN_IMPEDANCE,
    MoveControllerMode.EXTERNAL_CONTROLLER: ControllerMode.EXTERNAL_CONTROLLER,
}


def col_major(T: np.ndarray) -> list[float]:
    """A 4 x 4 matrix as libfranka sends it: column-major, 16 values."""
    return [float(x) for x in np.asarray(T, float).T.reshape(16)]


def from_col_major(values) -> np.ndarray:
    return np.asarray(values, float).reshape(4, 4).T


def _rotvec(R: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(R, float).reshape(9))
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, quat, 1.0)
    return out


def _rotmat(rv) -> np.ndarray:
    rv = np.asarray(rv, float)
    angle = float(np.linalg.norm(rv))
    quat = np.array([1.0, 0, 0, 0])
    if angle > 1e-12:
        mujoco.mju_axisAngle2Quat(quat, rv / angle, angle)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


@dataclass
class CollisionBehavior:
    """``setCollisionBehavior``'s eight arrays (N m for the joints, N and N m for the wrench)."""

    lower_torque_acceleration: np.ndarray
    upper_torque_acceleration: np.ndarray
    lower_torque_nominal: np.ndarray
    upper_torque_nominal: np.ndarray
    lower_force_acceleration: np.ndarray
    upper_force_acceleration: np.ndarray
    lower_force_nominal: np.ndarray
    upper_force_nominal: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict) -> CollisionBehavior:
        """The world's ``collision_behavior``: the four-array form (``lower_torque``, ``upper_torque``,
        ``lower_force``, ``upper_force``, applied to both phases as libfranka's 4-array overload does)
        or the eight-array form under the request's field names."""
        if "lower_torque" in cfg:
            lt, ut = np.asarray(cfg["lower_torque"], float), np.asarray(cfg["upper_torque"], float)
            lf, uf = np.asarray(cfg["lower_force"], float), np.asarray(cfg["upper_force"], float)
            return cls(lt, ut, lt, ut, lf, uf, lf, uf)
        return cls(*(np.asarray(cfg[k], float) for k in cls.__dataclass_fields__))

    @classmethod
    def from_request(cls, values: tuple) -> CollisionBehavior:
        a = np.asarray(values, float)
        cuts = [0, 7, 14, 21, 28, 34, 40, 46, 52]
        return cls(*(a[cuts[i] : cuts[i + 1]] for i in range(8)))

    def errors(self) -> list[str]:
        out = []
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            n = 7 if "torque" in name else 6
            if v.shape != (n,):
                out.append(f"collision_behavior.{name}: {n} values, got {v.shape}")
            elif not np.all(np.isfinite(v)) or np.any(v < 0):
                out.append(f"collision_behavior.{name}: non-negative finite values")
        return out


@dataclass
class Motion:
    """A running ``Move``: the controller, the generator, and what the generator last commanded."""

    controller: ControllerMode
    generator: MotionGeneratorMode
    command_id: int | None = None  # the TCP Move's id, answered when the motion ends
    q_c: np.ndarray | None = None
    dq_c: np.ndarray | None = None
    ddq_c: np.ndarray | None = None
    T_c: np.ndarray | None = None  # O_T_EE_c
    v_c: np.ndarray | None = None  # O_dP_EE_c
    a_c: np.ndarray | None = None
    elbow_c: np.ndarray | None = None
    tau_c: np.ndarray | None = None
    first: bool = True


@dataclass
class Ending:
    """How a motion ended, for whoever waits on it (the TCP Move reply, an in-process caller)."""

    command_id: int | None
    status: MoveStatus
    errors: list[str] = field(default_factory=list)


class FrankaCore:
    """One simulated FR3 controller. ``tick`` once per millisecond, ``torques`` every physics step."""

    def __init__(
        self,
        robot_model: RobotModel,
        *,
        collision_behavior: CollisionBehavior,
        joint_impedance=EXAMPLE_JOINT_IMPEDANCE,
        cartesian_impedance=EXAMPLE_CARTESIAN_IMPEDANCE,
        end_effector: dict | None = None,
        observer_gain: float = 200.0,
        accel_switch: float = 1.0,
        singular_threshold: float = 0.01,
    ):
        self.model = robot_model
        self.collision = collision_behavior
        self.K_theta = np.asarray(joint_impedance, float)
        self.K_x = np.asarray(cartesian_impedance, float)
        ee = end_effector or {}
        # Desk's end effector: its mass, centre of mass and inertia in the flange frame, and F_T_NE.
        self.m_ee = float(ee.get("mass", 0.0))
        self.F_x_Cee = np.asarray(ee.get("com", [0, 0, 0]), float)
        self.I_ee = np.asarray(ee.get("inertia", np.zeros(9)), float).reshape(3, 3)
        self.F_T_NE = np.asarray(ee.get("F_T_NE", np.eye(4)), float).reshape(4, 4)
        self.m_load, self.F_x_Cload, self.I_load = 0.0, np.zeros(3), np.zeros((3, 3))
        self.NE_T_EE = np.eye(4)
        self.EE_T_K = np.eye(4)
        self.observer_gain = float(observer_gain)
        self.accel_switch = float(accel_switch)
        self.singular_threshold = float(singular_threshold)
        self._apply_load()
        self.reset(np.zeros(7))

    # -- configuration (the setters libfranka sends) ----------------------------------------------

    def _apply_load(self) -> None:
        # The load is given about its own centre of mass in the flange frame, as libfranka's
        # setLoad documents ("inertia matrix in [kg*m^2], column-major") for F_x_Cload.
        m, c, inertia = combine_loads(
            [(self.m_ee, self.F_x_Cee, self.I_ee), (self.m_load, self.F_x_Cload, self.I_load)]
        )
        self.m_total, self.F_x_Ctotal, self.I_total = m, c, inertia
        self.model.set_load(m, c, inertia)

    def set_load(self, mass: float, com, inertia) -> None:
        self.m_load = float(mass)
        self.F_x_Cload = np.asarray(com, float)
        self.I_load = np.asarray(inertia, float).reshape(
            3, 3
        )  # symmetric: row- or column-major alike
        self._apply_load()

    def set_ee(self, NE_T_EE) -> None:
        self.NE_T_EE = np.asarray(NE_T_EE, float).reshape(4, 4)

    def set_k(self, EE_T_K) -> None:
        self.EE_T_K = np.asarray(EE_T_K, float).reshape(4, 4)

    def set_joint_impedance(self, K) -> bool:
        K = np.asarray(K, float)
        lo, hi = limits.JOINT_IMPEDANCE_RANGE
        if K.shape != (7,) or np.any(K < lo) or np.any(K > hi):
            return False
        self.K_theta = K
        return True

    def set_cartesian_impedance(self, K) -> bool:
        K = np.asarray(K, float)
        tl, th = limits.CARTESIAN_TRANSLATIONAL_STIFFNESS_RANGE
        rl, rh = limits.CARTESIAN_ROTATIONAL_STIFFNESS_RANGE
        if (
            K.shape != (6,)
            or np.any(K[:3] < tl)
            or np.any(K[:3] > th)
            or np.any(K[3:] < rl)
            or np.any(K[3:] > rh)
        ):
            return False
        self.K_x = K
        return True

    def set_collision_behavior(self, behavior: CollisionBehavior) -> bool:
        if behavior.errors():
            return False
        self.collision = behavior
        return True

    # -- frames -------------------------------------------------------------------------------------

    @property
    def F_T_EE(self) -> np.ndarray:
        return self.F_T_NE @ self.NE_T_EE

    @property
    def F_T_K(self) -> np.ndarray:
        return self.F_T_EE @ self.EE_T_K

    # -- lifecycle ----------------------------------------------------------------------------------

    def reset(self, q) -> None:
        q = np.asarray(q, float)
        self.q = q.copy()
        self.dq = np.zeros(7)
        self.tau_J = np.zeros(7)
        self.tau_J_prev = np.zeros(7)
        self.dtau_J = np.zeros(7)
        self.robot_mode = RobotMode.IDLE
        self.motion: Motion | None = None
        self.hold_q = q.copy()
        self.q_d = q.copy()
        self.dq_d = np.zeros(7)
        self.ddq_d = np.zeros(7)
        self.tau_J_d = np.zeros(7)
        self.errors = np.zeros(len(ERRORS), bool)
        self.reflex_reason = np.zeros(len(ERRORS), bool)
        self.tau_ext = np.zeros(7)
        self.F_ext_O = np.zeros(6)
        self.F_ext_K = np.zeros(6)
        self.joint_contact = np.zeros(7)
        self.cartesian_contact = np.zeros(6)
        self.joint_collision = np.zeros(7)
        self.cartesian_collision = np.zeros(6)
        self._obs_p0: np.ndarray | None = None
        self._obs_int = np.zeros(7)
        self._obs_r = np.zeros(7)
        self._M_prev: np.ndarray | None = None
        self.endings: list[Ending] = []
        self.model.update(q, np.zeros(7))
        self._gravity = self.model.gravity()
        self.T_d = self._pose(self.F_T_EE)
        self.elbow_d = np.array([q[2], np.sign(q[3]) or -1.0])
        self.control_command_success_rate = 1.0

    def _pose(self, F_T_X: np.ndarray) -> np.ndarray:
        return self.model.flange_pose() @ F_T_X

    # -- one millisecond ----------------------------------------------------------------------------

    def measure(self, q, dq, tau_J) -> None:
        """The arm at this tick: joint positions, velocities and link-side torques (``tau_J``)."""
        self.q = np.asarray(q, float).copy()
        self.dq = np.asarray(dq, float).copy()
        self.tau_J_prev = self.tau_J
        self.tau_J = np.asarray(tau_J, float).copy()
        self.dtau_J = (self.tau_J - self.tau_J_prev) / limits.DELTA_T
        self.model.update(self.q, self.dq)
        self._gravity = self.model.gravity()
        self._estimate()
        self._contacts_and_reflexes()

    def _estimate(self) -> None:
        """The generalized-momentum observer: r -> the external joint torque, lag 1 / observer_gain."""
        dt = limits.DELTA_T
        M = self.model.mass()
        c = self.model.coriolis()
        g = self._gravity
        Mdot = np.zeros_like(M) if self._M_prev is None else (M - self._M_prev) / dt
        self._M_prev = M
        p = M @ self.dq
        if self._obs_p0 is None:
            self._obs_p0 = p.copy()
            self._obs_int[:] = 0.0
            self._obs_r[:] = 0.0
        self._obs_int += (self.tau_J + Mdot @ self.dq - c - g + self._obs_r) * dt
        self._obs_r = self.observer_gain * (p - self._obs_p0 - self._obs_int)
        # The observer's r is the environment's torque on the robot; Franka reports the robot's on
        # the environment.
        self.tau_ext = -self._obs_r
        J = self.model.zero_jacobian(self.F_T_K)
        smallest = float(np.linalg.svd(J, compute_uv=False)[-1])
        if smallest < self.singular_threshold:
            self.F_ext_O = np.zeros(6)
        else:
            self.F_ext_O = np.linalg.pinv(J.T) @ self.tau_ext
        R_K = self._pose(self.F_T_K)[:3, :3]
        self.F_ext_K = np.concatenate([R_K.T @ self.F_ext_O[:3], R_K.T @ self.F_ext_O[3:]])

    def _thresholds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cb = self.collision
        if float(np.linalg.norm(self.ddq_d)) > self.accel_switch:
            return (
                cb.lower_torque_acceleration,
                cb.upper_torque_acceleration,
                cb.lower_force_acceleration,
                cb.upper_force_acceleration,
            )
        return (
            cb.lower_torque_nominal,
            cb.upper_torque_nominal,
            cb.lower_force_nominal,
            cb.upper_force_nominal,
        )

    def _contacts_and_reflexes(self) -> None:
        lt, ut, lf, uf = self._thresholds()
        self.joint_contact = (np.abs(self.tau_ext) > lt).astype(float)
        self.cartesian_contact = (np.abs(self.F_ext_O) > lf).astype(float)
        joint_hit = np.abs(self.tau_ext) > ut
        cart_hit = np.abs(self.F_ext_O) > uf
        self.joint_collision = np.maximum(self.joint_collision, joint_hit.astype(float))
        self.cartesian_collision = np.maximum(self.cartesian_collision, cart_hit.astype(float))
        if self.robot_mode != RobotMode.MOVE:
            return
        reasons = []
        if np.any(joint_hit):
            reasons.append("joint_reflex")
        if np.any(cart_hit):
            reasons.append("cartesian_reflex")
        if np.any(self.q < limits.Q_LOWER) or np.any(self.q > limits.Q_UPPER):
            reasons.append("joint_position_limits_violation")
        if np.any(np.abs(self.dq) > limits.DQ_MAX):
            reasons.append("joint_velocity_violation")
        if reasons:
            self.abort(reasons, MoveStatus.REFLEX_ABORTED)

    # -- motions ------------------------------------------------------------------------------------

    def start_motion(
        self,
        controller: MoveControllerMode,
        generator: MoveMotionGeneratorMode,
        command_id: int | None = None,
    ) -> MoveStatus | None:
        """A ``Move``: ``None`` when it started (the reply comes when it ends), else the refusal."""
        if self.robot_mode == RobotMode.REFLEX or self.errors.any():
            return MoveStatus.COMMAND_REJECTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS
        if self.motion is not None:
            return MoveStatus.COMMAND_NOT_POSSIBLE_REJECTED
        if controller not in _CTRL_MAP or generator not in _GEN_MAP:
            return MoveStatus.INVALID_ARGUMENT_REJECTED
        if (
            generator == MoveMotionGeneratorMode.NONE
            and controller != MoveControllerMode.EXTERNAL_CONTROLLER
        ):
            return MoveStatus.INVALID_ARGUMENT_REJECTED
        self.motion = Motion(_CTRL_MAP[controller], _GEN_MAP[generator], command_id)
        self.robot_mode = RobotMode.MOVE
        self.q_d, self.dq_d, self.ddq_d = self.q.copy(), np.zeros(7), np.zeros(7)
        self.T_d = self._pose(self.F_T_EE)
        self.tau_J_d = np.zeros(7)
        self.reflex_reason[:] = False
        self._obs_p0 = None  # the observer restarts from the arm at rest in hand
        return None

    def _finish(self, status: MoveStatus, errors=()) -> None:
        motion = self.motion
        self.motion = None
        self.hold_q = self.q.copy()
        self.q_d, self.dq_d, self.ddq_d = self.q.copy(), np.zeros(7), np.zeros(7)
        self.tau_J_d = np.zeros(7)
        self.endings.append(Ending(motion.command_id if motion else None, status, list(errors)))

    def stop(self) -> None:
        """``StopMove``: the motion ends where it is, preempted."""
        if self.motion is not None:
            self._finish(MoveStatus.PREEMPTED)
            self.robot_mode = RobotMode.IDLE

    def abort(self, reasons: list[str], status: MoveStatus = MoveStatus.REFLEX_ABORTED) -> None:
        for r in reasons:
            self.errors[ERROR_INDEX[r]] = True
            self.reflex_reason[ERROR_INDEX[r]] = True
        if self.motion is not None:
            self._finish(status, reasons)
        self.robot_mode = RobotMode.REFLEX

    def automatic_error_recovery(self) -> bool:
        if self.motion is not None:
            return False
        self.errors[:] = False
        self.joint_collision[:] = 0.0
        self.cartesian_collision[:] = 0.0
        self.robot_mode = RobotMode.IDLE
        self.hold_q = self.q.copy()
        return True

    def apply(self, cmd: dict) -> None:
        """One ``RobotCommand`` (protocol field names) for the running motion."""
        motion = self.motion
        if motion is None:
            return
        dt = limits.DELTA_T
        gen, ctrl = motion.generator, motion.controller
        finished = bool(cmd.get("motion_generation_finished")) or bool(
            cmd.get("torque_command_finished")
        )
        problems: list[str] = []
        if gen in (MotionGeneratorMode.JOINT_POSITION, MotionGeneratorMode.JOINT_VELOCITY):
            if gen == MotionGeneratorMode.JOINT_POSITION:
                q_c = np.asarray(cmd["q_c"], float)
                prev_q = self.q_d if motion.first else motion.q_c
                dq_c = (q_c - prev_q) / dt
            else:
                dq_c = np.asarray(cmd["dq_c"], float)
                q_c = (self.q_d if motion.first else motion.q_c) + dq_c * dt
            prev_dq = np.zeros(7) if motion.first else motion.dq_c
            ddq_c = (dq_c - prev_dq) / dt
            prev_ddq = np.zeros(7) if motion.first else motion.ddq_c
            jerk = (ddq_c - prev_ddq) / dt
            if np.any(q_c < limits.Q_LOWER) or np.any(q_c > limits.Q_UPPER):
                problems.append("joint_motion_generator_position_limits_violation")
            tol = limits.JOINT_VELOCITY_LIMITS_TOLERANCE
            if np.any(dq_c > limits.upper_joint_velocity_limits(q_c) + tol) or np.any(
                dq_c < limits.lower_joint_velocity_limits(q_c) - tol
            ):
                problems.append("joint_motion_generator_velocity_limits_violation")
            if np.any(np.abs(ddq_c) > limits.MAX_JOINT_ACCELERATION + limits.LIMIT_EPS):
                problems.append("joint_motion_generator_velocity_discontinuity")
            if np.any(np.abs(jerk) > limits.MAX_JOINT_JERK + limits.LIMIT_EPS):
                problems.append("joint_motion_generator_acceleration_discontinuity")
            motion.q_c, motion.dq_c, motion.ddq_c = q_c, dq_c, ddq_c
            self.q_d, self.dq_d, self.ddq_d = q_c, dq_c, ddq_c
            self.model.update(q_c, np.zeros(7))
            self.T_d = self.model.flange_pose() @ self.F_T_EE
            self.model.update(self.q, self.dq)
        elif gen in (
            MotionGeneratorMode.CARTESIAN_POSITION,
            MotionGeneratorMode.CARTESIAN_VELOCITY,
        ):
            prev_T = self.T_d if motion.first else motion.T_c
            if gen == MotionGeneratorMode.CARTESIAN_POSITION:
                T_c = from_col_major(cmd["O_T_EE_c"])
                if motion.first and np.linalg.norm(T_c[:3, 3] - prev_T[:3, 3]) > 1e-3:
                    problems.append("cartesian_position_motion_generator_start_pose_invalid")
                v_c = np.concatenate(
                    [
                        (T_c[:3, 3] - prev_T[:3, 3]) / dt,
                        _rotvec(T_c[:3, :3] @ prev_T[:3, :3].T) / dt,
                    ]
                )
            else:
                v_c = np.asarray(cmd["O_dP_EE_c"], float)
                T_c = np.eye(4)
                T_c[:3, 3] = prev_T[:3, 3] + v_c[:3] * dt
                T_c[:3, :3] = _rotmat(v_c[3:] * dt) @ prev_T[:3, :3]
            prev_v = np.zeros(6) if motion.first else motion.v_c
            a_c = (v_c - prev_v) / dt
            prev_a = np.zeros(6) if motion.first else motion.a_c
            jerk = (a_c - prev_a) / dt
            if (
                np.linalg.norm(v_c[:3]) > limits.MAX_TRANSLATIONAL_VELOCITY
                or np.linalg.norm(v_c[3:]) > limits.MAX_ROTATIONAL_VELOCITY
            ):
                problems.append("cartesian_motion_generator_velocity_limits_violation")
            if (
                np.linalg.norm(a_c[:3]) > limits.MAX_TRANSLATIONAL_ACCELERATION
                or np.linalg.norm(a_c[3:]) > limits.MAX_ROTATIONAL_ACCELERATION
            ):
                problems.append("cartesian_motion_generator_velocity_discontinuity")
            if (
                np.linalg.norm(jerk[:3]) > limits.MAX_TRANSLATIONAL_JERK
                or np.linalg.norm(jerk[3:]) > limits.MAX_ROTATIONAL_JERK
            ):
                problems.append("cartesian_motion_generator_acceleration_discontinuity")
            motion.T_c, motion.v_c, motion.a_c = T_c, v_c, a_c
            if cmd.get("valid_elbow"):
                motion.elbow_c = np.asarray(cmd["elbow_c"], float)
            self.T_d = T_c
            if ctrl == ControllerMode.JOINT_IMPEDANCE:
                self.q_d = self._ik(T_c, self.q_d)
                self.dq_d = np.zeros(7)
        if ctrl == ControllerMode.EXTERNAL_CONTROLLER:
            tau = np.asarray(cmd["tau_J_d"], float)
            # Against what the client was sent: tau_J_d travels as float32, and a rate-limited
            # client steps from that value; the slack is that rounding, on both values, over dt.
            prev = self.tau_J_d.astype(np.float32).astype(float) if not motion.first else tau
            slack = 2 * np.finfo(np.float32).eps * np.maximum(np.abs(tau), np.abs(prev)) / dt
            if np.any(np.abs(tau - prev) / dt > limits.MAX_TORQUE_RATE + limits.LIMIT_EPS + slack):
                problems.append("controller_torque_discontinuity")
            motion.tau_c = tau
            self.tau_J_d = tau
        motion.first = False
        if problems:
            self.abort(problems, MoveStatus.REFLEX_ABORTED)
            return
        if finished:
            self._finish(MoveStatus.SUCCESS)
            self.robot_mode = RobotMode.IDLE

    def _ik(self, T: np.ndarray, q0: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Joint positions putting EE at T, by damped least squares from q0 (Cartesian under joint
        impedance: the robot's own inverse kinematics is not published)."""
        q = q0.copy()
        for _ in range(iterations):
            self.model.update(q, np.zeros(7))
            X = self.model.flange_pose() @ self.F_T_EE
            err = np.concatenate([T[:3, 3] - X[:3, 3], _rotvec(T[:3, :3] @ X[:3, :3].T)])
            if np.linalg.norm(err) < 1e-9:
                break
            J = self.model.zero_jacobian(self.F_T_EE)
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
        self.model.update(self.q, self.dq)
        return q

    # -- every physics step -------------------------------------------------------------------------

    def torques(self, q, dq) -> np.ndarray:
        """The motor torques for this physics step: the controller's command plus the model's gravity."""
        q, dq = np.asarray(q, float), np.asarray(dq, float)
        motion = self.motion
        if motion is not None and motion.controller == ControllerMode.EXTERNAL_CONTROLLER:
            tau = self.tau_J_d
        elif motion is not None and motion.controller == ControllerMode.CARTESIAN_IMPEDANCE:
            tau = self._cartesian_impedance(q, dq)
            self.tau_J_d = tau
        elif motion is not None:
            tau = self._joint_impedance(q, dq, self.q_d, self.dq_d)
            self.tau_J_d = tau
        else:
            tau = self._joint_impedance(q, dq, self.hold_q, np.zeros(7))
        return tau + self._gravity

    def _joint_impedance(self, q, dq, q_d, dq_d) -> np.ndarray:
        M = self.model.mass()
        D = 2.0 * np.sqrt(self.K_theta * np.diag(M))
        return self.K_theta * (q_d - q) + D * (dq_d - dq)

    def _cartesian_impedance(self, q, dq) -> np.ndarray:
        J = self.model.zero_jacobian(self.F_T_K)
        X = self._pose(self.F_T_K)
        T_dK = self.T_d @ self.EE_T_K
        err = np.concatenate([T_dK[:3, 3] - X[:3, 3], _rotvec(T_dK[:3, :3] @ X[:3, :3].T)])
        M = self.model.mass()
        Lambda = np.linalg.inv(J @ np.linalg.solve(M, J.T) + 1e-6 * np.eye(6))
        D = 2.0 * np.sqrt(self.K_x * np.maximum(np.diag(Lambda), 1e-6))
        F = self.K_x * err - D * (J @ dq)
        Jpinv_T = np.linalg.pinv(J.T)
        null = (np.eye(7) - J.T @ Jpinv_T) @ (10.0 * (self.hold_q - q) - 2.0 * np.sqrt(10.0) * dq)
        return J.T @ F + null

    # -- the state libfranka receives ---------------------------------------------------------------

    def state(self, message_id: int) -> dict:
        F_T_EE = self.F_T_EE
        O_T_EE = self._pose(F_T_EE)
        J = self.model.zero_jacobian(F_T_EE)
        v = J @ self.dq
        motion = self.motion
        return {
            "message_id": message_id,
            "O_T_EE": col_major(O_T_EE),
            "O_T_EE_d": col_major(self.T_d),
            "F_T_EE": col_major(F_T_EE),
            "EE_T_K": col_major(self.EE_T_K),
            "F_T_NE": col_major(self.F_T_NE),
            "NE_T_EE": col_major(self.NE_T_EE),
            "m_ee": self.m_ee,
            "I_ee": list(self.I_ee.T.reshape(9)),
            "F_x_Cee": list(self.F_x_Cee),
            "m_load": self.m_load,
            "I_load": list(self.I_load.T.reshape(9)),
            "F_x_Cload": list(self.F_x_Cload),
            "elbow": [self.q[2], -1.0 if self.q[3] < 0 else 1.0],
            "elbow_d": [self.q_d[2], -1.0 if self.q_d[3] < 0 else 1.0],
            "tau_J": list(self.tau_J),
            "tau_J_d": list(self.tau_J_d),
            "dtau_J": list(self.dtau_J),
            "q": list(self.q),
            "q_d": list(self.q_d),
            "dq": list(self.dq),
            "dq_d": list(self.dq_d),
            "ddq_d": list(self.ddq_d),
            "joint_contact": list(self.joint_contact),
            "cartesian_contact": list(self.cartesian_contact),
            "joint_collision": list(self.joint_collision),
            "cartesian_collision": list(self.cartesian_collision),
            "tau_ext_hat_filtered": list(self.tau_ext),
            "O_F_ext_hat_K": list(self.F_ext_O),
            "K_F_ext_hat_K": list(self.F_ext_K),
            "O_dP_EE_d": list(v),
            # robot_state.h documents O_ddP_O as constant gravity, whatever the base's mounting.
            "O_ddP_O": [0.0, 0.0, -9.81],
            "elbow_c": list(motion.elbow_c)
            if motion is not None and motion.elbow_c is not None
            else [0.0, 0.0],
            "delbow_c": [0.0, 0.0],
            "ddelbow_c": [0.0, 0.0],
            "O_T_EE_c": col_major(self.T_d),
            "O_dP_EE_c": list(motion.v_c)
            if motion is not None and motion.v_c is not None
            else [0.0] * 6,
            "O_ddP_EE_c": list(motion.a_c)
            if motion is not None and motion.a_c is not None
            else [0.0] * 6,
            "theta": list(self.q),
            "dtheta": list(self.dq),
            "accelerometer_top": [0.0] * 18,
            "accelerometer_bottom": [0.0] * 18,
            "motion_generator_mode": int(motion.generator)
            if motion is not None
            else int(MotionGeneratorMode.IDLE),
            "controller_mode": int(motion.controller)
            if motion is not None
            else int(ControllerMode.OTHER),
            "errors": list(self.errors),
            "reflex_reason": list(self.reflex_reason),
            "robot_mode": int(self.robot_mode),
            "control_command_success_rate": self.control_command_success_rate,
        }
