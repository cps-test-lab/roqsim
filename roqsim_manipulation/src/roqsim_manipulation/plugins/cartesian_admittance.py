"""Controller plugin: Cartesian end-effector control for an arm, with or without force feedback.

``arm_controller`` holds a joint vector. That is enough to replay a trajectory and not enough to do
anything in contact: a contact task is specified in the *task* frame ("press down at 10 N until the
part seats"), not in joint space, and the interesting laws all close a loop around a measured wrench.
This is the substrate's Cartesian layer, and its force-controlled half is the arm-family equivalent
of what a local planner is for a mobile base.

**The controller's name is its behaviour**, as it is under ros2_control: what a controller does is
decided by which one is loaded, never by a mode key on a single controller. One implementation backs
three real controller types, because they differ only in which terms of one law are live:

``cartesian_motion_controller``
    A Cartesian P controller toward ``target_frame``. Blind to contact by construction -- it commands
    the same motion whether the tool is in free space or crushing itself against an obstruction, which
    is what makes it the honest baseline for a force-control comparison.

``cartesian_force_controller``
    ``M xddot = (w_d - w_a) - D xdot``. The arm follows force: no stiffness, so no pose is tracked
    and no equilibrium is pulled back to.

``cartesian_compliance_controller``
    ``M xddot = (w_d - w_a) - D xdot - C (x - x_0)``, with ``x_0`` the commanded ``target_frame``.
    Force and motion superimposed, which is what a contact task needs: an axis given zero stiffness
    stays under pure force control while the rest track a frame. That superposition is the whole
    point -- it is how a task-space motion is layered on a running force loop on real hardware, so
    the thing driving the motion is an ordinary publisher rather than a second controller.

Both wrenches are what the TOOL applies -- ``target_wrench: [0, 0, -10, ...]`` means "press down with
10 N" -- and the measured one is converted into that convention using the reader's own ``measures``,
never assumed. A wrench has a direction as well as a frame, and the two conventions are negatives of
each other.

Everything after the twist is shared: clamp, resolve to joint velocities through a damped
least-squares inverse of the site Jacobian, integrate to joint position targets, and hand those to
``arm_controller``. The DLS solve is deliberate: near a singularity a plain pseudo-inverse produces
enormous joint velocities from a small Cartesian command, and in a contact task that reads as a
sudden force spike -- a physics artefact indistinguishable, in the metrics, from a real jam.

**Single-writer.** This plugin never touches ``data.ctrl``. It writes joint *targets* through the
``ArmHandle`` that ``arm_controller`` publishes, and ``arm_controller`` remains the only writer of
that arm's actuators.

Config -- a component of the entry that spawns the arm, whose ``ArmHandle`` it drives, since
ownership is where the entry sits rather than a config key::

    cartesian_admittance:
      controller_type: cartesian_compliance_controller   # which of the three above; see `law`
      controller_name: ""      # ROS name its topics sit under; defaults to controller_type
      initial_state: active    # active | inactive -- `inactive` is ros2_control's `spawner --inactive`
      site: tool_site          # site whose pose is controlled (prefixed with the arm's prefix)
      ft: ft                   # blackboard key suffix of the force_torque sensor (`ft:<key>`);
                               #   required by the force and compliance types, unused by motion
      rate_hz: 100.0           # control rate; the loop runs at this, not at the physics rate
      target_wrench: [0, 0, -10, 0, 0, 0]    # w_d, what the TOOL applies, so -10 on z presses DOWN
      mass: [1, 1, 1, 0.6, 0.6, 0.6]         # M, diagonal
      damping: [80, 80, 80, 160, 160, 160]   # D, diagonal
      stiffness: [0, 0, 0, 0, 0, 0]          # C, diagonal; a zero axis is pure force control
      axes: [1, 1, 1, 1, 1, 1]               # per-axis enable mask
      kp: [2, 2, 2, 2, 2, 2]                 # motion type only: proportional gain on the pose error
      max_linear_vel: 0.1      # m/s, clamp on the commanded twist MAGNITUDE
      max_angular_vel: 1.0     # rad/s
      ik_damping: 0.01         # damped-least-squares lambda

``law: admittance | position`` is the older spelling and still works, deriving a ``controller_type``:
``position`` is the motion controller, and ``admittance`` is the force controller, or the compliance
controller where a non-zero ``stiffness`` is configured. Prefer ``controller_type`` -- a controller
that changes its law on command is not something any real robot offers.

Endpoints, named as FZI's ``cartesian_controllers`` name them, so a node written against this runs
unchanged against that stack: ``<controller>/target_wrench`` (in, ``geometry_msgs/WrenchStamped``),
``<controller>/target_frame`` (in, ``geometry_msgs/PoseStamped``) and ``<controller>/current_pose``
(out). A commanded value overrides its configured default; until one arrives the config stands, so a
world that publishes nothing behaves exactly as it always did.

Also publishes a ``CartesianHandle`` on the blackboard under ``cartesian:<arm>`` for an in-process
task plugin, with the same reach as the endpoints.

**Frames.** The commanded twist is applied in the WORLD frame, and the measured wrench is used as
given. Configure ``force_torque``'s ``frame:`` to match how the task defines its working axis; a
wrench reported in the sensor frame and integrated as if it were world-frame produces a controller
that drifts sideways under load and looks like a friction problem.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.controllers import ACTIVE, INACTIVE, Controller, registry_for
from roqsim.plugin import Plugin

#: The three ros2_control-shaped identities this implementation backs, and which terms each makes
#: live. ``needs_ft`` is what decides whether a missing wrench sensor is an error.
_TYPES = {
    "cartesian_motion_controller": {"stiffness": True, "wrench": False, "needs_ft": False},
    "cartesian_force_controller": {"stiffness": False, "wrench": True, "needs_ft": True},
    "cartesian_compliance_controller": {"stiffness": True, "wrench": True, "needs_ft": True},
}

#: The class a real ros2_control deployment would report for each of the three.
_TYPE_CLASSES = {
    "cartesian_motion_controller": "CartesianMotionController",
    "cartesian_force_controller": "CartesianForceController",
    "cartesian_compliance_controller": "CartesianComplianceController",
}

#: Older config spelling, kept working. ``admittance`` resolves by whether a stiffness is configured.
_LAWS = ("admittance", "position")


def _type_from_law(law: str, stiffness) -> str:
    if law == "position":
        return "cartesian_motion_controller"
    if any(float(v) != 0.0 for v in stiffness):
        return "cartesian_compliance_controller"
    return "cartesian_force_controller"


def _limit(vec: np.ndarray, limit: float) -> np.ndarray:
    """Scale ``vec`` down to ``limit`` in MAGNITUDE, keeping its direction.

    Per-axis clipping would cap each component at the limit and so permit sqrt(3) times it in
    magnitude -- and worse, it turns the commanded direction, because clipping one component of a
    diagonal motion and not the others points the result somewhere nobody asked for. A speed limit
    is a limit on speed.
    """
    norm = float(np.linalg.norm(vec))
    return vec * (limit / norm) if norm > limit > 0.0 else vec


def _rotvec(mat_from: np.ndarray, mat_to: np.ndarray) -> np.ndarray:
    """Rotation from ``mat_from`` to ``mat_to`` as an axis-angle vector."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (mat_to @ mat_from.T).reshape(9))
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, quat, 1.0)
    return out


@dataclass
class CartesianHandle:
    """Blackboard handle under ``cartesian:<arm>``; all callables run on the physics thread."""

    arm: str
    set_goal: Callable[[np.ndarray, np.ndarray], None]
    read_pose: Callable[[], tuple[np.ndarray, np.ndarray]]
    set_law: Callable[[str], None]
    set_active: Callable[[bool], None]
    #: The controller this instance is, by its ros2_control-shaped name.
    controller_name: str = ""
    #: Command the target wrench, the in-process twin of the ``target_wrench`` endpoint.
    set_target_wrench: Callable[[np.ndarray], None] | None = None
    is_active: Callable[[], bool] | None = None


class CartesianAdmittancePlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.arm = self.entity
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

        # The identity is primary and the law follows from it: `controller_type` decides which terms
        # are live, and the legacy `law` key only picks a type when no type was named.
        self.controller_type = str(
            self.config.get("controller_type", "")
            or _type_from_law(self.law, self.config.get("stiffness", [0, 0, 0, 0, 0, 0]))
        )
        self.controller_name = str(self.config.get("controller_name", "") or self.controller_type)
        terms = _TYPES.get(self.controller_type, _TYPES["cartesian_compliance_controller"])
        self._uses_wrench = terms["wrench"]
        self._uses_stiffness = terms["stiffness"]
        self._needs_ft = terms["needs_ft"]
        # A controller the world declares inactive comes up holding nothing, the way
        # `spawner --inactive` leaves one. Default active, so a world that never switches is unchanged.
        self._active = str(self.config.get("initial_state", "active")) != "inactive"

        self._ctx: SimContext | None = None
        self._arm_handle = None
        self._ft = None
        self._site_id = -1
        self._dofs: np.ndarray = np.zeros(0, dtype=int)
        self._joint_names: list[str] = []
        self._twist = np.zeros(6)
        self._goal_pos: np.ndarray | None = None
        self._goal_mat: np.ndarray | None = None
        # x_0 for the stiffness term when no target_frame has been commanded: the pose the arm was
        # in when this controller last became responsible for it.
        self._rest_pos: np.ndarray | None = None
        self._rest_mat: np.ndarray | None = None
        self._next_t = 0.0
        self._q_target: np.ndarray | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if config.get("law", "admittance") not in _LAWS:
            errors.append(f"'law' must be one of {', '.join(_LAWS)}")
        if config.get("controller_type") and config["controller_type"] not in _TYPES:
            errors.append(f"'controller_type' must be one of {', '.join(sorted(_TYPES))}")
        if str(config.get("initial_state", "active")) not in ("active", "inactive"):
            errors.append("'initial_state' must be 'active' or 'inactive'")
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
        if self._needs_ft:
            self._ft = ctx.blackboard.get(f"ft:{self.ft_key}")
            if self._ft is None:
                raise RuntimeError(
                    f"cartesian_admittance: {self.controller_type} closes a loop around a measured "
                    f"wrench, but no force_torque sensor is registered at 'ft:{self.ft_key}'. Add a "
                    f"`force_torque` plugin (its `name` is the key) before this one, or use "
                    f"controller_type 'cartesian_motion_controller'."
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

        ns = entity.meta.get("namespace", "") if entity else ""

        # Listed and switched like any other controller. It claims the SAME command interfaces as
        # the trajectory controller, which is what makes handing the arm from one to the other a
        # switch rather than a race.
        registry_for(ctx).register(
            Controller(
                name=self.controller_name,
                type=f"cartesian_controllers/{_TYPE_CLASSES[self.controller_type]}",
                claims=tuple(f"{j}/position" for j in self._joint_names),
                state=ACTIVE if self._active else INACTIVE,
                namespace=ns,
                owner=self.arm,
                apply=self.set_active,
            )
        )

        # The arm pulls this each step. An older arm_controller has no such slot, so this stays
        # optional -- the plugin then runs from its own `pre_step` as it always did.
        if getattr(self._arm_handle, "set_command_source", None) is not None:
            self._arm_handle.set_command_source(self.update, self.label or "cartesian_admittance")

        ctx.blackboard.set(
            f"cartesian:{self.arm}",
            CartesianHandle(
                arm=self.arm,
                set_goal=self.set_goal,
                read_pose=self.read_pose,
                set_law=self.set_law,
                set_active=self.set_active,
                controller_name=self.controller_name,
                set_target_wrench=self.set_target_wrench,
                is_active=lambda: self._active,
            ),
        )

        # Named as FZI's cartesian_controllers name them: a node that drives this controller drives
        # the real one unchanged. The setpoints are topics rather than services because they are a
        # stream -- a reference a task republishes as it moves, not a command with an outcome.
        ctx.interface.add(
            Endpoint(
                name="target_frame",
                direction="in",
                owner=self.arm,
                namespace=ns,
                write=lambda payload: self.set_goal(payload[0], payload[1]),
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.PoseStamped",
                        "topic": self.topic_override("target_frame")
                        or f"{self.controller_name}/target_frame",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="target_wrench",
                direction="in",
                owner=self.arm,
                namespace=ns,
                write=lambda payload: self.set_target_wrench(
                    np.concatenate([payload[0], payload[1]])
                ),
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.WrenchStamped",
                        "topic": self.topic_override("target_wrench")
                        or f"{self.controller_name}/target_wrench",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="current_pose",
                direction="out",
                owner=self.arm,
                namespace=ns,
                read=self.read_pose_quat,
                rate_hz=self.config.get("pose_rate_hz", 50.0),
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.PoseStamped",
                        "topic": self.topic_override("current_pose")
                        or f"{self.controller_name}/current_pose",
                        "frame_id": "world",
                    }
                },
            )
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

    def read_pose_quat(self) -> tuple[list[float], list[float]]:
        """Endpoint ``read``: the controlled pose as transport-neutral ``(position, quaternion)``."""
        pos, mat = self.read_pose()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.reshape(9))
        return (pos.tolist(), quat.tolist())

    def set_target_wrench(self, wrench) -> None:
        """Command ``w_d``: what the TOOL is to apply, the convention ``target_wrench`` config uses."""
        self.w_d = np.array(wrench, dtype=float)

    def set_law(self, law: str) -> None:
        """Deprecated: a real controller does not change its law, you switch to another controller.

        Kept because worlds and task plugins call it. Prefer declaring ``controller_type`` and, where
        a run really must change behaviour part-way, switching controllers.
        """
        if law not in _LAWS:
            raise ValueError(f"cartesian_admittance: unknown law {law!r}")
        self.law = law
        self.controller_type = _type_from_law(law, self.C)
        terms = _TYPES[self.controller_type]
        self._uses_wrench, self._uses_stiffness = terms["wrench"], terms["stiffness"]
        self._twist = np.zeros(6)

    def set_active(self, active: bool) -> None:
        active = bool(active)
        if active and not self._active:
            # Activating starts from the arm's state NOW, which is what makes the hand-over a
            # well-defined instant: a controller resuming with a stale integrator, a stale joint
            # target and an equilibrium from wherever the episode began would command a step change
            # the moment it took over. Real hardware does the same on activation.
            self._anchor_here()
            self._twist = np.zeros(6)
            self._q_target = None
        elif not active:
            self._twist = np.zeros(6)
        self._active = active

    def _anchor_here(self) -> None:
        """Capture the current pose as the resting equilibrium for the stiffness term."""
        pos, mat = self.read_pose()
        self._rest_pos, self._rest_mat = pos.copy(), mat.copy()

    # -- lifecycle -------------------------------------------------------------------------------

    def on_reset(self, ctx: SimContext) -> None:
        self._twist = np.zeros(6)
        # A commanded frame belongs to the episode that commanded it: carrying one across a reset
        # would make a repetition start where the previous one left off.
        self._goal_pos = None
        self._goal_mat = None
        self._next_t = 0.0
        self._q_target = None
        self._anchor_here()

    def pre_step(self, ctx: SimContext) -> None:
        # Ask rather than act: the arm runs this through `ensure_updated` too, and whichever plugin
        # reaches it first in this step does the work. Declaring this one before or after the arm
        # therefore changes nothing -- which it used to, by a whole step.
        if self._arm_handle is not None and self._arm_handle.ensure_updated is not None:
            self._arm_handle.ensure_updated(ctx)
        else:
            self.update(ctx)

    def update(self, ctx: SimContext) -> None:
        """Compute this step's joint targets. Pulled by the arm; never called twice in a step."""
        if ctx.manual_control or not self._active:
            return
        # Fixed-rate control loop over a finer physics loop: a controller tuned at 100 Hz behaves
        # differently when run at the 1 kHz physics rate (the integrated twist grows ten times as
        # fast per unit time), so the rate is honoured rather than being whatever the world's
        # timestep happens to be.
        # Due within half a physics step of the scheduled time, and the next tick scheduled on the
        # fixed grid rather than one period after this one. The sim clock is a floating-point sum of
        # timesteps, so a tick due at exactly k periods can read a hair early; compared strictly and
        # re-anchored on the current time, each such hair became a whole missed physics step.
        half_step = 0.5 * ctx.model.opt.timestep
        if ctx.sim_time < self._next_t - half_step:
            return
        dt = 1.0 / self.rate_hz
        self._next_t += dt
        if self._next_t < ctx.sim_time + half_step:
            # Behind by more than a period (the controller was inactive): resume at the rate from
            # here rather than catching up on the ticks that were missed.
            self._next_t = ctx.sim_time + dt

        twist = self._wrench_twist(dt) if self._uses_wrench else self._position_twist()
        self._apply(ctx, self._clamp(twist), dt)

    # -- laws ------------------------------------------------------------------------------------

    def _wrench_twist(self, dt: float) -> np.ndarray:
        force, torque = self._ft.read()
        w_a = self._as_applied_by_tool(np.concatenate([force, torque]))
        forcing = self.w_d - w_a
        if self._uses_stiffness:
            forcing = forcing - self.C * self._deflection()
        # The mask is applied to the FORCING term, not to the resulting twist, and the stored twist
        # is masked with it. Masking only the output leaves a disabled axis integrating to the clamp
        # behind the mask, so enabling it later dumps a saturated velocity into the arm in one step.
        accel = (forcing * self.axes - self.D * self._twist) / self.M
        self._twist = self._clamp(self._twist + accel * dt) * self.axes
        return self._twist

    def _deflection(self) -> np.ndarray:
        """``x - x_0``: how far the tool has been pushed off its equilibrium, translation and rotation.

        The equilibrium is the commanded ``target_frame`` where one has been commanded, and otherwise
        the pose captured when this controller took the arm. Both halves are present: with the
        rotational half missing, a configured rotational stiffness did nothing and "orientation is
        held" quietly meant "orientation is unregulated", with the DLS solve free to accumulate drift
        across a long run.
        """
        pos, mat = self.read_pose()
        anchor_pos = self._goal_pos if self._goal_pos is not None else self._rest_pos
        anchor_mat = self._goal_mat if self._goal_mat is not None else self._rest_mat
        out = np.zeros(6)
        out[:3] = pos - (anchor_pos if anchor_pos is not None else pos)
        if anchor_mat is not None:
            out[3:] = _rotvec(anchor_mat, mat)
        return out

    def _as_applied_by_tool(self, wrench: np.ndarray) -> np.ndarray:
        """The measured wrench in ``target_wrench``'s convention: what the TOOL applies.

        The two conventions are negatives of each other, and mixing them does not read as a sign
        error. With the sensor reporting the reaction (its default, and what a real FT sensor does),
        `w_d - w_a` for a downward target grew as the contact resisted: pressing harder raised the
        reaction, which raised the commanded push. Positive feedback with no equilibrium anywhere --
        measured running to roughly 300 N against a target of 10.

        Taken from the reader rather than assumed, because `invert` is the world's to set: a sensor
        configured the other way is already in this convention and must not be flipped twice.
        """
        if getattr(self._ft, "measures", "environment_on_tool") == "environment_on_tool":
            return -wrench
        return wrench

    def _position_twist(self) -> np.ndarray:
        pos, mat = self.read_pose()
        if self._goal_pos is None:
            return np.zeros(6)
        twist = np.zeros(6)
        twist[:3] = self.kp[:3] * (self._goal_pos - pos)
        if self._goal_mat is not None:
            # Orientation error as a rotation vector, from where the tool is to where it should be.
            twist[3:] = self.kp[3:] * _rotvec(mat, self._goal_mat)
        # No integrator in this law, so masking the result is enough -- nothing accumulates behind it.
        return twist * self.axes

    # -- shared downstream -----------------------------------------------------------------------

    def _clamp(self, twist: np.ndarray) -> np.ndarray:
        out = np.array(twist, dtype=float)
        out[:3] = _limit(out[:3], self.v_lin)
        out[3:] = _limit(out[3:], self.v_ang)
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
