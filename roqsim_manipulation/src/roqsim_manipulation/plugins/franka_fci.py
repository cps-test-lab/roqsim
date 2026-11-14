"""Plugin: a Franka Research 3 as its controller and its FCI serve it, in lock-step with the physics.

A component of the ``spawn_arm`` entry of an ``fr3``. It is the arm's controller: every physics step
it writes the seven joint torques (the command plus the robot's own gravity model), and once per
millisecond of simulated time it measures the arm, estimates the external torques and wrench,
checks the reflexes and takes the next command (:mod:`roqsim_manipulation.franka.core`).

Commands come from two places, and the robot side is the same for both:

- **The network** (``bind:`` set; not in a preview, which only looks at the world): the robot's side of the FCI (:mod:`roqsim_manipulation.franka.server`),
  so an unmodified libfranka -- a C++ program, pylibfranka, franka_ros2's hardware interface --
  connects to ``<bind>`` as to a robot and runs in lock-step: one control callback per millisecond
  of simulated time. Two arms in one world are two components with two addresses (libfranka always
  connects to port 1337), stepped by one physics, so their clients see one clock.
- **In process**: the handle published at ``franka_fci:<arm>`` (:class:`FrankaHandle`) runs a law
  ``law(state, period) -> command`` with libfranka's field names on the physics thread, with no
  socket -- how a roqsim plugin (an executor, a task) drives the arm.

**The timestep.** One tick is 1 ms, so the world's timestep must be 1 ms / N for an integer N; the
command is held over the N steps of a tick. The internal controllers run every step, on that
step's joint positions and velocities, with the mass matrix, Jacobian, flange pose and gravity of
the tick, as the robot's model is updated once per tick. Any other timestep is refused at
configure.

**Settings only the world can give**, with no invented default:

- ``collision_behavior`` (required): the thresholds ``setCollisionBehavior`` takes -- the robot's own
  start-up values are not published, and whether the arm stops on a contact turns on them.
  Four-array form ``{lower_torque, upper_torque, lower_force, upper_force}`` (both phases) or the
  request's eight arrays.
- ``end_effector``: what Desk is told is mounted (``mass``, ``com`` and ``inertia`` about the centre of
  mass, in the flange frame, and ``F_T_NE``, a 4 x 4 row-major list); none by default, as on an arm
  without a hand. ``load``: an initial ``setLoad``.

Every key, its type, unit and default: ``roqsim plugins describe franka_fci``.

**What is measured.** ``tau_J`` is the link-side joint torque -- the actuator torque less the rotor
inertia, damping and friction, which act before the torque sensor -- plus the configured sensor
offset and noise (``tau_J_offset``, ``tau_J_noise``, per joint, drawn from ``ctx.rng_for``; zero by
default). Everything the arm touches therefore reaches the estimate through its dynamics, as on the
robot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.models import resolve_model
from roqsim.plugin import Plugin
from roqsim.schema import Field

from ..franka import limits
from ..franka.core import CollisionBehavior, Ending, FrankaCore
from ..franka.protocol import (
    COMMAND_PORT,
    ERRORS,
    MOVE_REQUEST,
    SET_CARTESIAN_IMPEDANCE_REQUEST,
    SET_COLLISION_BEHAVIOR_REQUEST,
    SET_EE_TO_K_REQUEST,
    SET_JOINT_IMPEDANCE_REQUEST,
    SET_LOAD_REQUEST,
    SET_NE_TO_EE_REQUEST,
    STATUS_RESPONSE,
    AutomaticErrorRecoveryStatus,
    Command,
    GetterSetterStatus,
    MoveControllerMode,
    MoveMotionGeneratorMode,
    MoveStatus,
    StopMoveStatus,
    pack_robot_state,
)
from ..franka.robot_model import RobotModel
from ..franka.server import FCIServer, Request

JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


@dataclass
class InProcessMotion:
    """A law running in process; ``ending`` is set when the motion ends (``status``, ``errors``)."""

    law: Callable[[dict, float], dict]
    ending: Ending | None = None

    @property
    def done(self) -> bool:
        return self.ending is not None


@dataclass
class FrankaHandle:
    """What ``franka_fci:<arm>`` offers in process (all calls on the physics thread, e.g. from a plugin)."""

    arm: str
    state: Callable[[], dict]
    control: Callable[..., InProcessMotion]
    stop: Callable[[], None]
    recover: Callable[[], bool]
    set_load: Callable[..., None]
    set_collision_behavior: Callable[[dict], bool]
    set_cartesian_impedance: Callable[[list], bool]
    set_joint_impedance: Callable[[list], bool]
    set_k: Callable[[list], None]
    core: FrankaCore


def _matrix(value, name: str) -> np.ndarray:
    a = np.asarray(value, float)
    if a.shape == (16,):
        a = a.reshape(4, 4)
    if a.shape != (4, 4):
        raise ValueError(f"franka_fci: {name} must be a 4 x 4 matrix (row-major)")
    return a


class FrankaFCIPlugin(Plugin):
    CONFIG_SCHEMA = {
        "collision_behavior": Field(
            dict,
            required=True,
            doc="setCollisionBehavior's thresholds; the robot's own are unpublished",
        ),
        "bind": Field(str, default="", doc="address the FCI is served on (empty: in-process only)"),
        "port": Field(
            int, default=COMMAND_PORT, doc="the FCI command port; libfranka always uses 1337"
        ),
        "joint_impedance": Field(
            list,
            default=None,
            doc="setJointImpedance's K (N m/rad); default: libfranka's examples' values",
        ),
        "cartesian_impedance": Field(
            list,
            default=None,
            doc="setCartesianImpedance's K; default: libfranka's examples' values",
        ),
        "end_effector": Field(
            dict, default=None, doc="Desk's end effector: mass, com, inertia, F_T_NE"
        ),
        "load": Field(dict, default=None, doc="an initial setLoad: mass, com, inertia"),
        "observer_gain": Field(
            float,
            default=200.0,
            minimum=1.0,
            unit="1/s",
            doc="the external-torque observer's gain (a first-order lag of 1/gain)",
        ),
        "accel_switch": Field(
            float,
            default=1.0,
            minimum=0.0,
            unit="rad/s^2",
            doc="|ddq_d| above which the 'acceleration' thresholds apply (unpublished)",
        ),
        "tau_J_noise": Field(
            (float, list),
            default=0.0,
            minimum=0.0,
            unit="N m",
            doc="standard deviation of the joint torque sensors' noise, one or per joint",
        ),
        "tau_J_offset": Field(
            (float, list), default=0.0, unit="N m", doc="the sensors' offset, one or per joint"
        ),
        "client_timeout_s": Field(
            float,
            default=30.0,
            minimum=0.1,
            unit="s",
            doc="wall time a client may take to answer a state before the run stops",
        ),
        "urdf": Field(
            str,
            default="",
            doc="the URDF GetRobotModel serves (default: <model>.robot_model.urdf beside the MJCF)",
        ),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.server: FCIServer | None = None
        self.core: FrankaCore | None = None
        self._inproc: InProcessMotion | None = None
        self._net_motion = False
        self._first_command = False
        self._message_id = 0
        self._step = 0

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        cb = config.get("collision_behavior")
        if not isinstance(cb, dict):
            errors.append(
                "'collision_behavior' is required: the thresholds setCollisionBehavior takes "
                "(the robot's own start-up values are not published)"
            )
        else:
            try:
                errors += CollisionBehavior.from_config(cb).errors()
            except (KeyError, ValueError) as exc:
                errors.append(f"collision_behavior: {exc!r}")
        return errors

    # -- set-up -------------------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        steps = 1e-3 / float(m.opt.timestep)
        self._per_tick = int(round(steps))
        if self._per_tick < 1 or abs(steps - self._per_tick) > 1e-9:
            raise RuntimeError(
                f"franka_fci: the timestep {m.opt.timestep} s does not divide the robot's 1 ms control "
                "period; use 1 ms / N for an integer N (0.001, 0.0005, 1/3 ms, ...)"
            )
        entity = ctx.entities.get(self.entity)
        if entity is None:
            raise RuntimeError(
                f"franka_fci: no arm entity {self.entity!r}; nest it under the fr3's spawn_arm"
            )
        prefix = entity.meta.get("prefix", "")
        self._dofs, self._qadr, self._acts = [], [], []
        for j in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + j)
            if jid < 0 or aid < 0:
                raise RuntimeError(
                    f"franka_fci: {prefix + j!r} is not a joint with an actuator of this arm"
                )
            self._dofs.append(int(m.jnt_dofadr[jid]))
            self._qadr.append(int(m.jnt_qposadr[jid]))
            self._acts.append(aid)
        self._dofs, self._qadr, self._acts = map(np.array, (self._dofs, self._qadr, self._acts))
        self._base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "fr3_link0")
        mujoco.mj_kinematics(m, ctx.data)
        R_base = ctx.data.xmat[self._base].reshape(3, 3)
        model_file = Path(resolve_model(entity.meta["model"]).path)
        urdf = Path(self.config.get("urdf") or model_file.with_suffix(".robot_model.urdf"))
        if not urdf.is_file():
            raise RuntimeError(f"franka_fci: no URDF for GetRobotModel at {urdf}")
        robot_model = RobotModel(model_file, R_base.T @ np.asarray(m.opt.gravity, float))
        cfg = self.config
        self.core = FrankaCore(
            robot_model,
            collision_behavior=CollisionBehavior.from_config(cfg["collision_behavior"]),
            **({"joint_impedance": cfg["joint_impedance"]} if cfg.get("joint_impedance") else {}),
            **(
                {"cartesian_impedance": cfg["cartesian_impedance"]}
                if cfg.get("cartesian_impedance")
                else {}
            ),
            end_effector=self._end_effector(cfg.get("end_effector")),
            observer_gain=float(cfg.get("observer_gain", 200.0)),
            accel_switch=float(cfg.get("accel_switch", 1.0)),
        )
        load = cfg.get("load")
        if load:
            self.core.set_load(
                load["mass"], load.get("com", [0, 0, 0]), load.get("inertia", np.zeros(9))
            )
        self._noise = np.broadcast_to(np.asarray(cfg.get("tau_J_noise", 0.0), float), (7,)).copy()
        self._offset = np.broadcast_to(np.asarray(cfg.get("tau_J_offset", 0.0), float), (7,)).copy()
        self.core.reset(ctx.data.qpos[self._qadr])
        # A preview (a render, an export, a catalog read) builds the world to look at it: it has no
        # client, and must not take the arm's address from a run that has one.
        if cfg.get("bind") and not ctx.preview:
            self.server = FCIServer(
                cfg["bind"],
                urdf.read_text(encoding="utf-8"),
                port=int(cfg.get("port", COMMAND_PORT)),
            )
        ctx.blackboard.set(
            f"franka_fci:{self.entity}",
            FrankaHandle(
                arm=self.entity,
                state=lambda: self._last_state,
                control=self.control,
                stop=self._stop_inproc,
                recover=self.core.automatic_error_recovery,
                set_load=self.core.set_load,
                set_collision_behavior=lambda cb: self.core.set_collision_behavior(
                    CollisionBehavior.from_config(cb)
                ),
                set_cartesian_impedance=self.core.set_cartesian_impedance,
                set_joint_impedance=self.core.set_joint_impedance,
                set_k=lambda T: self.core.set_k(_matrix(T, "EE_T_K")),
                core=self.core,
            ),
        )
        self._last_state = self.core.state(0)

    @staticmethod
    def _end_effector(ee: dict | None) -> dict | None:
        if not ee:
            return None
        out = dict(ee)
        if "F_T_NE" in out:
            out["F_T_NE"] = _matrix(out["F_T_NE"], "end_effector.F_T_NE")
        if "inertia" in out:
            out["inertia"] = np.asarray(out["inertia"], float).reshape(3, 3)
        return out

    def on_reset(self, ctx: SimContext) -> None:
        # A motion the reset ends is answered as aborted: a network client blocks on its Move's
        # outcome (libfranka's throwOnMotionError) and would otherwise wait for it forever.
        self._settle_endings()
        motion = self.core.motion
        if self._net_motion and self.server is not None and motion is not None:
            self.server.reply(
                Command.MOVE, motion.command_id, STATUS_RESPONSE.pack(MoveStatus.ABORTED)
            )
        if self._inproc is not None and self._inproc.ending is None:
            self._inproc.ending = Ending(None, MoveStatus.ABORTED, ["reset"])
        self._inproc = None
        self.core.reset(ctx.data.qpos[self._qadr])
        self._net_motion = False
        self._step = 0
        self._last_state = self.core.state(self._message_id)

    def shutdown(self, ctx: SimContext | None = None) -> None:
        if self.server is not None:
            self.server.close()

    # -- in process ---------------------------------------------------------------------------------

    def control(
        self,
        law: Callable[[dict, float], dict],
        controller: str = "external",
        generator: str = "none",
    ) -> InProcessMotion:
        """Start ``law`` as a motion: ``controller`` ``external | joint_impedance | cartesian_impedance``,
        ``generator`` ``none | joint_position | joint_velocity | cartesian_position | cartesian_velocity``."""
        ctrl = {
            "external": MoveControllerMode.EXTERNAL_CONTROLLER,
            "joint_impedance": MoveControllerMode.JOINT_IMPEDANCE,
            "cartesian_impedance": MoveControllerMode.CARTESIAN_IMPEDANCE,
        }[controller]
        gen = {
            "none": MoveMotionGeneratorMode.NONE,
            "joint_position": MoveMotionGeneratorMode.JOINT_POSITION,
            "joint_velocity": MoveMotionGeneratorMode.JOINT_VELOCITY,
            "cartesian_position": MoveMotionGeneratorMode.CARTESIAN_POSITION,
            "cartesian_velocity": MoveMotionGeneratorMode.CARTESIAN_VELOCITY,
        }[generator]
        motion = InProcessMotion(law)
        if self._net_motion:
            motion.ending = Ending(
                None,
                MoveStatus.COMMAND_NOT_POSSIBLE_REJECTED,
                ["a network client is moving the arm"],
            )
            return motion
        refused = self.core.start_motion(ctrl, gen)
        if refused is not None:
            motion.ending = Ending(
                None, refused, [e for e, on in zip(ERRORS, self.core.errors, strict=True) if on]
            )
            return motion
        self._inproc = motion
        return motion

    def _stop_inproc(self) -> None:
        if self._inproc is not None:
            self.core.stop()

    # -- the physics --------------------------------------------------------------------------------

    def _tau_J(self, ctx: SimContext) -> np.ndarray:
        """The link-side joint torque: the actuator's, less what acts on the motor side of the sensor."""
        m, d = ctx.model, ctx.data
        dofs = self._dofs
        tau = (
            d.qfrc_actuator[dofs]
            - m.dof_armature[dofs] * d.qacc[dofs]
            - m.dof_damping[dofs] * d.qvel[dofs]
        )
        friction = np.zeros(7)
        if d.nefc:
            rows = np.nonzero(d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)[0]
            index = {int(dof): k for k, dof in enumerate(dofs)}
            for r in rows:
                k = index.get(int(d.efc_id[r]))
                if k is not None:
                    friction[k] += d.efc_force[r]
        tau = tau + friction + self._offset
        if np.any(self._noise > 0):
            rng = ctx.rng_for(self.draw_key("tau_J"))
            tau = tau + rng.normal(0.0, 1.0, 7) * self._noise
        return tau

    def pre_step(self, ctx: SimContext) -> None:
        d = ctx.data
        if self._step % self._per_tick == 0:
            self._tick(ctx)
        self._step += 1
        d.ctrl[self._acts] = self.core.torques(d.qpos[self._qadr], d.qvel[self._dofs])

    def _tick(self, ctx: SimContext) -> None:
        d = ctx.data
        core = self.core
        replies: list = []
        if self.server is not None:
            while not self.server.requests.empty():
                self._execute(self.server.requests.get_nowait(), replies)
        core.measure(d.qpos[self._qadr], d.qvel[self._dofs], self._tau_J(ctx))
        self._settle_endings()
        self._message_id += 1
        state = core.state(self._message_id)
        self._last_state = state
        data = pack_robot_state(state) if self.server is not None else b""
        if self.server is not None:
            self.server.send_state(data)
            # A setting's reply follows the state that shows it, so a client reading on finds it.
            for reply in replies:
                self.server.reply(*reply)
        if core.motion is None:
            return
        if self._inproc is not None:
            core.apply(self._inproc.law(state, limits.DELTA_T))
        elif self._net_motion and self.server is not None and self.server.connected:
            cmd = self.server.await_command(
                self._message_id,
                data,
                strict=self._first_command,
                client_timeout_s=float(self.config.get("client_timeout_s", 30.0)),
            )
            if cmd is not None:
                self._first_command = True
                core.apply(cmd)
        self._settle_endings()

    def _settle_endings(self) -> None:
        core = self.core
        for ending in core.endings:
            if self._inproc is not None and ending.command_id is None:
                self._inproc.ending = ending
                self._inproc = None
            elif self.server is not None:
                self.server.reply(
                    Command.MOVE, ending.command_id, STATUS_RESPONSE.pack(ending.status)
                )
                self._net_motion = False
        core.endings.clear()

    # -- the FCI's commands, on the physics thread --------------------------------------------------

    def _execute(self, req: Request, replies: list) -> None:
        core = self.core
        ok, bad = GetterSetterStatus.SUCCESS, GetterSetterStatus.INVALID_ARGUMENT_REJECTED
        busy = GetterSetterStatus.COMMAND_NOT_POSSIBLE_REJECTED
        c, p = req.command, req.payload
        if c == Command.MOVE:
            values = MOVE_REQUEST.unpack(p)
            try:
                modes = MoveControllerMode(values[0]), MoveMotionGeneratorMode(values[1])
            except ValueError:  # a mode the protocol does not define, refused as the robot does
                modes = None
            if modes is None:
                status = MoveStatus.INVALID_ARGUMENT_REJECTED
            elif self._inproc is not None:
                status = MoveStatus.COMMAND_NOT_POSSIBLE_REJECTED
            else:
                status = core.start_motion(*modes, req.command_id)
            if status is not None:
                replies.append((c, req.command_id, STATUS_RESPONSE.pack(status)))
            else:
                # The robot answers a Move twice: "motion started" now (libfranka's executeCommand
                # blocks on it), and the outcome when the motion ends (_settle_endings).
                replies.append((c, req.command_id, STATUS_RESPONSE.pack(MoveStatus.MOTION_STARTED)))
                self._net_motion, self._first_command = True, False
            return
        if c == Command.STOP_MOVE:
            if self._net_motion:
                core.stop()
            replies.append((c, req.command_id, STATUS_RESPONSE.pack(StopMoveStatus.SUCCESS)))
            return
        if c == Command.AUTOMATIC_ERROR_RECOVERY:
            done = core.automatic_error_recovery()
            replies.append(
                (
                    c,
                    req.command_id,
                    STATUS_RESPONSE.pack(
                        AutomaticErrorRecoveryStatus.SUCCESS
                        if done
                        else AutomaticErrorRecoveryStatus.COMMAND_NOT_POSSIBLE_REJECTED
                    ),
                )
            )
            return
        if core.motion is not None:  # the robot takes no settings while it moves
            replies.append((c, req.command_id, STATUS_RESPONSE.pack(busy)))
            return
        status = ok
        if c == Command.SET_COLLISION_BEHAVIOR:
            status = (
                ok
                if core.set_collision_behavior(
                    CollisionBehavior.from_request(SET_COLLISION_BEHAVIOR_REQUEST.unpack(p))
                )
                else bad
            )
        elif c == Command.SET_JOINT_IMPEDANCE:
            status = ok if core.set_joint_impedance(SET_JOINT_IMPEDANCE_REQUEST.unpack(p)) else bad
        elif c == Command.SET_CARTESIAN_IMPEDANCE:
            status = (
                ok
                if core.set_cartesian_impedance(SET_CARTESIAN_IMPEDANCE_REQUEST.unpack(p))
                else bad
            )
        elif c == Command.SET_EE_TO_K:
            core.set_k(np.asarray(SET_EE_TO_K_REQUEST.unpack(p)).reshape(4, 4).T)
        elif c == Command.SET_NE_TO_EE:
            core.set_ee(np.asarray(SET_NE_TO_EE_REQUEST.unpack(p)).reshape(4, 4).T)
        elif c == Command.SET_LOAD:
            v = SET_LOAD_REQUEST.unpack(p)
            core.set_load(v[0], v[1:4], np.asarray(v[4:13]).reshape(3, 3))
        else:  # SetGuidingMode: guiding is not served
            status = busy
        replies.append((c, req.command_id, STATUS_RESPONSE.pack(status)))
