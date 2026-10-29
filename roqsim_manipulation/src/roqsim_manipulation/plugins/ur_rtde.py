"""Plugin: a simulated UR arm driven through the ``ur_rtde`` client API, in lock-step with the physics.

A program that controls a real Universal Robots arm from a PC over RTDE (Real-Time Data Exchange,
port 30004) is most often written against the ``ur_rtde`` library: an ``RTDEControlInterface`` that
moves the arm (``speedL``, ``moveL``, ``zeroFtSensor``, the ``initPeriod``/``waitPeriod`` control
loop) and an ``RTDEReceiveInterface`` that reads it (``getActualTCPPose``, ``getActualTCPForce``,
...). This plugin serves a subset of those two interfaces, under the same method names, signatures,
units and frames, against the simulated arm. A control loop written as a function of the two objects
then runs unchanged on the robot (built from ``rtde_control`` / ``rtde_receive`` with the robot's
address) and here (built by this plugin)::

    def press(c, r, force):
        c.zeroFtSensor()
        while r.getActualTCPForce()[2] < force:
            t0 = c.initPeriod()
            c.speedL([0, 0, -0.01, 0, 0, 0], 0.5)
            c.waitPeriod(t0)
        c.speedStop()

    ctx.blackboard.get("ur_rtde:ur5e").run(press, 10.0)          # in a simulation
    press(RTDEControlInterface(host), RTDEReceiveInterface(host), 10.0)   # on the robot

**Lock-step, on simulated time.** The program runs in a thread of its own, as it does on a PC, and
it and the physics take turns: while the program holds the turn the physics waits, and the program
gives it back by a call that waits on the robot -- ``waitPeriod`` until its period has passed in
simulated time, a blocking ``moveL`` until the motion is done. One control period of the program is
therefore one ``1 / rate_hz`` of simulated time, however slow the physics is in wall time, and a run
is reproducible. A loop that never waits would hold the turn forever; the physics gives up after
``client_timeout_s`` of wall time and raises, naming ``waitPeriod``. ``run`` hands the program the
turn at the sim time it is called, so where in the trial it starts is the caller's, not a thread
scheduler's.

**What the controller is emulated as.**

* ``speedL(xd, acceleration)``: the TCP twist ``xd`` in the base frame is approached at
  ``acceleration`` (m/s^2 on the linear part, rad/s^2 on the angular part) and held until the next
  command. ``speedStop(a)`` decelerates to rest. A ``time`` other than 0 is refused: ``ur_rtde``
  documents it as the time before the call returns, and what the controller does with it is not
  emulated.
* ``moveL(pose, speed, acceleration, asynchronous)``: a straight line of the TCP to ``pose`` with a
  trapezoidal speed profile; the orientation turns along the shortest rotation in proportion. It
  blocks until the profile is done unless ``asynchronous``; ``getAsyncOperationProgress`` is
  negative once no asynchronous move is running.
* ``setWatchdog(min_frequency)``: from then on, a period of sim time longer than
  ``1 / min_frequency`` without a ``kickWatchdog`` shuts the control down, as ``ur_rtde`` documents
  the watchdog's default action: the arm decelerates to rest and every later call raises
  :class:`RTDEError`. The emulation counts sim time during a blocking move as well, so a program
  that sets a watchdog kicks it through an asynchronous move.
* ``setPayload(mass, cog)`` is accepted and checked, and changes nothing: the simulated arm carries
  the tool the model gives it, whatever the program declares.
* A command reaches the arm ``command_delay_s`` after the state it was computed from, by default
  one control period: what a script on the controller reading the command registers sees at the
  next cycle.
* The twist is resolved to joint targets through a damped least-squares inverse of the TCP
  Jacobian, taken at the commanded joint targets so a lagging servo does not bend a straight
  line, and handed to ``arm_controller``. The arm's own servo and motion limits act as they do for
  every other controller.

Not emulated, and so absent rather than approximated: the controller's own force law
(``forceMode``), ``servoL``/``servoJ`` and the other motion commands, payload compensation of the
wrench (``actual_TCP_force`` here is the wrist sensor's tared reading, so tare at the pose of contact),
the safety limits and protective stops, the speed slider. A program calling a method that is not
served fails with an ``AttributeError`` naming it. The interfaces are in-process: a client in another
process, speaking the RTDE protocol on a socket, is not served.

**Frames and conventions are the controller's.** Poses are ``[x, y, z, rx, ry, rz]`` (metres, a
rotation vector) of the TCP in the robot's base frame, which on a UR is ``ur_description``'s
``base_link`` turned half a turn about z; ``base`` names the body and that turn. The TCP is the
flange (``flange_site``) offset by ``setTcp``'s pose. ``getActualTCPForce`` reports the wrench at
the TCP in the base frame, as the force ``tcp_force_measures`` names -- the environment's on the
tool by default, the convention of the ``force_torque`` reader it comes from. Check that sign on
the robot (a push on the tool along base +z) before a law relies on it.

It is a component of the entry that spawns the arm, after the ``force_torque`` entry it names,
which reports in the world frame. Every key, with its type, unit and default, is the plugin's
``CONFIG_SCHEMA`` (``roqsim plugins describe ur_rtde``); a world declares it, for instance::

    - force_torque: {site: fts_site, frame: world}
      name: ft
    - ur_rtde: {ft: ft}

The handle is published under ``ur_rtde:<arm>`` (:class:`RTDEHandle`).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.controllers import ACTIVE, Controller, registry_for
from roqsim.plugin import Plugin
from roqsim.schema import Field

_MEASURES = ("environment_on_tool", "tool_on_environment")


class RTDEError(RuntimeError):
    """The interface is gone: the episode was reset, the script stopped, or the simulation ended."""


def _rotvec(R: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=float).reshape(9))
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, quat, 1.0)
    return out


def _rotmat(rotvec) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rv))
    quat = np.zeros(4)
    if angle < 1e-12:
        quat[0] = 1.0
    else:
        mujoco.mju_axisAngle2Quat(quat, rv / angle, angle)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


def _toward(v: np.ndarray, goal: np.ndarray, step: float) -> np.ndarray:
    """*v* moved toward *goal* by at most *step* in norm."""
    dv = goal - v
    n = float(np.linalg.norm(dv))
    return goal.copy() if n <= step else v + dv * (step / n)


class _Line:
    """A straight TCP motion with a trapezoidal speed profile, in the world frame."""

    def __init__(self, p0, R0, p1, R1, speed: float, accel: float):
        self.p0, self.R0 = p0, R0
        self.dp = p1 - p0
        self.w = _rotvec(R0.T @ R1)  # in the start frame
        length, angle = float(np.linalg.norm(self.dp)), float(np.linalg.norm(self.w))
        # The profile runs over the path length, or over the angle for a pure rotation, as a UR's
        # movel does.
        self.dist = length if length > 1e-6 else angle
        v, a = float(speed), float(accel)
        if self.dist < v * v / a:
            v = math.sqrt(self.dist * a)
        self.v, self.a = v, a
        self.t_acc = v / a if a > 0 else 0.0
        self.T = 2 * self.t_acc + ((self.dist - v * self.t_acc) / v if v > 0 else 0.0)

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Desired position, orientation and world twist at *t* seconds into the line."""
        if self.dist <= 0 or t >= self.T:
            s, sd = self.dist, 0.0
        elif t < self.t_acc:
            s, sd = 0.5 * self.a * t * t, self.a * t
        elif t < self.T - self.t_acc:
            s, sd = self.v * (t - 0.5 * self.t_acc), self.v
        else:
            r = self.T - t
            s, sd = self.dist - 0.5 * self.a * r * r, self.a * r
        frac = s / self.dist if self.dist > 0 else 1.0
        rate = sd / self.dist if self.dist > 0 else 0.0
        R = self.R0 @ _rotmat(frac * self.w)
        twist = np.concatenate([rate * self.dp, rate * (self.R0 @ self.w)])
        return self.p0 + frac * self.dp, R, twist


class _Link:
    """What the physics thread and the program's thread share, and the turn they pass."""

    def __init__(self):
        self.cv = threading.Condition()
        self.client_turn = False
        self.attached = False
        self.closed: str | None = None
        self.wake_at: float | None = None
        self.wait_motion: int | None = None
        self.error: BaseException | None = None
        self.result = None
        self.state: dict = {}
        self.commands: list[tuple[float, int, Callable[[], None]]] = []
        self.order = 0
        self.motion_issued = 0
        self.motion_done = 0
        self.watchdog_s: float | None = None
        self.kicked_at = 0.0


class RTDEControlInterface:
    """The served part of ``ur_rtde``'s ``RTDEControlInterface``. Built by :class:`URRTDEPlugin`."""

    def __init__(self, plugin: URRTDEPlugin, link: _Link):
        self._p, self._l = plugin, link

    # -- the turn -------------------------------------------------------------------------------

    def _check(self) -> None:
        if self._l.closed is not None:
            raise RTDEError(f"ur_rtde: the interface is closed ({self._l.closed})")

    def _send(self, apply: Callable[[], None]) -> None:
        with self._l.cv:
            self._check()
            at = self._l.state["timestamp"] + self._p.command_delay_s
            self._l.order += 1
            self._l.commands.append((at, self._l.order, apply))

    def _yield(self, wake_at: float | None = None, motion: int | None = None) -> None:
        link = self._l
        with link.cv:
            self._check()
            link.wake_at, link.wait_motion = wake_at, motion
            link.client_turn = False
            link.cv.notify_all()
            while not link.client_turn and link.closed is None:
                link.cv.wait()
            self._check()

    # -- the control loop -----------------------------------------------------------------------

    def initPeriod(self) -> float:
        """The start of this control period: the sim time of the state the program reads."""
        with self._l.cv:
            self._check()
            return float(self._l.state["timestamp"])

    def waitPeriod(self, t_cycle_start: float) -> None:
        """Give the turn to the physics until one period after *t_cycle_start*, in sim time."""
        self._yield(wake_at=float(t_cycle_start) + self._p.period)

    # -- motion ---------------------------------------------------------------------------------

    def speedL(self, xd, acceleration: float = 0.25, time: float = 0.0) -> bool:
        xd = np.asarray(xd, dtype=float)
        if xd.shape != (6,) or not np.all(np.isfinite(xd)):
            raise ValueError(f"speedL: xd must be 6 finite numbers, got {xd!r}")
        if acceleration <= 0:
            raise ValueError("speedL: acceleration must be > 0")
        if time != 0.0:
            raise ValueError("speedL: a time other than 0 is not emulated")
        self._send(lambda: self._p._speed(xd, float(acceleration)))
        return True

    def speedStop(self, a: float = 10.0) -> bool:
        self._send(lambda: self._p._speed(np.zeros(6), float(a)))
        return True

    def moveL(
        self, pose, speed: float = 0.25, acceleration: float = 1.2, asynchronous: bool = False
    ) -> bool:
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"moveL: pose must be 6 finite numbers, got {pose!r}")
        if speed <= 0 or acceleration <= 0:
            raise ValueError("moveL: speed and acceleration must be > 0")
        with self._l.cv:
            self._l.motion_issued += 1
            motion = self._l.motion_issued
        self._send(lambda: self._p._line_to(pose, float(speed), float(acceleration), motion))
        if not asynchronous:
            self._yield(motion=motion)
        return True

    def getAsyncOperationProgress(self) -> int:
        """Negative once the last asynchronous move is done (or none was sent), else 0."""
        with self._l.cv:
            self._check()
            return -1 if self._l.motion_done >= self._l.motion_issued else 0

    # -- the watchdog ---------------------------------------------------------------------------

    def setWatchdog(self, min_frequency: float = 10.0) -> bool:
        if min_frequency <= 0:
            raise ValueError("setWatchdog: min_frequency must be > 0")
        with self._l.cv:
            self._check()
            self._l.watchdog_s = 1.0 / float(min_frequency)
            self._l.kicked_at = float(self._l.state["timestamp"])
        return True

    def kickWatchdog(self) -> bool:
        with self._l.cv:
            self._check()
            self._l.kicked_at = float(self._l.state["timestamp"])
        return True

    # -- the tool -------------------------------------------------------------------------------

    def zeroFtSensor(self) -> bool:
        self._send(self._p._tare)
        return True

    def setTcp(self, tcp_offset) -> bool:
        off = np.asarray(tcp_offset, dtype=float)
        if off.shape != (6,):
            raise ValueError(f"setTcp: the offset is 6 numbers [x, y, z, rx, ry, rz], got {off!r}")
        self._send(lambda: self._p._set_tcp(off))
        return True

    def setPayload(self, mass: float, cog=()) -> bool:
        cog = list(cog)
        if not mass >= 0 or len(cog) not in (0, 3):
            raise ValueError(
                f"setPayload: a mass >= 0 kg and a cog of 3 numbers or none, got {mass!r}, {cog!r}"
            )
        return True

    def getTCPOffset(self) -> list[float]:
        with self._l.cv:
            return list(self._l.state["tcp_offset"])

    # -- the connection -------------------------------------------------------------------------

    def isConnected(self) -> bool:
        return self._l.closed is None

    def isProgramRunning(self) -> bool:
        return self._l.closed is None

    def stopScript(self) -> None:
        """End the program's control: the arm decelerates to rest and every later call raises."""
        with self._l.cv:
            if self._l.closed is None:
                self._l.closed = "stopScript"
                self._l.cv.notify_all()

    disconnect = stopScript


class RTDEReceiveInterface:
    """The served part of ``ur_rtde``'s ``RTDEReceiveInterface``: the state at the period's start."""

    def __init__(self, link: _Link):
        self._l = link

    def _get(self, key: str) -> list[float]:
        with self._l.cv:
            if self._l.closed is not None:
                raise RTDEError(f"ur_rtde: the interface is closed ({self._l.closed})")
            return list(self._l.state[key])

    def getTimestamp(self) -> float:
        with self._l.cv:
            return float(self._l.state["timestamp"])

    def getActualTCPPose(self) -> list[float]:
        return self._get("actual_TCP_pose")

    def getActualTCPSpeed(self) -> list[float]:
        return self._get("actual_TCP_speed")

    def getActualTCPForce(self) -> list[float]:
        return self._get("actual_TCP_force")

    def getTargetTCPSpeed(self) -> list[float]:
        return self._get("target_TCP_speed")

    def getActualQ(self) -> list[float]:
        return self._get("actual_q")

    def getActualQd(self) -> list[float]:
        return self._get("actual_qd")

    def isConnected(self) -> bool:
        return self._l.closed is None


@dataclass
class RTDEHandle:
    """Published under ``ur_rtde:<arm>``. All of it runs on the physics thread."""

    arm: str
    #: ``run(program, *args, **kwargs)``: start ``program(control, receive, *args, **kwargs)`` in a
    #: thread holding the turn from now. One program at a time.
    run: Callable[..., None]
    #: Whether the program is still running.
    running: Callable[[], bool]
    #: The program's return value once it has ended; re-raises what it raised.
    result: Callable[[], object]


class URRTDEPlugin(Plugin):
    requires_owner = True

    CONFIG_SCHEMA = {
        "ft": Field(
            str, required=True, doc="label of the force_torque entry, reporting in the world frame"
        ),
        "ft_site": Field(
            str, default="fts_site", doc="the site that sensor measures at (prefixed)"
        ),
        "flange_site": Field(
            str, default="tool0_site", doc="the tool flange: the TCP offset's origin"
        ),
        "base": Field(
            dict,
            default={"body": "base", "yaw": math.pi},
            doc="the controller's base frame: a body (prefixed) turned by yaw (rad) about its z",
        ),
        "rate_hz": Field(float, default=500.0, minimum=1.0, unit="Hz", doc="the control cycle"),
        "command_delay_s": Field(
            float,
            default=0.002,
            minimum=0.0,
            unit="s",
            doc="how long after the state it was computed from a command acts",
        ),
        "tcp_force_measures": Field(
            str,
            default="environment_on_tool",
            choices=_MEASURES,
            doc="which force getActualTCPForce reports",
        ),
        "ik_damping": Field(float, default=0.01, minimum=0.0, doc="damped least-squares lambda"),
        "line_gain": Field(
            float, default=50.0, minimum=0.0, unit="1/s", doc="moveL's closure on its profile"
        ),
        "client_timeout_s": Field(
            float,
            default=120.0,
            minimum=0.0,
            unit="s",
            doc="wall seconds the physics waits for a program that holds the turn",
        ),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        s = self.settings
        self.ft_label = s.ft
        self.period = 1.0 / float(s.rate_hz)
        self.command_delay_s = float(s.command_delay_s)
        self.ik_damping = float(s.ik_damping)
        self.line_gain = float(s.line_gain)
        self.client_timeout_s = float(s.client_timeout_s)
        self.tcp_force_sign = 1.0 if s.tcp_force_measures == "environment_on_tool" else -1.0
        base = dict(s.base)
        self._base_body_name = str(base.get("body", "base"))
        self._base_yaw = float(base.get("yaw", math.pi))
        self._ctx: SimContext | None = None
        self._link = _Link()
        self._thread: threading.Thread | None = None
        self._reset_motion()

    def validate_config(self, config: dict) -> list[str]:
        base = config.get("base", {})
        if isinstance(base, dict) and set(base) - {"body", "yaw"}:
            return [f"'base' takes 'body' and 'yaw', got {sorted(base)}"]
        return []

    def _reset_motion(self) -> None:
        self._v = np.zeros(6)  # commanded TCP twist, world frame
        self._vd = np.zeros(6)
        self._acc = 0.0
        self._mode = "idle"
        self._line: _Line | None = None
        self._line_t0 = 0.0
        self._line_id = 0
        self._q_target: np.ndarray | None = None
        self._stopped = True

    # -- set-up ---------------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.entity)
        prefix = entity.meta.get("prefix", "") if entity else ""
        self._arm = ctx.blackboard.get(f"arm:{self.entity}")
        if self._arm is None:
            raise RuntimeError(
                f"ur_rtde: no ArmHandle at 'arm:{self.entity}'. Nest ur_rtde under the arm's spawn, "
                "after its arm_controller."
            )
        self._ft = ctx.blackboard.get(f"ft:{self.ft_label}")
        if self._ft is None or self._ft.tare is None:
            raise RuntimeError(
                f"ur_rtde: no force_torque sensor labelled {self.ft_label!r} with a tare. Declare "
                "that force_torque entry under the arm BEFORE this one."
            )
        if self._ft.frame != "world":
            raise RuntimeError(
                f"ur_rtde: force_torque {self.ft_label!r} reports in {self._ft.frame!r}; set its "
                "frame: world, from which this plugin turns the wrench into the controller's base frame"
            )
        self._ft_sign = 1.0 if self._ft.measures == "environment_on_tool" else -1.0

        def _id(kind, name, what):
            i = mujoco.mj_name2id(m, kind, f"{prefix}{name}")
            if i < 0:
                raise RuntimeError(f"ur_rtde: {what} {prefix}{name!r} not found")
            return i

        self._base_id = _id(mujoco.mjtObj.mjOBJ_BODY, self._base_body_name, "base body")
        self._flange_id = _id(mujoco.mjtObj.mjOBJ_SITE, self.settings.flange_site, "flange site")
        self._ft_site_id = _id(mujoco.mjtObj.mjOBJ_SITE, self.settings.ft_site, "sensor site")
        self._flange_body = int(m.site_bodyid[self._flange_id])
        c, s = math.cos(self._base_yaw), math.sin(self._base_yaw)
        self._yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self._tcp_offset = np.zeros(6)

        self._joint_names = list(self._arm.joint_names)
        dofs, qposadr = [], []
        for jname in self._joint_names:
            jid = _id(mujoco.mjtObj.mjOBJ_JOINT, jname, "joint")
            dofs.append(int(m.jnt_dofadr[jid]))
            qposadr.append(int(m.jnt_qposadr[jid]))
        self._dofs, self._qposadr = np.array(dofs), np.array(qposadr)
        self._kin = mujoco.MjData(m)

        # The program's control script owns the arm while it runs, as the external control script
        # does on the robot: the trajectory role is released, and listed as a controller that
        # claims the joints so a switch back is a switch rather than a race.
        self._arm.set_command_source(self.update, self.label or "ur_rtde")
        registry_for(ctx).register(
            Controller(
                name=self.label or "ur_rtde",
                type="ur_rtde/RTDEControlInterface",
                claims=tuple(f"{j}/position" for j in self._joint_names),
                state=ACTIVE,
                namespace=entity.meta.get("namespace", "") if entity else "",
                owner=self.entity,
            )
        )
        ctx.blackboard.set(
            f"ur_rtde:{self.entity}",
            RTDEHandle(arm=self.entity, run=self.run, running=self.running, result=self.result),
        )

    # -- frames ---------------------------------------------------------------------------------

    def _base(self, d) -> tuple[np.ndarray, np.ndarray]:
        R = np.array(d.xmat[self._base_id]).reshape(3, 3) @ self._yaw
        return np.array(d.xpos[self._base_id]), R

    def _tcp(self, d) -> tuple[np.ndarray, np.ndarray]:
        Rf = np.array(d.site_xmat[self._flange_id]).reshape(3, 3)
        p = np.array(d.site_xpos[self._flange_id]) + Rf @ self._tcp_offset[:3]
        return p, Rf @ _rotmat(self._tcp_offset[3:])

    def _snapshot(self, ctx: SimContext) -> dict:
        m, d = ctx.model, ctx.data
        pb, Rb = self._base(d)
        p, R = self._tcp(d)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self._flange_id, vel, 0)
        w, v_f = vel[:3], vel[3:]
        v = v_f + np.cross(w, p - np.array(d.site_xpos[self._flange_id]))
        f, tau = self._ft.read()
        f, tau = self._ft_sign * np.asarray(f), self._ft_sign * np.asarray(tau)
        # The torque about the TCP rather than about the sensor's point.
        tau = tau + np.cross(np.array(d.site_xpos[self._ft_site_id]) - p, f)
        f, tau = self.tcp_force_sign * f, self.tcp_force_sign * tau
        _, q, qd, _ = self._arm.read_state()
        return {
            "timestamp": float(ctx.sim_time),
            "actual_TCP_pose": [*(Rb.T @ (p - pb)), *_rotvec(Rb.T @ R)],
            "actual_TCP_speed": [*(Rb.T @ v), *(Rb.T @ w)],
            "target_TCP_speed": [*(Rb.T @ self._v[:3]), *(Rb.T @ self._v[3:])],
            "actual_TCP_force": [*(Rb.T @ f), *(Rb.T @ tau)],
            "actual_q": [float(x) for x in q],
            "actual_qd": [float(x) for x in qd],
            "tcp_offset": [float(x) for x in self._tcp_offset],
        }

    # -- commands, applied on the physics thread ------------------------------------------------

    def _speed(self, xd: np.ndarray, acc: float) -> None:
        _, Rb = self._base(self._ctx.data)
        self._mode = "speed"
        self._line = None
        self._vd = np.concatenate([Rb @ xd[:3], Rb @ xd[3:]])
        self._acc = acc

    def _line_to(self, pose: np.ndarray, speed: float, acc: float, motion: int) -> None:
        pb, Rb = self._base(self._ctx.data)
        p0, R0 = self._commanded_tcp()
        self._mode = "line"
        self._line = _Line(p0, R0, pb + Rb @ pose[:3], Rb @ _rotmat(pose[3:]), speed, acc)
        self._line_t0 = self._ctx.sim_time
        self._line_id = motion

    def _tare(self) -> None:
        self._ft.tare()

    def _set_tcp(self, offset: np.ndarray) -> None:
        self._tcp_offset = offset.copy()

    # -- the program ----------------------------------------------------------------------------

    def run(self, program: Callable, *args, **kwargs) -> None:
        """Start *program(control, receive, ...)* holding the turn from this sim time."""
        if self.running():
            raise RuntimeError("ur_rtde: a program is already running on this arm")
        link = _Link()
        link.attached = True
        link.client_turn = True
        link.state = self._snapshot(self._ctx)
        self._link = link
        self._stopped = False
        control, receive = RTDEControlInterface(self, link), RTDEReceiveInterface(link)

        def main():
            try:
                link.result = program(control, receive, *args, **kwargs)
            except RTDEError as e:
                # Ended by the simulation (a reset, the end of the run): nothing to report. Ended
                # by the program's own stopScript or by the watchdog: result() re-raises it.
                link.error = None if link.closed in ("reset", "shutdown") else e
            except BaseException as e:  # noqa: BLE001 - handed to the physics thread, which raises it
                link.error = e
            finally:
                with link.cv:
                    link.attached = False
                    link.client_turn = False
                    link.cv.notify_all()

        self._thread = threading.Thread(target=main, name=f"ur_rtde:{self.entity}", daemon=True)
        self._thread.start()

    def running(self) -> bool:
        return self._link.attached

    def result(self):
        if self._link.attached:
            raise RuntimeError("ur_rtde: the program is still running")
        if self._link.error is not None:
            raise self._link.error
        return self._link.result

    def _await_program(self) -> bool:
        """Wait while the program holds the turn; whether it has stopped controlling the arm."""
        link = self._link
        with link.cv:
            deadline = time.monotonic() + self.client_timeout_s
            while link.attached and link.client_turn and link.closed is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError(
                        f"ur_rtde: the program held the turn for {self.client_timeout_s:.0f} s of "
                        "wall time without waiting on the robot. A control loop gives the turn back "
                        "with waitPeriod (or a blocking move); one that never does stops the simulation."
                    )
                link.cv.wait(left)
            if (
                link.error is not None
                and not link.attached
                and not isinstance(link.error, RTDEError)
            ):
                # A bug in the program fails the run; an RTDEError is the control being shut down,
                # which the arm survives and result() reports.
                error, link.error = link.error, None
                raise RuntimeError(f"ur_rtde: the program raised {error!r}") from error
            if (
                link.attached
                and link.closed is None
                and link.watchdog_s is not None
                and self._ctx.sim_time - link.kicked_at > link.watchdog_s + 1e-9
            ):
                link.closed = "watchdog"
                link.cv.notify_all()
            return not link.attached or link.closed is not None

    def _speed_world_stop(self) -> None:
        # The script has ended: the arm comes to rest, as it does when the control script stops.
        self._mode, self._line = "speed", None
        self._vd = np.zeros(6)
        self._acc = max(self._acc, 2.0)

    def _apply_due(self, t: float, dt: float) -> None:
        link = self._link
        with link.cv:
            due = [c for c in link.commands if c[0] <= t + 0.5 * dt]
            link.commands = [c for c in link.commands if c[0] > t + 0.5 * dt]
        for _, _, apply in sorted(due, key=lambda c: (c[0], c[1])):
            apply()

    # -- the controller -------------------------------------------------------------------------

    def pre_step(self, ctx: SimContext) -> None:
        if self._arm.ensure_updated is not None:
            self._arm.ensure_updated(ctx)
        else:
            self.update(ctx)

    def update(self, ctx: SimContext) -> None:
        """This step's joint targets. Pulled by the arm; once per step."""
        if ctx.manual_control:
            return
        stopped = self._await_program()
        t, dt = ctx.sim_time, float(ctx.model.opt.timestep)
        self._apply_due(t, dt)
        if stopped and not self._stopped:
            self._stopped = True
            with self._link.cv:
                self._link.commands.clear()
            if self._mode != "idle":
                self._speed_world_stop()
        if self._mode == "idle":
            return
        if self._mode == "speed":
            self._v = np.concatenate(
                [
                    _toward(self._v[:3], self._vd[:3], self._acc * dt),
                    _toward(self._v[3:], self._vd[3:], self._acc * dt),
                ]
            )
            twist = self._v
            if not np.any(self._v) and not np.any(self._vd):
                self._mode = "idle"
                return
        else:
            p_d, R_d, ff = self._line.at(t + dt - self._line_t0)
            p, R = self._commanded_tcp()
            err = np.concatenate([p_d - p, R @ _rotvec(R.T @ R_d)])
            twist = ff + self.line_gain * err
            self._v = ff
            if t + dt - self._line_t0 >= self._line.T:
                self._finish_line()
        self._resolve(ctx, twist, dt)

    def _finish_line(self) -> None:
        self._mode, self._v = "idle", np.zeros(6)
        done = self._line_id
        self._line = None
        with self._link.cv:
            self._link.motion_done = max(self._link.motion_done, done)

    def _commanded_kinematics(self) -> mujoco.MjData:
        ctx = self._ctx
        if self._q_target is None:
            _, positions, _, _ = self._arm.read_state()
            by_name = dict(zip(self._arm.joint_names, positions, strict=False))
            self._q_target = np.array([by_name[n] for n in self._joint_names], dtype=float)
        kin = self._kin
        kin.qpos[:] = ctx.data.qpos
        kin.mocap_pos[:] = ctx.data.mocap_pos
        kin.mocap_quat[:] = ctx.data.mocap_quat
        kin.qpos[self._qposadr] = self._q_target
        mujoco.mj_kinematics(ctx.model, kin)
        mujoco.mj_comPos(ctx.model, kin)
        return kin

    def _commanded_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        return self._tcp(self._commanded_kinematics())

    def _resolve(self, ctx: SimContext, twist: np.ndarray, dt: float) -> None:
        """A world-frame TCP twist to joint targets, by damped least squares at the commanded joints."""
        m = ctx.model
        kin = self._commanded_kinematics()
        p, _ = self._tcp(kin)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, kin, jacp, jacr, p, self._flange_body)
        jac = np.vstack([jacp, jacr])[:, self._dofs]
        jjt = jac @ jac.T + self.ik_damping**2 * np.eye(6)
        self._q_target = self._q_target + (jac.T @ np.linalg.solve(jjt, twist)) * dt
        self._arm.set_targets(self._joint_names, self._q_target.tolist())

    def post_step(self, ctx: SimContext) -> None:
        link = self._link
        if not link.attached:
            return
        with link.cv:
            if link.client_turn or link.closed is not None:
                return
            t = ctx.sim_time
            half = 0.5 * float(ctx.model.opt.timestep)
            woken = (link.wake_at is not None and t >= link.wake_at - half) or (
                link.wait_motion is not None and link.motion_done >= link.wait_motion
            )
            if woken:
                link.state = self._snapshot(ctx)
                link.wake_at, link.wait_motion = None, None
                link.client_turn = True
                link.cv.notify_all()

    # -- lifecycle ------------------------------------------------------------------------------

    def _close(self, reason: str) -> None:
        link = self._link
        with link.cv:
            if link.closed is None:
                link.closed = reason
            link.cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def on_reset(self, ctx: SimContext) -> None:
        # A program belongs to the episode that started it: a reset ends it, and its next call
        # raises RTDEError rather than steering the next episode.
        self._close("reset")
        self._link = _Link()
        self._reset_motion()
        self._tcp_offset = np.zeros(6)

    def shutdown(self, ctx: SimContext) -> None:
        self._close("shutdown")
