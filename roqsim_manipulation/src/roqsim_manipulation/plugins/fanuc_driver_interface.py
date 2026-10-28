# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A FANUC arm under the names FANUC's ROS 2 driver gives it (``fanuc_driver``).

A stack written for a real FANUC arm finds the driver's controllers, topics and services here, with
their types, units and replies, so it runs against the simulated arm unchanged. The reference is
FANUC-CORPORATION/fanuc_driver (Apache-2.0) at the commit ``docs/interfaces.rst`` names; every
behaviour below cites the driver file it reproduces. The arm itself is ``arm_controller`` with
``controller_type: fanuc_controllers/ScaledJointTrajectoryController``, which this plugin feeds:

- **Speed scaling** (``speed_scaling_factor``, ``std_msgs/Int32``, in the robot's namespace): the
  value is clamped to [0, 100] silently and filtered by a first-order lag of 0.2 s stepped every
  control period while the controller is active and motion is possible; trajectory time advances
  at ``0.01 * lag * collaborative_speed_scaling`` (``scaled_joint_trajectory_controller.cpp``). At 0
  the trajectory stops and its goal stays active.
- **The motion gate** (``arm_controller``'s): closed while ``motion_possible`` is 0 -- after
  ``switch_control_state`` STOP, in an alarm (a CRX contact stop), and on a CRX whose payload was
  changed, which ends Stream Motion on the robot. A goal is then aborted with the driver's
  "Aborted: STMO is inactive (!motion_possible)".
- **I/O** from the driver's own ``gpio_topic_config`` (its gpio yaml, inline): ``io_state``,
  ``num_reg_state`` and ``analog_io_state`` publish the blocks it lists every control period;
  ``io_cmd``, ``num_reg_cmd`` and ``analog_io_cmd`` take the values for the blocks it lists and
  ignore any other entry, and those outputs are written every period from the latest command (all
  0 until one arrives), so a service setting such an output is overwritten a period later, as on
  the robot. The analog command's indexing is the driver's as written, which mis-indexes a second
  block (``fanuc_gpio_controller.cpp`` ``on_configure``).
- **Services** with the driver's request and reply, ``result`` 0 or 1: I/O, numeric and position
  registers, payload (``get_payload_*`` divides the centre of gravity by 100 and the inertia by
  10000, as the driver does), UFrame/UTool, the Cartesian position and TCP speed (mm and degrees, in
  FANUC's world frame, the description's ``wbase``), ``read_error``, ``reset``,
  ``switch_control_state``, ``set_gen_override``, ``cfg_force_sensor``.
- **Force sensing.** On a collaborative arm (``collaborative: true``, the CRX) the embedded sensor
  (``fs_type`` 1) reports the external wrench at ``fanuc_flange``, estimated from the contact forces
  on the arm through its Jacobian -- as joint torque sensing sees contact on any link -- plus the
  weight of what is mounted beyond the flange less the declared payload. Above ``contact_stop_force``
  the arm stops in an alarm until ``reset``. An industrial arm has no sensor selected (``fs_type``
  0, zeros) until ``cfg_force_sensor`` selects an external one (2), which reads the contact wrench
  on the tool alone.

Not served, so a stack that relies on it fails at the missing name: ``fanuc_rmi_controller`` and
``rmi_status`` (no RMI session), the hardware plugin's connection parameters.

The values the controller computes that the simulator has no source for are stated, not invented:
the configuration flags of a Cartesian position are 0, an alarm reads ``SIM-001``, and
``speed_clamp_limit`` is 0.

Config::

    fanuc_driver_interface:
      collaborative: false       # true for a CRX: embedded force sensing, contact stop, collaborative
                                 #   speed scaling, payload compensation (the manifest sets it)
      contact_stop_force: 150.0  # N of external force a collaborative arm stops at (FANUC publishes
                                 #   no threshold; the manifest states its assumption)
      prefix: ""                 # the driver's prefix, for the frames it names (<prefix>fanuc_flange)
      gpio_topic_config: {}      # the driver's gpio yaml's block of that name: io_state, io_cmd,
                                 #   analog_io_state, analog_io_cmd, num_reg_state, num_reg_cmd
      io_rate_hz: 0.0            # Hz of the GPIO and force topics (0: every control period, as the
                                 #   driver publishes them)
      status_ext_rate_hz: 30.0   # Hz of robot_status_ext (the driver's 33 ms timer)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import Endpoint, SimContext
from roqsim.controllers import ACTIVE, Controller, registry_for
from roqsim.plugin import Plugin
from roqsim.types import Wrench

from .arm_controller import motion_gate_key, speed_scaling_key, state_interfaces_key

#: ``scaled_joint_trajectory_controller.hpp``: the speed scaling's first-order lag time constant.
SCALING_TAU_S = 0.2
#: The driver's reply when motion is not possible (``scaled_joint_trajectory_controller.cpp``).
STMO_INACTIVE = "Aborted: STMO is inactive (!motion_possible)"
#: ``fanuc_msgs/ForceSensor.fs_type``.
FS_UNSELECTED, FS_EMBEDDED, FS_EXTERNAL = 0, 1, 2
#: ``fanuc_msgs/RobotStatus.contact_stop_mode``.
CONTACT_STOP_NONE, CONTACT_STOP_STOP = 0, 2
#: The I/O types each service takes (``fanuc_gpio_controller.cpp``).
GET_BOOL_TYPES = ("DI", "RI", "DO", "RO", "F", "UO", "UI")
SET_BOOL_TYPES = ("DO", "RO", "F")
#: The override range a FANUC controller takes: 1..100 %, as the teach pendant's override.
GEN_OVERRIDE_RANGE = (1, 100)


def _blocks(config: dict, key: str) -> list[dict]:
    return list((config.get("gpio_topic_config") or {}).get(key) or [])


def _wpr(rot: np.ndarray) -> tuple[float, float, float]:
    """FANUC's W, P, R (deg) of a rotation: fixed-axis X, Y, Z, so ``R = Rz(r) Ry(p) Rx(w)``."""
    p = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    w = math.atan2(rot[2, 1], rot[2, 2])
    r = math.atan2(rot[1, 0], rot[0, 0])
    return math.degrees(w), math.degrees(p), math.degrees(r)


def _rot_wpr(w: float, p: float, r: float) -> np.ndarray:
    w, p, r = (math.radians(v) for v in (w, p, r))
    cw, sw, cp, sp, cr, sr = (
        math.cos(w),
        math.sin(w),
        math.cos(p),
        math.sin(p),
        math.cos(r),
        math.sin(r),
    )
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    return rz @ ry @ rx


def _frame_matrix(frame: dict) -> np.ndarray:
    """A FANUC frame (mm, deg) as a 4x4 transform in metres."""
    t = np.eye(4)
    t[:3, :3] = _rot_wpr(frame.get("w", 0.0), frame.get("p", 0.0), frame.get("r", 0.0))
    t[:3, 3] = np.array([frame.get("x", 0.0), frame.get("y", 0.0), frame.get("z", 0.0)]) / 1000.0
    return t


class FanucDriverInterfacePlugin(Plugin):
    """``fanuc_driver``'s controllers, topics and services for an arm (see the module docstring)."""

    requires_owner = True

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        gpio = config.get("gpio_topic_config")
        if gpio is not None and not isinstance(gpio, dict):
            errors.append("'gpio_topic_config' is the driver's gpio yaml's block of that name")
        for key in ("io_state", "io_cmd", "analog_io_state", "analog_io_cmd"):
            for block in _blocks(config, key):
                if not isinstance(block, dict) or not {"type", "start", "length"} <= set(block):
                    errors.append(f"gpio_topic_config.{key}: each block has type, start and length")
        for key in ("num_reg_state", "num_reg_cmd"):
            for block in _blocks(config, key):
                if not isinstance(block, dict) or not {"start", "length"} <= set(block):
                    errors.append(f"gpio_topic_config.{key}: each block has start and length")
        if float(config.get("contact_stop_force", 150.0)) <= 0:
            errors.append("'contact_stop_force' must be > 0 (N)")
        for key in ("io_rate_hz", "status_ext_rate_hz"):
            if float(config.get(key, 1.0)) < 0:
                errors.append(f"'{key}' must be >= 0 (Hz; 0 is every step)")
        return errors

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.collaborative = bool(self.config.get("collaborative", False))
        self.prefix = str(self.config.get("prefix", ""))
        self.contact_stop_force = float(self.config.get("contact_stop_force", 150.0))
        self.io_rate_hz = float(self.config.get("io_rate_hz", 0.0))  # every control period
        self.status_ext_rate_hz = float(self.config.get("status_ext_rate_hz", 30.0))
        self._ctx: SimContext | None = None
        self._reset_state()

    # -- state ---------------------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._scaling_value = 100  # the topic's last value, clamped
        self._scaling_lag = 100.0  # the filter's state
        self._collab = 1.0
        self._gen_override = 100
        self._bool: dict[tuple[str, int], bool] = {}
        self._analog: dict[tuple[str, int], float] = {}
        self._group: dict[tuple[str, int], int] = {}
        self._numreg: dict[int, float] = {}
        self._posreg: dict[int, dict] = {}
        self._io_cmd: dict[tuple[str, int], bool] = {}
        self._num_cmd: dict[int, float] = {}
        self._analog_cmd_msg: list[float] = [0.0] * len(self._analog_cmd_index)
        self._uframe, self._utool = 0, 0
        self._uframes: dict[int, dict] = {}
        self._utools: dict[int, dict] = {}
        self._payload_id = 1
        self._payload_value: dict[int, dict] = {}
        self._payload_comp: dict[int, dict] = {}
        self._fs_type = FS_EMBEDDED if self.collaborative else FS_UNSELECTED
        self._force = np.zeros(6)
        self._streaming = True  # the hardware's Stream Motion: ends on a CRX payload change
        self._motion_enabled = True  # switch_control_state
        self._in_error = False
        self._alarms: deque[str] = deque(maxlen=5)
        self._error_code = ""
        self._last_tcp: tuple[float, np.ndarray] | None = None
        self._tcp_speed = 0.0

    @property
    def _analog_cmd_index(self) -> list[int]:
        """The driver's ``index_analog_io_cmd_`` as its ``on_configure`` builds it: each block
        writes from position 0 again, so a later block overwrites an earlier one's entries and a
        position past the longest block keeps 0."""
        blocks = _blocks(self.config, "analog_io_cmd")
        index = [0] * sum(int(b["length"]) for b in blocks)
        interface = 0
        for block in blocks:
            for i in range(int(block["length"])):
                index[i] = interface
                interface += 1
        return index

    def motion_possible(self) -> bool:
        return self._streaming and self._motion_enabled and not self._in_error

    def _gate(self) -> str | None:
        return None if self.motion_possible() else STMO_INACTIVE

    def speed_scaling_factor(self) -> float:
        """Seconds of trajectory time per second, what the scaled controller advances at."""
        return 0.01 * self._scaling_lag * self._collab

    # -- lifecycle -----------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.entity)
        mprefix = entity.meta.get("prefix", "") if entity else ""
        self._mprefix = mprefix
        for key, value in (
            (speed_scaling_key(self.entity), self.speed_scaling_factor),
            (motion_gate_key(self.entity), self._gate),
            (state_interfaces_key(self.entity), self.state_interfaces),
        ):
            if ctx.blackboard.get(key) is not None:
                raise RuntimeError(
                    f"fanuc_driver_interface: {key!r} is already registered; an arm has one driver"
                )
            ctx.blackboard.set(key, value)

        def body(name: str) -> int:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mprefix + name)
            if bid < 0:
                raise RuntimeError(f"fanuc_driver_interface: the arm has no body {name!r}")
            return bid

        def site(name: str) -> int:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, mprefix + name)
            if sid < 0:
                raise RuntimeError(f"fanuc_driver_interface: the arm has no site {name!r}")
            return sid

        self._base = body("base_link")
        self._j6 = body("J6_link")
        self._flange = site("fanuc_flange")
        self._wbase = site("wbase")
        self._dofs = np.array(
            [
                int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{mprefix}J{i}")])
                for i in range(1, 7)
            ]
        )
        # The arm's bodies, and of them the tool: everything mounted beyond J6_link.
        self._arm_bodies = {b for b in range(m.nbody) if self._descends(m, b, self._base)}
        self._tool_bodies = {
            b for b in self._arm_bodies if self._descends(m, b, self._j6) and b != self._j6
        }
        self._tool_mass = float(sum(m.body_mass[b] for b in self._tool_bodies))
        self._jac = np.zeros((6, m.nv))

        ns = self.endpoint_namespace(ctx)
        registry = registry_for(ctx)
        gpio_claims = tuple(
            [
                f"AO/{b['start'] + i}"
                for b in _blocks(self.config, "analog_io_cmd")
                for i in range(int(b["length"]))
            ]
            + [
                f"{b['type']}/{b['start'] + i}"
                for b in _blocks(self.config, "io_cmd")
                for i in range(int(b["length"]))
            ]
            + [
                f"FloatReg/{b['start'] + i}"
                for b in _blocks(self.config, "num_reg_cmd")
                for i in range(int(b["length"]))
            ]
        )
        for name, kind, claims, reads in (
            (
                "fanuc_gpio_controller",
                "fanuc_controllers/FanucGPIOController",
                gpio_claims,
                ("ConnectionStatus/is_connected", "ConnectionStatus/motion_command_type"),
            ),
            (
                "fanuc_force_sensor_broadcaster",
                "fanuc_controllers/FanucForceSensorBroadcaster",
                (),
                tuple(
                    f"Force/{f}"
                    for f in (
                        "force_x",
                        "force_y",
                        "force_z",
                        "moment_x",
                        "moment_y",
                        "moment_z",
                        "fs_type",
                    )
                ),
            ),
            (
                "force_torque_sensor_broadcaster",
                "force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster",
                (),
                tuple(f"ft_sensor/{a}.{x}" for a in ("force", "torque") for x in "xyz"),
            ),
        ):
            registry.register(
                Controller(
                    name=name,
                    type=kind,
                    claims=claims,
                    reads=reads,
                    state=ACTIVE,
                    namespace=ns,
                    owner=self.entity,
                )
            )
        self._add_endpoints(ctx, ns)

    @staticmethod
    def _descends(m, body: int, ancestor: int) -> bool:
        while body > 0:
            if body == ancestor:
                return True
            body = int(m.body_parentid[body])
        return ancestor == 0

    def on_reset(self, ctx: SimContext) -> None:
        self._reset_state()

    def pre_step(self, ctx: SimContext) -> None:
        # The scaled controller's update: no scaling while motion is impossible (it returns early),
        # nor while it is inactive.
        jtc = registry_for(ctx).get("joint_trajectory_controller", self.endpoint_namespace(ctx))
        if jtc is not None and "Status/motion_possible" not in jtc.reads:
            # It reads the two status interfaces besides the joints. Here rather than at configure,
            # since a world may declare the arm's controller after this plugin.
            jtc.reads = tuple(jtc.reads) + (
                "Status/collaborative_speed_scaling",
                "Status/motion_possible",
            )
        if self.motion_possible() and (jtc is None or jtc.state == ACTIVE):
            h = float(ctx.model.opt.timestep)
            self._scaling_lag = (1 - h / SCALING_TAU_S) * self._scaling_lag + (
                h / SCALING_TAU_S
            ) * float(self._scaling_value)
        # The GPIO controller writes its command interfaces every period.
        for key, value in self._io_cmd_values().items():
            self._bool[key] = value
        for index, value in self._num_cmd_values().items():
            self._numreg[index] = value
        for interface, value in self._analog_cmd_values().items():
            self._analog[("AO", interface)] = value

    def post_step(self, ctx: SimContext) -> None:
        self._force = self._sense(ctx)
        if (
            self.collaborative
            and not self._in_error
            and np.linalg.norm(self._force[:3]) > self.contact_stop_force
        ):
            self._in_error = True
            self._error_code = "SIM-001"
            self._alarms.appendleft(
                f"SIM-001 Contact stop: external force {np.linalg.norm(self._force[:3]):.0f} N "
                f"above {self.contact_stop_force:.0f} N"
            )
        tcp = self._tcp(ctx)
        if self._last_tcp is not None:
            dt = float(ctx.data.time) - self._last_tcp[0]
            if dt > 0:
                self._tcp_speed = float(np.linalg.norm(tcp[:3, 3] - self._last_tcp[1]) / dt)
        self._last_tcp = (float(ctx.data.time), tcp[:3, 3].copy())

    # -- what is commanded ---------------------------------------------------------------------

    def _io_cmd_values(self) -> dict[tuple[str, int], bool]:
        out = {}
        for block in _blocks(self.config, "io_cmd"):
            for i in range(int(block["length"])):
                key = (str(block["type"]), int(block["start"]) + i)
                out[key] = self._io_cmd.get(key, False)
        return out

    def _num_cmd_values(self) -> dict[int, float]:
        out = {}
        for block in _blocks(self.config, "num_reg_cmd"):
            for i in range(int(block["length"])):
                index = int(block["start"]) + i
                out[index] = self._num_cmd.get(index, 0.0)
        return out

    def _analog_cmd_values(self) -> dict[int, float]:
        """AO index -> value, through the driver's index array (see :attr:`_analog_cmd_index`)."""
        ao = [
            int(b["start"]) + i
            for b in _blocks(self.config, "analog_io_cmd")
            for i in range(int(b["length"]))
        ]
        out = {}
        for position, interface in enumerate(self._analog_cmd_index):
            out[ao[interface]] = self._analog_cmd_msg[position]
        return out

    # -- sensing -------------------------------------------------------------------------------

    def _sense(self, ctx: SimContext) -> np.ndarray:
        """The force sensor's wrench at fanuc_flange, in its frame: (fx, fy, fz, mx, my, mz)."""
        if self._fs_type == FS_UNSELECTED:
            return np.zeros(6)
        m, d = ctx.model, ctx.data
        flange = d.site_xpos[self._flange]
        tau = np.zeros(6)  # the external torque on the arm's joints (embedded)
        wrench = np.zeros(6)  # the wrench on the tool about the flange (external)
        force6 = np.zeros(6)
        jacp = np.zeros((3, m.nv))
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            in1, in2 = b1 in self._arm_bodies, b2 in self._arm_bodies
            if in1 == in2:
                continue  # with the world outside the arm, or the arm's own
            mujoco.mj_contactForce(m, d, i, force6)
            f = c.frame.reshape(3, 3).T @ force6[:3]  # on geom2, in world
            body, force = (b2, f) if in2 else (b1, -f)
            point = c.pos
            mujoco.mj_jac(m, d, jacp, None, point, body)
            tau += jacp[:, self._dofs].T @ force
            if body in self._tool_bodies:
                wrench[:3] += force
                wrench[3:] += np.cross(point - flange, force)
        if self._fs_type == FS_EMBEDDED:
            # What is mounted beyond the flange weighs on the sensor; the declared payload is what
            # the controller takes off. Gravity on the tool, about the flange, through the Jacobian.
            declared = self._payload_value.get(self._payload_id, {}).get("mass", 0.0)
            residual = (self._tool_mass - float(declared)) * m.opt.gravity
            if self._tool_mass > 0:
                com = sum(m.body_mass[b] * d.xipos[b] for b in self._tool_bodies) / self._tool_mass
            else:
                com = flange
            mujoco.mj_jac(m, d, jacp, None, com, self._j6)
            tau += jacp[:, self._dofs].T @ residual
            mujoco.mj_jacSite(m, d, self._jac[:3], self._jac[3:], self._flange)
            j = self._jac[:, self._dofs]
            wrench, *_ = np.linalg.lstsq(j.T, tau, rcond=None)
        rot = d.site_xmat[self._flange].reshape(3, 3)
        return np.concatenate([rot.T @ wrench[:3], rot.T @ wrench[3:]])

    def _tcp(self, ctx: SimContext) -> np.ndarray:
        """The TCP (fanuc_flange with the active UTool) in the active UFrame, as a 4x4 in metres."""
        d = ctx.data

        def pose(site: int) -> np.ndarray:
            t = np.eye(4)
            t[:3, :3] = d.site_xmat[site].reshape(3, 3)
            t[:3, 3] = d.site_xpos[site]
            return t

        world = pose(self._wbase)  # FANUC's world frame: the J1 axis at J2's height
        tool = pose(self._flange) @ _frame_matrix(self._utools.get(self._utool, {}))
        frame = world @ _frame_matrix(self._uframes.get(self._uframe, {}))
        return np.linalg.inv(frame) @ tool

    def state_interfaces(self) -> list[tuple[str, dict[str, float]]]:
        """The hardware's state interfaces beyond the joints, as ``dynamic_joint_states`` lists them."""
        rows: dict[str, dict[str, float]] = {}
        for block in _blocks(self.config, "io_state"):
            for i in range(int(block["length"])):
                key = (str(block["type"]), int(block["start"]) + i)
                rows.setdefault(key[0], {})[str(key[1])] = float(self._bool.get(key, False))
        for block in _blocks(self.config, "num_reg_state"):
            for i in range(int(block["length"])):
                index = int(block["start"]) + i
                rows.setdefault("FloatReg", {})[str(index)] = float(self._numreg.get(index, 0.0))
        status = self._robot_status()
        rows["Status"] = {
            "in_error": float(status["in_error"]),
            "tp_enabled": float(status["tp_enabled"]),
            "e_stopped": float(status["e_stopped"]),
            "motion_possible": float(status["motion_possible"]),
            "contact_stop_mode": float(status["contact_stop_mode"]),
            "collaborative_speed_scaling": self._collab,
        }
        rows["ConnectionStatus"] = {
            "is_connected": float(self._streaming),
            "motion_command_type": 2.0 if self._streaming else 0.0,  # 2: position
        }
        names = ("force_x", "force_y", "force_z", "moment_x", "moment_y", "moment_z")
        rows["Force"] = {
            **dict(zip(names, map(float, self._force), strict=True)),
            "fs_type": float(self._fs_type),
        }
        rows["ft_sensor"] = {
            f"{a}.{x}": float(self._force[3 * k + j])
            for k, a in enumerate(("force", "torque"))
            for j, x in enumerate("xyz")
        }
        return list(rows.items())

    # -- what is published ---------------------------------------------------------------------

    def _robot_status(self) -> dict:
        return {
            "in_error": self._in_error,
            "tp_enabled": False,  # no teach pendant is simulated
            "e_stopped": False,  # no e-stop chain is simulated
            "motion_possible": self.motion_possible(),
            "contact_stop_mode": CONTACT_STOP_STOP if self.collaborative else CONTACT_STOP_NONE,
        }

    def _bool_values(self, key: str) -> list[dict]:
        return [
            {
                "io_type": {"type": str(b["type"])},
                "index": int(b["start"]) + i,
                "value": self._bool.get((str(b["type"]), int(b["start"]) + i), False),
            }
            for b in _blocks(self.config, key)
            for i in range(int(b["length"]))
        ]

    def _add_endpoints(self, ctx: SimContext, ns: str) -> None:
        reliable_1 = {"depth": 1}
        controller = "fanuc_gpio_controller"
        # (name, type, read, rate) of the outputs
        outs = [
            (
                f"{controller}/io_state",
                "IOState",
                lambda: {"values": self._bool_values("io_state")},
                self.io_rate_hz,
            ),
            (
                f"{controller}/analog_io_state",
                "AnalogIOState",
                lambda: {
                    "values": [
                        {
                            "io_type": {"type": str(b["type"])},
                            "index": int(b["start"]) + i,
                            "value": self._analog.get((str(b["type"]), int(b["start"]) + i), 0.0),
                        }
                        for b in _blocks(self.config, "analog_io_state")
                        for i in range(int(b["length"]))
                    ]
                },
                self.io_rate_hz,
            ),
            (
                f"{controller}/num_reg_state",
                "NumRegState",
                lambda: {
                    "values": [
                        {
                            "index": int(b["start"]) + i,
                            "value": self._numreg.get(int(b["start"]) + i, 0.0),
                        }
                        for b in _blocks(self.config, "num_reg_state")
                        for i in range(int(b["length"]))
                    ]
                },
                self.io_rate_hz,
            ),
            (
                f"{controller}/connection_status",
                "ConnectionStatus",
                lambda: {"is_connected": self._streaming},
                self.io_rate_hz,
            ),
            (f"{controller}/robot_status", "RobotStatus", self._robot_status, self.io_rate_hz),
            (
                f"{controller}/collaborative_speed_scaling",
                "CollaborativeSpeedScaling",
                lambda: {"collaborative_speed_scaling": self._collab},
                self.io_rate_hz,
            ),
            (
                f"{controller}/robot_status_ext",
                "RobotStatusExt",
                self._robot_status_ext,
                self.status_ext_rate_hz,
            ),
            (
                "fanuc_force_sensor_broadcaster/force_sensor",
                "ForceSensor",
                self._force_sensor,
                self.io_rate_hz,
            ),
        ]
        for topic, kind, read, rate in outs:
            ctx.interface.add(
                Endpoint(
                    name=topic.replace("/", "."),
                    direction="out",
                    owner=self.entity,
                    namespace=ns,
                    read=read,
                    rate_hz=rate,
                    backend={
                        "ros2": {
                            "type": f"fanuc_msgs.msg.{kind}",
                            "topic": topic,
                            "qos": reliable_1,
                        }
                    },
                )
            )
        # Inputs: commands to the GPIO controller, the speed scaling.
        for topic, kind, write in (
            (f"{controller}/io_cmd", "IOCmd", self._on_io_cmd),
            (f"{controller}/analog_io_cmd", "AnalogIOCmd", self._on_analog_io_cmd),
            (f"{controller}/num_reg_cmd", "NumRegCmd", self._on_num_reg_cmd),
        ):
            ctx.interface.add(
                Endpoint(
                    name=topic.replace("/", "."),
                    direction="in",
                    owner=self.entity,
                    namespace=ns,
                    write=write,
                    backend={
                        "ros2": {
                            "type": f"fanuc_msgs.msg.{kind}",
                            "topic": topic,
                            "qos": reliable_1,
                        }
                    },
                )
            )
        ctx.interface.add(
            Endpoint(
                name="speed_scaling_factor",
                direction="in",
                owner=self.entity,
                namespace=ns,
                write=self.set_speed_scaling_factor,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Int32",
                        "topic": "speed_scaling_factor",
                        "qos": {"depth": 10},
                    }
                },
            )
        )
        services = {
            "get_bool_io": ("GetBoolIO", self.get_bool_io),
            "set_bool_io": ("SetBoolIO", self.set_bool_io),
            "get_analog_io": ("GetAnalogIO", self.get_analog_io),
            "set_analog_io": ("SetAnalogIO", self.set_analog_io),
            "get_group_io": ("GetGroupIO", self.get_group_io),
            "set_group_io": ("SetGroupIO", self.set_group_io),
            "get_num_reg": ("GetNumReg", self.get_num_reg),
            "set_num_reg": ("SetNumReg", self.set_num_reg),
            "get_pos_reg": ("GetPosReg", self.get_pos_reg),
            "set_pos_reg": ("SetPosReg", self.set_pos_reg),
            "set_gen_override": ("SetGenOverride", self.set_gen_override),
            "set_payload_id": ("SetPayloadID", self.set_payload_id),
            "get_payload_id": ("GetPayloadID", self.get_payload_id),
            "set_payload_value": ("SetPayloadValue", self.set_payload_value),
            "get_payload_value": ("GetPayloadValue", self.get_payload_value),
            "set_payload_comp": ("SetPayloadComp", self.set_payload_comp),
            "get_payload_comp": ("GetPayloadComp", self.get_payload_comp),
            "read_error": ("ReadError", self.read_error),
            "get_uframe_utool": ("GetUFrameUTool", self.get_uframe_utool),
            "set_uframe_utool": ("SetUFrameUTool", self.set_uframe_utool),
            "get_uframe_data": ("GetUFrameData", self.get_uframe_data),
            "set_uframe_data": ("SetUFrameData", self.set_uframe_data),
            "get_utool_data": ("GetUToolData", self.get_utool_data),
            "set_utool_data": ("SetUToolData", self.set_utool_data),
            "get_cartesian_position": ("GetCartesianPosition", self.get_cartesian_position),
            "get_tcp_speed": ("GetTCPSpeed", self.get_tcp_speed),
            "switch_control_state": ("SwitchControlState", self.switch_control_state),
            "reset": ("Reset", self.reset),
        }
        for name, (kind, handler) in services.items():
            self._service(ctx, ns, f"{controller}/{name}", kind, handler)
        self._service(
            ctx,
            ns,
            "fanuc_force_sensor_broadcaster/cfg_force_sensor",
            "CfgForceSensor",
            self.cfg_force_sensor,
        )

    def _service(self, ctx, ns, name, kind, handler) -> None:
        ctx.interface.add(
            Endpoint(
                name=name.replace("/", "."),
                direction="in",
                owner=self.entity,
                namespace=ns,
                write=lambda request, h=handler: h(dict(request or {})),
                backend={"ros2": {"service": f"fanuc_msgs.srv.{kind}", "name": name}},
            )
        )

    def _robot_status_ext(self) -> dict:
        moving = bool(np.abs(self._ctx.data.qvel[self._dofs]).max() > 1e-3) if self._ctx else False
        return {
            "error_code": self._error_code,
            "in_motion": moving,
            "drives_powered": True,
            "gen_override": self._gen_override,
            "speed_clamp_limit": 0.0,
            "control_mode": "",
        }

    def _force_sensor(self) -> dict:
        names = ("force_x", "force_y", "force_z", "moment_x", "moment_y", "moment_z")
        return {**dict(zip(names, map(float, self._force), strict=True)), "fs_type": self._fs_type}

    @endpoint.out(
        rate="io_rate_hz",
        ros2=lambda self: {
            "type": "geometry_msgs.msg.WrenchStamped",
            "topic": "force_torque_sensor_broadcaster/wrench",
            "frame_id": f"{self.prefix}fanuc_flange",
        },
    )
    def ft_wrench(self) -> Wrench:
        """``force_torque_sensor_broadcaster``'s wrench: the force sensor's (the hardware's
        ``ft_sensor/*`` and ``Force/*`` are one store), in ``<prefix>fanuc_flange``."""
        return Wrench(force=tuple(self._force[:3]), torque=tuple(self._force[3:]))

    # -- inputs --------------------------------------------------------------------------------

    def set_speed_scaling_factor(self, value: Any) -> None:
        """``speed_scaling_factor``: clamped to [0, 100], silently (the driver's ``std::clamp``)."""
        self._scaling_value = min(max(int(value), 0), 100)

    def set_collaborative_speed_scaling(self, value: float) -> None:
        """The CRX's collaborative speed scaling, which a safety function sets on the robot.

        A roqsim-side input (a scenario's, a separation monitor's), not a driver name; an
        industrial arm has none and refuses it."""
        if not self.collaborative:
            raise RuntimeError("an industrial arm has no collaborative speed scaling")
        if not math.isfinite(value):
            return  # the driver keeps the last finite value
        self._collab = float(value)

    def set_input(self, io_type: str, index: int, value: bool) -> None:
        """Drive an input (DI, RI, UI) from the simulation: what the cell's wiring does on the robot.

        A roqsim-side input, not a driver name."""
        if io_type not in ("DI", "RI", "UI", "WI", "SI"):
            raise ValueError(f"{io_type} is not an input")
        self._bool[(io_type, int(index))] = bool(value)

    def _on_io_cmd(self, values) -> None:
        configured = set(self._io_cmd_values())
        for v in values or ():
            key = (str(v["io_type"]["type"]), int(v["index"]))
            if key in configured:  # anything else is ignored, as the driver ignores it
                self._io_cmd[key] = bool(v["value"])

    def _on_num_reg_cmd(self, values) -> None:
        configured = set(self._num_cmd_values())
        for v in values or ():
            if int(v["index"]) in configured:
                self._num_cmd[int(v["index"])] = float(v["value"])

    def _on_analog_io_cmd(self, values) -> None:
        ao = [
            int(b["start"]) + i
            for b in _blocks(self.config, "analog_io_cmd")
            for i in range(int(b["length"]))
        ]
        for v in values or ():
            key = (str(v["io_type"]["type"]), int(v["index"]))
            if key[0] == "AO" and key[1] in ao:
                self._analog_cmd_msg[ao.index(key[1])] = float(v["value"])

    # -- services (fanuc_gpio_controller.cpp) --------------------------------------------------

    def get_bool_io(self, r: dict) -> dict:
        kind = r["io_type"]["type"]
        if kind not in GET_BOOL_TYPES:
            return {"result": 1}
        return {"result": 0, "value": self._bool.get((kind, int(r["index"])), False)}

    def set_bool_io(self, r: dict) -> dict:
        kind = r["io_type"]["type"]
        if kind not in SET_BOOL_TYPES:
            return {"result": 1}
        self._bool[(kind, int(r["index"]))] = bool(r["value"])
        return {"result": 0}

    def get_analog_io(self, r: dict) -> dict:
        kind = r["io_type"]["type"]
        if kind not in ("AI", "AO"):
            return {"result": 1}
        return {"result": 0, "value": self._analog.get((kind, int(r["index"])), 0.0)}

    def set_analog_io(self, r: dict) -> dict:
        if r["io_type"]["type"] != "AO":
            return {"result": 1}
        self._analog[("AO", int(r["index"]))] = float(r["value"])
        return {"result": 0}

    def get_group_io(self, r: dict) -> dict:
        kind = r["io_type"]["type"]
        if kind not in ("GI", "GO"):
            return {"result": 1}
        return {"result": 0, "value": self._group.get((kind, int(r["index"])), 0)}

    def set_group_io(self, r: dict) -> dict:
        if r["io_type"]["type"] != "GO":
            return {"result": 1}
        self._group[("GO", int(r["index"]))] = int(r["value"]) & 0xFFFF
        return {"result": 0}

    def get_num_reg(self, r: dict) -> dict:
        return {"result": 0, "value": self._numreg.get(int(r["index"]), 0.0)}

    def set_num_reg(self, r: dict) -> dict:
        self._numreg[int(r["index"])] = float(r["value"])
        return {"result": 0}

    def get_pos_reg(self, r: dict) -> dict:
        stored = self._posreg.get(int(r["index"]), {"representation": "Cartesian"})
        return {"result": 0, **stored}

    def set_pos_reg(self, r: dict) -> dict:
        stored = {k: v for k, v in r.items() if k != "index"}
        for flag in ("front", "up", "left", "flip"):
            stored[flag] = min(int(stored.get(flag, 0)), 1)  # the driver clamps them to 1
        self._posreg[int(r["index"])] = stored
        return {"result": 0}

    def set_gen_override(self, r: dict) -> dict:
        """The controller's general override: stored and reported, retiming nothing (FANUC: it
        "needs to stay at 100 when the hardware interface is in active state")."""
        value = int(r["value"])
        low, high = GEN_OVERRIDE_RANGE
        if not low <= value <= high:
            self._alarms.appendleft(f"SIM-002 Invalid Speed Override: {value}")
            return {"result": 1}
        self._gen_override = value
        return {"result": 0}

    def set_payload_id(self, r: dict) -> dict:
        self._payload_id = int(r["payload_schedule_id"])
        return {"result": 0}

    def get_payload_id(self, r: dict) -> dict:
        return {"result": 0, "payload_schedule_id": self._payload_id}

    def set_payload_value(self, r: dict) -> dict:
        self._payload_value[int(r["payload_schedule_id"])] = {
            k: float(r.get(k, 0.0))
            for k in ("mass", "cg_x", "cg_y", "cg_z", "in_x", "in_y", "in_z")
        }
        if self.collaborative:
            # A CRX takes a payload change through its DCS, which ends Stream Motion.
            self._streaming = False
        return {"result": 0}

    def get_payload_value(self, r: dict) -> dict:
        return {"result": 0, **self._scaled_payload(self._payload_value, r)}

    def set_payload_comp(self, r: dict) -> dict:
        if not self.collaborative:
            return {"result": 1}  # collaborative robots only
        self._payload_comp[int(r["payload_schedule_id"])] = {
            k: float(r.get(k, 0.0))
            for k in ("mass", "cg_x", "cg_y", "cg_z", "in_x", "in_y", "in_z")
        }
        return {"result": 0}

    def get_payload_comp(self, r: dict) -> dict:
        if not self.collaborative:
            return {"result": 1}
        return {"result": 0, **self._scaled_payload(self._payload_comp, r)}

    @staticmethod
    def _scaled_payload(store: dict, r: dict) -> dict:
        """As the driver returns it: the centre of gravity / 100, the inertia / 10000."""
        p = store.get(int(r["payload_schedule_id"]), {})
        out = {"mass": p.get("mass", 0.0)}
        out.update({k: p.get(k, 0.0) / 100 for k in ("cg_x", "cg_y", "cg_z")})
        out.update({k: p.get(k, 0.0) / 10000 for k in ("in_x", "in_y", "in_z")})
        return out

    def read_error(self, r: dict) -> dict:
        count = int(r.get("count", 1))
        if not 1 <= count <= 5:
            return {"result": 1}
        alarms = list(self._alarms)[:count]
        return {"result": 0, "count": len(alarms), "error_data": alarms + [""] * (5 - len(alarms))}

    @staticmethod
    def _group_of(r: dict) -> int:
        return int(r.get("group", 0)) or 1

    def get_uframe_utool(self, r: dict) -> dict:
        return {
            "result": 0,
            "uframe_number": self._uframe,
            "utool_number": self._utool,
            "group": self._group_of(r),
        }

    def set_uframe_utool(self, r: dict) -> dict:
        self._uframe, self._utool = int(r["uframe_number"]), int(r["utool_number"])
        return {"result": 0, "group": self._group_of(r)}

    def get_uframe_data(self, r: dict) -> dict:
        n = int(r["uframe_number"])
        return {
            "result": 0,
            "uframe_number": n,
            "frame": dict(self._uframes.get(n, {})),
            "group": self._group_of(r),
        }

    def set_uframe_data(self, r: dict) -> dict:
        self._uframes[int(r["uframe_number"])] = dict(r["frame"])
        return {"result": 0, "group": self._group_of(r)}

    def get_utool_data(self, r: dict) -> dict:
        n = int(r["utool_number"])
        return {
            "result": 0,
            "utool_number": n,
            "frame": dict(self._utools.get(n, {})),
            "group": self._group_of(r),
        }

    def set_utool_data(self, r: dict) -> dict:
        self._utools[int(r["utool_number"])] = dict(r["frame"])
        return {"result": 0, "group": self._group_of(r)}

    def get_cartesian_position(self, r: dict) -> dict:
        """The TCP in the active UFrame, mm and deg; the configuration flags are not computed (0)."""
        t = self._tcp(self._ctx)
        w, p, rr = _wpr(t[:3, :3])
        x, y, z = (float(v) * 1000.0 for v in t[:3, 3])
        q = self._ctx.data.qpos[
            [
                int(
                    self._ctx.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self._ctx.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._mprefix}J{i}"
                        )
                    ]
                )
                for i in (4, 5, 6)
            ]
        ]
        turns = [int(math.floor((math.degrees(v) + 180.0) / 360.0)) for v in q]
        return {
            "result": 0,
            "timetag": int(self._ctx.data.time * 1000),
            "utool": self._utool,
            "uframe": self._uframe,
            "front": 0,
            "up": 0,
            "left": 0,
            "flip": 0,
            "turn4": turns[0],
            "turn5": turns[1],
            "turn6": turns[2],
            "x": x,
            "y": y,
            "z": z,
            "w": w,
            "p": p,
            "r": rr,
            "group": self._group_of(r),
        }

    def get_tcp_speed(self, r: dict) -> dict:
        return {
            "result": 0,
            "timetag": int(self._ctx.data.time * 1000),
            "speed": self._tcp_speed * 1000.0,
            "group": self._group_of(r),
        }

    def switch_control_state(self, r: dict) -> dict:
        status = int(r["status"])
        if status == 1:  # START
            self._motion_enabled = True
            return {"result": 0}
        if status == 0:  # STOP
            self._motion_enabled = False
            return {"result": 0}
        return {"result": -1}

    def reset(self, r: dict) -> dict:  # noqa: ARG002
        self._in_error = False
        self._error_code = ""
        return {"result": 0}

    def cfg_force_sensor(self, r: dict) -> dict:
        """Select the sensor; no validation, ``result`` 0 always, as the driver answers."""
        if int(r.get("do_reset", 0)):
            self._force = np.zeros(6)
        fs = int(r.get("fs_type", self._fs_type))
        if fs in (FS_UNSELECTED, FS_EMBEDDED, FS_EXTERNAL):
            # An industrial arm has no embedded sensor to select; the robot ignores a type it
            # cannot take, as FANUC's doc says it ignores an invalid one.
            if fs != FS_EMBEDDED or self.collaborative:
                self._fs_type = fs
        return {"result": 0}
