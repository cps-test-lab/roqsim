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

"""Observation plugin: what a robot's actuators cost it, integrated over the run.

"Energy per metre", "how far on a charge" and "which planner is cheaper" are ordinary conclusions in
the mobile-robotics literature, and nothing here could produce the number they rest on. Reconstructed
afterwards from a recording it is worse in the two ways :mod:`roqsim.plugins.clearance_monitor`'s
docstring already argues about clearance: the integrand is sampled at the recording's rate rather
than the physics rate, and it needs a model of the drivetrain to turn poses back into effort, which
puts a fitted constant between the simulator and the result. MuJoCo already computes the actuator
force and the velocity it acts through; their product is mechanical power, exactly, every step.

**What is measured, and what is assumed.** The measured part is mechanical: ``sum(|force *
velocity|)`` over the actuators that move this robot. Everything between that and a battery current
is an assumption an experiment has to state, so each is config with a documented default that
changes nothing:

* ``efficiency`` (default ``1.0``) -- drivetrain and driver losses. Electrical draw is mechanical
  power divided by it.
* ``idle_w`` (default ``0.0``) -- what the robot draws standing still: compute, sensors, brakes.
  On a real platform this dominates a slow trial, and it is a per-platform datasheet number.
* ``regenerative`` (default ``false``) -- whether braking returns energy. False clamps negative
  mechanical power to zero, which is what a robot without regenerative drive does; true integrates
  it as a credit.

Defaults that model nothing are deliberate. A plausible efficiency curve shipped as a default would
silently change every energy figure a campaign reported, and no reader would know which paper's robot
it came from.

**A capacity is optional, and the state of charge only exists with one.** Given ``capacity_wh``, the
plugin reports the fraction remaining and latches ``depleted`` when it reaches zero. It does **not**
stop the robot: that is trial logic, and a substrate that decides when a run ends has taken the
experiment's decision (the same line ``contact_monitor`` draws about a collision). A scenario reads
the endpoint and ends the trial itself.

Config::

    energy_monitor:
      # The entity is the one this entry is NESTED UNDER (`requires_owner`): a battery belongs to a
      # robot, and which actuators count is decided by which ones move it.
      actuators: []            # names to meter (default: every actuator driving this entity's bodies)
      efficiency: 1.0          # mechanical -> electrical; 0 < e <= 1
      idle_w: 0.0              # W drawn regardless of motion (compute, sensors)
      regenerative: false      # credit negative mechanical power back
      capacity_wh: 0.0         # 0 = no battery modelled: energy is still reported, charge is not
      voltage: 0.0             # V, nominal; 0 = unknown, and the current is then not reported
      rate_hz: 5.0             # endpoint publish rate

Endpoint ``battery`` (out) reads an :class:`EnergyReport` and carries a ``sensor_msgs/BatteryState``
hint on ``battery_state`` -- the message a real platform publishes, so a stack that already watches a
battery needs no change. An :class:`EnergyReader` is published on the blackboard under
``energy:<address>`` for an in-process consumer, and the report carries the raw joules as well as the
derived state of charge, because the metric a paper quotes is usually the integral, not the fraction.

**The integral is accumulated on the physics thread, every step**, not on read: a rate-limited or
subscriber-gated sample would silently integrate a different signal depending on who was listening.
It is integrated against elapsed *sim* time rather than a fixed ``dt`` so a replay over recorded
samples (see :mod:`roqsim.recording`) accumulates the same way, at its own spacing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

#: Joules per watt-hour, so a datasheet number (Wh) and the integral (J) can be one quantity.
JOULES_PER_WH = 3600.0


@dataclass
class EnergyReport:
    """Neutral payload for the ``battery`` endpoint: the integral, the rate, and the charge left.

    ``charge_fraction`` and ``depleted`` are meaningful only when a ``capacity_wh`` was configured;
    without one ``charge_fraction`` is ``-1.0``, the "unknown" convention ``sensor_msgs/BatteryState``
    uses for a value a device cannot report, rather than a plausible-looking 1.0.
    """

    energy_j: float = 0.0
    power_w: float = 0.0
    mechanical_w: float = 0.0
    charge_fraction: float = -1.0
    depleted: bool = False
    voltage: float = 0.0
    current_a: float = 0.0
    capacity_wh: float = 0.0


@dataclass
class EnergyReader:
    """Blackboard handle published under ``energy:<address>``; read on the physics thread."""

    name: str
    read: Callable[[], EnergyReport]


class EnergyMonitorPlugin(Plugin):
    """See the module docstring."""

    parallel_safe = False  # post_step accumulates state

    #: A battery belongs to the robot it powers.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.actuator_names = list(self.config.get("actuators") or [])
        self.efficiency = float(self.config.get("efficiency", 1.0))
        self.idle_w = float(self.config.get("idle_w", 0.0))
        self.regenerative = bool(self.config.get("regenerative", False))
        self.capacity_wh = float(self.config.get("capacity_wh", 0.0))
        self.voltage = float(self.config.get("voltage", 0.0))
        self.rate_hz = float(self.config.get("rate_hz", 5.0))
        self._ctx: SimContext | None = None
        self._actuators: np.ndarray | None = None
        self._energy_j = 0.0
        self._power_w = 0.0
        self._mech_w = 0.0
        self._depleted = False
        self._last_time = 0.0

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        efficiency = float(config.get("efficiency", 1.0))
        if not 0.0 < efficiency <= 1.0:
            errors.append("'efficiency' must be in (0, 1] -- it divides the mechanical power")
        for key in ("idle_w", "capacity_wh", "voltage"):
            if float(config.get(key, 0.0)) < 0:
                errors.append(f"'{key}' must be >= 0")
        if float(config.get("rate_hz", 5.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if config.get("actuators") is not None and not isinstance(config["actuators"], list):
            errors.append("'actuators' must be a list of actuator names")
        return errors

    # -- lifecycle ----------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        self._actuators = (
            self._named_actuators(m, prefix)
            if self.actuator_names
            else self._actuators_of(m, entity, prefix)
        )
        if self._actuators.size == 0:
            # A meter reading zero forever looks exactly like a robot that costs nothing to drive.
            raise RuntimeError(
                f"energy_monitor[{self.label}]: no actuators to meter for entity "
                f"{self.robot!r}. Name them with 'actuators:', or check that this entry is nested "
                f"under the spawn that owns them."
            )

        ctx.blackboard.set(f"energy:{self.address}", EnergyReader(name=self.label, read=self.read))
        ctx.interface.add(
            Endpoint(
                name="battery",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.BatteryState",
                        "topic": self.topic_override("battery") or "battery_state",
                        "frame_id": entity.body if entity and entity.body else "base_link",
                    }
                },
            )
        )

    def _named_actuators(self, m, prefix: str) -> np.ndarray:
        """The actuators a world named explicitly, prefixed like every other name it gives."""
        ids = []
        for name in self.actuator_names:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
            if aid < 0:
                raise RuntimeError(
                    f"energy_monitor[{self.label}]: actuator {prefix + name!r} not found"
                )
            ids.append(aid)
        return np.asarray(sorted(set(ids)), dtype=int)

    def _actuators_of(self, m, entity, prefix: str) -> np.ndarray:
        """Every actuator that moves a body of this entity's kinematic subtree.

        Derived rather than configured, because "which motors are on this robot" is a fact about the
        model and a world that had to list them would get it wrong the first time a model gained a
        joint. The subtree is the same notion ``contact_monitor`` watches: a robot is its base and
        everything descended from it.
        """
        root = -1
        if entity is not None and entity.body:
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if root < 0:
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")
        if root < 0:
            raise RuntimeError(
                f"energy_monitor[{self.label}]: entity {self.robot!r} registered no base body, so "
                f"the actuators that drive it cannot be found. Name them with 'actuators:'."
            )
        subtree = {root}
        for body in range(root + 1, m.nbody):
            if int(m.body_parentid[body]) in subtree:
                subtree.add(body)

        ids = []
        for aid in range(m.nu):
            body = self._actuator_body(m, aid)
            if body in subtree:
                ids.append(aid)
        return np.asarray(ids, dtype=int)

    @staticmethod
    def _actuator_body(m, aid: int) -> int:
        """The body an actuator acts on, whatever it is attached to.

        MuJoCo's transmissions do not share one id space -- ``actuator_trnid`` is a joint for a
        joint/jointinparent motor, a tendon for a tendon drive, a body for an adhesion or body
        transmission, a site for a general one -- so the id is read through its ``trntype``. Guessing
        it is a joint id (the common case) silently attributes a tendon-driven gripper's power to
        whichever body happens to hold that joint number.
        """
        trntype = int(m.actuator_trntype[aid])
        trnid = int(m.actuator_trnid[aid, 0])
        if trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            return int(m.jnt_bodyid[trnid])
        if trntype == mujoco.mjtTrn.mjTRN_SITE:
            return int(m.site_bodyid[trnid])
        if trntype == mujoco.mjtTrn.mjTRN_BODY:
            return trnid
        if trntype == mujoco.mjtTrn.mjTRN_TENDON:
            # A tendon spans bodies; attribute it to the body its first wrapping point sits on.
            adr = int(m.tendon_adr[trnid])
            wrap_objid = int(m.wrap_objid[adr])
            wrap_type = int(m.wrap_type[adr])
            if wrap_type == mujoco.mjtWrap.mjWRAP_JOINT:
                return int(m.jnt_bodyid[wrap_objid])
            if wrap_type == mujoco.mjtWrap.mjWRAP_SITE:
                return int(m.site_bodyid[wrap_objid])
        return -1

    def on_reset(self, ctx: SimContext) -> None:
        # A trial starts on a full battery: without this, trial 2 of one process inherits trial 1's
        # consumption and the second cell of a campaign reports a robot that started half-empty.
        self._energy_j = 0.0
        self._power_w = 0.0
        self._mech_w = 0.0
        self._depleted = False
        self._last_time = ctx.sim_time

    # -- the integral -------------------------------------------------------------------------

    def post_step(self, ctx: SimContext) -> None:
        dt = ctx.sim_time - self._last_time
        self._last_time = ctx.sim_time
        if dt <= 0.0:
            return
        d = ctx.data
        # Mechanical power, exactly: force through the velocity it acts at, per actuator.
        mech = float(
            np.dot(d.actuator_force[self._actuators], d.actuator_velocity[self._actuators])
        )
        if not self.regenerative:
            # A robot without regenerative drive pays for braking too; it does not get paid for it.
            mech = abs(mech)
        self._mech_w = mech
        self._power_w = mech / self.efficiency + self.idle_w
        self._energy_j += self._power_w * dt
        if self.capacity_wh and self._energy_j >= self.capacity_wh * JOULES_PER_WH:
            # Latched, like contact_monitor's verdict: a battery that reports itself empty and then
            # full again on the next downhill metre is not a fact a trial can act on.
            self._depleted = True

    def read(self) -> EnergyReport:
        """The report as it stands. Runs on the physics thread."""
        capacity_j = self.capacity_wh * JOULES_PER_WH
        fraction = -1.0
        if capacity_j > 0.0:
            fraction = max(0.0, min(1.0, 1.0 - self._energy_j / capacity_j))
        return EnergyReport(
            energy_j=self._energy_j,
            power_w=self._power_w,
            mechanical_w=self._mech_w,
            charge_fraction=fraction,
            depleted=self._depleted,
            voltage=self.voltage,
            # Only where a nominal voltage was stated: current is power/voltage, and inventing a
            # voltage to be able to report a current would put a made-up number on a real field.
            current_a=(self._power_w / self.voltage) if self.voltage > 0.0 else 0.0,
            capacity_wh=self.capacity_wh,
        )
