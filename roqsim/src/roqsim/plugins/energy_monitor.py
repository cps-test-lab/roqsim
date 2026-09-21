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

**What is measured, and what is assumed.** The measured part is mechanical and is evaluated **per
actuator**: ``force * velocity`` for each of the actuators that move this robot, where the force is
the one a real drive would supply -- the actuator's own force **plus its share of the
gravity-compensation force**. MuJoCo's ``body_gravcomp`` carries a compensated arm's weight outside
the actuator, so ``actuator_force`` reads exactly zero on a joint holding a payload against gravity;
metering it alone reports an arm that costs nothing to hold a load up, and nothing to lift one. A
real drive supplies that torque, which is why ``arm_controller`` already reports
``qfrc_actuator + qfrc_gravcomp`` as a joint's effort. Where nothing is compensated -- ``control:
effort``, whose controller supplies the gravity term itself -- the share is zero and nothing
changes. The sum is taken
after the per-actuator split, never before -- on an arm, one joint descending while another lifts is
the ordinary case rather than an edge case, and a net taken first lets the descent pay for the lift
and reports a pose change as free. Everything between that measurement and a battery current is an
assumption an experiment has to state, so each is config with a documented default that changes
nothing:

* ``efficiency`` (default ``1.0``) -- drivetrain and driver losses. Positive mechanical power is
  divided by it to get the draw; a regenerative credit is multiplied by it, because a recovered
  joule crosses the same losses on its way back into the pack.
* ``idle_w`` (default ``0.0``) -- what the robot draws regardless of motion: compute, sensors,
  brakes. On a real platform this dominates a slow trial, and it is a per-platform datasheet number.
* ``resistive_w_per_nm2`` (default ``0.0``) -- winding loss, the ``k`` in ``k * tau^2``, summed over
  the actuators. A motor torque is a motor current, so this is the term that survives a standstill:
  an arm holding a payload against gravity has exactly zero mechanical power and still dissipates
  ``I^2 R``, and on a manipulator that term is often the larger part of a slow trial's bill. A number
  applies to every metered actuator; a mapping of actuator name to coefficient gives each its own,
  for a machine whose motors are not one class, and an actuator the mapping omits contributes
  nothing. The unit follows MuJoCo's actuator space -- W per (N*m)^2 for the rotary transmissions
  that it almost always is, W per N^2 for a linear one.
* ``regenerative`` (default ``false``) -- whether braking returns energy. False drops negative
  mechanical power, which is what a robot without regenerative drive does: the load's kinetic energy
  is dissipated on the way out, not drawn from the pack, and billing the pack for it would charge an
  experiment for joules the pack never supplied. True integrates it as a credit.

Dropping negative mechanical power rather than taking its magnitude is what keeps the two loss terms
from counting the same joule twice. What a braking or a lowering motor genuinely pays for is
dissipation in its windings, and that arrives through ``resistive_w_per_nm2`` -- once, and scaled by
the torque that causes it.

Defaults that model nothing are deliberate. A plausible efficiency curve shipped as a default would
silently change every energy figure a campaign reported, and no reader would know which paper's robot
it came from.

**A capacity is optional, and the state of charge only exists with one.** Given ``capacity_wh``, the
plugin reports the fraction remaining and latches ``depleted`` when it reaches zero. It does **not**
stop the robot: that is trial logic, and a substrate that decides when a run ends has taken the
experiment's decision (the same line ``contact_monitor`` draws about a collision). A scenario reads
the endpoint and ends the trial itself.

Every key is declared in :attr:`EnergyMonitorPlugin.CONFIG_SCHEMA`, with its default, unit and
bound, which is what ``roqsim plugins describe energy_monitor`` and the plugin catalog publish. The
entity is the one this entry is nested under (``requires_owner``): a battery belongs to a robot, and
which actuators count is decided by which ones move it.

Endpoint ``battery`` (out) reads an :class:`EnergyReport` and carries a ``sensor_msgs/BatteryState``
hint on ``battery_state`` -- the message a real platform publishes, so a stack that already watches a
battery needs no change. An :class:`EnergyReader` is published on the blackboard under
``energy:<address>`` for an in-process consumer, and the report carries the raw joules as well as the
derived state of charge, because the metric a paper quotes is usually the integral, not the fraction.
For the same reason it carries ``torque_integral_nms``, the integral of the summed absolute actuator
forces: where a platform's electrical constants are not published, that effort integral is the metric
a paper falls back on, and accumulated here it is the physics-rate quantity rather than a sum over
whatever rate ``/joint_states`` happened to be published at.

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
from roqsim.schema import Field

#: Joules per watt-hour, so a datasheet number (Wh) and the integral (J) can be one quantity.
JOULES_PER_WH = 3600.0


@dataclass
class EnergyReport:
    """Neutral payload for the ``battery`` endpoint: the integral, the rate, and the charge left.

    ``charge_fraction`` and ``depleted`` are meaningful only when a ``capacity_wh`` was configured;
    without one ``charge_fraction`` is ``-1.0``, the "unknown" convention ``sensor_msgs/BatteryState``
    uses for a value a device cannot report, rather than a plausible-looking 1.0.

    ``mechanical_w`` is the measurement alone and is **signed**: the net of what the actuators deliver
    and what is delivered back into them, before any assumption in this plugin is applied. It can be
    negative on a machine whose load is driving it. ``power_w`` is what reaches the pack, and
    ``resistive_w`` is the part of it that is winding loss, reported separately so a trial can say how
    much of its bill was holding rather than moving.
    """

    energy_j: float = 0.0
    power_w: float = 0.0
    mechanical_w: float = 0.0
    resistive_w: float = 0.0
    torque_integral_nms: float = 0.0
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
    parallel_safe = False  # post_step accumulates state

    #: A battery belongs to the robot it powers.
    requires_owner = True

    #: Every key this plugin reads, besides the transport ones every component may carry
    #: (:data:`roqsim.schema.INJECTED_KEYS`) -- which is what makes refusing any other key safe. A misspelt
    #: coefficient would otherwise meter the robot at the default that models nothing, and report an
    #: energy figure that looks measured.
    CONFIG_SCHEMA = {
        "actuators": Field(
            list, default=[], doc="names to meter (default: every actuator driving this entity)"
        ),
        "efficiency": Field(
            float, default=1.0, maximum=1.0, doc="mechanical -> electrical, in (0, 1]"
        ),
        "idle_w": Field(
            float, default=0.0, minimum=0.0, unit="W", doc="drawn regardless of motion"
        ),
        "resistive_w_per_nm2": Field(
            (float, dict),
            default=0.0,
            minimum=0.0,
            unit="W/(N*m)^2",
            doc="winding loss k in k*tau^2; a number, or {actuator_name: k}",
        ),
        "regenerative": Field(bool, default=False, doc="credit negative mechanical power back"),
        "capacity_wh": Field(
            float, default=0.0, minimum=0.0, unit="Wh", doc="0: no battery modelled, no charge"
        ),
        "voltage": Field(
            float, default=0.0, minimum=0.0, unit="V", doc="nominal; 0: unknown, no current"
        ),
        "rate_hz": Field(float, default=5.0, unit="Hz", doc="endpoint publish rate, > 0"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        # Copied once, because post_step reads them every physics step. Taken from the settings as
        # they are rather than converted here: a value of the wrong type is the schema check's to
        # report, by name, and a float() here would raise first and report nothing.
        settings = self.settings
        self.actuator_names = settings.actuators
        self.efficiency = settings.efficiency
        self.idle_w = settings.idle_w
        self.resistive = settings.resistive_w_per_nm2
        self.regenerative = settings.regenerative
        self.capacity_wh = settings.capacity_wh
        self.voltage = settings.voltage
        self.rate_hz = settings.rate_hz
        self._ctx: SimContext | None = None
        self._actuators: np.ndarray | None = None
        self._resistive_k: np.ndarray | None = None
        self._energy_j = 0.0
        self._power_w = 0.0
        self._mech_w = 0.0
        self._resistive_w = 0.0
        self._torque_integral = 0.0
        self._depleted = False
        self._last_time = 0.0

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        # Types, defaults and the closed bounds are the schema's; what is left is what it has no word
        # for: two open bounds, and the entries of a per-actuator mapping.
        errors = self.validate_topics(config)
        settings = self.settings_for(config)
        if isinstance(settings.efficiency, float) and settings.efficiency <= 0.0:
            errors.append("'efficiency' must be > 0 -- it divides the mechanical power")
        if isinstance(settings.rate_hz, float) and settings.rate_hz <= 0:
            errors.append("'rate_hz' must be > 0")
        errors.extend(self._resistive_errors(settings.resistive_w_per_nm2))
        return errors

    @staticmethod
    def _resistive_errors(spec) -> list[str]:
        """Each entry of a per-actuator ``resistive_w_per_nm2`` is a number, and never negative.

        The schema checks the value's shape and a single coefficient's bound; a mapping's entries are
        left to this. A negative coefficient is a motor that is paid to produce torque, so it is
        refused here rather than left to show up as an energy figure that falls while the arm works.
        """
        if not isinstance(spec, dict):
            return []
        key = "'resistive_w_per_nm2'"
        for actuator, value in spec.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return [f"{key}[{actuator!r}] must be a number, got {value!r}"]
            if value < 0:
                return [f"{key}[{actuator!r}] must be >= 0, got {value}"]
        return []

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

        self._resistive_k = self._resistive_coefficients(m, prefix)

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

    def _resistive_coefficients(self, m, prefix: str) -> np.ndarray:
        """``k`` per metered actuator, aligned with :attr:`_actuators`.

        A number covers a machine whose motors are one class. The mapping is for one that is not: an
        arm's shoulder and its wrist carry different motors, and a single coefficient would have to be
        wrong for one of them. Keys are prefixed like every other name a world gives, and a key that
        names an actuator this monitor does not meter is an error -- silently ignored it would read as
        a wrist that costs nothing to hold.
        """
        if not isinstance(self.resistive, dict):
            return np.full(self._actuators.shape, float(self.resistive), dtype=float)
        metered = {int(aid): i for i, aid in enumerate(self._actuators)}
        coefficients = np.zeros(self._actuators.shape, dtype=float)
        for name, value in self.resistive.items():
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
            if aid not in metered:
                raise RuntimeError(
                    f"energy_monitor[{self.label}]: 'resistive_w_per_nm2' names actuator "
                    f"{prefix + name!r}, which this monitor does not meter."
                )
            coefficients[metered[aid]] = float(value)
        return coefficients

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

    def _gravcomp_share(self, d) -> np.ndarray:
        """Each metered actuator's share of the gravity-compensation force, in actuator space.

        ``qfrc_gravcomp`` is a DOF-space force, so it is projected onto each actuator's transmission
        row: for the ordinary joint drive that is a division by the gear, and in general it is the
        least-squares share, so ``moment^T @ share`` reproduces the DOF-space force it came from.
        The row is read from ``mjData`` every step rather than cached, because a transmission's
        moment is a function of the configuration.
        """
        moment = d.actuator_moment
        share = np.zeros(self._actuators.shape, dtype=float)
        for i, aid in enumerate(self._actuators):
            nnz = int(d.moment_rownnz[aid])
            if nnz == 0:
                continue
            adr = int(d.moment_rowadr[aid])
            cols = d.moment_colind[adr : adr + nnz]
            vals = moment[adr : adr + nnz]
            denominator = float(vals @ vals)
            if denominator > 0.0:
                share[i] = float(vals @ d.qfrc_gravcomp[cols]) / denominator
        return share

    def on_reset(self, ctx: SimContext) -> None:
        # A trial starts on a full battery: without this, trial 2 of one process inherits trial 1's
        # consumption and the second cell of a campaign reports a robot that started half-empty.
        self._energy_j = 0.0
        self._power_w = 0.0
        self._mech_w = 0.0
        self._resistive_w = 0.0
        self._torque_integral = 0.0
        self._depleted = False
        self._last_time = ctx.sim_time

    # -- the integral -------------------------------------------------------------------------

    def post_step(self, ctx: SimContext) -> None:
        dt = ctx.sim_time - self._last_time
        self._last_time = ctx.sim_time
        if dt <= 0.0:
            return
        d = ctx.data
        # Mechanical power, exactly: force through the velocity it acts at, per actuator. The split
        # into driving and driven happens BEFORE the sum -- netting first would let one joint's
        # descent pay for another's lift, which on an arm is the ordinary case.
        # The torque a real drive supplies: the actuator's own force plus the share of the
        # weight-carrying force MuJoCo applies outside it. Without the second term a compensated
        # arm reads as costing nothing to hold a payload, or to lift one.
        torque = d.actuator_force[self._actuators] + self._gravcomp_share(d)
        per_actuator = torque * d.actuator_velocity[self._actuators]
        driving = float(np.sum(np.maximum(per_actuator, 0.0)))
        driven = float(np.sum(np.minimum(per_actuator, 0.0)))  # <= 0
        self._mech_w = driving + driven
        # Winding loss: a motor torque is a motor current, so this is the one term that survives a
        # standstill, and the only one that bills a braking or a holding motor.
        self._resistive_w = float(np.dot(self._resistive_k, torque * torque))
        self._power_w = (
            driving / self.efficiency
            # A recovered joule crosses the drivetrain's losses on the way back, so it is scaled
            # down by the efficiency; dividing would make a lossier machine recover more.
            + (driven * self.efficiency if self.regenerative else 0.0)
            + self._resistive_w
            + self.idle_w
        )
        self._energy_j += self._power_w * dt
        self._torque_integral += float(np.sum(np.abs(torque))) * dt
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
            resistive_w=self._resistive_w,
            torque_integral_nms=self._torque_integral,
            charge_fraction=fraction,
            depleted=self._depleted,
            voltage=self.voltage,
            # Only where a nominal voltage was stated: current is power/voltage, and inventing a
            # voltage to be able to report a current would put a made-up number on a real field.
            current_a=(self._power_w / self.voltage) if self.voltage > 0.0 else 0.0,
            capacity_wh=self.capacity_wh,
        )
