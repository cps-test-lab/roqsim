# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A sensor's own fault: values it takes on while degraded, switched at run time.

roqsim keeps faults on two channels, and :mod:`roqsim.plugins.model_override` states the split --
that plugin changes the *physics*, while "a perturbation of a reported value is sensor noise and
belongs in a sensor's own config". The physics channel has had a runtime trigger since
``set_model_override``. This is the missing half, so a sensor can be degraded **during** a run
instead of only for the whole of it: a lidar that fails halfway down a corridor rather than one that
was always noisy.

It is deliberately **not** a plugin. A component belongs to the entry it is nested under, and a
sensor registers no entity, so a separate ``sensor_override`` entry could not be nested under the
sensor it faults -- it would have to name its target in a config key, which is the
ownership-as-a-value pattern the component model removed. The fault therefore lives in the sensor's
own config, where the value being perturbed already lives::

    components:
      - spawn_robot: {model: turtlebot4}
        name: robot
        components:
          - lidar:
              range_stddev: 0.01
              fault: {dropout_percent: 60.0, range_stddev: 0.35}

and a scenario switches it by the sensor's **address**::

    set_sensor_override(instance: 'robot.lidar', active: true)

Three properties follow, each mirroring the physics channel rather than re-deciding it.

**It never decides when to fire.** No time trigger, no condition trigger. One bit crosses the wire:
apply, or restore. A fault's timing is the experiment's independent variable, and a sensor that owned
it would put trial logic in the substrate.

**Severity is configured, not sent.** The ``fault:`` block is ordinary config, so sweeping how bad
the fault gets is ``components.robot.lidar.fault.dropout_percent`` -- an ordinary campaign factor,
deterministic per cell, and in the run's provenance. Nothing about severity is on the wire.

**Only keys read per frame may be written.** A sensor declares ``LIVE_WRITABLE`` (config key ->
attribute) and, for the keys someone will reach for first, ``REFUSED_WRITES`` (key -> why). Anything
in neither is refused as undeclared, so a sensor that has not thought about this is safe by default
rather than silently writable. This is not politeness: ``model_override`` learned it from
``geom_size``, which writes fine, takes effect nowhere, and reads back as though it had. The sensor
analogue is ``rays`` -- writing it mid-run changes a ``LaserScan``'s length, which every consumer of
a fixed-length array reads as corruption.

Both dicts merge across the class hierarchy, so a device adds its own rows without restating its
base's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from roqsim.context import Endpoint

# The verdict vocabulary is IMPORTED rather than restated. A grader reads both channels, and two
# spellings of "it did not land" would be two columns meaning one thing.
from roqsim.plugins.model_override import LANDED, NO_EFFECT, UNTESTED

#: Config key holding the faulted values.
FAULT_KEY = "fault"


def blackboard_key(address: str) -> str:
    """One spelling of the key, shared by the publisher and every consumer."""
    return f"sensor_fault:{address}"


@dataclass
class FaultReport:
    """What the two out endpoints carry, and what the service reply is built from."""

    active: bool
    since: float  # sim time the fault last became active; -1.0 if it never has
    changes: int  # how many times the state actually changed since reset
    verified: str  # LANDED | NO_EFFECT | UNTESTED


@dataclass
class SensorFaultHandle:
    """Published on the blackboard under ``sensor_fault:<address>``.

    The same three members ``ModelOverrideHandle`` offers, so the scenario-execution access layer
    drives either channel through one code path.
    """

    name: str
    set_active: Callable[[bool], None]
    is_active: Callable[[], bool]
    read_state: Callable[[], FaultReport]


class FaultableSensorMixin:
    """Opt-in for a sensor plugin: a ``fault:`` block, an allowlist, and the switch.

    Mixed into a sensor base rather than into :class:`~roqsim.plugin.Plugin`, because "which of my
    config keys survive being written at runtime" is a question only a component that reads config
    per frame can answer, and answering it wrongly for everything else would be worse than not
    offering it at all.
    """

    #: config key -> instance attribute. Extended, not replaced, by a subclass.
    LIVE_WRITABLE: dict[str, str] = {}
    #: config key -> the reason writing it at runtime is refused.
    REFUSED_WRITES: dict[str, str] = {}

    # -- the allowlist -------------------------------------------------------------------------
    @classmethod
    def live_writable(cls) -> dict[str, str]:
        """Every writable key on this class and its bases, base first so a subclass may override."""
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "LIVE_WRITABLE", {}) or {})
        return merged

    @classmethod
    def refused_writes(cls) -> dict[str, str]:
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "REFUSED_WRITES", {}) or {})
        return merged

    @classmethod
    def refusal_reason(cls, key: str) -> str:
        """Why *key* may not be written at run time, or ``''`` if it may be."""
        if key in cls.live_writable():
            return ""
        named = cls.refused_writes().get(key)
        if named:
            return f"'{key}' cannot be written while the run is in progress: {named}"
        return (
            f"'{key}' is not declared live-writable by {cls.__name__}. Writable: "
            f"{', '.join(sorted(cls.live_writable())) or '(none)'}. Add it to LIVE_WRITABLE only "
            "once it is read per frame -- a key consumed once at configure() takes effect nowhere "
            "and reads back as if it had."
        )

    # -- validation ----------------------------------------------------------------------------
    def validate_fault(self, config: dict) -> list[str]:
        """Config errors for the ``fault:`` block. Call from ``validate_config``."""
        block = config.get(FAULT_KEY)
        if block is None:
            return []
        if not isinstance(block, dict):
            return [f"'{FAULT_KEY}' must be a mapping of config key -> faulted value"]
        if not block:
            return [
                f"'{FAULT_KEY}' is empty, so applying it would change nothing. Give it at least one "
                f"key ({', '.join(sorted(self.live_writable()))}), or remove it."
            ]
        errors = [reason for key in block if (reason := self.refusal_reason(key))]
        # The faulted values are checked by the device that owns their meaning, not by a second copy
        # of its rules -- so a fault cannot set a dropout of 400% where the nominal config could not.
        merged = {k: v for k, v in config.items() if k != FAULT_KEY}
        merged.update(block)
        errors += [f"'{FAULT_KEY}': {e}" for e in self._validate_nominal(merged)]
        return errors

    def _validate_nominal(self, config: dict) -> list[str]:
        """The device's own validation, minus anything a fault cannot reach.

        Overridden nowhere so far; a device whose ``validate_config`` checks something a partial
        merge cannot satisfy narrows it here rather than skipping validation entirely.
        """
        return [e for e in self.validate_config(config) if FAULT_KEY not in e]

    # -- state ---------------------------------------------------------------------------------
    def _fault_init(self) -> None:
        """Call from ``__init__``, after the config-derived attributes are set."""
        self._fault: dict[str, Any] = dict(self.config.get(FAULT_KEY) or {})
        self._fault_active = False
        self._fault_nominal: dict[str, Any] = {}
        self._fault_report = FaultReport(False, -1.0, 0, UNTESTED)

    def has_fault(self) -> bool:
        return bool(getattr(self, "_fault", None))

    def _read_live(self, key: str) -> Any:
        return getattr(self, self.live_writable()[key])

    def _write_live(self, key: str, value: Any) -> None:
        attr = self.live_writable()[key]
        nominal = getattr(self, attr)
        # Coerce to the nominal's type: a YAML `60` for a float key would otherwise make the
        # attribute an int and change the arithmetic downstream of it.
        setattr(self, attr, type(nominal)(value) if nominal is not None else value)

    # -- the trigger ---------------------------------------------------------------------------
    def set_fault_active(self, on: bool, sim_time: float = 0.0) -> None:
        """Apply the configured faulted values, or restore nominal. Physics thread only.

        The ROS side reaches this through the bridge, which marshals every inbound payload through
        ``ctx.post``, so this always runs on the physics thread and the single-writer rule holds
        without this mixin doing anything about it.
        """
        on = bool(on)
        if not self.has_fault() or on == self._fault_active:
            return
        if on:
            # Saved on APPLY rather than at configure: a campaign may have overridden the nominal
            # between the two, and restoring to a stale value would silently un-apply that override.
            self._fault_nominal = {k: self._read_live(k) for k in self._fault}
            for key, value in self._fault.items():
                self._write_live(key, value)
        else:
            for key, value in self._fault_nominal.items():
                self._write_live(key, value)
        self._fault_active = on
        self._fault_report = FaultReport(
            active=on,
            since=float(sim_time) if on else self._fault_report.since,
            changes=self._fault_report.changes + 1,
            verified=self._verify_fault() if on else UNTESTED,
        )

    def _verify_fault(self) -> str:
        """Read back what the sensor now holds, rather than predicting what it should hold.

        ``NO_EFFECT`` is the case that matters: a fault whose target equals the nominal changes
        nothing, and a run that believes it was faulted would otherwise carry an unfaulted outcome
        under a faulted label.
        """
        changed = [k for k in self._fault if self._read_live(k) != self._fault_nominal.get(k)]
        return LANDED if changed else NO_EFFECT

    def read_fault_state(self) -> FaultReport:
        return self._fault_report

    def fault_detail(self) -> str:
        """Human-readable current state, for a service reply."""
        if not self.has_fault():
            return "no fault configured"
        keys = ", ".join(f"{k}={self._read_live(k)!r}" for k in self._fault)
        return f"{'faulted' if self._fault_active else 'nominal'} ({keys})"

    def on_reset_fault(self) -> None:
        """Back to nominal. Call from ``on_reset``.

        ``Engine.reset`` does not re-instantiate plugins, so without this a fault applied in trial 1
        survives into trials 2..N of the same process: the nominal control cell silently becomes a
        faulted one and nothing crashes.
        """
        if self._fault_active:
            for key, value in self._fault_nominal.items():
                self._write_live(key, value)
        self._fault_active = False
        self._fault_nominal = {}
        self._fault_report = FaultReport(False, -1.0, 0, UNTESTED)

    # -- wiring --------------------------------------------------------------------------------
    def register_fault_endpoints(self, ctx, namespace: str) -> None:
        """Publish the blackboard handle and the three endpoints. Call from ``configure``.

        Endpoint-for-endpoint what ``model_override`` serves, so the bridge needs no new handler and
        a scenario drives either channel the same way. Scoped by the sensor's **address** with dots
        as slashes (``robot.lidar`` -> ``robot/lidar/override``): the address is what identifies a
        component now, and a dot is not legal in a ROS name.
        """
        if not self.has_fault():
            # A sensor with no `fault:` block advertises no service. The alternative -- a service
            # that always replies "nothing configured" -- would make a scenario's typo look like a
            # working call.
            return
        scope = self.address.replace(".", "/")

        ctx.blackboard.set(
            blackboard_key(self.address),
            SensorFaultHandle(
                name=self.address,
                set_active=lambda on: self.set_fault_active(on, ctx.sim_time),
                is_active=lambda: self._fault_active,
                read_state=self.read_fault_state,
            ),
        )
        ctx.interface.add(
            Endpoint(
                name="override",
                direction="in",
                owner=self.entity or self.label,
                namespace=namespace,
                write=lambda payload: self.set_fault_active(bool(payload), ctx.sim_time),
                backend={
                    "ros2": {
                        # A service, not a topic: apply/restore is a command with an outcome, and
                        # the reply is what lets a scenario fail the trial when a fault did not land.
                        "service": "std_srvs.srv.SetBool",
                        "name": f"{scope}/override",
                        "state_key": blackboard_key(self.address),
                    }
                },
            )
        )
        for endpoint_name, field, msg in (
            ("override_state", "active", "std_msgs.msg.Bool"),
            ("override_verified", "verified", "std_msgs.msg.String"),
        ):
            ctx.interface.add(
                Endpoint(
                    name=endpoint_name,
                    direction="out",
                    owner=self.entity or self.label,
                    namespace=namespace,
                    read=self.read_fault_state,
                    rate_hz=10.0,
                    backend={
                        "ros2": {
                            "type": msg,
                            # The report is a structure and these types carry one value, so the
                            # endpoint says WHICH field rather than the bridge holding a converter
                            # that knows this mixin's attribute names.
                            "field": field,
                            "topic": f"{scope}/{endpoint_name}",
                        }
                    },
                )
            )
