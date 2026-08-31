# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``energy_monitor``: the integral, and what it is allowed to assume.

The scene is the smallest thing that costs a measurable amount to drive: one hinge, one motor, held
at a constant velocity by damping, so the mechanical power is a number that can be written down --
``force * velocity`` -- rather than one this test reads back from the plugin it is checking.

The assertions that matter are the ones about what is NOT modelled: an unconfigured monitor reports
mechanical work and nothing else, a state of charge exists only where a capacity was given, and a
depleted battery does not stop the robot.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.plugins.energy_monitor import JOULES_PER_WH, EnergyMonitorPlugin

#: Constant control on the motor; with heavy damping the joint settles at a constant rate, so power
#: settles too and the integral over a known time is predictable.
CTRL = 1.0
DAMPING = 10.0


class _RobotScene(Plugin):
    """A driven hinge (the robot) plus an undriven one on a separate body (someone else's motor)."""

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0.5])
        base.add_joint(
            name="wheel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=DAMPING
        )
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1], mass=2.0)
        actuator = spec.add_actuator()
        actuator.name = "wheel_motor"
        actuator.target = "wheel"
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT

        # A second machine in the same world: its motor must not land on this robot's bill.
        other = spec.worldbody.add_body(name="conveyor", pos=[2, 0, 0.5])
        other.add_joint(
            name="belt", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=DAMPING
        )
        other.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1], mass=2.0)
        belt = spec.add_actuator()
        belt.name = "belt_motor"
        belt.target = "belt"
        belt.trntype = mujoco.mjtTrn.mjTRN_JOINT

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name, kind="robot", body="base_link", meta={"prefix": "", "namespace": ""}
            )
        )


class _UnpoweredScene(_RobotScene):
    """An entity with no actuators at all -- nothing to meter, which must be said rather than shown
    as a bill of zero."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0.5])
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1], mass=2.0)


def _engine(
    *, scene: str = f"{__name__}:_RobotScene", steps: int = 500, drive: float = CTRL, **config
):
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    scene: {},
                    "name": "robot",
                    "components": [{"energy_monitor": dict(config)}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(steps):
        engine.ctx.data.ctrl[:] = drive
        engine.step()
    return engine


def _plugin(engine) -> EnergyMonitorPlugin:
    return next(p for p in engine.plugins if isinstance(p, EnergyMonitorPlugin))


# -- what is measured ------------------------------------------------------------------------


def test_the_power_is_force_times_velocity_and_the_energy_is_its_integral():
    engine = _engine()
    plugin = _plugin(engine)
    report = plugin.read()
    d = engine.ctx.data
    aid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_motor")
    expected_w = abs(float(d.actuator_force[aid] * d.actuator_velocity[aid]))
    assert report.power_w == pytest.approx(expected_w, rel=1e-9)
    assert report.mechanical_w == pytest.approx(expected_w, rel=1e-9)
    # The integral is the average power over the elapsed time, and the run settles quickly, so it
    # lands close to (power * time) without this test having to model the transient.
    assert report.energy_j == pytest.approx(expected_w * engine.ctx.sim_time, rel=0.1)


def test_a_standing_robot_costs_only_what_it_was_told_it_costs():
    """The default models nothing: no motion, no draw. `idle_w` is the platform's own number."""
    assert _plugin(_engine(drive=0.0)).read().power_w == pytest.approx(0.0, abs=1e-9)
    idle = _plugin(_engine(drive=0.0, idle_w=12.0)).read()
    assert idle.power_w == pytest.approx(12.0)
    assert idle.energy_j == pytest.approx(12.0 * 500 * 0.002, rel=1e-6)


def test_efficiency_divides_the_mechanical_power_and_leaves_it_reported():
    plain = _plugin(_engine()).read()
    lossy = _plugin(_engine(efficiency=0.5)).read()
    assert lossy.mechanical_w == pytest.approx(plain.mechanical_w, rel=1e-6)
    assert lossy.power_w == pytest.approx(plain.power_w * 2.0, rel=1e-6)


def test_braking_is_paid_for_unless_the_drive_is_regenerative():
    """A robot without regenerative drive does not get paid to slow down."""
    braking = _engine(steps=200, drive=-CTRL)
    assert _plugin(braking).read().energy_j > 0.0


def test_only_this_robots_actuators_are_metered():
    """A world's other machines are not on this robot's bill -- the subtree decides, not the model."""
    plugin = _plugin(_engine())
    metered = {
        mujoco.mj_id2name(_engine().ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))
        for a in plugin._actuators
    }
    assert metered == {"wheel_motor"}


def test_an_entity_with_no_actuators_is_an_error_not_a_zero_reading():
    """A meter reading zero forever looks exactly like a robot that costs nothing to drive."""
    with pytest.raises(RuntimeError, match="no actuators to meter"):
        _engine(scene=f"{__name__}:_UnpoweredScene", steps=1, drive=0.0)


# -- the battery, where there is one ---------------------------------------------------------


def test_without_a_capacity_the_charge_is_unknown_rather_than_full():
    report = _plugin(_engine()).read()
    assert report.charge_fraction == -1.0
    assert report.depleted is False


def test_a_capacity_gives_a_state_of_charge_that_falls():
    report = _plugin(_engine(capacity_wh=0.001)).read()
    assert 0.0 <= report.charge_fraction < 1.0
    assert report.energy_j > 0.0


def test_depletion_latches_and_does_not_stop_the_robot():
    """The substrate reports; the trial decides. Ending a run is the experiment's call."""
    engine = _engine(capacity_wh=1e-7)
    plugin = _plugin(engine)
    assert plugin.read().depleted is True
    assert plugin.read().charge_fraction == 0.0
    before = float(engine.ctx.data.qvel[0])
    for _ in range(50):
        engine.ctx.data.ctrl[:] = CTRL
        engine.step()
    assert engine.ctx.data.qvel[0] == pytest.approx(before, rel=0.2), "the wheel still turns"


def test_a_reset_starts_the_next_trial_on_a_full_battery():
    """One process serves several trials; a leaked integral makes cell 2 start half-empty."""
    engine = _engine(capacity_wh=0.01)
    assert _plugin(engine).read().energy_j > 0.0
    engine.reset()
    assert _plugin(engine).read().energy_j == 0.0
    assert _plugin(engine).read().depleted is False


# -- wiring ------------------------------------------------------------------------------------


def test_the_endpoint_and_the_blackboard_reader_agree():
    engine = _engine()
    reader = engine.ctx.blackboard.get("energy:robot.energy_monitor")
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "battery")
    assert reader is not None
    assert reader.read().energy_j == endpoint.read().energy_j
    assert endpoint.backend["ros2"]["type"] == "sensor_msgs.msg.BatteryState"
    assert endpoint.backend["ros2"]["topic"] == "battery_state"


def test_it_belongs_to_a_robot():
    with pytest.raises(PluginError):
        load_config_from_dict({"sim": {}, "components": [{"energy_monitor": {}}]})


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"efficiency": 0.0}, "'efficiency' must be in (0, 1]"),
        ({"efficiency": 1.5}, "'efficiency' must be in (0, 1]"),
        ({"idle_w": -1}, "'idle_w' must be >= 0"),
        ({"capacity_wh": -1}, "'capacity_wh' must be >= 0"),
        ({"rate_hz": 0}, "'rate_hz' must be > 0"),
        ({"actuators": "wheel_motor"}, "must be a list"),
    ],
)
def test_config_errors_are_reported_by_name(config, expected):
    errors = EnergyMonitorPlugin(config, entity="robot", label="energy").validate_config(config)
    assert any(expected in e for e in errors), errors


def test_watt_hours_and_joules_are_one_quantity():
    assert JOULES_PER_WH == 3600.0
