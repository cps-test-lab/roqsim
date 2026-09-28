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


class _TwoJointScene(_RobotScene):
    """Two motors on one robot, so the per-actuator split is observable.

    One actuator alone cannot tell a sum of magnitudes from the magnitude of a sum, which is why the
    single-motor scene above cannot check the arithmetic an arm depends on.
    """

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 1.0])
        base.add_joint(name="j1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=DAMPING)
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1], mass=2.0)
        link = base.add_body(name="link2", pos=[0.4, 0, 0])
        link.add_joint(name="j2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=DAMPING)
        link.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1], mass=2.0)
        for joint in ("j1", "j2"):
            actuator = spec.add_actuator()
            actuator.name = f"{joint}_motor"
            actuator.target = joint
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT


class _HoldingScene(_RobotScene):
    """A gravity-loaded joint held at a pose by a position servo: torque without motion.

    This is the manipulator case the mechanical integral alone cannot see -- the arm is still, so
    ``force * velocity`` is zero however much of the payload the motor is carrying.
    """

    #: Gravity's moment on the held joint: the offset mass, its lever arm, and MuJoCo's default g.
    HOLD_NM = 5.0 * 9.81 * 0.4

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 1.0])
        base.add_joint(
            name="shoulder", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=DAMPING
        )
        # The mass hangs out along +x, so gravity puts a constant moment on the joint.
        base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.4, 0.05, 0.05], pos=[0.4, 0, 0], mass=5.0
        )
        actuator = spec.add_actuator()
        actuator.name = "shoulder_motor"
        actuator.target = "shoulder"
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gaintype = mujoco.mjtGain.mjGAIN_AFFINE
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = 2000.0
        actuator.biasprm[1] = -2000.0
        actuator.biasprm[2] = -200.0


class _CompensatedHoldingScene(_HoldingScene):
    """The same loaded joint, with MuJoCo carrying its weight.

    This is what every position- or impedance-driven arm in roqsim is: `apply_gravity_compensation`
    sets `gravcomp` on the arm's bodies, so the weight-carrying force arrives OUTSIDE the actuator
    and `actuator_force` reads zero on a joint holding a payload. A meter reading only that reports
    an arm that is free to hold a load up, and free to lift one.
    """

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        super().build(spec, ctx)
        spec.body("base_link").gravcomp = 1.0


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


def test_one_joints_descent_does_not_pay_for_anothers_lift():
    """The split into driving and driven happens per actuator, before the sum.

    Netted first, the two cancel and an arm changing pose reports as free. The scene puts the two
    motors' work in opposite directions deliberately: that is an arm's ordinary case, not a corner.
    """
    engine = _engine(scene=f"{__name__}:_TwoJointScene", steps=0)
    plugin = _plugin(engine)
    d = engine.ctx.data
    d.qvel[:] = [20.0, -20.0]
    d.ctrl[:] = [CTRL, CTRL]
    engine.step()

    work = d.actuator_force[plugin._actuators] * d.actuator_velocity[plugin._actuators]
    assert work.min() < 0.0 < work.max(), "the two actuators do opposite-sign work"
    report = plugin.read()
    assert report.power_w == pytest.approx(float(work[work > 0.0].sum()), rel=1e-9)
    assert abs(float(work.sum())) < report.power_w, "the net alone would have understated the draw"


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


def _braking(**config):
    """One step with the load moving and the motor opposing it: negative mechanical power."""
    engine = _engine(steps=0, **config)
    engine.ctx.data.qvel[0] = 20.0
    engine.ctx.data.ctrl[:] = -CTRL
    engine.step()
    return _plugin(engine).read()


def test_braking_is_not_billed_to_the_pack_unless_the_drive_is_regenerative():
    """A motor braking a load dissipates the load's energy; it does not draw it from the pack.

    Billing the magnitude would charge the experiment for joules the pack never supplied -- and
    charge them twice over once a winding loss is configured, which is where a braking motor's real
    cost is.
    """
    report = _braking(idle_w=2.0)
    assert report.mechanical_w < 0.0, "the motor is braking"
    assert report.power_w == pytest.approx(2.0), "only the idle draw reaches the pack"


def test_a_regenerative_credit_crosses_the_drivetrain_losses_on_the_way_back():
    """A recovered joule is scaled DOWN by the efficiency. Dividing makes a lossier machine recover
    more, which is the wrong direction."""
    report = _braking(regenerative=True, efficiency=0.5)
    assert report.mechanical_w < 0.0
    assert report.power_w == pytest.approx(report.mechanical_w * 0.5, rel=1e-9)


def test_a_reverse_drive_is_not_braking_and_is_paid_for():
    """Driving the other way is still driving: force and velocity share a sign."""
    assert _plugin(_engine(steps=200, drive=-CTRL)).read().energy_j > 0.0


# -- holding, which is where a manipulator's bill comes from -----------------------------------


def _held(**config):
    """The loaded joint, settled at its held pose."""
    return _plugin(
        _engine(scene=f"{__name__}:_HoldingScene", steps=1500, drive=0.0, **config)
    ).read()


def test_a_held_load_is_free_until_a_winding_loss_says_it_is_not():
    """Mechanical power is exactly zero at a standstill, whatever the motor is carrying.

    Unconfigured the plugin says so rather than guessing a motor; `resistive_w_per_nm2` is the
    platform's own number, and it is the term that makes a manipulator's slow trial add up.
    """
    still = _held()
    assert still.mechanical_w == pytest.approx(0.0, abs=1e-2), "it is not moving"
    assert still.power_w == pytest.approx(0.0, abs=1e-2), "and nothing was assumed about its motor"

    lossy = _held(resistive_w_per_nm2=0.01)
    expected = 0.01 * _HoldingScene.HOLD_NM**2
    assert lossy.power_w == pytest.approx(expected, rel=0.02)
    assert lossy.resistive_w == pytest.approx(expected, rel=0.02)
    assert lossy.mechanical_w == pytest.approx(0.0, abs=1e-2), "still not moving"
    assert lossy.energy_j > 0.0, "and the holding is integrated, not only reported"


def test_a_compensated_arm_is_billed_for_the_load_it_holds():
    """The torque metered is the one a real drive supplies, not the residue MuJoCo leaves.

    `actuator_force` is zero here by construction, so this is the case a meter reading it alone
    cannot see -- and it is the ordinary case, because every position- and impedance-driven arm is
    gravity-compensated.
    """
    scene = f"{__name__}:_CompensatedHoldingScene"
    engine = _engine(scene=scene, steps=1500, drive=0.0, resistive_w_per_nm2=0.01)
    plugin = _plugin(engine)
    d = engine.ctx.data
    assert d.actuator_force[plugin._actuators[0]] == pytest.approx(0.0, abs=1e-6), (
        "MuJoCo carries the weight outside the actuator -- the premise of this test"
    )
    expected = 0.01 * _HoldingScene.HOLD_NM**2
    assert plugin.read().resistive_w == pytest.approx(expected, rel=0.02)
    assert plugin.read().energy_j > 0.0


def test_a_coefficient_per_actuator_bills_each_motor_its_own_loss():
    """An arm's shoulder and its wrist are not the same motor, so one coefficient cannot serve both.
    An actuator the mapping omits contributes nothing."""
    engine = _engine(
        scene=f"{__name__}:_TwoJointScene", steps=200, resistive_w_per_nm2={"j2_motor": 0.5}
    )
    plugin = _plugin(engine)
    d = engine.ctx.data
    aid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "j2_motor")
    assert plugin.read().resistive_w == pytest.approx(0.5 * float(d.actuator_force[aid]) ** 2)


def test_a_coefficient_for_an_unmetered_actuator_is_an_error():
    """Ignored, it would read as a joint that costs nothing to hold."""
    with pytest.raises(RuntimeError, match="does not meter"):
        _engine(steps=1, resistive_w_per_nm2={"belt_motor": 0.5})


def test_the_torque_integral_is_accumulated_beside_the_energy():
    """The effort metric a paper falls back on where its motor constants are not published."""
    report = _plugin(_engine()).read()
    assert report.torque_integral_nms == pytest.approx(CTRL * 500 * 0.002, rel=1e-6)


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
    assert _plugin(engine).read().torque_integral_nms == 0.0
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
        ({"resistive_w_per_nm2": -1}, "'resistive_w_per_nm2' must be >= 0"),
        ({"resistive_w_per_nm2": {"wheel_motor": -1}}, "'resistive_w_per_nm2' must be >= 0"),
        ({"resistive_w_per_nm2": "lots"}, "must be a number, or a mapping"),
    ],
)
def test_config_errors_are_reported_by_name(config, expected):
    errors = EnergyMonitorPlugin(config, entity="robot", label="energy").validate_config(config)
    assert any(expected in e for e in errors), errors


def test_watt_hours_and_joules_are_one_quantity():
    assert JOULES_PER_WH == 3600.0
