"""The rotor actuation model: scaling, lag, moment signs, and the spin table.

``test_equal_thrust_produces_no_yaw_torque`` is the one that earns its keep: a permuted spin table
is invisible in hover, in climb and in roll, and only shows up as a drone that yaws away whenever it
is asked to do anything. Every other test here would still pass with the table scrambled.

The plugin is instantiated directly rather than through world loading, so these tests do not depend
on the entry point being registered.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.context import SimContext
from roqsim.models import resolve_model
from roqsim_aerial.plugins.multirotor_motors import MultirotorMotorsPlugin

#: PX4's MPC_THR_HOVER 0.6 for 4001_gz_x500; see the MJCF header.
MAX_THRUST = 8.2
#: PX4's CA_ROTOR*_KM for the same airframe, and the plugin's default.
MOMENT_CONSTANT = 0.05
DT = 0.002


def _harness(config=None, *, air=True):
    """A compiled x500 plus a configured plugin, with no engine in the way."""
    asset = resolve_model("roqsim_aerial:x500")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    spec.option.timestep = DT
    if air:
        spec.option.density = 1.225
        spec.option.viscosity = 1.8e-5
    model = spec.compile()
    ctx = SimContext({})
    ctx.model = model
    ctx.data = mujoco.MjData(model)
    plugin = MultirotorMotorsPlugin(config or {}, entity="drone")
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _settle(ctx, plugin, seconds=0.5):
    for _ in range(int(seconds / DT)):
        plugin.pre_step(ctx)
        mujoco.mj_step(ctx.model, ctx.data)


def _bid(ctx):
    return mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "x500")


def test_full_command_reaches_max_thrust():
    ctx, plugin = _harness()
    plugin.set_normalized([1.0] * 4)
    _settle(ctx, plugin, 1.0)
    assert ctx.data.ctrl == pytest.approx(np.full(4, MAX_THRUST), rel=1e-3)
    assert plugin.read_normalized() == pytest.approx(np.ones(4), rel=1e-3)


def test_commands_are_clipped():
    ctx, plugin = _harness()
    plugin.set_normalized([2.0, -1.0, 0.5, 0.0])
    _settle(ctx, plugin, 1.0)
    assert ctx.data.ctrl == pytest.approx(
        np.array([1.0, 0.0, 0.5, 0.0]) * MAX_THRUST, rel=1e-3
    )


def test_the_lag_actually_lags():
    """One tick of a 2 ms world against a 20 ms motor must move about a tenth of the way."""
    ctx, plugin = _harness()
    plugin.set_normalized([1.0] * 4)
    plugin.pre_step(ctx)
    alpha = DT / (0.02 + DT)
    assert ctx.data.ctrl[0] == pytest.approx(alpha * MAX_THRUST, rel=1e-6)
    assert ctx.data.ctrl[0] < 0.2 * MAX_THRUST, "a step command must not become a step force"


def test_zero_time_constant_is_instantaneous():
    ctx, plugin = _harness({"time_constant": 0.0})
    plugin.set_normalized([1.0] * 4)
    plugin.pre_step(ctx)
    assert ctx.data.ctrl == pytest.approx(np.full(4, MAX_THRUST))


def test_roll_and_pitch_moment_signs():
    """Differential thrust must move the airframe the way the geometry says it should.

    tau = r x F with F along +z gives tau = (y*T, -x*T, 0): more thrust on the +y side rolls
    positive about x, more thrust at the front (+x) pitches negative about y (nose up).
    """
    # rotor1 (rear-left) and rotor2 (front-left) are the +y pair.
    ctx, plugin = _harness()
    plugin.set_normalized([0.0, 1.0, 1.0, 0.0])
    _settle(ctx, plugin, 0.3)
    roll_rate = float(ctx.data.qvel[3])
    assert roll_rate > 0.1, f"+y thrust must roll positive about x, got {roll_rate:.3f} rad/s"

    # rotor0 (front-right) and rotor2 (front-left) are the +x pair.
    ctx, plugin = _harness()
    plugin.set_normalized([1.0, 0.0, 1.0, 0.0])
    _settle(ctx, plugin, 0.3)
    pitch_rate = float(ctx.data.qvel[4])
    assert pitch_rate < -0.1, f"front thrust must pitch negative about y, got {pitch_rate:.3f}"


def test_equal_thrust_produces_no_yaw_torque():
    """The test a permuted spin table fails, and the only one that does.

    Two CCW and two CW rotors at equal thrust must cancel exactly. If they do not, the airframe
    yaws whenever it climbs -- which reads as a flight-stack tuning problem, not as a table.
    """
    ctx, plugin = _harness()
    plugin.set_normalized([1.0] * 4)
    _settle(ctx, plugin, 1.0)
    torque = ctx.data.xfrc_applied[_bid(ctx), 3:6]
    assert torque == pytest.approx(np.zeros(3), abs=1e-12)
    assert abs(float(ctx.data.qvel[5])) < 1e-6, "no yaw rate may appear out of a symmetric climb"


def test_all_ccw_spin_yaws():
    """Sanity check on the cancellation above: four same-handed rotors must NOT cancel.

    Without this, a plugin that simply never applied any torque would pass the zero-yaw test.
    """
    ctx, plugin = _harness({"spin": [1, 1, 1, 1]})
    plugin.set_normalized([1.0] * 4)
    _settle(ctx, plugin, 1.0)
    tau_z = float(ctx.data.xfrc_applied[_bid(ctx), 5])
    # -spin * k_m * T summed: -4 * 0.05 * 8.2 = -1.64 N*m. CCW rotors drag the airframe CW.
    assert tau_z == pytest.approx(-4 * MOMENT_CONSTANT * MAX_THRUST, rel=1e-3)
    assert float(ctx.data.qvel[5]) < -0.1, "a CW reaction torque must yaw the drone negatively"


def test_yaw_torque_is_expressed_in_the_world_frame():
    """A body-frame torque written straight in is wrong exactly when the drone is tilted."""
    ctx, plugin = _harness({"spin": [1, 1, 1, 1]})
    # Roll 90 degrees about x: body z now points along world -y.
    ctx.data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(ctx.model, ctx.data)
    plugin.set_normalized([1.0] * 4)
    for _ in range(int(1.0 / DT)):
        plugin.pre_step(ctx)
        mujoco.mj_forward(ctx.model, ctx.data)  # no integration: hold the attitude
    torque = ctx.data.xfrc_applied[_bid(ctx), 3:6]
    expected = -4 * MOMENT_CONSTANT * MAX_THRUST
    assert torque[1] == pytest.approx(-expected, rel=1e-3), "body +z is world -y at 90 deg roll"
    assert abs(torque[2]) < 1e-9, "nothing may remain on world z"


def test_missing_actuator_fails_loudly():
    asset = resolve_model("roqsim_aerial:x500")
    model = mujoco.MjModel.from_xml_path(str(asset.path))
    ctx = SimContext({})
    ctx.model = model
    ctx.data = mujoco.MjData(model)
    plugin = MultirotorMotorsPlugin({"rotors": ["rotor0_thrust", "nope", "also_nope"],
                                     "spin": [1, 1, -1]}, entity="drone")
    with pytest.raises(RuntimeError, match="nope"):
        plugin.configure(ctx)


def test_spin_length_mismatch_is_a_config_error():
    plugin = MultirotorMotorsPlugin({"spin": [1, -1]}, entity="drone")
    errors = plugin.validate_config({"spin": [1, -1]})
    assert any("spin" in e for e in errors)


def test_max_thrust_defaults_to_the_models_ctrlrange():
    ctx, plugin = _harness()
    handle = ctx.blackboard.require("motors:drone")
    assert handle.max_thrust == pytest.approx((MAX_THRUST,) * 4)
    assert handle.count == 4


def test_handle_and_endpoint_are_registered():
    ctx, plugin = _harness()
    handle = ctx.blackboard.require("motors:drone")
    handle.set_normalized([0.3, 0.3, 0.3, 0.3])
    _settle(ctx, plugin, 1.0)
    assert handle.read_normalized() == pytest.approx(np.full(4, 0.3), rel=1e-3)

    endpoint = next(e for e in ctx.interface.all() if e.name == "motor_cmd")
    assert endpoint.direction == "in"
    assert endpoint.backend["ros2"]["type"] == "std_msgs.msg.Float32MultiArray"
    # The bridge hands over the message; a bare sequence must work too, for in-process callers.
    endpoint.write([1.0, 1.0, 1.0, 1.0])
    assert plugin._cmd == pytest.approx(np.ones(4))


def test_reset_clears_the_external_torque():
    """Leaving a stale xfrc_applied across a reset starts the next episode under a phantom torque."""
    ctx, plugin = _harness({"spin": [1, 1, 1, 1]})
    plugin.set_normalized([1.0] * 4)
    _settle(ctx, plugin, 0.5)
    assert abs(float(ctx.data.xfrc_applied[_bid(ctx), 5])) > 0.1
    plugin.on_reset(ctx)
    assert ctx.data.xfrc_applied[_bid(ctx)] == pytest.approx(np.zeros(6))
    assert ctx.data.ctrl == pytest.approx(np.zeros(4))
    assert plugin.read_normalized() == pytest.approx(np.zeros(4))


def test_warns_in_a_vacuum(caplog):
    with caplog.at_level(logging.WARNING):
        _harness(air=False)
    assert any("vacuum" in r.message for r in caplog.records)
