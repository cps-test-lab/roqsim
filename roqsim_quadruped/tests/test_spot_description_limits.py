"""Spot's leg geometry, joint ranges and joint speed limits are spot_description's.

Fixtures from rai-opensource/spot_description @ 156d1802bfb117f219dbfce7597d283d5fabc968 (the
submodule bdaiinstitute/spot_ros2 pins), urdf/spot_macro.xacro: the hip_x, hip_y and knee joints of
each leg (lines 85-541), their origins and their <limit> lower/upper/velocity.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from roqsim_quadruped.plugins.spot_locomotion import SpotLocomotionPlugin
from roqsim_quadruped.policy import DEFAULT_POLICY

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

POLICY = os.environ.get("SPOT_POLICY_PATH") or str(DEFAULT_POLICY)
LEGS = {"fl": "front_left", "fr": "front_right", "hl": "rear_left", "hr": "rear_right"}
HIP_X = (-0.78539816339744827899, 0.78539816339744827899)
HIP_Y = (-0.89884456477707963539, 2.2951079663725435509)
KNEE_UPPER = {"fl": -0.254801, "fr": -0.247563, "hl": -0.267153, "hr": -0.257725}
KNEE_LOWER = -2.7929
UPPER_LEG_Y = 0.110945  # front_left_hip_y origin y; negated on the right
KNEE_ORIGIN = (0.025, 0.0, -0.3205)
VELOCITY = {"hx": 17.647, "hy": 17.647, "kn": 12.0}


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(resolve_model("spot").path))


def _range(m, joint):
    return tuple(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)])


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_joint_ranges_are_the_descriptions(model, leg):
    assert _range(model, f"{leg}_hx") == pytest.approx(HIP_X, abs=1e-6)
    assert _range(model, f"{leg}_hy") == pytest.approx(HIP_Y, abs=1e-5)
    assert _range(model, f"{leg}_kn") == pytest.approx((KNEE_LOWER, KNEE_UPPER[leg]), abs=1e-6)


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_leg_link_offsets_are_the_descriptions(model, leg):
    side = 1.0 if leg in ("fl", "hl") else -1.0
    uleg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_uleg")
    lleg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_lleg")
    np.testing.assert_allclose(model.body_pos[uleg], (0.0, side * UPPER_LEG_Y, 0.0), atol=1e-9)
    np.testing.assert_allclose(model.body_pos[lleg], KNEE_ORIGIN, atol=1e-9)


@pytest.mark.parametrize(
    "limits", [[12.0], {"fl_kn": 0.0}, {"fl_kn": -1.0}, {"fl_kn": True}, {"fl_kn": "12"}]
)
def test_a_joint_velocity_limit_that_is_not_a_positive_speed_is_refused(limits):
    errors = SpotLocomotionPlugin({}).validate_config({"joint_velocity_limits": limits})
    assert any("joint_velocity_limits" in e for e in errors), errors


def test_a_positive_joint_velocity_limit_is_accepted():
    assert SpotLocomotionPlugin({}).validate_config({"joint_velocity_limits": {"fl_kn": 12}}) == []


def _walking(**locomotion) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.002},
            "components": [
                {
                    "spawn_robot": {"model": "spot"},
                    "name": "robot",
                    "components": [{"spot_locomotion": locomotion}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)
def test_no_joint_is_commanded_faster_than_its_motor_and_spot_still_walks():
    engine = _walking(test_cmd=[1.0, 0.0, 0.3])
    plugin = next(p for p in engine.plugins if p.address == "robot.spot_locomotion")
    m, d = engine.ctx.model, engine.ctx.data
    limits = np.array([VELOCITY[j.split("_")[1]] for j in plugin._joint_order])
    prev = d.ctrl[plugin._leg_aid].copy()
    fastest = np.zeros(len(limits))
    for _ in range(1500):  # 3 s
        engine.step()
        now = d.ctrl[plugin._leg_aid].copy()
        fastest = np.maximum(fastest, np.abs(now - prev) / m.opt.timestep)
        prev = now
    assert np.all(fastest <= limits * (1 + 1e-9)), dict(
        zip(plugin._joint_order, fastest, strict=True)
    )
    odom = next(e for e in engine.ctx.interface.all() if e.name == "odom").read()
    assert odom.position[0] > 1.0, "Spot did not walk forward"
    assert odom.position[2] > 0.4, "Spot did not stay standing"
    engine.shutdown()
