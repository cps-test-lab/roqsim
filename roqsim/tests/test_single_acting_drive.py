# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A single-acting drive's one-sided rating belongs on the joint, not on the actuator.

A lift carriage on a vertical slide, driven by a position servo and compensated as ``spawn_arm`` and
``spawn_robot`` compensate it. MuJoCo clamps the actuator's own force by its ``forcerange`` and only
then adds the gravity term, so a floor of 0 there cancels the weight: the empty carriage can never
lower. The joint's ``actuatorfrcrange`` clamps the sum, which is what a cylinder that can only push
does: the carriage lowers under its own weight, and a load above the rating stalls it.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.actuators import apply_gravity_compensation, joint_force_range
from roqsim.actuators import resolve as resolve_actuators

CARRIAGE_KG = 150.0
RATING_N = 25000.0
START, LOW, HIGH = 0.0, -0.5, 0.5


def _lift(actuator_range: str | None, joint_range: str | None, load_kg: float = 0.0):
    act = f'forcerange="{actuator_range}"' if actuator_range else ""
    jnt = f'actuatorfrcrange="{joint_range}"' if joint_range else ""
    load = (
        f'<body name="load" pos="0 0 .1"><geom type="box" size=".1 .1 .05" mass="{load_kg}" '
        'contype="0" conaffinity="0"/></body>'
        if load_kg
        else ""
    )
    spec = mujoco.MjSpec.from_string(f"""
<mujoco>
  <worldbody>
    <body name="mast">
      <geom type="box" size=".05 .05 .05" contype="0" conaffinity="0"/>
      <body name="carriage" pos="0 0 1">
        <joint name="lift" type="slide" axis="0 0 1" range="-.9 1" {jnt}/>
        <geom type="box" size=".3 .3 .05" mass="{CARRIAGE_KG}" contype="0" conaffinity="0"/>
        {load}
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="200000" dampratio="1" {act}/>
  </actuator>
</mujoco>""")
    apply_gravity_compensation(spec, resolve_actuators(spec, None, model_name="lift"))
    return spec.compile()


def _drive_to(model, target: float, seconds: float = 3.0) -> float:
    data = mujoco.MjData(model)
    data.ctrl[0] = target
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return float(data.qpos[0])


def _rated_on_the_joint(load_kg: float = 0.0):
    return _lift(f"-{RATING_N:g} {RATING_N:g}", f"0 {RATING_N:g}", load_kg)


def test_the_carriage_lowers_under_its_own_weight():
    assert _drive_to(_rated_on_the_joint(), LOW) == pytest.approx(LOW, abs=1e-3)


def test_the_carriage_lifts_a_load_within_the_rating():
    # 2000 kg plus the carriage is 21.1 kN, inside 25 kN.
    assert _drive_to(_rated_on_the_joint(load_kg=2000.0), HIGH) == pytest.approx(HIGH, abs=1e-3)


def test_a_load_above_the_rating_stalls():
    # 3000 kg plus the carriage is 30.9 kN: the cylinder cannot hold it, and it sinks to the stop.
    assert _drive_to(_rated_on_the_joint(load_kg=3000.0), HIGH) < START - 0.5


def test_a_one_sided_actuator_range_leaves_the_carriage_unable_to_lower():
    """Why the rating does not go on the actuator: its clamp comes before the gravity term."""
    model = _lift(f"0 {RATING_N:g}", None)
    assert _drive_to(model, LOW) == pytest.approx(START, abs=1e-3)


def test_the_joint_range_is_the_force_range_the_drive_reports():
    assert joint_force_range(_rated_on_the_joint(), 0) == pytest.approx((0.0, RATING_N))
