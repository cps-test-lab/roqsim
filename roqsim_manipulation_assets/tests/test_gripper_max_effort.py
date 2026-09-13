# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""GripperCommand's ``max_effort``: the force each jaw may apply, in newtons.

Three layers, because each catches a different mistake:

* **the transmission** -- newtons at a jaw per unit of actuator force, read from the model. The
  Schunk PG+70 drives two slide jaws through a fixed tendon at 0.5 each; the Robotiq 2F-85 closes
  revolute knuckles, where no constant exists and a newton figure would be a guess.
* **the plugin** -- what ``arm_controller`` publishes and does: the endpoint's ``effort_key`` hint,
  the rating, the force range a cap writes, the refusals, and the reset that restores the model's own.
* **the physics** -- that capping the force range at that figure is the force the jaws put into an
  object. This is the check a force gauge would make on a real gripper.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import jaw_force_per_actuator_force

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import apply_assets, resolve_model

#: The PG+70's tendon coefficient on each jaw (schunk_pg70.xml, `split`).
PG70_JAW_GAIN = 0.5

#: Its rated force range on that tendon, carried to one jaw.
PG70_RATED_N = 200.0 * PG70_JAW_GAIN

#: The two caps the physics check closes at, well inside the rating.
CAPS_N = (20.0, 50.0)

#: How far each jaw's pressing force may sit from the cap. Measured on the cap to the newton at both
#: levels; the band leaves room for solver settings, not for a transmission that is off by a factor.
FORCE_TOLERANCE = 0.05

# The PG+70's grasp geometry, as test_grippers.py measures it.
PALM_Z = 0.5
JAW_PLANE = 0.0789
JAW_HALF_Z = 0.005
JAW_HOME = 0.0301
CTRL_OPEN, CTRL_CLOSE = -0.0301, 0.001
PARCEL_HALF = (0.03, 0.0225, 0.01875)
PARCEL_MASS = 0.211

_PARCEL_WORLD = """<mujoco model="grip_force">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"
          noslip_iterations="10" gravity="0 0 0"/>
  <worldbody>
    <body name="parcel" pos="0 0 {pz}">
      <freejoint/>
      <geom name="parcel" type="box" size="{hx} {hy} {hz}" mass="{mass}" condim="4"
            friction="1.2 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.005 1"/>
    </body>
  </worldbody>
</mujoco>"""


def _gripper_model(model: str) -> mujoco.MjModel:
    asset = resolve_model(model)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    return spec.compile()


def _ids(model, obj, *names):
    return [mujoco.mj_name2id(model, obj, n) for n in names]


def _engine(tmp_path, gripper: str) -> Engine:
    config = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "ur5e_",
                        "end_effector": {"model": gripper},
                    },
                    "name": "ur5e",
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(config)
    engine.setup()
    return engine


def _gripper_actuator(engine: Engine, name: str) -> int:
    model = engine.ctx.model
    for aid in range(model.nu):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or "").endswith(name):
            return aid
    raise AssertionError(f"no actuator ending in {name!r}")


def test_a_slide_jaw_tendon_carries_its_coefficient():
    m = _gripper_model("schunk_pg70")
    (aid,) = _ids(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_actuator")
    for joint in _ids(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_left_joint", "finger_right_joint"):
        assert jaw_force_per_actuator_force(m, aid, joint) == pytest.approx(PG70_JAW_GAIN)


def test_a_revolute_linkage_has_no_newton_figure():
    m = _gripper_model("robotiq_2f85")
    (aid,) = _ids(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    (joint,) = _ids(m, mujoco.mjtObj.mjOBJ_JOINT, "robotiq_85_left_knuckle_joint")
    assert jaw_force_per_actuator_force(m, aid, joint) == 0.0, (
        "a knuckle's force at the pads changes with the grasp width, so no constant may be claimed"
    )


def test_the_endpoint_names_the_force_entry_and_its_rating(tmp_path):
    engine = _engine(tmp_path, "schunk_pg70")
    endpoint = {e.name: e for e in engine.ctx.interface.all()}["gripper_cmd"]
    key = endpoint.backend["ros2"]["effort_key"]
    effort = engine.ctx.blackboard.require(key)
    assert effort.rated == pytest.approx(PG70_RATED_N)


def test_a_linkage_gripper_is_rated_zero_and_refuses_a_cap(tmp_path):
    engine = _engine(tmp_path, "robotiq_2f85")
    endpoint = {e.name: e for e in engine.ctx.interface.all()}["gripper_cmd"]
    effort = engine.ctx.blackboard.require(endpoint.backend["ros2"]["effort_key"])
    assert effort.rated == 0.0
    assert math.isnan(effort.read_effort()), "a force with no rating must not read as 0 N"
    with pytest.raises(ValueError, match="no force rating"):
        effort.set_max_effort(10.0)
    effort.set_max_effort(0.0)  # restoring the model's own range is always allowed


def test_a_cap_writes_the_force_range_and_a_reset_restores_the_model(tmp_path):
    engine = _engine(tmp_path, "schunk_pg70")
    engine.reset()
    effort = engine.ctx.blackboard.require("gripper_effort:ur5e")
    aid = _gripper_actuator(engine, "finger_actuator")
    model_range = tuple(engine.ctx.model.actuator_forcerange[aid])

    effort.set_max_effort(CAPS_N[0])
    limit = CAPS_N[0] / PG70_JAW_GAIN
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx((-limit, limit))

    effort.set_max_effort(0.0)
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(model_range)

    effort.set_max_effort(CAPS_N[1])
    engine.reset()
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(model_range), (
        "a trial's grip force must not leak into the next trial"
    )


def test_a_cap_above_the_rating_is_refused_not_clamped(tmp_path):
    engine = _engine(tmp_path, "schunk_pg70")
    effort = engine.ctx.blackboard.require("gripper_effort:ur5e")
    with pytest.raises(ValueError, match="exceeds"):
        effort.set_max_effort(PG70_RATED_N + 1.0)


@pytest.mark.parametrize("cap", CAPS_N)
def test_the_jaws_press_with_the_force_asked_for(cap):
    """Closed on a parcel with the range capped at ``cap`` per jaw, the jaws push with ``cap``."""
    parcel_z = PALM_Z - (JAW_PLANE + PARCEL_HALF[2] - JAW_HALF_Z)
    world = mujoco.MjSpec.from_string(
        _PARCEL_WORLD.format(
            pz=parcel_z, hx=PARCEL_HALF[0], hy=PARCEL_HALF[1], hz=PARCEL_HALF[2], mass=PARCEL_MASS
        )
    )
    frame = world.worldbody.add_frame()
    frame.pos = [0.0, 0.0, PALM_Z]
    frame.quat = [0.0, 1.0, 0.0, 0.0]  # approach axis pointing down at the parcel
    asset = resolve_model("schunk_pg70")
    gripper = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(gripper, asset)
    world.attach(gripper, prefix="grip_", frame=frame)
    m = world.compile()
    d = mujoco.MjData(m)

    (aid,) = _ids(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_finger_actuator")
    joints = _ids(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_finger_left_joint", "grip_finger_right_joint")
    pads = _ids(m, mujoco.mjtObj.mjOBJ_GEOM, "grip_finger_left_pad", "grip_finger_right_pad")
    (parcel,) = _ids(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel")
    for joint in joints:
        d.qpos[m.jnt_qposadr[joint]] = JAW_HOME
    mujoco.mj_forward(m, d)

    limit = cap / jaw_force_per_actuator_force(m, aid, joints[0])
    m.actuator_forcerange[aid] = (-limit, limit)
    d.ctrl[aid] = CTRL_OPEN
    for _ in range(500):
        mujoco.mj_step(m, d)
    d.ctrl[aid] = CTRL_CLOSE
    for _ in range(1500):
        mujoco.mj_step(m, d)

    wrench = np.zeros(6)
    per_jaw = []
    for pad in pads:
        normal = 0.0
        for i in range(d.ncon):
            contact = d.contact[i]
            if {int(contact.geom1), int(contact.geom2)} == {pad, parcel}:
                mujoco.mj_contactForce(m, d, i, wrench)
                normal += abs(float(wrench[0]))
        per_jaw.append(normal)

    assert all(f > 0.0 for f in per_jaw), f"a jaw never touched the parcel: {per_jaw}"
    for force in per_jaw:
        assert force == pytest.approx(cap, rel=FORCE_TOLERANCE), per_jaw
