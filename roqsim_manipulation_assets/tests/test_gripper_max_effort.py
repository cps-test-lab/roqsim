# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""GripperCommand's ``max_effort``: a clamp on the gripper joint's effort, as ros2_control applies it.

Four layers, because each catches a different mistake:

* **the transmission** -- the gripper joint's effort per unit of actuator force, read from the model,
  for every gripper the substrate ships. A gripper without a constant moment could not take a clamp.
* **the plugin** -- what ``arm_controller`` publishes and does: the endpoint's ``effort_key`` hint, the
  limit, the force range a clamp writes, saturation at the model's own range, and the reset.
* **the callers** -- the values existing scenarios and MoveIt send are accepted, never refused.
* **the physics** -- that the clamp is what the gripper puts into an object: the force each PG+70 jaw
  presses with, and the effort a 2F-85 knuckle saturates at when closed on a parcel.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import joint_effort_per_actuator_force

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import apply_assets, resolve_model

#: Every shipped gripper: its gripper joint, the effort on it per unit of actuator force, and the
#: joint-effort limit its actuator's force range allows (N for a slide jaw, N*m for a knuckle).
SHIPPED = [
    pytest.param("robotiq_2f85", "robotiq_85_left_knuckle_joint", 0.5, 2.5, id="robotiq_2f85"),
    pytest.param("gen3", "robotiq_85_left_knuckle_joint", 0.5, 2.5, id="gen3"),
    pytest.param("schunk_pg70", "finger_left_joint", 0.5, 100.0, id="schunk_pg70"),
    pytest.param("unitree_g1_dex1", "left_dex1_finger_joint_1", 1.0, 20.0, id="unitree_g1_dex1"),
    pytest.param("vx300s", "left_finger", 1.0, 35.0, id="vx300s"),
    pytest.param("wx250s", "left_finger", 1.0, 35.0, id="wx250s"),
    pytest.param("tiago_pro", "gripper_left_finger_joint", 1.0, 10.0, id="tiago_pro"),
]

#: The PG+70's limit on one jaw, and the two clamps the physics check closes at, inside it.
PG70_LIMIT_N = 100.0
PG70_CLAMPS_N = (20.0, 50.0)

#: The 2F-85's knuckle limit, and a clamp inside it.
F85_LIMIT_NM = 2.5
F85_CLAMP_NM = 1.0

#: Efforts that scenarios and MoveIt configurations send a 2F-85, both above its limit: they run with
#: the model's own force range.
EXISTING_REQUESTS = (50.0, 80.0)

#: How far a measured force or effort may sit from the clamp. Measured on it; the band leaves room
#: for solver settings, not for a transmission that is off by a factor.
TOLERANCE = 0.05

_PARCEL_WORLD = """<mujoco model="grip_force">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"
          noslip_iterations="10" gravity="0 0 0"/>
  <worldbody>
    <body name="parcel" pos="0 0 {pz}">
      <freejoint/>
      <geom name="parcel" type="box" size="0.03 0.0225 0.01875" mass="0.211" condim="4"
            friction="1.2 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.005 1"/>
    </body>
  </worldbody>
</mujoco>"""

#: Each gripper's grasp geometry, as test_grippers.py measures it: jaw plane below the palm, jaw
#: half-height, the joints set to the open stance, that stance, and the actuator's (open, close) ctrl.
_GRASP = {
    "schunk_pg70": (
        0.0789,
        0.005,
        ("finger_left_joint", "finger_right_joint"),
        0.0301,
        (-0.0301, 0.001),
    ),
    "robotiq_2f85": (
        0.1488,
        0.019,
        ("robotiq_85_left_knuckle_joint", "right_driver_joint"),
        0.0,
        (0.0, 255.0),
    ),
}


def _compiled(model: str) -> mujoco.MjModel:
    asset = resolve_model(model)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    return spec.compile()


def _named(model, obj, name: str) -> int:
    """The id of the object whose name ends in ``name``: composite models prefix their parts."""
    count = {mujoco.mjtObj.mjOBJ_JOINT: model.njnt, mujoco.mjtObj.mjOBJ_GEOM: model.ngeom}[obj]
    for i in range(count):
        if (mujoco.mj_id2name(model, obj, i) or "").endswith(name):
            return i
    raise AssertionError(f"no {obj} ending in {name!r}")


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


def _effort(engine: Engine):
    endpoint = {e.name: e for e in engine.ctx.interface.all()}["gripper_cmd"]
    return engine.ctx.blackboard.require(endpoint.backend["ros2"]["effort_key"])


def _gripper_actuator(engine: Engine, name: str) -> int:
    model = engine.ctx.model
    for aid in range(model.nu):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or "").endswith(name):
            return aid
    raise AssertionError(f"no actuator ending in {name!r}")


@pytest.mark.parametrize("model,joint,gain,limit", SHIPPED)
def test_every_shipped_gripper_can_take_a_clamp(model, joint, gain, limit):
    """Each has a constant moment on its gripper joint, so the clamp is exact in every pose."""
    m = _compiled(model)
    jid = _named(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    reaching = [
        (joint_effort_per_actuator_force(m, aid, jid), aid)
        for aid in range(m.nu)
        if joint_effort_per_actuator_force(m, aid, jid) > 0.0
    ]
    assert reaching, f"{model}: no actuator reaches {joint} through a constant moment"
    found_gain, aid = reaching[0]
    assert found_gain == pytest.approx(gain)
    assert bool(m.actuator_forcelimited[aid])
    assert min(abs(v) for v in m.actuator_forcerange[aid]) * found_gain == pytest.approx(limit)


@pytest.mark.parametrize(
    "gripper,limit", [("schunk_pg70", PG70_LIMIT_N), ("robotiq_2f85", F85_LIMIT_NM)]
)
def test_the_endpoint_names_the_effort_entry_and_its_limit(tmp_path, gripper, limit):
    assert _effort(_engine(tmp_path, gripper)).limit == pytest.approx(limit)


def test_a_clamp_writes_the_force_range_and_saturates_at_the_model(tmp_path):
    engine = _engine(tmp_path, "schunk_pg70")
    engine.reset()
    effort = _effort(engine)
    aid = _gripper_actuator(engine, "finger_actuator")
    model_range = tuple(engine.ctx.model.actuator_forcerange[aid])

    effort.set_max_effort(PG70_CLAMPS_N[0])
    actuator_limit = PG70_CLAMPS_N[0] / 0.5
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(
        (-actuator_limit, actuator_limit)
    )

    effort.set_max_effort(PG70_LIMIT_N + 50.0)
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(model_range), (
        "a request above the limit saturates at the model's own range, as a drive does"
    )

    effort.set_max_effort(0.0)
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(model_range)

    effort.set_max_effort(PG70_CLAMPS_N[1])
    engine.reset()
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(model_range), (
        "a trial's grip force must not leak into the next trial"
    )


def test_a_knuckle_takes_a_clamp_in_newton_metres(tmp_path):
    engine = _engine(tmp_path, "robotiq_2f85")
    engine.reset()
    effort = _effort(engine)
    aid = _gripper_actuator(engine, "fingers_actuator")
    effort.set_max_effort(F85_CLAMP_NM)
    actuator_limit = F85_CLAMP_NM / 0.5
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(
        (-actuator_limit, actuator_limit)
    )


@pytest.mark.parametrize("request_value", EXISTING_REQUESTS)
def test_what_existing_callers_send_is_accepted(tmp_path, request_value):
    """Above the 2F-85's limit, so the grip is the model's own -- what these callers always had."""
    engine = _engine(tmp_path, "robotiq_2f85")
    engine.reset()
    aid = _gripper_actuator(engine, "fingers_actuator")
    model_range = tuple(engine.ctx.model.actuator_forcerange[aid])
    _effort(engine).set_max_effort(request_value)
    assert tuple(engine.ctx.model.actuator_forcerange[aid]) == pytest.approx(model_range)


def test_the_effort_read_is_the_one_joint_states_reports(tmp_path):
    engine = _engine(tmp_path, "schunk_pg70")
    engine.reset()
    for _ in range(50):
        engine.step()
    names, _, _, efforts = engine.ctx.blackboard.require("arm:ur5e").read_state()
    assert _effort(engine).read_effort() == pytest.approx(efforts[names.index("finger_left_joint")])


def _closed_on_a_parcel(gripper: str, effort_limit: float):
    """The gripper closed on a parcel in zero g, with its joint effort clamped at ``effort_limit``."""
    jaw_plane, jaw_half_z, joints, home, (ctrl_open, ctrl_close) = _GRASP[gripper]
    palm_z = 0.5
    world = mujoco.MjSpec.from_string(
        _PARCEL_WORLD.format(pz=palm_z - (jaw_plane + 0.01875 - jaw_half_z))
    )
    frame = world.worldbody.add_frame()
    frame.pos = [0.0, 0.0, palm_z]
    frame.quat = [0.0, 1.0, 0.0, 0.0]  # approach axis pointing down at the parcel
    asset = resolve_model(gripper)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    world.attach(spec, prefix="grip_", frame=frame)
    m = world.compile()
    d = mujoco.MjData(m)

    joint_ids = [_named(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_" + j) for j in joints]
    for jid in joint_ids:
        d.qpos[m.jnt_qposadr[jid]] = home
    mujoco.mj_forward(m, d)
    aid = next(a for a in range(m.nu) if joint_effort_per_actuator_force(m, a, joint_ids[0]) > 0.0)
    gain = joint_effort_per_actuator_force(m, aid, joint_ids[0])
    m.actuator_forcerange[aid] = (-effort_limit / gain, effort_limit / gain)
    d.ctrl[aid] = ctrl_open
    for _ in range(500):
        mujoco.mj_step(m, d)
    d.ctrl[aid] = ctrl_close
    for _ in range(1500):
        mujoco.mj_step(m, d)
    return m, d, joint_ids[0]


@pytest.mark.parametrize("clamp", PG70_CLAMPS_N)
def test_each_pg70_jaw_presses_with_the_clamp(clamp):
    m, d, _ = _closed_on_a_parcel("schunk_pg70", clamp)
    pads = [_named(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("finger_left_pad", "finger_right_pad")]
    parcel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel")
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
        assert force == pytest.approx(clamp, rel=TOLERANCE), per_jaw


def test_a_2f85_knuckle_saturates_at_the_clamp_on_a_parcel():
    """Blocked by the parcel, the drive saturates, and the knuckle carries exactly the clamp."""
    m, d, knuckle = _closed_on_a_parcel("robotiq_2f85", F85_CLAMP_NM)
    assert d.ncon > 0, "the jaws never touched the parcel"
    dof = m.jnt_dofadr[knuckle]
    effort = abs(float(d.qfrc_actuator[dof] + d.qfrc_gravcomp[dof]))
    assert effort == pytest.approx(F85_CLAMP_NM, rel=TOLERANCE)
