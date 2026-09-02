"""The X500 airframe: the numbers PX4's mixer and gains are valid against.

These are model-level tests on purpose. Everything here is a stated figure that another
workstream depends on -- the motor ORDER (a permuted table flies a drone stable in yaw and
divergent in roll), the ctrlrange (which fixes the thrust-to-weight margin the payload campaign
varies against), the mass and the sensor names. None of it is checked by "does it load".
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.models import resolve_model

#: Published X500 V2 figures and PX4's own airframe parameters; see the MJCF header for where each
#: came from. Written here as the PX4 values, not measured from our model, which would make the
#: assertion tautological -- the point of these tests is that the MJCF agrees with PX4's mixer.
MASS = 2.0
DIAG_INERTIA = (0.02, 0.02, 0.04)
#: PX4 CA_ROTOR*_PX / _PY for 4001_gz_x500, with the FRD -> FLU y-sign flip applied.
ROTOR_OFFSET = 0.174
#: MPC_THR_HOVER 0.6 -> full total 19.62/0.6 = 32.7 N -> 8.175 N per rotor, written 8.2.
MAX_THRUST = 8.2
THR_HOVER = 0.6
HOVER_PER_ROTOR = MASS * 9.81 / 4.0

#: The CONTRACT's motor table: ctrl index -> (sign of x, sign of y).
QUADRANTS = {
    "rotor0": (+1, -1),  # front-right
    "rotor1": (-1, +1),  # rear-left
    "rotor2": (+1, +1),  # front-left
    "rotor3": (-1, -1),  # rear-right
}


def _model():
    asset = resolve_model("roqsim_aerial:x500")
    return mujoco.MjModel.from_xml_path(str(asset.path))


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def test_model_compiles():
    model = _model()
    assert model.nq == 7 and model.nv == 6, "one free body, nothing else"


def test_mass_is_the_stated_value():
    # Stated on the root <inertial>, not integrated from geoms -- see the header. That is what makes
    # it a citable number rather than a side effect of where the landing legs were drawn.
    model = _model()
    assert model.body_mass.sum() == pytest.approx(MASS, abs=1e-9)
    bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "x500")
    assert model.body_inertia[bid] == pytest.approx(np.array(DIAG_INERTIA), rel=1e-9)


def test_free_joint_is_named():
    # spawn_robot places, resets and teleports through a joint it knows as `base_free`.
    model = _model()
    assert _id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free") >= 0


def test_no_option_block():
    # Integrator AND the medium terms are world-scoped: they belong to the world's sim block.
    import xml.etree.ElementTree as ET

    assert ET.parse(resolve_model("roqsim_aerial:x500").path).getroot().find("option") is None


def test_no_mesh_assets():
    """Hand-authored from primitives, deliberately: nothing here resolves against a mesh dir."""
    model = _model()
    assert model.nmesh == 0


def test_actuators_in_px4_motor_order():
    model = _model()
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)
    ]
    assert names == [f"rotor{i}_thrust" for i in range(4)], (
        "ctrl index -> rotor is PX4's mixer's contract, not ours to reorder"
    )
    for a in range(model.nu):
        assert tuple(model.actuator_ctrlrange[a]) == pytest.approx((0.0, MAX_THRUST))
        # gear "0 0 1 0 0 0": pure force along the site's +z, no torque. Yaw reaction is the
        # motors plugin's job, being a motor/propeller property rather than a frame one.
        assert model.actuator_gear[a][:6] == pytest.approx(np.array([0, 0, 1, 0, 0, 0]))


def test_thrust_to_weight_margin_matches_px4s_hover_throttle():
    """The ctrlrange is PX4's MPC_THR_HOVER, not a generous round number.

    PX4 boots expecting this airframe to hover at 60% throttle. If the model's full thrust does not
    put hover at 0.6, PX4's hover-throttle prior and its thrust feedforward are mis-scaled from the
    first tick -- at a 15 N ctrlrange, by nearly a factor of two. The payload campaign also varies
    mass against this margin, so it is doubly a pinned number.
    """
    model = _model()
    total_max = float(sum(model.actuator_ctrlrange[a][1] for a in range(model.nu)))
    assert HOVER_PER_ROTOR / MAX_THRUST == pytest.approx(THR_HOVER, abs=0.005), (
        "hover thrust over full thrust must be PX4's MPC_THR_HOVER"
    )
    assert total_max / (MASS * 9.81) == pytest.approx(1.0 / THR_HOVER, abs=0.02)


def test_rotor_sites_match_px4s_ca_rotor_table():
    """The MJCF's rotor geometry is derived from PX4's CA_ROTOR params; it must equal them.

    PX4's allocator inverts these positions to turn a commanded moment into differential thrust. A
    frame whose arms are longer or shorter than the mixer believes produces a drone that is stable
    but wrongly geared in roll and pitch, which reads as bad tuning.
    """
    model = _model()
    for name, (sx, sy) in QUADRANTS.items():
        sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid >= 0, f"missing rotor site {name}"
        pos = model.site_pos[sid]
        assert np.sign(pos[0]) == sx, f"{name} is on the wrong side in x"
        assert np.sign(pos[1]) == sy, f"{name} is on the wrong side in y"
        assert abs(pos[0]) == pytest.approx(ROTOR_OFFSET, abs=1e-6), "CA_ROTOR PX"
        assert abs(pos[1]) == pytest.approx(ROTOR_OFFSET, abs=1e-6), "CA_ROTOR PY (FRD -> FLU)"
        assert pos[2] == pytest.approx(0.040, abs=1e-6), (
            "the thrust-line-to-CoM offset is a flight-dynamics parameter, pinned in the MJCF"
        )
    # Arm length and wheelbase are consequences of the CA_ROTOR offsets, not independent choices.
    assert ROTOR_OFFSET * np.sqrt(2.0) == pytest.approx(0.246, abs=1e-3), "arm length"
    assert 2 * ROTOR_OFFSET * np.sqrt(2.0) == pytest.approx(0.492, abs=1e-3), "wheelbase"


def test_imu_and_sensors():
    model = _model()
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, "imu") >= 0
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in range(model.nsensor)
    }
    assert {"body_gyro", "body_linacc", "body_quat"} <= names
    imu = _id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    assert model.site_pos[imu] == pytest.approx(np.zeros(3)), (
        "PX4's EKF2 assumes the IMU sits at the body origin"
    )


def test_hover_keyframe_holds_the_drone():
    """Hover thrust exactly cancels weight, so the drone should stay where it is put."""
    model = _model()
    data = mujoco.MjData(model)
    key = _id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    assert key >= 0
    mujoco.mj_resetDataKeyframe(model, data, key)
    assert data.ctrl == pytest.approx(np.full(4, HOVER_PER_ROTOR), abs=1e-3)

    z0 = float(data.qpos[2])
    assert z0 == pytest.approx(0.1)
    for _ in range(int(1.0 / model.opt.timestep)):
        data.ctrl[:] = np.full(4, HOVER_PER_ROTOR)
        mujoco.mj_step(model, data)
    dz = float(data.qpos[2]) - z0
    # Exact cancellation still integrates a little numerically, and the keyframe rounds 4.905 N;
    # a few millimetres over a second is that, a centimetre is a wrong mass or a wrong ctrlrange.
    assert abs(dz) < 0.005, f"drifted {dz*1000:.2f} mm in one second of hover"
    assert np.linalg.norm(data.qpos[:2]) < 1e-6, "a symmetric quad must not translate at hover"
