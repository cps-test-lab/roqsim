"""The ``fr3`` model is franka_description's FR3, checked against the pinned URDF with Pinocchio.

The URDF is what ``robots/fr3/fr3.urdf.xacro hand:=false`` produces at the pinned release
(``upstream/franka/franka_description/fr3.urdf``); Pinocchio is the library libfranka computes its
``Model`` with. The MJCF must give the same joint limits, effort limits, masses, kinematics, mass
matrix (less the rotor inertia, which the URDF does not carry) and gravity torques, and the URDF
the simulated robot serves through ``GetRobotModel`` must be upstream's, byte for byte.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import pytest
from _franka_upstream import DESCRIPTION, joint_limits, text
from roqsim_manipulation.franka.robot_model import RobotModel

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

MJCF = Path(resolve_model("fr3").path)


@pytest.fixture(scope="module")
def pinocchio_model():
    model = pin.buildModelFromXML(text(DESCRIPTION / "fr3.urdf"))
    return model, model.createData()


@pytest.fixture(scope="module")
def spawned():
    world = {
        "sim": {"timestep": 0.001},
        "components": [{"spawn_arm": {"model": "fr3", "prefix": "left_"}, "name": "left"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")), preview=True)
    engine.setup()
    yield engine.ctx.model
    engine.shutdown()


def test_joint_names_ranges_and_effort_limits(spawned):
    m = spawned
    jl = joint_limits()
    for i in range(1, 8):
        limit = jl[f"joint{i}"]["limit"]
        jid = m.joint(f"left_fr3_joint{i}").id
        aid = m.actuator(f"left_fr3_joint{i}").id
        assert list(m.jnt_range[jid]) == pytest.approx([limit["lower"], limit["upper"]], abs=1e-12)
        assert list(m.actuator_forcerange[aid]) == pytest.approx(
            [-limit["effort"], limit["effort"]]
        )
        assert m.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_NONE  # a torque motor, no servo


def test_masses_are_the_urdfs(pinocchio_model):
    """To MJCF's printed precision: MuJoCo writes six significant digits."""
    pm, _ = pinocchio_model
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    for i in range(8):
        assert m.body(f"fr3_link{i}").mass[0] == pytest.approx(pm.inertias[i].mass, rel=1e-5)


def _q(rng):
    jl = joint_limits()
    lo = np.array([jl[f"joint{i}"]["limit"]["lower"] for i in range(1, 8)])
    hi = np.array([jl[f"joint{i}"]["limit"]["upper"] for i in range(1, 8)])
    return rng.uniform(lo, hi)


def test_kinematics_mass_matrix_and_gravity_match_pinocchio(pinocchio_model):
    pm, pd = pinocchio_model
    robot = RobotModel(MJCF, np.array([0.0, 0.0, -9.81]))
    flange = pm.getFrameId("fr3_link8")
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = _q(rng)
        robot.update(q, np.zeros(7))
        pin.forwardKinematics(pm, pd, q)
        pin.updateFramePlacements(pm, pd)
        T = robot.flange_pose()
        assert np.allclose(T[:3, 3], pd.oMf[flange].translation, atol=1e-9)
        assert np.allclose(T[:3, :3], pd.oMf[flange].rotation, atol=1e-9)
        assert np.allclose(robot.mass(), pin.crba(pm, pd, q), atol=1e-6)
        assert np.allclose(robot.gravity(), pin.computeGeneralizedGravity(pm, pd, q), atol=1e-6)


def test_the_served_urdf_is_upstreams():
    served = MJCF.with_suffix(".robot_model.urdf").read_text(encoding="utf-8")
    assert served == text(DESCRIPTION / "fr3_no_prefix.urdf")
    assert '<link name="link8"' in served  # libfranka's Model looks the flange up by this name


def test_the_rotor_inertia_is_on_the_motor_side_only():
    """The arm's own model carries the identified armature; the robot's model (link side) does not."""
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    assert np.all(m.dof_armature > 0)
    robot = RobotModel(MJCF, np.array([0.0, 0.0, -9.81]))
    assert np.all(robot.model.dof_armature == 0)
