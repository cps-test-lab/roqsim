"""A GripperCommand position puts the gripper joint there, for every arm that carries a gripper.

The command goes through the ``gripper_cmd`` endpoint, as the bridge sends it, and the result is read
from the joint itself rather than from the controller's reader. The open and closed positions are the
vendors' own, not the manifests', so a manifest that states them the wrong way round, or a controller
that maps them onto the wrong end of the actuator's range, fails here: an inverted gripper answers a
command at a quarter of its travel with three quarters.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: spawn_arm config, gripper joint, (open, closed) joint position from the vendor's description.
GRIPPERS = [
    # Interbotix xsarm descriptions: finger_limit_lower/upper; the xs_modules gripper `release()`
    # drives left_finger to its upper limit, `grasp()` to its lower.
    pytest.param({"model": "vx300s"}, "left_finger", (0.057, 0.021), id="vx300s"),
    pytest.param({"model": "wx250s"}, "left_finger", (0.037, 0.015), id="wx250s"),
    # robotiq_description: the knuckle's 0 is open, 0.8 closed.
    pytest.param(
        {"model": "gen3"}, "robotiq_85_left_knuckle_joint", (0.0, 0.8), id="gen3+robotiq_2f85"
    ),
    pytest.param(
        {"model": "ur5e", "end_effector": {"model": "robotiq_2f85"}},
        "robotiq_85_left_knuckle_joint",
        (0.0, 0.8),
        id="ur5e+robotiq_2f85",
    ),
    # PILZ prbt_pg70 URDF: each jaw 0.0301 m out when open.
    pytest.param(
        {"model": "ur5e", "end_effector": {"model": "schunk_pg70"}},
        "finger_left_joint",
        (0.0301, -0.001),
        id="ur5e+schunk_pg70",
    ),
    # franka_description: finger_joint1 0.04 m apart per finger when open.
    pytest.param({"model": "panda"}, "finger_joint1", (0.04, 0.0), id="panda"),
]

#: Fractions of the travel from open toward closed that are commanded. The closed end stays out: a
#: gripper closing on nothing may meet its own pads before its joint limit.
FRACTIONS = (0.0, 0.25, 0.75)
#: How far the joint may settle from the command, as a fraction of the travel.
TOLERANCE = 0.03
SETTLE_STEPS = 1500  # 3 s at 2 ms


def _joint(model, name: str) -> int:
    for jid in range(model.njnt):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").endswith(name):
            return jid
    raise AssertionError(f"no joint ending in {name!r}")


@pytest.mark.parametrize("spawn,joint,travel", GRIPPERS)
def test_a_gripper_command_puts_the_gripper_joint_there(spawn, joint, travel):
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"timestep": 0.002},
                "components": [{"spawn_arm": dict(spawn, prefix="a_"), "name": "a"}],
            },
            base_dir=Path("."),
        )
    )
    engine.setup()
    engine.reset()
    model, data = engine.ctx.model, engine.ctx.data
    endpoint = {e.name: e for e in engine.ctx.interface.all()}["gripper_cmd"]
    adr = model.jnt_qposadr[_joint(model, joint)]
    opened, closed = travel

    for fraction in FRACTIONS:
        command = opened + fraction * (closed - opened)
        endpoint.write({"position": command})
        for _ in range(SETTLE_STEPS):
            engine.step()
        assert float(data.qpos[adr]) == pytest.approx(
            command, abs=TOLERANCE * abs(closed - opened)
        ), f"commanded {command:.4f} ({fraction:.0%} closed)"
