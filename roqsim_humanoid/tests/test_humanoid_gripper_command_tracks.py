"""A GripperCommand position puts the gripper joint there, on each humanoid's hand.

The command goes through the ``gripper_cmd`` endpoint, as the bridge sends it, and the result is read
from the joint itself. The open and closed positions are the vendor's own, not the manifests', so a
manifest that states them the wrong way round, or a controller that maps them onto the wrong end of
the actuator's range, fails here: an inverted gripper answers a command at a quarter of its travel
with three quarters.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: Robot, gripper action name, gripper joint, (open, closed) joint position from the vendor.
GRIPPERS = [
    # Unitree's dex1 URDF: each finger slides -0.02 .. 0.0245 m; the upper end is the wide one
    # (94.9 mm between the pads, 5.9 mm at the lower).
    pytest.param(
        "unitree_g1_dex1",
        f"{side}_gripper_controller/gripper_cmd",
        f"{side}_dex1_finger_joint_1",
        (0.0245, -0.02),
        id=f"unitree_g1_dex1-{side}",
    )
    for side in ("left", "right")
]

#: Fractions of the travel from open toward closed that are commanded. The closed end stays out: a
#: gripper closing on nothing may meet its own pads before its joint limit.
FRACTIONS = (0.0, 0.25, 0.75)
#: How far the joint may settle from the command, as a fraction of the travel.
TOLERANCE = 0.03
SETTLE_STEPS = 1500  # 3 s at 2 ms


@pytest.mark.parametrize("robot,action,joint,travel", GRIPPERS)
def test_a_gripper_command_puts_the_gripper_joint_there(robot, action, joint, travel):
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {"spawn_robot": {"model": robot, "pose": {"position": {"x": 0, "y": 0}}}, "name": "r"}
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0  # a test driving an Engine is the driver, and the seed is driver-owned
    engine.setup()
    engine.reset()
    model, data = engine.ctx.model, engine.ctx.data
    endpoint = next(
        e
        for e in engine.ctx.interface.all()
        if e.name == "gripper_cmd" and e.backend["ros2"]["name"] == action
    )
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    opened, closed = travel

    for fraction in FRACTIONS:
        command = opened + fraction * (closed - opened)
        endpoint.write(command)
        for _ in range(SETTLE_STEPS):
            engine.step()
        assert float(data.qpos[adr]) == pytest.approx(
            command, abs=TOLERANCE * abs(closed - opened)
        ), f"commanded {command:.4f} ({fraction:.0%} closed)"
