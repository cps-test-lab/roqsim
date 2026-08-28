"""The mobile manipulators' wheels must ROLL, not spin backwards.

The sibling of ``roqsim_mobile/tests/test_wheels_roll.py``, which cannot reach these models: they
live in this package, and a test in ``roqsim_mobile`` importing them would invert the dependency
between the two.

The filename is deliberately not ``test_wheels_roll.py`` as well. These test directories have no
``__init__.py``, so pytest identifies a module by its basename and two files sharing one collide with
"import file mismatch" -- the same hazard that made the shared helper here ``mobile_scene_utils.py``
rather than a second ``conftest.py``.

It exists because ``tiago_pro`` was one of the five bases affected by ``omni_drive`` deriving its
wheel roll sign from ``-axis_y`` where rolling gives ``+axis_y`` — and it was the one the sibling
test could not have caught, so it was verified by hand at the time. Verified by hand once is not a
guard.

The invariant is the definition of rolling rather than a convention: without slip the **contact point
is stationary**, ``v_contact = v_centre + omega x r = 0``. Get the sign backwards and its magnitude
is twice the base speed.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: model -> the drive plugin's own name for a driven wheel joint is read at run time, so this is just
#: the list of wheeled robots in this package. `frankie` is a Panda on an Omron LD-60 whose base is
#: driven as a planar body with no wheel joints, so it has nothing to check.
WHEELED = ("tiago_pro",)
SPEED = 0.15


def _driven_wheel_joint(engine) -> str:
    for plugin in engine.plugins:
        kind = type(plugin).__name__
        if "DiffDrive" in kind:
            return plugin._lj_names[0]
        if "OmniDrive" in kind:
            return plugin._wj_names[0]
    raise AssertionError("no drive plugin attached")


@pytest.mark.parametrize("model", WHEELED)
def test_the_wheels_roll_rather_than_spin_backwards(model):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": model, "prefix": "z_"}, "name": "z"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=None))
    engine.setup()
    engine.reset()
    try:
        m, d = engine.ctx.model, engine.ctx.data
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"z_{_driven_wheel_joint(engine)}")
        assert jid >= 0, f"{model}: the drive plugin names a wheel joint the model does not have"
        body = int(m.jnt_bodyid[jid])
        gids = [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == body and int(m.geom_contype[g]) != 0]
        assert gids, f"{model}: the driven wheel body has no collision geom to roll on"
        gid = gids[0]

        handle = engine.ctx.blackboard.get("robot:z")
        for _ in range(400):
            engine.step()
        handle.drive(SPEED, 0.0, 0.0)
        for _ in range(1600):
            engine.step()

        base_speed = abs(float(d.qvel[0]))
        assert base_speed > 0.5 * SPEED, (
            f"{model} barely moved ({base_speed:.4f} m/s of {SPEED}); the rolling check below only "
            f"means something on a robot that is actually driving"
        )
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid], vel, 0)
        radius = float(m.geom_size[gid][0])
        contact = vel[3:] + np.cross(vel[:3], np.array([0.0, 0.0, -radius]))
        slip = float(np.linalg.norm(contact[:2]))
        assert slip < 0.4 * base_speed, (
            f"{model}: wheel contact point is moving at {slip:.4f} m/s while the base does "
            f"{base_speed:.4f}. Near twice the base speed means the wheel is spinning BACKWARDS -- "
            f"check the drive plugin's roll-sign derivation, not the model."
        )
    finally:
        engine.shutdown()
