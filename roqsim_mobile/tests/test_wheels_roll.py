"""Every wheeled base's wheels must ROLL, not spin backwards. One invariant, all models.

This file exists because a real defect survived a full suite. ``omni_drive`` derived its wheel roll
sign from ``-axis_y`` where the physics gives ``+axis_y``, so all five holonomic bases span their
wheels exactly backwards from 2026-08-28 and earlier. Nothing caught it: the wheels of an
``omni_drive`` base are deliberately near-frictionless load carriers and the base is driven through
planar actuators, so the sign never touched the dynamics. It was wrong only where those servos
actually matter — the viewer, and ``joint_states``.

The check is the definition of rolling rather than a convention, which is the point. For a wheel
rolling without slip the **contact point is stationary**:

    v_contact = v_centre + omega x r,  r = (0, 0, -R)  ->  0

Get the sign backwards and ``|v_contact|`` is twice the base speed instead of nearly zero. That is
measurable without agreeing on which way is positive, which is what made it able to settle a
question two plugins disagreed about: at 0.2 m/s the diff_drive bases read 0.0001-0.002 m/s and the
omni_drive bases read 0.379-0.383.

Per-model scene tests check each robot against its own vendor's numbers. This one checks every robot
against physics, and is deliberately indifferent to which drive plugin it uses.
"""

from __future__ import annotations

# `roqsim` selects MuJoCo's GL backend on import, and MUJOCO_GL is read once while `import mujoco`
# runs -- so roqsim must come first or a model with a camera (turtlebot4's OAK-D) fails to build an
# offscreen renderer. See roqsim/CLAUDE.md; the E402 ignore below is what keeps isort from "tidying"
# this back into one block.
import roqsim  # noqa: F401, I001
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mobile_scene_utils import named  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402

#: Every wheeled model in this package. No geom or joint names: those are read off the attached
#: drive plugin's own config, so this table cannot drift from a manifest and does not care that
#: turtlebot4's wheel geom is unnamed while everyone else's is named three different ways.
WHEELED = (
    "clearpath_jackal", "husky_a200", "lgdxrobot2", "makerspet_loki", "mp_400", "mpo_500",
    "mpo_700", "oomwoo_one", "panther", "raspimouse", "ridgeback", "rosbot",
    "turtlebot3_waffle", "turtlebot4",
)
#: Commanded forward speed. Low enough that every platform here reaches it well inside a 10 m room
#: and inside its own published limit -- the slowest is the OOMWOO vacuum at 0.2 m/s.
SPEED = 0.15


def _driven_wheel_joint(engine) -> str:
    """One driven wheel joint's name, taken from whichever drive plugin attached.

    Asking the plugin rather than hardcoding is what keeps this test honest: the thing under test is
    the plugin's roll-sign derivation, so the wheel it is asked about should be a wheel the plugin
    itself believes it is driving.
    """
    for plugin in engine.plugins:
        kind = type(plugin).__name__
        if "DiffDrive" in kind:
            # `_lj_names` rather than the raw config: diff_drive defaults the joint names when a
            # manifest omits them, and several here do. Reading the resolved list is the point --
            # it is what the plugin actually drives.
            return plugin._lj_names[0]
        if "OmniDrive" in kind:
            return plugin._wj_names[0]
    raise AssertionError("no drive plugin attached")


def _step(engine, n: int, model: str) -> None:
    """Step *n* times, skipping the test if this process has no offscreen GL backend.

    MuJoCo binds its GL backend during ``import mujoco``, once per process, and a camera-carrying
    model renders in ``post_step`` -- so turtlebot4's OAK-D needs one. This file imports roqsim
    first to get it, but when an earlier test module in the same session imported mujoco first the
    backend is already chosen and no import order here can change it. Skipping with the reason beats
    a failure that depends on pytest's collection order, and the check is unaffected on every model
    without a camera.
    """
    for _ in range(n):
        try:
            engine.step()
        except Exception as exc:  # noqa: BLE001 -- re-raised unless it is the backend error
            if type(exc).__name__ != "GLBackendError":
                raise
            pytest.skip(
                f"{model} carries a camera and this process bound an on-screen GL backend before "
                f"roqsim could choose one; run this file alone or set MUJOCO_GL=egl"
            )


@pytest.mark.parametrize("model", WHEELED)
def test_the_wheels_roll_rather_than_spin_backwards(model):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": model, "prefix": "z_"}, "name": "z"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    try:
        m, d = engine.ctx.model, engine.ctx.data
        jid = named(m, mujoco.mjtObj.mjOBJ_JOINT, f"z_{_driven_wheel_joint(engine)}")
        body = int(m.jnt_bodyid[jid])
        # The wheel's contacting geom, found by whether it can collide at all -- not by name and not
        # by group. Names differ three ways and turtlebot4's is unnamed; groups differ too, because
        # the older hand-written models put wheel collision in their own per-model class while the
        # generated ones use group 3. `contype != 0` is the one property every contacting geom has.
        gids = [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == body and int(m.geom_contype[g]) != 0]
        assert gids, f"{model}: the driven wheel body has no collision geom to roll on"
        gid = gids[0]
        handle = engine.ctx.blackboard.get("robot:z")
        _step(engine, 400, model)
        handle.drive(SPEED, 0.0, 0.0)
        _step(engine, 1600, model)

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
