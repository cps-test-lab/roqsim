"""The Crazyflie 2 model and its controller: it hovers, it translates, and it needs air.

Three of these tests pin decisions that are invisible to a "does it load" check and expensive to
rediscover:

* ``test_moment_authority_is_tuned`` guards the port's one substantive deviation from upstream.
  Menagerie ships an explicitly arbitrary 1e-5 N*m moment gear (its README says so and asks for
  tuning); at that value the drone hovers perfectly and flies away the instant it is asked to
  translate, because no attitude loop can track the tilt command. Reverting the gear would leave
  every other test here passing except the translation one, which is exactly the trap.
* ``test_warns_in_a_vacuum`` pins the loudness of a silent failure: with no ``density``/``viscosity`` the world
  is a vacuum and the drone is undamped.
* ``test_free_joint_is_named`` pins the substrate convention that makes the drone placeable and
  resettable at all -- upstream's freejoint is anonymous.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config, load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

WORLD = (
    Path(__file__).resolve().parents[1]
    / "src/roqsim_aerial/worlds/crazyflie_2_demo.yaml"
)

#: From MuJoCo Menagerie @ da76818e, which took them from the Crazyflie datasheet and MIT's system
#: identification. Not measured from our model -- that would make the test tautological.
MASS = 0.027
DIAG_INERTIA = (2.3951e-5, 2.3951e-5, 3.2347e-5)
#: Delta 5: derived from the airframe, not upstream's arbitrary 1e-5. See the converter.
MOMENT_GEARS = (4.3e-3, 4.3e-3, 8.0e-4)


def _model():
    asset = resolve_model("roqsim_aerial:crazyflie_2")
    return mujoco.MjModel.from_xml_path(str(asset.path))


def _flown(world=None):
    engine = Engine(load_config(WORLD) if world is None
                    else load_config_from_dict(world, base_dir=WORLD.parent))
    engine.setup()
    engine.reset()
    controller = next(
        p for p in engine.plugins if type(p).__name__ == "QuadrotorControllerPlugin"
    )
    return engine, controller


def _fly(engine, controller, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()
    return np.array(controller.read_state()[:3])


def test_mass_and_inertia_match_upstream():
    model = _model()
    assert model.body_mass.sum() == pytest.approx(MASS, abs=1e-6)
    assert model.body_inertia[1] == pytest.approx(np.array(DIAG_INERTIA), rel=1e-6)


def test_free_joint_is_named():
    # spawn_robot places, resets and teleports through a joint it knows as `base_free`. Upstream's
    # freejoint is anonymous, so without this the drone ignores the world's spawn pose entirely.
    model = _model()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free") >= 0


def test_no_option_block():
    # Integrator AND the medium terms are world-scoped; they live in the world's sim block. Parse rather than
    # grep -- the header comment says the word "<option>".
    import xml.etree.ElementTree as ET

    asset = resolve_model("roqsim_aerial:crazyflie_2")
    assert ET.parse(asset.path).getroot().find("option") is None


def test_moment_authority_is_tuned():
    """The moment gears must be the airframe-derived values, not upstream's arbitrary 1e-5.

    At 1e-5 N*m against a 2.4e-5 kg*m^2 inertia the drone gets 0.42 rad/s^2 -- one second to tilt
    12 degrees -- and cannot translate under position control at all.
    """
    model = _model()
    gears = [
        abs(float(model.actuator_gear[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)][3 + i]))
        for i, name in enumerate(("x_moment", "y_moment", "z_moment"))
    ]
    assert gears == pytest.approx(np.array(MOMENT_GEARS), rel=1e-6)
    ang_accel = gears[0] / DIAG_INERTIA[0]
    assert ang_accel > 100, f"only {ang_accel:.1f} rad/s^2 of roll authority; it cannot translate"


def test_demo_world_asks_for_air():
    model = Engine(load_config(WORLD))
    model.setup()
    try:
        assert float(model.ctx.model.opt.density) == pytest.approx(1.225)
        assert float(model.ctx.model.opt.viscosity) == pytest.approx(1.8e-5)
    finally:
        model.shutdown()


def test_warns_in_a_vacuum(caplog):
    world = {
        "sim": {},  # deliberately no physics block
        "components": [{
            "spawn_robot": {"model": "crazyflie_2", "prefix": "cf2_"},
            "name": "drone",
            "components": [{"quadrotor_controller": {}}],
        }],
    }
    with caplog.at_level(logging.WARNING):
        engine, _ = _flown(world)
        engine.shutdown()
    assert any("vacuum" in r.message for r in caplog.records), (
        "a world with no medium must say so: the drone hovers but is undamped, which reads as bad "
        "gains rather than as missing air"
    )


def test_takeoff_and_hover():
    engine, controller = _flown()
    try:
        pos = _fly(engine, controller, 6.0)
        assert pos[2] == pytest.approx(1.0, abs=0.02), f"did not reach 1 m, got {pos[2]:.3f}"
        assert np.hypot(*pos[:2]) < 0.02, f"drifted {np.hypot(*pos[:2]):.3f} m while hovering"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("target", [(1.0, 0.5, 1.2), (-0.5, -0.5, 0.6)])
def test_translates_to_setpoint(target):
    """The test the untuned moment gear fails: hovering is not flying."""
    engine, controller = _flown()
    try:
        _fly(engine, controller, 6.0)
        controller.set_target(*target)
        pos = _fly(engine, controller, 10.0)
        error = np.linalg.norm(pos - np.array(target))
        assert error < 0.05, f"settled {error:.3f} m from {target}, at {tuple(round(v,3) for v in pos)}"
    finally:
        engine.shutdown()


def test_reset_returns_to_spawn():
    engine, controller = _flown()
    try:
        _fly(engine, controller, 4.0)
        controller.set_target(1.0, 1.0, 1.5)
        _fly(engine, controller, 4.0)
        engine.reset()
        assert np.allclose(np.array(controller.read_state()[:3]), 0.0, atol=1e-6)
    finally:
        engine.shutdown()
