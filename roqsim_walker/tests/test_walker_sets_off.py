"""A walker that held for a robot sets off again only when the corridor it holds for is clear.

The encounter: a walker with ``avoidance: {steer: none, stop: true}`` holds for a
robot in its way, and the robot comes to stand beside its way, 40 degrees off its heading and close.
The robot is a TurtleBot's shape: a base lower than the walker's scan plane (its obstacle band's
floor plus 5 cm, 0.15 m) and above it only a lidar and a camera bracket, narrower than the rays'
spacing. Standing so, it is in the walker's way and no ray finds it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.presence import entity_geoms

#: A TurtleBot 4 as the probe and the walker's feet meet one: the base the feet can touch, and above
#: it, visual only, the lidar at its centre and the camera bracket at its back.
ROBOT = """<mujoco model="robot">
  <worldbody><body name="base">
    <geom name="body" type="cylinder" size=".164 .03" pos="0 0 .049"/>
    <geom type="cylinder" size=".164 .05" pos="0 0 .05" contype="0" conaffinity="0" group="2"/>
    <geom type="cylinder" size=".036 .025" pos="0 0 .16" contype="0" conaffinity="0" group="2"/>
    <geom type="box" size=".012 .04 .03" pos="-.118 0 .165" contype="0" conaffinity="0" group="2"/>
  </body></worldbody>
</mujoco>"""

START = (-2.0, 0.0)


def _engine(tmp_path):
    (tmp_path / "robot.xml").write_text(ROBOT)
    world = {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "skin": False,
                    "pose": {"position": {"x": START[0], "y": START[1]}},
                },
                "name": "pedestrian",
                "components": [
                    {
                        "navigator": {
                            "output": "walker",
                            "speed": 0.5,
                            "goals": [[4.0, START[1]]],
                            "avoidance": {"steer": "none", "stop": True},
                        }
                    }
                ],
            },
            {
                "spawn_model": {
                    "model": str(tmp_path / "robot.xml"),
                    "motion": "driven",
                    # In the walker's way, its lidar on the middle ray: the walker holds for it.
                    "pose": {
                        "position": {"x": START[0] + 1.0, "y": START[1]},
                        "orientation": {"yaw": -math.pi / 2},
                    },
                },
                "name": "robot",
            },
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=tmp_path))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _navigator(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")


def _run(engine, seconds, watch=None):
    """Step ``seconds``; with ``watch``, the least distance between the walker and the robot."""
    least = math.inf
    pairs = watch or []
    out = np.zeros(6)
    for i in range(int(seconds / engine.ctx.dt)):
        engine.step()
        if pairs and i % 10 == 0:
            m, d = engine.ctx.model, engine.ctx.data
            least = min(least, min(mujoco.mj_geomDistance(m, d, a, b, 1.0, out) for a, b in pairs))
    return least


def _place(engine, xy):
    model, data = engine.ctx.model, engine.ctx.data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("robot").body)
    data.mocap_pos[model.body_mocapid[body]][:2] = xy


def _collidable(model, entity):
    return [g for g in entity_geoms(model, entity, strict=False) if model.geom_contype[g]]


@pytest.fixture
def held(tmp_path):
    engine = _engine(tmp_path)
    try:
        _run(engine, 3.0)
        nav = _navigator(engine)
        assert nav._caution.blocked, "the walker never held for the robot"
        yield engine, nav._state.pos.copy()
    finally:
        engine.shutdown()


def test_a_held_walker_does_not_set_off_into_a_robot_standing_beside_its_way(held):
    engine, at = held
    model = engine.ctx.model
    walker = _collidable(model, engine.ctx.entities.get("pedestrian"))
    robot = _collidable(model, engine.ctx.entities.get("robot"))
    # 40 degrees off its heading, 0.39 m from its centre: the base reaches 0.09 m into its way.
    _place(engine, at + np.array([0.30, 0.25]))
    least = _run(engine, 4.0, watch=[(a, b) for a in walker for b in robot])
    assert least > 0.0, f"the walker walked into the robot ({least * 1000:.0f} mm into it)"
    assert _navigator(engine)._state.pos == pytest.approx(at, abs=0.01)


def test_a_held_walker_sets_off_once_the_robot_has_gone(held):
    engine, at = held
    _place(engine, at + np.array([0.30, 0.25]))
    _run(engine, 2.0)
    _place(engine, at + np.array([0.30, 3.0]))
    _run(engine, 3.0)
    assert _navigator(engine)._state.pos[0] > at[0] + 0.5, "it held for a robot that had gone"


def test_a_held_walker_does_not_pass_a_robot_inside_its_corridor_closer_than_it_stops_for(held):
    """73 degrees off, the base 2 cm clear of where its feet go but inside the corridor it stops for.

    Walking on, it would pass without touching, but closer than it stops for. It holds, or steps
    round the robot (``sidestep``, a walker's default) away from it: it never comes nearer the
    robot than where the robot was put, 2 cm from its feet.
    """
    engine, at = held
    model, data = engine.ctx.model, engine.ctx.data
    walker = _collidable(model, engine.ctx.entities.get("pedestrian"))
    robot = _collidable(model, engine.ctx.entities.get("robot"))
    pairs = [(a, b) for a in walker for b in robot]
    _place(engine, at + np.array([0.10, 0.36]))
    mujoco.mj_forward(model, data)
    put = min(mujoco.mj_geomDistance(model, data, a, b, 1.0, np.zeros(6)) for a, b in pairs)
    least = _run(engine, 6.0, watch=pairs)
    assert least > 0.0, f"it walked into the robot ({least * 1000:.0f} mm into it)"
    assert least > put - 0.01, f"it came to {least * 1000:.0f} mm of it, from {put * 1000:.0f}"


def test_a_held_walker_sets_off_past_a_robot_beside_it_and_clear_of_its_corridor(held):
    """The base's edge 0.34 m to its side, outside the 0.3 m half width it stops for: it walks on."""
    engine, at = held
    _place(engine, at + np.array([0.10, 0.50]))
    _run(engine, 3.0)
    assert _navigator(engine)._state.pos[0] > at[0] + 0.5, "it held for a robot clear of its way"
