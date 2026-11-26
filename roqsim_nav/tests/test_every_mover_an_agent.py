"""Steering goes round everything that moves on the floor, with or without a navigator of its own.

A robot driven by its own stack, one standing still, a pushed cart: each joins the world's
avoidance model as an agent the others go round, measured from its state every tick. The pass keeps
``clearance`` (and ``clearance_time`` more per m/s of closing speed); ``give_way`` passes a mover
crossing its way behind it, on the side it comes from; a mover's ``ignore`` applies to its steering.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim_nav.avoidance import RegistryError

DATA = Path(__file__).parent / "data"

CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".2 .2 .25"/></body></worldbody>
</mujoco>"""

#: A TurtleBot's shape: a low base and a slim post.
ROBOT = """<mujoco model="robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".17 .04" pos="0 0 .04"/>
    <geom name="post" type="cylinder" size=".03 .15" pos="0 0 .2" contype="0" conaffinity="0"/>
  </body></worldbody>
</mujoco>"""


def _world(
    tmp_path,
    *,
    model="give_way",
    robot_at=(0.0, 0.0),
    robot_route=None,
    avoidance=None,
    params=None,
):
    (tmp_path / "crate.xml").write_text(CRATE)
    (tmp_path / "robot.xml").write_text(ROBOT)
    robot = {
        "spawn_model": {
            "model": str(tmp_path / "robot.xml"),
            "prefix": "r_",
            "motion": "driven",
            "pose": {"position": {"x": robot_at[0], "y": robot_at[1], "z": 0.0}},
        },
        "name": "robot",
    }
    if robot_route:
        robot["components"] = [
            {
                "navigator": {
                    "output": "mocap",
                    "speed": robot_route[1],
                    "route_mode": "exact",
                    "goals": robot_route[0],
                    "avoidance": {"stop": False},
                    "recovery": {"enabled": False},
                }
            }
        ]
    mover = {
        "spawn_model": {
            "model": str(tmp_path / "crate.xml"),
            "motion": "driven",
            "pose": {"position": {"x": -3.0, "y": 0.0, "z": 0.25}},
        },
        "name": "cart",
        "components": [
            {
                "navigator": {
                    "speed": 0.6,
                    "goals": [[3.0, 0.0]],
                    "obstacle_height": [0.05, 0.6],
                    "recovery": {"enabled": False},
                    "avoidance": {
                        "steer": model,
                        "stop": False,
                        **({"params": params} if params else {}),
                        **(avoidance or {}),
                    },
                }
            }
        ],
    }
    return load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": str(DATA / "open_room.xml")},
            "components": [robot, mover],
        },
        base_dir=tmp_path,
    )


def _track(engine, seconds):
    model = engine.ctx.model
    ids = {}
    for name in ("cart", "robot"):
        body = engine.ctx.entities.get(name).body
        ids[name] = int(
            model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)]
        )
    cart, robot = [], []
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()
        cart.append(engine.ctx.data.mocap_pos[ids["cart"]][:2].copy())
        robot.append(engine.ctx.data.mocap_pos[ids["robot"]][:2].copy())
    return np.array(cart), np.array(robot)


def _run(world, seconds=12.0):
    engine = Engine(world)
    engine.setup()
    engine.reset()
    try:
        return _track(engine, seconds)
    finally:
        engine.shutdown()


def test_a_robot_with_no_navigator_is_steered_round(tmp_path):
    """Standing on the cart's line, with nothing that would make it an agent but being there."""
    cart, robot = _run(_world(tmp_path))
    least = np.linalg.norm(cart - robot, axis=1).min()
    # the crate's corner (0.28 m) and the base (0.17 m), and a clearance between
    assert least > 0.2 + 0.17 + 0.1, f"it came {least:.2f} m from the robot's centre"
    assert cart[-1][0] > 2.5, "it did not get past"


def test_a_robot_crossing_is_passed_behind(tmp_path):
    """Crossing from the cart's left to its right, the robot is passed on the side it came from."""
    route = ([[0.0, 1.5], [0.0, -3.0]], 0.3)
    cart, _ = _run(_world(tmp_path, robot_at=(0.0, 1.5), robot_route=route), seconds=10.0)
    assert cart[:, 1].max() > 0.1, "it never moved to the side the robot came from"
    assert cart[:, 1].min() > -0.05, "it went the way the robot was going"


def test_give_ways_old_margin_is_refused_naming_its_replacement(tmp_path):
    with pytest.raises(PluginError, match=r"params\.margin.*`clearance`"):
        Engine(_world(tmp_path, params={"margin": 0.2})).setup()


def test_an_ignored_robot_is_not_steered_round(tmp_path):
    cart, robot = _run(_world(tmp_path, avoidance={"ignore": ["robot"]}))
    assert np.abs(cart[:, 1]).max() < 0.02, "it steered round a robot it ignores"


def test_orca_steers_round_a_robot_with_no_navigator(tmp_path):
    pytest.importorskip("rvo2")
    cart, robot = _run(_world(tmp_path, model="orca"))
    least = np.linalg.norm(cart - robot, axis=1).min()
    assert least > 0.2 + 0.17 + 0.1, f"it came {least:.2f} m from the robot's centre"


def test_orca_refuses_an_ignore_it_cannot_apply_to_steering(tmp_path):
    pytest.importorskip("rvo2")
    engine = Engine(_world(tmp_path, model="orca", avoidance={"ignore": ["robot"]}))
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RegistryError, match=r"cannot leave one agent out"):
            for _ in range(int(1.0 / engine.ctx.dt)):  # past the navigator's first tick
                engine.step()
    finally:
        engine.shutdown()
