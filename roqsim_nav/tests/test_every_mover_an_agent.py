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
from roqsim_nav.avoidance import SERVICE_KEY

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
    cart, robot = _run(_world(tmp_path, model="orca"))
    least = np.linalg.norm(cart - robot, axis=1).min()
    assert least > 0.2 + 0.17 + 0.1, f"it came {least:.2f} m from the robot's centre"


def test_orca_does_not_steer_round_a_robot_it_ignores(tmp_path):
    cart, robot = _run(_world(tmp_path, model="orca", avoidance={"ignore": ["robot"]}))
    assert np.abs(cart[:, 1]).max() < 0.02, "it steered round a robot it ignores"


#: A door 0.9 m wide on a vertical hinge at one end: it moves, but only where it is fixed.
DOOR = """<mujoco model="door">
  <worldbody><body name="leaf">
    <joint name="hinge" type="hinge" axis="0 0 1" pos="-.45 0 0" damping="5"/>
    <geom name="leaf" type="box" size=".45 .02 1" pos="0 0 1" mass="10"/>
  </body></worldbody>
</mujoco>"""

#: A cart on a free joint, 0.6 m by 0.4 m in plan and 0.3 m high.
CART = """<mujoco model="cart">
  <worldbody><body name="cart">
    <freejoint/>
    <geom name="cart" type="box" size=".3 .2 .15" pos="0 0 .15" mass="20"/>
  </body></worldbody>
</mujoco>"""


def _with(tmp_path, name, xml, at):
    """The cart of :func:`_world` with no robot, and a body ``xml`` spawned at ``at``."""
    world = _world(tmp_path, robot_at=(0.0, 8.0))
    (tmp_path / f"{name}.xml").write_text(xml)
    raw = world.raw
    raw["components"].append(
        {
            "spawn_model": {
                "model": str(tmp_path / f"{name}.xml"),
                "prefix": f"{name}_",
                "motion": "static",  # the model brings its own root joint
                "pose": {"position": {"x": at[0], "y": at[1], "z": 0.0}},
            },
            "name": name,
        }
    )
    return load_config_from_dict(raw, base_dir=tmp_path)


def test_a_hinged_door_beside_the_way_is_not_steered_round(tmp_path):
    """Open along the way, 0.4 m beside it: the planner and the probe deal with a door, and a disc
    round it would push the mover a metre off its line."""
    cart, _ = _run(_with(tmp_path, "door", DOOR, (0.0, 0.5)))
    assert np.abs(cart[:, 1]).max() < 0.05, f"it swerved {np.abs(cart[:, 1]).max():.2f} m"


def test_a_free_cart_joins_as_the_disc_of_its_plan_view(tmp_path):
    engine = Engine(_with(tmp_path, "trolley", CART, (0.0, 3.0)))
    engine.setup()
    engine.reset()
    try:
        for _ in range(int(0.2 / engine.ctx.dt)):  # past the first solve
            engine.step()
        model = engine.ctx.blackboard.get(SERVICE_KEY).model
        (agent,) = [a for a in model._agents.values() if a["key"] == "trolley"]
        assert agent["radius"] == pytest.approx(np.hypot(0.3, 0.2), abs=0.005)
    finally:
        engine.shutdown()


def test_orca_grows_each_disc_by_half_the_clearance_and_clearance_time_times_its_speed():
    """Two agents keep `clearance` and `clearance_time` times the sum of their speeds."""
    from types import SimpleNamespace

    from roqsim_nav.avoidance.orca import OrcaModel

    model = OrcaModel()
    model.configure(SimpleNamespace(model=None), {"clearance": 0.24, "clearance_time": 0.1})
    aid = model.add_agent("a", radius=0.23, max_speed=2.0, yields=True, params={})
    model.submit(aid, (0.0, 0.0), (1.2, 0.0), (1.2, 0.0))
    assert model._agents[aid]["disc"] == pytest.approx(0.23 + 0.12 + 0.1 * 1.2)
