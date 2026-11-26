"""A walker on its route steps aside round what stands in its way, and back.

With ``steer: none`` a walker keeps its route; what stands in it -- a robot, a walker coming the
other way -- it steps round within ``sidestep.max`` of the route, keeping ``sidestep.clearance``,
and it is back on its route after. Two crossing take turns: the one ahead goes first.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

#: A TurtleBot's shape: a low base that collides, a slim tower that does not.
ROBOT = """<mujoco model="robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".164 .03" pos="0 0 .049"/>
    <geom name="tower" type="cylinder" size=".03 .15" pos="0 0 .25" contype="0" conaffinity="0"/>
  </body></worldbody>
</mujoco>"""


def _walker(name, start, goal, speed=0.5, avoidance=None, route_mode=None):
    nav = {
        "output": "walker",
        "speed": speed,
        "goals": [list(goal)],
        "avoidance": {"steer": "none", "stop": True, **(avoidance or {})},
    }
    if route_mode:
        nav["route_mode"] = route_mode
    yaw = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    return {
        "walker": {
            "walker": "MaleVisitorWalk",
            "pose": {"position": {"x": start[0], "y": start[1]}, "orientation": {"yaw": yaw}},
        },
        "name": name,
        "components": [{"navigator": nav}],
    }


def _engine(tmp_path, components):
    (tmp_path / "robot.xml").write_text(ROBOT)
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"}, "components": components}, base_dir=tmp_path
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _xy(engine, name):
    model = engine.ctx.model
    body = engine.ctx.entities.get(name).body
    return engine.ctx.data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()


def _run(engine, seconds, names):
    track = {n: [] for n in names}
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()
        for n in names:
            track[n].append(_xy(engine, n))
    return {n: np.array(v) for n, v in track.items()}


def test_it_steps_round_a_robot_standing_on_its_route_and_back(tmp_path):
    robot = {
        "spawn_model": {
            "model": "robot.xml",
            "prefix": "r_",
            "motion": "driven",
            "pose": {"position": {"x": 0.0, "y": 0.0}},
        },
        "name": "robot",
    }
    engine = _engine(tmp_path, [_walker("w", (-3.0, 0.0), (4.0, 0.0)), robot])
    try:
        track = _run(engine, 18.0, ["w"])["w"]
        least = float(np.linalg.norm(track - (0.0, 0.0), axis=1).min())
        # its legs (0.18 m from its centre) and the base (0.164 m) keep a gap between them
        assert least > 0.18 + 0.164 + 0.15, f"it came {least:.2f} m from the robot's centre"
        assert np.abs(track[:, 1]).max() <= 1.0 + 0.05, "it left its route by more than max"
        assert np.linalg.norm(track[-1] - (4.0, 0.0)) < 0.3, "it did not arrive"
        assert abs(track[-1][1]) < 0.05, "it did not get back on its route"
        events = [e["event"] for e in engine.ctx.events.all() if e["source"] == "w"]
        assert events.count("sidestep") == 1 and events.count("sidestep_back") == 1
    finally:
        engine.shutdown()


def test_two_walkers_head_on_on_one_line_pass_each_other(tmp_path):
    engine = _engine(
        tmp_path,
        [_walker("a", (-3.0, 0.0), (3.0, 0.0), 1.4), _walker("b", (3.0, 0.0), (-3.0, 0.0), 1.4)],
    )
    try:
        track = _run(engine, 8.0, ["a", "b"])
        assert np.linalg.norm(track["a"][-1] - (3.0, 0.0)) < 0.3, "a did not get past"
        assert np.linalg.norm(track["b"][-1] - (-3.0, 0.0)) < 0.3, "b did not get past"
        least = float(np.linalg.norm(track["a"] - track["b"], axis=1).min())
        assert least > 0.6, f"they passed {least:.2f} m apart"
    finally:
        engine.shutdown()


def test_two_walkers_crossing_take_turns_without_touching(tmp_path):
    engine = _engine(
        tmp_path,
        [_walker("a", (-3.0, 0.0), (3.0, 0.0), 1.4), _walker("b", (0.0, -2.9), (0.0, 3.0), 1.4)],
    )
    try:
        track = _run(engine, 9.0, ["a", "b"])
        assert np.linalg.norm(track["a"][-1] - (3.0, 0.0)) < 0.3
        assert np.linalg.norm(track["b"][-1] - (0.0, 3.0)) < 0.3
        least = float(np.linalg.norm(track["a"] - track["b"], axis=1).min())
        assert least > 0.6, f"they crossed {least:.2f} m apart"
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("avoidance", "route_mode", "says"),
    [
        ({"steer": "give_way"}, None, "a mover that steers is taken round by its model"),
        ({"stop": False}, None, "needs 'stop'"),
        ({}, "exact", "route_mode: exact"),
    ],
)
def test_sidestep_is_refused_where_it_cannot_act(tmp_path, avoidance, route_mode, says):
    walker = _walker(
        "w",
        (0.0, 0.0),
        (2.0, 0.0),
        avoidance={"sidestep": {"max": 0.5}, **avoidance},
        route_mode=route_mode,
    )
    with pytest.raises(PluginError, match=says):
        _engine(tmp_path, [walker])
