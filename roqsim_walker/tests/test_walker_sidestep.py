"""A walker on its route steps aside round what stands in its way, and back.

With ``steer: none`` a walker keeps its route; what stands in it -- a robot, a walker coming the
other way -- it steps round within ``sidestep.max`` of the route, keeping ``sidestep.clearance``,
and it is back on its route after. Two crossing take turns: the one ahead goes first. Two meeting
head-on step to opposite sides.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim.presence import entity_geoms

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


def test_with_sidestep_false_it_stands_before_a_robot_on_its_route_as_before(tmp_path):
    """The walker declares stepping aside; ``sidestep: false`` keeps a world's walker on its route,
    standing before what is in it."""
    robot = {
        "spawn_model": {
            "model": "robot.xml",
            "prefix": "r_",
            "motion": "driven",
            "pose": {"position": {"x": 0.0, "y": 0.0}},
        },
        "name": "robot",
    }
    walker = _walker("w", (-3.0, 0.0), (4.0, 0.0), avoidance={"sidestep": False})
    engine = _engine(tmp_path, [walker, robot])
    try:
        track = _run(engine, 12.0, ["w"])["w"]
        assert np.abs(track[:, 1]).max() < 0.05, "it left its route"
        assert -1.0 < track[-1][0] < -0.3, (
            f"it did not stand before the robot: x={track[-1][0]:.2f}"
        )
        events = [e["event"] for e in engine.ctx.events.all() if e["source"] == "w"]
        assert "sidestep" not in events
    finally:
        engine.shutdown()


def _body(engine, name):
    """The walker's geoms that collide: its legs, hips and torso, not its swinging arms' visuals."""
    model = engine.ctx.model
    geoms = entity_geoms(model, engine.ctx.entities.get(name), strict=False)
    return [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]


def _least_gap(engine, seconds, a, b):
    """The least gap (m) between the bodies of walkers ``a`` and ``b`` over ``seconds``, every
    0.05 s, and where each ends."""
    model, data = engine.ctx.model, engine.ctx.data
    ga, gb = _body(engine, a), _body(engine, b)
    fromto = np.zeros(6)
    every = max(1, round(0.05 / engine.ctx.dt))
    least = np.inf
    for i in range(int(seconds / engine.ctx.dt)):
        engine.step()
        if i % every == 0:
            gap = min(
                mujoco.mj_geomDistance(model, data, x, y, 1.0, fromto) for x in ga for y in gb
            )
            least = min(least, gap)
    return least, _xy(engine, a), _xy(engine, b)


@pytest.mark.parametrize("speed", [0.4, 0.5, 0.8, 1.2, 1.4])
@pytest.mark.parametrize("half", [2.0, 3.0, 5.0])
def test_two_walkers_head_on_on_one_line_pass_each_other(tmp_path, half, speed):
    """Started on one line, ``2 * half`` apart: each steps to its right, they pass without their
    bodies meeting, and both arrive."""
    engine = _engine(
        tmp_path,
        [
            _walker("a", (-half, 0.0), (half, 0.0), speed),
            _walker("b", (half, 0.0), (-half, 0.0), speed),
        ],
    )
    try:
        least, a, b = _least_gap(engine, 2.0 * half / speed + 6.0, "a", "b")
        assert least > 0.0, f"their bodies met, {-least * 100:.1f} cm into each other"
        assert np.linalg.norm(a - (half, 0.0)) < 0.3, f"a did not get past: {a}"
        assert np.linalg.norm(b - (-half, 0.0)) < 0.3, f"b did not get past: {b}"
    finally:
        engine.shutdown()


def test_two_walkers_head_on_beside_each_others_line_pass_on_the_side_they_are(tmp_path):
    """Coming the other way 0.3 m to a walker's right, the other is passed on the walker's left,
    the side away from it, and the other passes on its own left too: neither crosses the other's
    way."""
    engine = _engine(
        tmp_path,
        [_walker("a", (-3.0, 0.0), (3.0, 0.0), 1.0), _walker("b", (3.0, -0.3), (-3.0, -0.3), 1.0)],
    )
    try:
        track = _run(engine, 10.0, ["a", "b"])
        meet = int(np.argmin(np.abs(track["a"][:, 0] - track["b"][:, 0])))
        assert track["a"][meet, 1] > track["b"][meet, 1], "they swapped sides to pass"
        assert np.linalg.norm(track["a"][-1] - (3.0, 0.0)) < 0.3
        assert np.linalg.norm(track["b"][-1] - (-3.0, -0.3)) < 0.3
    finally:
        engine.shutdown()


def test_two_walkers_crossing_take_turns_without_touching(tmp_path):
    engine = _engine(
        tmp_path,
        [_walker("a", (-3.0, 0.0), (3.0, 0.0), 1.4), _walker("b", (0.0, -2.9), (0.0, 3.0), 1.4)],
    )
    try:
        least, a, b = _least_gap(engine, 9.0, "a", "b")
        assert least > 0.0, f"their bodies met, {-least * 100:.1f} cm into each other"
        assert np.linalg.norm(a - (3.0, 0.0)) < 0.3
        assert np.linalg.norm(b - (0.0, 3.0)) < 0.3
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("avoidance", "route_mode", "says"),
    [
        ({"steer": "give_way"}, None, "a mover that steers is taken round by its model"),
        ({"stop": False}, None, "needs 'stop'"),
        ({}, "exact", "route_mode: exact"),
        ({"sidestep": True}, None, "write {} for the block's defaults"),
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
