"""The ORCA model on its own: the solve RVO2 does, with walls, and what an agent ignores.

Agents are moved by the velocity the model chose, as a navigator's body would follow it. Where
``rvo2`` is installed, the velocities are compared with its own over the same scenes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from roqsim_nav.avoidance.orca import OrcaModel

DT = 0.02


def _model(**params):
    model = OrcaModel()
    model.configure(None, params)
    return model


def _run(model, agents, seconds=8.0):
    """Move ``agents`` (aid -> [pos, goal velocity]) by what the model chooses; their paths."""
    paths = {aid: [np.array(p, dtype=float)] for aid, (p, _) in agents.items()}
    for _ in range(int(seconds / DT)):
        for aid, (_, pref) in agents.items():
            model.submit(aid, paths[aid][-1], np.zeros(2), pref)
        model.solve(DT)
        for aid in agents:
            paths[aid].append(paths[aid][-1] + model.result(aid) * DT)
    return {aid: np.array(p) for aid, p in paths.items()}


def _two(model, *, offset=0.02):
    a = model.add_agent("a", radius=0.23, max_speed=1.5, yields=True, params={})
    b = model.add_agent("b", radius=0.23, max_speed=1.5, yields=True, params={})
    return a, b, {a: [(-3.0, 0.0), (1.0, 0.0)], b: [(3.0, offset), (-1.0, 0.0)]}


def test_two_head_on_pass_each_other_at_the_clearance():
    model = _model()
    a, b, agents = _two(model)
    paths = _run(model, agents)
    least = np.linalg.norm(paths[a] - paths[b], axis=1).min()
    # Both discs and the clearance (speeds are submitted as zero, so no clearance_time).
    assert least == pytest.approx(0.23 + 0.23 + 0.24, abs=0.02)
    assert paths[a][-1, 0] > 3.0 and paths[b][-1, 0] < -3.0, "they did not get past each other"


def test_both_give_way_by_the_same_amount():
    model = _model()
    a, b, agents = _two(model)
    paths = _run(model, agents)
    off_a = np.abs(paths[a][:, 1]).max()
    off_b = np.abs(paths[b][:, 1] - 0.02).max()
    assert off_a == pytest.approx(off_b, rel=0.1), (off_a, off_b)


def test_an_agent_that_does_not_yield_goes_its_way_and_is_gone_round():
    model = _model()
    a = model.add_agent("a", radius=0.23, max_speed=1.5, yields=True, params={})
    robot = model.add_agent("robot", radius=0.2, max_speed=0.0, yields=False, params={})
    paths = _run(model, {a: [(-3.0, 0.0), (1.0, 0.0)], robot: [(0.0, 0.01), (0.0, 0.0)]})
    assert np.allclose(paths[robot], paths[robot][0]), "the model moved an agent it may not move"
    least = np.linalg.norm(paths[a] - paths[robot], axis=1).min()
    assert least >= 0.23 + 0.2 + 0.24 - 0.02


def test_an_agent_goes_through_one_it_ignores_and_that_one_still_gives_way():
    model = _model()
    a, b, agents = _two(model)
    model.exclude(a, b)
    paths = _run(model, agents)
    assert np.abs(paths[a][:, 1]).max() < 1e-9, "it gave way to an agent it ignores"
    assert np.abs(paths[b][:, 1] - 0.02).max() > 0.2, "the other did not give way"


def test_an_absent_agent_deflects_nobody():
    model = _model()
    a, b, agents = _two(model)
    for _ in range(int(6.0 / DT)):
        model.submit(a, (-3.0, 0.0), np.zeros(2), (1.0, 0.0))
        model.submit(b, (-2.0, 0.0), np.zeros(2), (0.0, 0.0), present=False)
        model.solve(DT)
    np.testing.assert_allclose(model.result(a), (1.0, 0.0))
    np.testing.assert_allclose(model.result(b), (0.0, 0.0))


def _box(cx, cy, hx, hy):
    """A wall footprint, counter-clockwise, as the planner gives it."""
    return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]


def test_an_agent_sent_into_a_wall_stops_short_of_it():
    model = _model()
    model.add_static([_box(1.0, 0.0, 0.1, 1.0)])
    a = model.add_agent("a", radius=0.23, max_speed=1.5, yields=True, params={})
    paths = _run(model, {a: [(-1.0, 0.0), (1.0, 0.0)]}, seconds=5.0)
    assert paths[a][:, 0].max() <= 0.9 - 0.23 + 1e-3, "it went into the wall"


def test_an_agent_pushed_toward_a_wall_by_another_does_not_cross_it():
    """Head-on beside a wall: the one nearer the wall gives way, never through it."""
    model = _model()
    model.add_static([_box(0.0, -0.5, 4.0, 0.05)])  # a wall 0.45 m below both lines
    a, b, agents = _two(model, offset=-0.02)
    paths = _run(model, agents)
    assert paths[b][:, 1].min() >= -0.45 + 0.23 - 1e-3, "it was pushed into the wall"
    assert paths[a][-1, 0] - paths[b][-1, 0] > 2.0, "they did not get past each other"


def test_an_agent_rounds_a_corner_without_cutting_it():
    model = _model()
    model.add_static([_box(0.0, 0.0, 0.5, 0.5)])
    a = model.add_agent("a", radius=0.23, max_speed=1.5, yields=True, params={})
    paths = _run(model, {a: [(-2.0, 0.6), (1.0, 0.0)]}, seconds=4.0)
    inside = (np.abs(paths[a][:, 0]) < 0.5 + 0.23 - 1e-3) & (
        np.abs(paths[a][:, 1]) < 0.5 + 0.23 - 1e-3
    )
    assert not inside.any(), "it cut through the box"
    assert paths[a][-1, 0] > 1.0


def test_reset_forgets_the_velocities():
    model = _model()
    a, b, agents = _two(model)
    _run(model, agents, seconds=2.0)
    model.reset()
    model.submit(a, (-3.0, 0.0), np.zeros(2), (0.0, 0.0))
    model.submit(b, (3.0, 0.0), np.zeros(2), (0.0, 0.0))
    model.solve(DT)
    np.testing.assert_allclose(model.result(a), (0.0, 0.0))


# -- against RVO2 itself ---------------------------------------------------------------------------
SCENES = {
    "head-on": ([], [((-3.0, 0.0), (1.0, 0.0)), ((3.0, 0.02), (-1.0, 0.0))]),
    "crossing": ([], [((-3.0, 0.0), (1.0, 0.0)), ((0.0, -3.0), (0.0, 1.2))]),
    "four": (
        [],
        [
            ((-3.0, 0.0), (1.0, 0.0)),
            ((3.0, 0.1), (-1.0, 0.0)),
            ((0.0, -3.0), (0.0, 1.0)),
            ((0.1, 3.0), (0.0, -1.0)),
        ],
    ),
    "wall": (
        [_box(0.0, -0.5, 4.0, 0.05)],
        [((-3.0, -0.02), (1.0, 0.0)), ((3.0, 0.0), (-1.0, 0.0))],
    ),
    "corner": ([_box(0.0, 0.0, 0.5, 0.5)], [((-2.0, 0.6), (1.0, 0.0))]),
    "corridor": (
        [_box(0.0, 0.8, 4.0, 0.05), _box(0.0, -0.8, 4.0, 0.05)],
        [((-3.0, 0.0), (1.0, 0.0)), ((3.0, 0.05), (-1.0, 0.0))],
    ),
}


RADIUS, MAX_SPEED, HORIZON, STEPS = 0.23, 1.5, 3.0, int(6.0 / DT)


def rvo2_trace(scene):
    """rvo2's velocity for every agent at every step of ``scene``, moved by its own choice."""
    import rvo2  # noqa: PLC0415 - only where it is installed

    walls, agents = SCENES[scene]
    sim = rvo2.PyRVOSimulator(DT, 4.0, 10, HORIZON, 2.0, RADIUS, MAX_SPEED)
    for poly in walls:
        sim.addObstacle(poly)
    if walls:
        sim.processObstacles()
    for pos, _ in agents:
        sim.addAgent(pos, 4.0, 10, HORIZON, 2.0, RADIUS, MAX_SPEED, (0.0, 0.0))
    pos = [np.array(p, dtype=float) for p, _ in agents]
    trace = []
    for _ in range(STEPS):
        for i, (_, pref) in enumerate(agents):
            sim.setAgentPosition(i, tuple(pos[i]))
            sim.setAgentPrefVelocity(i, pref)
        sim.doStep()
        step = [list(sim.getAgentVelocity(i)) for i in range(len(agents))]
        for i, v in enumerate(step):
            pos[i] = pos[i] + np.array(v) * DT
        trace.append(step)
    return trace


def _differs_from(scene, trace):
    """The largest difference from ``trace`` of the port's velocity, along rvo2's own path."""
    walls, agents = SCENES[scene]
    ours = _model(clearance=0.0, clearance_time=0.0)
    for poly in walls:
        ours.add_static([poly])
    for _ in agents:
        ours.add_agent("x", radius=RADIUS, max_speed=MAX_SPEED, yields=True, params={})
    pos = [np.array(p, dtype=float) for p, _ in agents]
    worst = 0.0
    for step in trace:
        for i, (_, pref) in enumerate(agents):
            ours.submit(i, pos[i], np.zeros(2), pref)
        ours.solve(DT)
        for i, v in enumerate(step):
            v = np.array(v)
            worst = max(worst, float(np.abs(ours.result(i) - v).max()))
            # Both go on from RVO2's choice, so a difference cannot carry into the next step.
            ours._agents[i]["vel"] = v
            pos[i] = pos[i] + v * DT
    return worst


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_it_chooses_what_rvo2_chose(scene):
    """Against rvo2's velocities, recorded in ``data/orca_rvo2.json`` (``data/orca_rvo2.py``)."""
    trace = json.loads((Path(__file__).parent / "data" / "orca_rvo2.json").read_text())[scene]
    worst = _differs_from(scene, trace)
    # RVO2 solves in single precision.
    assert worst < 1e-3, f"{scene}: differs from rvo2 by up to {worst:.2e} m/s"


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_it_chooses_what_rvo2_chooses(scene):
    pytest.importorskip("rvo2")
    worst = _differs_from(scene, rvo2_trace(scene))
    assert worst < 1e-3, f"{scene}: differs from rvo2 by up to {worst:.2e} m/s"


def _random_scene(seed):
    rng = np.random.default_rng(seed)
    walls = [
        _box(*rng.uniform(-3, 3, 2), *rng.uniform(0.05, 0.8, 2)) for _ in range(rng.integers(0, 5))
    ]

    def clear(p):
        return all(
            not (
                abs(p[0] - (w[0][0] + w[1][0]) / 2) < (w[1][0] - w[0][0]) / 2 + 0.4
                and abs(p[1] - (w[0][1] + w[2][1]) / 2) < (w[2][1] - w[0][1]) / 2 + 0.4
            )
            for w in walls
        )

    agents = []
    count = rng.integers(2, 9)
    while len(agents) < count:
        p = rng.uniform(-4, 4, 2)
        if clear(p) and all(np.linalg.norm(p - q) > 0.6 for q, _ in agents):
            agents.append((p, -p / np.linalg.norm(p) * rng.uniform(0.5, 1.4)))
    return walls, agents


def _at_contact(i, pos, edges, radius):
    """How near agent ``i`` is to touching another agent or a wall, where ORCA changes case."""
    gaps = [abs(np.linalg.norm(pos[j] - pos[i]) - 2 * radius) for j in range(len(pos)) if j != i]
    if len(edges.point):
        a, b = edges.point, edges.point[edges.nxt]
        ab, ap = b - a, pos[i] - a
        r = np.clip((ap * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
        gaps.append(np.abs(np.linalg.norm(a + r[:, None] * ab - pos[i], axis=1) - radius).min())
    return min(gaps)


def test_it_chooses_what_rvo2_chooses_in_random_rooms():
    """Random walls and crowds, following RVO2's own trajectory. The two differ only where an agent
    is touching another or a wall: ORCA changes case there, and RVO2 decides it in single
    precision."""
    rvo2 = pytest.importorskip("rvo2")
    radius, elsewhere = 0.23, []
    for seed in range(60):
        walls, agents = _random_scene(seed)
        ours = _model(clearance=0.0, clearance_time=0.0)
        theirs = rvo2.PyRVOSimulator(DT, 4.0, 10, 3.0, 2.0, radius, 1.5)
        for poly in walls:
            ours.add_static([poly])
            theirs.addObstacle(poly)
        if walls:
            theirs.processObstacles()
        for pos, _ in agents:
            ours.add_agent("x", radius=radius, max_speed=1.5, yields=True, params={})
            theirs.addAgent(tuple(pos), 4.0, 10, 3.0, 2.0, radius, 1.5, (0.0, 0.0))
        pos = [p.copy() for p, _ in agents]
        for step in range(300):
            for i, (_, pref) in enumerate(agents):
                ours.submit(i, pos[i], np.zeros(2), pref)
                theirs.setAgentPosition(i, tuple(pos[i]))
                theirs.setAgentPrefVelocity(i, tuple(pref))
            ours.solve(DT)
            theirs.doStep()
            for i in range(len(agents)):
                v = np.array(theirs.getAgentVelocity(i))
                if np.abs(ours.result(i) - v).max() > 1e-3 and (
                    _at_contact(i, pos, ours._edges, radius) > 1e-4
                ):
                    elsewhere.append((seed, step, i))
                # Both go on from RVO2's choice, so a difference cannot carry into the next step.
                ours._agents[i]["vel"] = v
                pos[i] = pos[i] + v * DT
    assert not elsewhere, f"differs from rvo2 away from contact at {elsewhere[:5]}"
