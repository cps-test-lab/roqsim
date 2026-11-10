"""A humanoid robot handling things through roqsim_motion: the G1-dex1's and the Oli's retargeting
tables, and scene 4 -- the walker's box swap by the G1-dex1, physical, in Python and in
OpenSCENARIO -- whose moves are the walker's own."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from roqsim_motion.retarget import find_table

from roqsim.config import load_config, load_config_from_dict
from roqsim.engine import Engine
from roqsim_motion import quat as Q

WORLDS = Path(__file__).resolve().parents[1] / "src" / "roqsim_humanoid" / "worlds"


@pytest.mark.parametrize("model", ["unitree_g1_dex1", "oli"])
def test_a_robots_table_loads_and_matches_its_model(model):
    from roqsim.models import resolve_model

    table = find_table(model)
    table.check(mujoco.MjModel.from_xml_path(str(resolve_model(model).path)))
    assert set(table.hands) == {"l", "r"}


@pytest.mark.parametrize("model", ["unitree_g1_dex1", "oli"])
def test_a_robot_kinematic_puts_a_hand_on_a_pose(model):
    world = {
        "sim": {"pacing": "asap", "timestep": 0.002},
        "components": [
            {
                "spawn_model": {"model": model, "prefix": "r_", "motion": "static"},
                "name": "bot",
                "components": [
                    {"humanoid_robot": {"drive": "kinematic"}},
                    {"navigator": {"output": "humanoid", "speed": 0.5, "arrival_radius": 0.05}},
                    {"humanoid": {}},
                ],
            },
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    try:
        engine.reset()
        body = engine.ctx.blackboard.get("humanoid_body:bot")
        d = engine.ctx.data
        for _ in range(200):
            engine.step()
        p, q = body.grip(d, "r")
        goal = p + np.array([0.15, -0.05, 0.15])
        body.set_hand("r", goal, q)
        for _ in range(2500):
            engine.step()
        p2, q2 = body.grip(d, "r")
        assert np.linalg.norm(p2 - goal) < 0.005
        assert Q.angle_between(q2, q) < np.radians(5)
    finally:
        engine.shutdown()


def _run(world: str, seconds: float = 120.0):
    engine = Engine(load_config(WORLDS / f"{world}.yaml"))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    scene = next(p for p in engine.plugins if p.address in ("scene", "scene_osc"))
    while engine.ctx.sim_time < seconds and not scene.done():
        engine.step()
    assert scene.done(), f"{world}: the scene did not end in {seconds} s"
    return engine


def _timeline(engine):
    return [
        (e["source"], e["event"], e.get("program"), e.get("error"))
        for e in engine.ctx.events.all()
        if not e["source"].startswith("scene")
    ]


def test_scene_4_the_box_swap_by_a_humanoid_robot_physical_in_both_forms():
    python = _run("handling_box_swap_g1")
    try:
        d, m = python.ctx.data, python.ctx.model
        full, empty = d.body("tote_full").xpos, d.body("tote_empty").xpos
        assert (
            abs(full[2] - 0.70) < 0.02 and np.hypot(full[0] + 1.2, full[1] - 1.0) < 0.45
        )  # on the trolley
        assert abs(empty[2] - 0.645) < 0.02  # in the drawer
        slide = m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cell/drawer_1/slide")
        ]
        assert d.qpos[slide] < 0.01  # pushed shut by the robot's hand, its lag taken up
        # physical: the robot's joints were tracked by servos and its base hung on its anchor; the
        # base stood up throughout, as nothing balanced it but the anchor
        assert abs(d.body("g1_base_link").xpos[2] - 0.793) < 0.05
        expected = _timeline(python)
    finally:
        python.shutdown()
    try:
        import scenario_execution  # noqa: F401 -- the OSC twin runs on it (scenario_execution_roqsim[osc])
    except ImportError:
        pytest.skip(
            "handling_box_swap_g1: scenario_execution is not installed, so the OSC twin's timeline "
            "is not compared"
        )
    osc = _run("handling_box_swap_g1_osc")
    try:
        assert _timeline(osc) == expected
    finally:
        osc.shutdown()


def test_the_robots_legs_walk_while_its_navigator_moves_it():
    """The gait plays all the while the navigator moves the robot, not only on the body's updates
    that follow one of the navigator's: each hip swings forward and back by a stride."""
    engine = Engine(load_config(WORLDS / "handling_box_swap_g1.yaml"))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        m, d = engine.ctx.model, engine.ctx.data
        hips = {
            s: m.jnt_qposadr[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"g1_{s}_hip_pitch_joint")
            ]
            for s in ("left", "right")
        }
        seen = {s: [] for s in hips}
        while engine.ctx.sim_time < 3.4:  # walking to the cell's drawer
            engine.step()
            if engine.ctx.sim_time > 1.0:
                for s, a in hips.items():
                    seen[s].append(float(d.qpos[a]))
        for s, q in seen.items():
            assert min(q) < -0.08 and max(q) > 0.2, (
                f"the {s} hip swings only {min(q):.2f}..{max(q):.2f} rad"
            )
    finally:
        engine.shutdown()
