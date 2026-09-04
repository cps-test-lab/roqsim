"""The navigator driving a mocap prop end to end: plan, follow, arrive, reset.

A crate that goes somewhere. The divided room is the load-bearing part of the fixture: a wall across
the middle means the straight line is blocked, so reaching the goal is evidence that A* searched and
the follower followed, not that a velocity happened to point the right way.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

DATA = Path(__file__).parent / "data"
ROOM = DATA / "divided_room.xml"  # walls at +/-4, a divider spanning y in [-2.5, 2.5] at x = 0
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""

START = (-2.5, 0.0)
GOAL = (2.5, 0.0)


def _world(tmp_path, *, world=ROOM, nav=None, **prop):
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    navigator = {"speed": 0.8, "goals": [list(GOAL)], "obstacle_height": [0.05, 0.6]}
    navigator.update(nav or {})
    return load_config_from_dict(
        {
            "sim": {"pacing": "asap", **({"world": str(world)} if world else {})},
            "components": [
                {
                    "spawn_model": {
                        "model": str(crate),
                        "pos": [START[0], START[1], 0.25],
                        "mocap": True,
                        **prop,
                    },
                    "name": "cart",
                    "components": [{"navigator": navigator}],
                }
            ],
        },
        base_dir=tmp_path,
    )


def _mocapid(engine):
    model = engine.ctx.model
    body = engine.ctx.entities.get("cart").body
    return int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])


def _drive(engine, steps=6000, every=50):
    mid = _mocapid(engine)
    track = []
    for i in range(steps):
        engine.step()
        if i % every == 0:
            track.append(engine.ctx.data.mocap_pos[mid][:2].copy())
    return np.array(track)


@pytest.fixture
def sim(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


def test_it_routes_around_the_wall_and_arrives(sim):
    track = _drive(sim)
    assert np.linalg.norm(track[-1] - np.asarray(GOAL)) < 0.3, "never reached the goal"
    # The divider spans y in [-2.5, 2.5]; a straight run would never leave y = 0.
    assert np.abs(track[:, 1]).max() > 2.5, "went through the wall instead of around it"


def test_it_writes_mocap_and_never_touches_qpos(sim):
    """A mocap prop contributes no generalised coordinate; writing qpos would mean it had one."""
    assert sim.ctx.model.nq == 0
    _drive(sim, steps=200)


def test_reset_returns_it_to_the_start_and_re_arms_the_route(sim):
    _drive(sim, steps=2000)
    moved = sim.ctx.data.mocap_pos[_mocapid(sim)][:2].copy()
    assert np.linalg.norm(moved - np.asarray(START)) > 0.5, "it never left the start"

    sim.reset()
    back = sim.ctx.data.mocap_pos[_mocapid(sim)][:2].copy()
    assert back == pytest.approx(START, abs=1e-9)
    # Re-armed, not merely re-seated: it must run the route again rather than stand still.
    assert np.linalg.norm(_drive(sim, steps=2000)[-1] - np.asarray(START)) > 0.5


def test_with_no_walls_it_still_reaches_the_goal(tmp_path):
    """A wall-less world yields no grid at all. That is a straight-line fallback, not an error."""
    engine = Engine(_world(tmp_path, world=None))
    engine.setup()
    engine.reset()
    try:
        assert np.linalg.norm(_drive(engine)[-1] - np.asarray(GOAL)) < 0.3
    finally:
        engine.shutdown()


def test_two_movers_share_one_rasterized_grid(tmp_path):
    """Building the grid walks every geom in the model; doing it per mover repeats identical work.

    Asserted by object identity rather than by timing, so it is deterministic: both planners must be
    searching the *same* raster, however many movers agree on the resolution and the height band.
    """
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    nav = {"speed": 0.8, "goals": [list(GOAL)], "obstacle_height": [0.05, 0.6]}
    entries = [
        {
            "spawn_model": {
                "model": str(crate),
                "prefix": f"c{i}_",
                "pos": [-2.5, y, 0.25],
                "mocap": True,
            },
            "name": f"cart{i}",
            "components": [{"navigator": dict(nav)}],
        }
        for i, y in enumerate((0.0, 1.0))
    ]
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap", "world": str(ROOM)}, "components": entries},
            base_dir=tmp_path,
        )
    )
    engine.setup()
    try:
        navigators = [p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin"]
        assert len(navigators) == 2
        grids = [n._core.planner.grid for n in navigators]
        assert grids[0] is grids[1], "each mover rasterized its own copy of the same walls"
    finally:
        engine.shutdown()


def test_a_second_navigator_on_one_entity_is_refused(tmp_path):
    """Two would write the same body every step, and the last one would silently win."""
    cfg = _world(tmp_path)
    engine = Engine(cfg)
    engine.plugins  # noqa: B018 - built at construction
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    doubled = load_config_from_dict(
        {
            "sim": {"pacing": "asap"},
            "components": [
                {
                    "spawn_model": {"model": str(crate), "pos": [0, 0, 0.25], "mocap": True},
                    "name": "cart",
                    "components": [
                        {"navigator": {"speed": 0.5, "goals": [[1.0, 0.0]]}, "name": "a"},
                        {"navigator": {"speed": 0.5, "goals": [[2.0, 0.0]]}, "name": "b"},
                    ],
                }
            ],
        },
        base_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="already has a navigator"):
        Engine(doubled).setup()


def test_speed_is_required(tmp_path):
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    with pytest.raises(PluginError, match="'speed' is required"):
        Engine(
            load_config_from_dict(
                {
                    "sim": {},
                    "components": [
                        {
                            "spawn_model": {
                                "model": str(crate),
                                "pos": [0, 0, 0.25],
                                "mocap": True,
                            },
                            "name": "cart",
                            "components": [{"navigator": {"goals": [[1.0, 0.0]]}}],
                        }
                    ],
                },
                base_dir=tmp_path,
            )
        )


def test_a_navigator_on_a_welded_prop_is_refused_with_the_fix(tmp_path):
    """The common mistake, and the error has to name the cure rather than the symptom."""
    engine = Engine(_world(tmp_path, mocap=False, nav={"output": "mocap"}))
    with pytest.raises(Exception, match="mocap: true"):
        engine.setup()
