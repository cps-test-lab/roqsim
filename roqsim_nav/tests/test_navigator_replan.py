"""`caution.on_blocked: replan` -- routing around what the planner could not have known about.

The fixture is the whole argument. The room is open, so A* plans straight through the middle; the
barrier across that line is a **mocap** body, which is exactly what the planner's grid refuses to
rasterize (:func:`~roqsim_nav.obstacles.wall_polygons` excludes anything that could move). So the
mover plans a route that is genuinely wrong, discovers it with its caution probe, and has to do
something about it.

The two modes are tested against the same fixture on purpose, because the contrast IS the feature:

* ``stop``    -- drives up to the barrier and holds. Never arrives, and that is correct: it is still
                 on the path it was given, and an experiment holding that path fixed gets it.
* ``replan``  -- remembers where it was stopped, plans around it, and arrives.

Without the memory, ``replan`` would be indistinguishable from ``stop``: the grid is static, so
re-planning from the same place to the same goal returns the same path.

What ``replan`` does NOT promise is clearance. A mark is a disc where the mover was stopped, not an
outline of what stopped it, so the route it finds rounds the *mark* and can still pass close to a
long obstacle's far end. Caution is what keeps that from becoming a collision -- it is still looking
ahead the whole way -- and these tests assert arrival and which side it went, never a gap.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

DATA = Path(__file__).parent / "data"
ROOM = DATA / "open_room.xml"

CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""
#: Spans y in [-1.5, 1.5] at x = 0: it blocks the straight line and leaves a gap at either side.
BARRIER = """<mujoco model="barrier">
  <worldbody><body name="barrier"><geom name="bar" type="box" size=".15 1.5 .5"/></body></worldbody>
</mujoco>"""

START = (-2.5, 0.0)
GOAL = (2.5, 0.0)


def _world(tmp_path, on_blocked):
    (tmp_path / "crate.xml").write_text(CRATE)
    (tmp_path / "barrier.xml").write_text(BARRIER)
    return load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": str(ROOM)},
            "components": [
                {
                    "spawn_model": {
                        "model": str(tmp_path / "barrier.xml"),
                        "pos": [0.0, 0.0, 0.5],
                        "mocap": True,
                    },
                    "name": "barrier",
                },
                {
                    "spawn_model": {
                        "model": str(tmp_path / "crate.xml"),
                        "pos": [START[0], START[1], 0.25],
                        "mocap": True,
                    },
                    "name": "cart",
                    "components": [
                        {
                            "navigator": {
                                "speed": 0.8,
                                "goals": [list(GOAL)],
                                "obstacle_height": [0.05, 0.6],
                                "caution": {
                                    "on_blocked": on_blocked,
                                    "lookahead": 0.8,
                                    "forget_after": 30.0,
                                    "blockage_radius": 0.6,
                                },
                            }
                        }
                    ],
                },
            ],
        },
        base_dir=tmp_path,
    )


def _run(tmp_path, on_blocked, steps=12000):
    engine = Engine(_world(tmp_path, on_blocked))
    engine.setup()
    engine.reset()
    model = engine.ctx.model
    body = engine.ctx.entities.get("cart").body
    mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
    track = []
    try:
        for i in range(steps):
            engine.step()
            if i % 25 == 0:
                track.append(engine.ctx.data.mocap_pos[mid][:2].copy())
    finally:
        engine.shutdown()
    return np.array(track)


def test_stop_holds_at_the_barrier_and_never_arrives(tmp_path):
    track = _run(tmp_path, "stop")
    end = track[-1]
    assert end[0] < 0.0, "it should be held on the near side of the barrier"
    assert np.hypot(*(end - np.array(GOAL))) > 1.0, "stopping must not report itself past the wall"
    # Held, not wandering: the last second of the run goes nowhere.
    assert np.linalg.norm(track[-1] - track[-20]) < 0.05


def test_replan_routes_around_the_barrier_and_arrives(tmp_path):
    track = _run(tmp_path, "replan")
    assert np.hypot(*(track[-1] - np.array(GOAL))) < 0.4, "it should reach the goal the long way"
    # It rounds an END of the 3 m barrier rather than crossing anywhere near the middle. Not a
    # clearance assertion: marks are discs where the mover was stopped, so the route hugs the end it
    # discovered -- here to within a few centimetres of it. See the module docstring.
    crossing = track[np.argmin(np.abs(track[:, 0]))]
    assert abs(crossing[1]) > 1.0, f"crossed the barrier's plane at y={crossing[1]:.2f}"


def test_replan_is_refused_where_there_is_nothing_to_replan(tmp_path):
    """An exact route IS the given polyline, so routing around something means not walking it."""
    from roqsim_nav.plugins.navigator import NavigatorPlugin

    def check(**extra):
        cfg = {"speed": 1.0, "goals": [[1.0, 0.0]], "caution": {"on_blocked": "replan"}, **extra}
        return NavigatorPlugin(cfg).validate_config(cfg)

    assert any("route_mode: exact" in e for e in check(route_mode="exact"))
    assert any("traffic: respect" in e for e in check(traffic="ignore"))
