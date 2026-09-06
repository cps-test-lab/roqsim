"""`dwell`: standing still on arrival before moving on.

The route's one stochastic element, and the reason a repeated trial is a sample rather than a
duplicate. `NavState` has carried a `dwell` field and `behavior.py` has read it through the seeded
`uniform` all along; what was missing was any way for a world to set it, so it was always `None`.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim_nav.plugins.navigator import _dwell_list, _dwell_pair

DATA = Path(__file__).parent / "data"
ROOM = DATA / "divided_room.xml"
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""

START, GOAL = (-2.5, 2.8), (2.5, 2.8)  # clear of the divider, so the route is a straight shuttle


def _engine(tmp_path, seed=None, **nav):
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    navigator = {"speed": 1.0, "goals": [list(GOAL)], "loop": True, "route_mode": "exact"}
    navigator.update(nav)
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap", "world": str(ROOM)},
                "components": [
                    {
                        "spawn_model": {
                            "model": str(crate),
                            "pose": {"position": {"x": START[0], "y": START[1], "z": 0.25}},
                            "mocap": True,
                        },
                        "name": "cart",
                        "components": [{"navigator": navigator}],
                    }
                ],
            },
            base_dir=tmp_path,
        )
    )
    # Before setup: the navigator draws its generator in `configure`, so a seed set afterwards
    # would never reach it.
    if seed is not None:
        engine.ctx.seed = seed
    return engine


def _core(engine):
    nav = next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")
    return nav._core


def _run(engine, seconds=30.0):
    """Run, returning (fraction of steps with the dwell timer armed, final x)."""
    core = _core(engine)
    model, ctx = engine.ctx.model, engine.ctx
    mid = int(
        model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ctx.entities.get("cart").body)
        ]
    )
    armed = total = 0
    for _ in range(int(seconds / ctx.dt)):
        engine.step()
        total += 1
        armed += core._dwell_until > 0.0
    return armed / total, float(ctx.data.mocap_pos[mid][0])


# -- the coercion ---------------------------------------------------------------------------------
def test_a_number_is_a_fixed_pause_and_a_pair_is_a_random_one():
    assert _dwell_pair(2.0) == (2.0, 2.0)
    assert _dwell_pair([1.0, 4.0]) == (1.0, 4.0)


def test_one_value_applies_to_every_route_point():
    assert _dwell_list(1.5, 3) == [(1.5, 1.5)] * 3
    assert _dwell_list([0.0, 3.0], 2) == [(0.0, 3.0)] * 2


def test_a_per_point_list_may_mix_scalars_and_pairs():
    """What a patrol with a pause at only SOME of its waypoints looks like, and the shape the
    shipped `walker_patrol` world writes. Requiring every entry to be nested rejected exactly this,
    which is how two shipped worlds stopped loading."""
    assert _dwell_list([0.0, [2.0, 4.0], 0.0, [1.0, 3.0]], 4) == [
        (0.0, 0.0),
        (2.0, 4.0),
        (0.0, 0.0),
        (1.0, 3.0),
    ]


def test_bare_numbers_are_per_point_when_the_count_is_unambiguous():
    """Three numbers on a three-point route cannot be a `[lo, hi]` pause, so they are per-point."""
    assert _dwell_list([1.0, 2.0, 3.0], 3) == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_nested_entries_are_per_point():
    """A bare `[lo, hi]` is ambiguous against a two-point route's per-point list, and reads as the
    random pause; nesting is how a world says the other thing."""
    assert _dwell_list([[0.0, 3.0], [1.0, 2.0]], 2) == [(0.0, 3.0), (1.0, 2.0)]


@pytest.mark.parametrize(
    "bad, message",
    [
        ([[0.0, 1.0]], "one dwell per route point"),
        ([1.0, 2.0, 3.0], "one dwell per route point"),
        (-1.0, "cannot be negative"),
        ([3.0, 1.0], "must be ordered"),
    ],
)
def test_a_malformed_dwell_is_refused(bad, message):
    with pytest.raises((ValueError, TypeError), match=message):
        _dwell_list(bad, 2)


def test_the_plugin_reports_a_malformed_dwell_before_the_run(tmp_path):
    """Refused while the config is validated -- i.e. during `Engine(...)`, before a step is taken
    and before any compute is spent on a world that would have dwelt for a nonsense time."""
    with pytest.raises(PluginError, match="dwell"):
        _engine(tmp_path, dwell=[3.0, 1.0])


# -- the behaviour --------------------------------------------------------------------------------
def test_without_a_dwell_the_mover_never_stops(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    assert _core(engine).st.dwell == [(0.0, 0.0), (0.0, 0.0)]
    armed, _ = _run(engine)
    assert armed == 0.0


def test_a_dwell_makes_the_mover_stand_at_each_goal(tmp_path):
    """The load-bearing check. `dwell` reaching `NavState` is not the same as a pause happening --
    the field was present and unread for exactly that reason."""
    engine = _engine(tmp_path, seed=5, dwell=[0.0, 3.0])
    engine.setup()
    engine.reset()
    assert _core(engine).st.dwell == [(0.0, 3.0), (0.0, 3.0)]
    armed, _ = _run(engine)
    assert armed > 0.10, f"the dwell timer was armed for only {armed:.1%} of the run"


def test_the_pause_is_a_function_of_the_seed(tmp_path):
    """Which is the point of a random dwell: repetitions of one trial differ, and any one of them
    can be replayed. A pause drawn from process entropy would give neither."""

    def where(seed):
        engine = _engine(tmp_path, seed=seed, dwell=[0.0, 3.0])
        engine.setup()
        engine.reset()
        return _run(engine, seconds=20.0)[1]

    assert where(5) == pytest.approx(where(5)), "the same seed must replay the same pauses"
    assert where(5) != pytest.approx(where(9)), "a different seed must give different pauses"


def test_a_fixed_dwell_needs_no_seed(tmp_path):
    """`dwell: 2.0` is a stated pause, not a draw, so it must not depend on the seed at all."""

    def where(seed):
        engine = _engine(tmp_path, seed=seed, dwell=2.0)
        engine.setup()
        engine.reset()
        return _run(engine, seconds=20.0)[1]

    assert where(5) == pytest.approx(where(9))
    assert not np.isnan(where(5))
