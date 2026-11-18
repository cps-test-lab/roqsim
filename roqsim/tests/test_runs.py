"""A run's identity: its name, what a reset without one is called, and the draws keyed on it.

The draws are a function of ``(seed, run, sim_time, name)``. What has to hold: two runs never
draw the same values, whatever the process; the same seed and run draw the same values in any
process; and a name is what the runner gave, or one the runner can give back to run it again.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.runs import RunNames, check_run, run_key

WORLD = {"sim": {"timestep": 0.005}, "components": [{"dummy": {}, "name": "d0"}]}


def _engine(tmp_path, seed=7) -> Engine:
    engine = Engine(load_config_from_dict(WORLD, base_dir=tmp_path))
    engine.ctx.seed = seed
    engine.setup()
    return engine


def _draws(ctx: SimContext) -> np.ndarray:
    """A time-keyed and a per-run draw of one name, as a plugin would make them."""
    return np.concatenate(
        [
            ctx.rng_for("probe").standard_normal(4),
            ctx.rng_for("probe", per_run=True).standard_normal(4),
        ]
    )


# -- names ------------------------------------------------------------------------------------


def test_unnamed_runs_are_numbered_and_a_named_one_starts_its_own_series():
    names = RunNames()
    assert [names.next() for _ in range(3)] == ["#0", "#1", "#2"]
    assert names.next("cfg-a/0") == "cfg-a/0"
    assert [names.next(), names.next()] == ["cfg-a/0#1", "cfg-a/0#2"]
    assert names.next("cfg-a/1") == "cfg-a/1"
    assert names.next() == "cfg-a/1#1"


def test_a_first_name_names_only_the_first_unnamed_reset():
    names = RunNames(first="b/3")
    assert [names.next(), names.next()] == ["b/3", "b/3#1"]
    assert RunNames(first="b/3").next("c/0") == "c/0", "a reset's own name beats the first"


@pytest.mark.parametrize("bad", ["", 3, None])
def test_a_name_that_is_no_name_is_refused(bad):
    with pytest.raises(ValueError):
        check_run(bad)


def test_an_unnamed_run_s_name_given_back_draws_what_that_run_drew(tmp_path):
    """The only way to repeat a run nobody named: give its name back."""
    engine = _engine(tmp_path)
    try:
        for _ in range(3):
            engine.reset()
        assert engine.ctx.run == "#2"
        drawn = _draws(engine.ctx)
    finally:
        engine.shutdown()
    again = _engine(tmp_path)
    try:
        again.reset(run="#2")
        assert np.array_equal(_draws(again.ctx), drawn)
    finally:
        again.shutdown()


def test_every_name_of_a_process_is_distinct():
    names = RunNames()
    given = [None, None, "a/0", None, "a/1", None, None, "b/0", None]
    produced = [names.next(run) for run in given]
    assert len(set(produced)) == len(produced)


def test_the_key_is_stable_and_no_two_names_of_a_campaign_share_one():
    # Pinned: the key must not depend on the process (Python's hash() is salted per process).
    assert run_key("cfg-a/3") == 12071998984734756596
    names = [f"cfg-{i}/{j}" for i, j in itertools.product(range(1000), range(100))]
    names += [f"#{n}" for n in range(1000)]
    assert len({run_key(n) for n in names}) == len(names)
    assert all(0 <= run_key(n) < 2**64 for n in names[:100])


# -- draws ------------------------------------------------------------------------------------


def test_two_runs_under_one_seed_never_draw_the_same_values():
    ctx = SimContext(config={})
    ctx.seed = 7
    seen = []
    for run in ["#0", "#1", "cfg-a/0", "cfg-b/0", "cfg-a/1"]:
        ctx.run = run
        seen.append(_draws(ctx))
    for a, b in itertools.combinations(seen, 2):
        assert not np.allclose(a, b)


def test_the_same_seed_and_run_draw_the_same_values_in_another_context():
    a, b = SimContext(config={}), SimContext(config={})
    a.seed = b.seed = 7
    a.run = "cfg-b/1"
    for other in ["#0", "#1", "x/9"]:  # whatever ran before in b
        b.run = other
        _draws(b)
    b.run = "cfg-b/1"
    assert np.array_equal(_draws(a), _draws(b))


# -- the engine -------------------------------------------------------------------------------


def test_a_reset_names_its_run_before_any_plugin_runs_and_counts_itself(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.reset()
        assert (engine.ctx.run, engine.ctx.reset_count) == ("#0", 1)
        engine.reset(run="cfg-a/0")
        assert (engine.ctx.run, engine.ctx.reset_count) == ("cfg-a/0", 2)
        engine.reset(run="cfg-a/0")
        assert (engine.ctx.run, engine.ctx.reset_count) == ("cfg-a/0", 3), "same run, new reset"
        engine.reset()
        assert engine.ctx.run == "cfg-a/0#1"
        assert "run" not in engine.ctx.blackboard.get("reset_params")
    finally:
        engine.shutdown()


def test_a_reset_with_a_bad_name_resets_nothing(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.reset()
        with pytest.raises(ValueError, match="empty"):
            engine.reset(run="")
        assert (engine.ctx.run, engine.ctx.reset_count) == ("#0", 1)
    finally:
        engine.shutdown()


def test_a_driver_that_rebuilds_keeps_counting_its_unnamed_runs(tmp_path):
    """A new engine handed the driver's names continues them, so no unnamed run repeats a name."""
    names = RunNames()
    produced = []
    for _ in range(2):
        engine = _engine(tmp_path)
        engine.runs = names
        try:
            for _ in range(2):
                engine.reset()
                produced.append(engine.ctx.run)
        finally:
            engine.shutdown()
    assert produced == ["#0", "#1", "#2", "#3"]


def test_events_name_the_run_they_belong_to(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.reset(run="cfg-a/0")
        first = engine.ctx.events.emit("probe", "start")
        engine.reset()
        second = engine.ctx.events.emit("probe", "start")
    finally:
        engine.shutdown()
    assert (first["run"], second["run"]) == ("cfg-a/0", "cfg-a/0#1")
