"""A run either has a seed it reports, or it fails -- it never silently draws from a default.

The defect this pins: the scenario adapter built an ``Engine`` and called ``setup()`` without ever
assigning ``ctx.seed``, and ``rng_for`` stood in a 0. Every repeated run of one world then drew the
same numbers -- each looking like its own, ``seed: None`` in every recording, and no signal anywhere
except the drawn values themselves. Two properties close it: an unset seed raises
(so no future driver can repeat the omission), and the adapter resolves one (so the shipped path
never reaches that raise).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.scenario_adapter import MujocoSim
from roqsim.seed import SEED_ENV, SeedError

WORLD = """
sim:
  timestep: 0.005
plugins:
  - dummy: {}
    name: d0
"""


def _world(tmp_path: Path) -> str:
    p = tmp_path / "w.yaml"
    p.write_text(WORLD)
    return str(p)


def _seeded_run(tmp_path: Path, **kwargs) -> tuple[int, np.ndarray]:
    """Drive one adapter run and return the seed it resolved plus a draw made under it."""
    sim = MujocoSim(world=_world(tmp_path), **kwargs)
    sim.setup(logger=logging.getLogger("t"))
    sim.reset()
    try:
        for _ in range(5):
            sim.step()
        return sim._engine.ctx.seed, sim._engine.ctx.rng_for("probe").standard_normal(4)
    finally:
        sim.shutdown()


# -- the class-closing guard ------------------------------------------------------------------


def test_an_unset_seed_raises_rather_than_drawing_from_a_default():
    """The root cause. Standing in a 0 is what made a forgetful driver invisible."""
    with pytest.raises(SeedError) as err:
        SimContext(config={}).rng_for("lidar")
    # The message has to say whose job it is, or the next driver repeats the omission.
    assert "lidar" in str(err.value)
    assert "ctx.seed" in str(err.value)


def test_a_seed_of_zero_is_a_chosen_seed_not_an_absent_one():
    """``0`` is a seed like any other; only ``None`` is "nobody resolved one"."""
    ctx = SimContext(config={})
    ctx.seed = 0
    assert ctx.rng_for("lidar").standard_normal(2).shape == (2,)


# -- the adapter resolves one -----------------------------------------------------------------


def test_an_adapter_driven_run_never_leaves_the_seed_unset(tmp_path: Path, caplog):
    """A recording from this path can no longer say ``seed: None``."""
    with caplog.at_level(logging.INFO):
        seed, _ = _seeded_run(tmp_path)
    assert isinstance(seed, int)
    assert "drawn" in caplog.text  # reported, not just set -- otherwise it cannot be replayed


def test_two_unseeded_runs_draw_differently(tmp_path: Path):
    """The reported symptom, as a test: repetitions of one configuration must be samples.

    This is the assertion that failed before the fix -- both runs drew from 0.
    """
    seed_a, draw_a = _seeded_run(tmp_path)
    seed_b, draw_b = _seeded_run(tmp_path)
    assert seed_a != seed_b
    assert (draw_a != draw_b).any()


@pytest.mark.parametrize("via", ["argument", "environment"])
def test_the_same_given_seed_reproduces_the_draws(tmp_path: Path, monkeypatch, via: str):
    """Both channels a deployment has: a constructor argument and the environment."""

    def once():
        if via == "environment":
            monkeypatch.setenv(SEED_ENV, "4242")
            return _seeded_run(tmp_path)
        return _seeded_run(tmp_path, seed=4242)

    seed_a, draw_a = once()
    seed_b, draw_b = once()
    assert seed_a == seed_b == 4242
    assert (draw_a == draw_b).all()


def test_the_worlds_own_seed_is_honoured_and_an_explicit_one_beats_it(tmp_path: Path):
    """``sim.seed`` was validated at load and then discarded on this path."""
    p = tmp_path / "seeded.yaml"
    p.write_text(
        WORLD + "\nsim:\n  seed: 11\n" if False else WORLD.replace("sim:\n", "sim:\n  seed: 11\n")
    )
    sim = MujocoSim(world=str(p))
    sim.setup(logger=logging.getLogger("t"))
    sim.reset()
    assert sim._engine.ctx.seed == 11
    sim.shutdown()

    sim = MujocoSim(world=str(p), seed=12)
    sim.setup(logger=logging.getLogger("t"))
    sim.reset()
    assert sim._engine.ctx.seed == 12, "the caller states what THIS run uses"
    sim.shutdown()


def test_a_rebuild_does_not_redraw(tmp_path: Path):
    """One run, one seed. A reset that rebuilds must not change the value the recording carries."""
    sim = MujocoSim(world=_world(tmp_path))
    sim.setup(logger=logging.getLogger("t"))
    sim.reset()
    first = sim._engine.ctx.seed
    sim.reset(world_overrides={"sim": {"timestep": 0.004}})
    assert sim._engine.ctx.seed == first
    sim.shutdown()


# -- the ordering the fix depends on ----------------------------------------------------------


class _ReadsSeedAtConfigure(Plugin):
    """Draws in ``configure``, which is what makes the assignment order load-bearing."""

    PLUGIN_LABEL = "seed_probe"

    def configure(self, ctx: SimContext) -> None:
        ctx.blackboard.set("seed_probe::at_configure", ctx.seed)


def test_the_seed_is_set_before_setup_so_configure_can_read_it():
    """`configure` may draw -- the navigator did -- so a seed applied later is applied too late."""
    cfg = load_config_from_dict({"sim": {}, "plugins": [{f"{__name__}:_ReadsSeedAtConfigure": {}}]})
    engine = Engine(cfg)
    engine.ctx.seed = 17
    engine.setup()
    try:
        assert engine.ctx.blackboard.get("seed_probe::at_configure") == 17
    finally:
        engine.shutdown()


# -- the compile-only drivers -------------------------------------------------------------------


class _DrawsAtConfigure(Plugin):
    """A randomised world, in one line: something is drawn while the world is being set up."""

    PLUGIN_LABEL = "drawer"

    def configure(self, ctx: SimContext) -> None:
        ctx.blackboard.set("drawer::value", float(ctx.rng_for("drawer").normal(0.0, 1.0)))


def _drawing_world():
    return load_config_from_dict({"sim": {}, "plugins": [{f"{__name__}:_DrawsAtConfigure": {}}]})


def test_a_driver_that_never_runs_still_cannot_forget_the_seed():
    """The rule is unchanged where it means something: no seed, no draw, no silent zero."""
    engine = Engine(_drawing_world())
    with pytest.raises(SeedError):
        engine.setup()
    engine.shutdown()


def test_preview_compiles_a_world_that_draws():
    """The hole the rule left: a tool that produces a PICTURE refused a world it could draw.

    ``roqsim render``, ``roqsim check``, the map and MoveIt exports and the scene preview all
    compile a world in order to look at it. There is no run to replay and no seed the caller could
    supply, so the refusal named a remedy that did not exist -- and the world check that a campaign
    runs before spending anything reported a perfectly good randomised world as broken.
    """
    engine = Engine(_drawing_world(), preview=True)
    engine.setup()
    try:
        assert engine.ctx.blackboard.get("drawer::value") is not None
    finally:
        engine.shutdown()


def test_two_previews_of_one_world_are_the_same_picture():
    """Fixed, not drawn: a render that differed run to run would be a diff nobody could review."""
    values = []
    for _ in range(2):
        engine = Engine(_drawing_world(), preview=True)
        engine.setup()
        values.append(engine.ctx.blackboard.get("drawer::value"))
        engine.shutdown()
    assert values[0] == values[1]


def test_a_preview_seed_is_still_the_drivers_to_override():
    """Assigned at construction, so a driver that wants another seed can say so before setup.

    A scene preview that wanted to look at a second draw of the same world would otherwise have to
    reach past the flag it just used.
    """
    engine = Engine(_drawing_world(), preview=True)
    engine.ctx.seed = 12345
    engine.setup()
    try:
        assert engine.ctx.seed == 12345
    finally:
        engine.shutdown()
