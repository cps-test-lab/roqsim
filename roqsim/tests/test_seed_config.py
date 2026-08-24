# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``sim.seed`` — the run's noise seed as a world setting, not only a CLI flag.

Sensor noise is seeded through ``SimContext.rng_for``, but the seed could only be
given on the command line. Everything else about a run is configurable in the world
YAML and therefore overridable by ``--set`` / ``--override``; the seed was the one
thing that was not, so the noise a run uses could not be chosen from a world file the
way every other simulator setting can.
"""

import logging

import pytest

from roqsim.config import PluginError, load_config
from roqsim.runner import _resolve_seed


def _cfg(tmp_path, body: str):
    p = tmp_path / "w.yaml"
    p.write_text(body)
    return load_config(str(p))


# -- the config key ---------------------------------------------------------


def test_seed_is_read_from_the_sim_block(tmp_path):
    assert _cfg(tmp_path, "sim: {seed: 7}\nplugins: []\n").seed == 7


def test_seed_absent_is_none(tmp_path):
    """Absent means 'draw one', which is not the same as seed 0."""
    assert _cfg(tmp_path, "sim: {}\nplugins: []\n").seed is None


def test_seed_zero_is_a_seed_and_not_absence(tmp_path):
    assert _cfg(tmp_path, "sim: {seed: 0}\nplugins: []\n").seed == 0


@pytest.mark.parametrize("value", ["'abc'", "-1", "1.5"])
def test_a_malformed_seed_is_rejected_at_load_time(tmp_path, value):
    """Loudly, and at load -- a seed that silently became 0 would make every run of a
    campaign share one noise draw while looking varied."""
    with pytest.raises(PluginError, match="sim.seed"):
        _cfg(tmp_path, f"sim: {{seed: {value}}}\nplugins: []\n")


def test_seed_survives_an_override(tmp_path):
    """The point of the feature: the seed is reachable through the same override
    channel as every other simulator setting."""
    from roqsim.config import apply_overrides

    p = tmp_path / "w.yaml"
    p.write_text("sim: {seed: 1}\nplugins: []\n")
    merged = apply_overrides({"sim": {"seed": 1}, "plugins": []}, {"sim": {"seed": 99}})
    assert merged["sim"]["seed"] == 99


# -- precedence -------------------------------------------------------------


def test_explicit_seed_beats_the_config(caplog):
    with caplog.at_level(logging.INFO):
        assert _resolve_seed(5, logging.getLogger("t"), config_seed=7) == 5


def test_config_seed_is_used_when_no_explicit_one(caplog):
    with caplog.at_level(logging.INFO):
        assert _resolve_seed(None, logging.getLogger("t"), config_seed=7) == 7


def test_a_seed_is_drawn_when_neither_is_given(caplog):
    """Unchanged behaviour: an unseeded run is still varied, and the draw is announced
    so it can be repeated."""
    with caplog.at_level(logging.INFO):
        drawn = _resolve_seed(None, logging.getLogger("t"), config_seed=None)
    assert isinstance(drawn, int) and 0 <= drawn < 2**31
    assert "drawn" in caplog.text
