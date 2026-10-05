"""The floorplan's light hangs under its walls' top: a spot's cone edge is the horizontal plane through
it, and walls whose tops lie in that plane flicker in a moving camera."""

from __future__ import annotations

import logging

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

ROOM = [
    {"id": 0, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 6.0, "y1_m": 0.0},
    {"id": 1, "x0_m": 6.0, "y0_m": 0.0, "x1_m": 6.0, "y1_m": 4.0},
    {"id": 2, "x0_m": 6.0, "y0_m": 4.0, "x1_m": 0.0, "y1_m": 4.0},
    {"id": 3, "x0_m": 0.0, "y0_m": 4.0, "x1_m": 0.0, "y1_m": 0.0},
]


def _light_z(**config) -> float:
    engine = Engine(
        load_config_from_dict({"sim": {}, "components": [{"floorplan": {"lines": ROOM, **config}}]})
    )
    engine.setup()
    (z,) = [float(engine.ctx.model.light_pos[i][2]) for i in range(engine.ctx.model.nlight)]
    return z


def test_the_default_light_hangs_under_default_walls():
    assert _light_z() == pytest.approx(2.35)


def test_the_default_light_hangs_under_walls_of_its_own_height():
    assert _light_z(height=2.5) == pytest.approx(2.35)


def test_the_default_light_stays_where_the_walls_are_taller():
    assert _light_z(height=6.0) == pytest.approx(2.5)


def test_a_light_the_world_places_at_the_walls_top_stays_and_is_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="roqsim_mobile.floorplan"):
        assert _light_z(height=6.0, light={"height": 6.0}) == pytest.approx(6.0)
    assert any("cone edge" in r.getMessage() for r in caplog.records)


def test_a_light_the_world_places_elsewhere_is_left_alone(caplog):
    with caplog.at_level(logging.WARNING, logger="roqsim_mobile.floorplan"):
        assert _light_z(height=6.0, light={"height": 5.85}) == pytest.approx(5.85)
    assert not any("cone edge" in r.getMessage() for r in caplog.records)
