"""The lights MuJoCo's renderer leaves dark: the eight it draws in index order, the headlight among
them, and the warning the engine logs for the rest."""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim import render
from roqsim.lights import RENDERED_LIGHTS, summary, undrawn_lights


def _row_of_lamps(n: int, *, headlight: bool = True, inactive: tuple[int, ...] = ()) -> str:
    """A floor under a row of spot lights, each lighting its own patch, 2 m apart."""
    lamps = "".join(
        f'<light name="lamp_{i}" pos="{2 * i} 0 1" dir="0 0 -1" cutoff="25" diffuse=".5 .5 .5" '
        f'castshadow="false" active="{"false" if i in inactive else "true"}"/>'
        for i in range(n)
    )
    return (
        f'<mujoco><visual><headlight active="{int(headlight)}"/></visual>'
        f'<worldbody>{lamps}<geom type="plane" size="30 5 .1" rgba="1 1 1 1"/></worldbody></mujoco>'
    )


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def test_eight_lights_without_a_headlight_are_all_drawn():
    assert undrawn_lights(_model(_row_of_lamps(RENDERED_LIGHTS, headlight=False))) == []


def test_the_headlight_takes_one_of_the_eight():
    assert undrawn_lights(_model(_row_of_lamps(RENDERED_LIGHTS))) == ["lamp_7"]


def test_the_lights_past_the_eighth_are_named_in_index_order():
    assert undrawn_lights(_model(_row_of_lamps(10, headlight=False))) == ["lamp_8", "lamp_9"]


def test_an_inactive_light_takes_no_place():
    assert undrawn_lights(_model(_row_of_lamps(9, headlight=False, inactive=(0,)))) == []


def test_the_summary_names_the_count_and_the_lights():
    model = _model(_row_of_lamps(10))
    line = summary(undrawn_lights(model), model)
    assert "10 active lights plus the headlight" in line and "3 are never drawn" in line
    assert "lamp_7, lamp_8, lamp_9" in line


def test_the_engine_says_it_when_the_world_compiles(tmp_path, caplog):
    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    scene = tmp_path / "lamps.xml"
    scene.write_text(_row_of_lamps(10))
    engine = Engine(
        load_config_from_dict(
            {"sim": {}, "components": [{"spawn_model": {"model": str(scene), "motion": "static"}}]}
        )
    )
    with caplog.at_level(logging.WARNING, logger="roqsim.engine"):
        engine.setup()
    assert any("never drawn" in r.getMessage() for r in caplog.records)


def _shot(tmp_path, xml: str, name: str) -> np.ndarray:
    from PIL import Image

    scene = tmp_path / f"{name}.xml"
    scene.write_text(xml)
    out = tmp_path / f"{name}.png"
    try:
        render.render_target(
            str(scene),
            out,
            size="320x120",
            view=["lookat=9,0,0", "distance=30", "azimuth=90", "elevation=-60"],
        )
    except render.RenderError as err:
        pytest.skip(f"no usable offscreen GL here: {err}")
    return np.asarray(Image.open(out).convert("RGB"), dtype=int)


def test_mujoco_draws_no_light_past_the_ones_named(tmp_path, monkeypatch):
    """What the module claims about the renderer, checked against it: switching off a light it calls
    undrawn changes no pixel, switching off one it draws does."""
    monkeypatch.setenv("MUJOCO_GL", __import__("os").environ.get("MUJOCO_GL", "egl"))
    base = _shot(tmp_path, _row_of_lamps(10), "all")
    assert np.array_equal(base, _shot(tmp_path, _row_of_lamps(10, inactive=(9,)), "last_off"))
    assert not np.array_equal(
        base, _shot(tmp_path, _row_of_lamps(10, inactive=(6,)), "seventh_off")
    )
