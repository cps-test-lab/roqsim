"""The parametric window: a glazed pane in a slim frame, built from boxes to the configured size and
placed by its opening centre -- the counterpart to ``door`` for an opening that is glazed, not hung."""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _window(tmp_path, extra=None):
    plugins = [{"window": dict(extra or {}), "name": "window"}]
    return load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path)


def _built(tmp_path, extra=None):
    engine = Engine(_window(tmp_path, extra))
    engine.setup()
    engine.reset()
    return engine


def _geom(engine, name):
    gid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, name
    return gid


def _size(engine, name):
    return np.asarray(engine.ctx.model.geom_size[_geom(engine, name)], dtype=float)


def _pos(engine, name):
    """World position of a geom (from data, after the frame placement is applied)."""
    return np.asarray(engine.ctx.data.geom_xpos[_geom(engine, name)], dtype=float)


def test_defaults_match_the_door_unit(tmp_path):
    # The defaults exist so a window lines up with this library's door: 2.06 m is door_frame's outer
    # casing height, and the frame is as deep as the generator's default wall.
    engine = _built(tmp_path)
    assert engine.ctx.entities.get("window").kind == "prop"
    head = _size(engine, "head")
    stile = _size(engine, "stile_p")
    assert np.isclose(2 * stile[2], 2.06)  # overall height
    assert np.isclose(2 * stile[1], 0.10)  # depth through the wall
    assert np.isclose(2 * head[0], 0.94 - 2 * 0.05)  # glass span between the stiles


def test_width_and_height_are_parametric(tmp_path):
    engine = _built(tmp_path, {"width": 1.6, "height": 1.2, "frame": 0.08, "depth": 0.2})
    assert np.isclose(2 * _size(engine, "stile_p")[2], 1.2)
    assert np.isclose(2 * _size(engine, "stile_p")[1], 0.2)
    # Stiles sit inside the width, glass and rails span what the border leaves.
    assert np.isclose(_pos(engine, "stile_p")[0] - _pos(engine, "stile_n")[0], 1.6 - 0.08)
    assert np.isclose(2 * _size(engine, "glass")[0], 1.6 - 2 * 0.08)
    assert np.isclose(2 * _size(engine, "glass")[2], 1.2 - 2 * 0.08)


def test_sits_on_the_floor_at_the_opening_centre(tmp_path):
    # Placed by the opening CENTRE like a door, base on the floor: the sill's underside is at z=0.
    engine = _built(tmp_path, {"pos": [3.0, -1.5], "width": 1.0, "frame": 0.05})
    sill = _pos(engine, "sill")
    assert np.allclose(sill[:2], [3.0, -1.5])
    assert np.isclose(sill[2] - _size(engine, "sill")[2], 0.0)


def test_yaw_turns_the_pane_onto_its_wall(tmp_path):
    # At yaw 90 deg the width runs along world Y instead of X.
    engine = _built(tmp_path, {"width": 1.2, "rpy": [0.0, 0.0, np.pi / 2]})
    span = _pos(engine, "stile_p") - _pos(engine, "stile_n")
    assert abs(span[1]) > 1.0 and abs(span[0]) < 1e-6


def test_glass_is_translucent_and_collides(tmp_path):
    engine = _built(tmp_path)
    m = engine.ctx.model
    gid = _geom(engine, "glass")
    assert m.mat_rgba[m.geom_matid[gid]][3] < 1.0  # see-through
    assert m.geom_contype[gid] != 0  # but a robot cannot drive through it


def test_colors_are_configurable(tmp_path):
    engine = _built(tmp_path, {"color": [0.1, 0.2, 0.3], "glass_color": [0.4, 0.5, 0.6, 0.5]})
    m = engine.ctx.model
    assert np.allclose(m.mat_rgba[m.geom_matid[_geom(engine, "head")]], [0.1, 0.2, 0.3, 1.0])
    assert np.allclose(m.mat_rgba[m.geom_matid[_geom(engine, "glass")]], [0.4, 0.5, 0.6, 0.5])


def test_validate_config_flags_bad_input(tmp_path):
    from roqsim_assets.plugins.window import WindowPlugin

    bad = {"width": -1, "frame": 0.5, "height": 0.9, "color": "grey"}
    errors = " ".join(WindowPlugin(bad).validate_config(bad))
    assert "'width' must be > 0" in errors
    assert "leaves no glass" in errors  # a 0.5 m border in a 0.9 m height
    assert "'color' must be" in errors
