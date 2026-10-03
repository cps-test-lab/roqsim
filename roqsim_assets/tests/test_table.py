"""The parametric table: the top's surface is at `height`, its size is `width` x `depth`, the legs stand
inside it on the floor, and what is set on it rests there."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_assets.plugins.table import TablePlugin


def _engine(tmp_path, table=None, extra=()):
    plugins = [{"table": dict(table or {}), "name": "bench"}, *extra]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    return engine


def _geom(engine, name):
    gid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} not found"
    return gid


def _aabb(engine, name):
    m, d = engine.ctx.model, engine.ctx.data
    g = _geom(engine, name)
    half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
    return d.geom_xpos[g] - half, d.geom_xpos[g] + half


@pytest.mark.parametrize("height", [0.50, 0.62, 0.75, 0.97])
def test_the_top_surface_is_at_the_height(tmp_path, height):
    engine = _engine(tmp_path, {"height": height})
    assert _aabb(engine, "top")[1][2] == pytest.approx(height, abs=1e-9)


def test_the_top_is_width_by_depth_and_the_legs_stand_inside_it_on_the_floor(tmp_path):
    engine = _engine(tmp_path, {"width": 1.6, "depth": 0.6, "height": 0.62, "leg": 0.06, "leg_inset": 0.02})
    lo, hi = _aabb(engine, "top")
    assert hi[0] - lo[0] == pytest.approx(1.6) and hi[1] - lo[1] == pytest.approx(0.6)
    for tag in ("rl", "fl", "fr", "rr"):
        llo, lhi = _aabb(engine, f"leg_{tag}")
        assert llo[2] == pytest.approx(0.0, abs=1e-9)
        assert lhi[2] == pytest.approx(lo[2], abs=1e-9)  # up to the top's underside
        assert lo[0] + 0.02 - 1e-9 <= llo[0] and lhi[0] <= hi[0] - 0.02 + 1e-9
        assert lo[1] + 0.02 - 1e-9 <= llo[1] and lhi[1] <= hi[1] - 0.02 + 1e-9


def test_without_an_apron_the_space_between_the_legs_is_open(tmp_path):
    engine = _engine(tmp_path, {"apron": False})
    names = {engine.ctx.model.geom(i).name for i in range(engine.ctx.model.ngeom)}
    assert not any("apron" in n for n in names)
    with_apron = _engine(tmp_path, {"apron": True})
    names = {with_apron.ctx.model.geom(i).name for i in range(with_apron.ctx.model.ngeom)}
    assert sum("apron" in n for n in names) == 4


def test_the_pose_places_and_turns_it(tmp_path):
    engine = _engine(tmp_path, {"width": 2.0, "depth": 0.5, "pose": {"position": {"x": 3.0, "y": -1.0},
                                                                     "orientation": {"yaw": 1.5707963}}})
    lo, hi = _aabb(engine, "top")
    assert (lo[0] + hi[0]) / 2 == pytest.approx(3.0, abs=1e-6)
    assert hi[0] - lo[0] == pytest.approx(0.5, abs=1e-6)  # turned: the long side along Y
    assert hi[1] - lo[1] == pytest.approx(2.0, abs=1e-6)


def test_a_box_set_on_it_rests_on_the_top(tmp_path):
    box = {"spawn_model": {"model": "graspable_box", "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.70}}},
           "name": "parcel"}
    engine = _engine(tmp_path, {"height": 0.62}, extra=[box])
    for _ in range(1000):
        engine.step()
    m, d = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "graspable_box")
    if bid < 0:  # the spawned body carries the plugin's prefix
        bid = next(i for i in range(m.nbody) if m.body(i).name.endswith("graspable_box"))
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == bid)
    bottom = d.geom_xpos[g][2] - (np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g])[2]
    assert bottom == pytest.approx(0.62, abs=0.003)


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"height": -1}, "'height' must be > 0"),
        ({"width": "wide"}, "'width' must be a number"),
        ({"top_thickness": 0.8, "height": 0.75}, "must be less than 'height'"),
        ({"width": 0.12, "leg": 0.05, "leg_inset": 0.02}, "no room for two legs"),
        ({"leg_inset": -0.1}, "'leg_inset' must be >= 0"),
        ({"top_rgba": [1, 0, 0]}, "'top_rgba' must be a list of 4 numbers"),
    ],
)
def test_bad_config_is_reported(bad, fragment):
    errors = TablePlugin(bad).validate_config(bad)
    assert any(fragment in e for e in errors), errors


def test_the_default_config_is_valid():
    assert TablePlugin({}).validate_config({}) == []
