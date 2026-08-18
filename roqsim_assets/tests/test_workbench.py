"""The parametric ESD workbench: the worktop height is config (the lift column's 0.695..0.995 m
stroke), so the two catalogue settings are one YAML key apart and a campaign can sweep it."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_assets.plugins.workbench import WorkbenchPlugin


def _built(tmp_path, extra=None):
    plugins = [{"workbench": {"name": "bench", **(extra or {})}}]
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    return engine


def _gid(engine, name):
    gid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} not found"
    return gid


def _pos(engine, name):
    return engine.ctx.model.geom_pos[_gid(engine, name)]


def _size(engine, name):
    return engine.ctx.model.geom_size[_gid(engine, name)]


def _has(engine, name):
    return mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


def test_default_is_the_lowest_column_position(tmp_path):
    # 2.00 x 0.70 m worktop, top face at the datasheet's nominal 695 mm.
    engine = _built(tmp_path)
    assert np.allclose(_size(engine, "worktop"), [1.0, 0.35, 0.0125])
    assert _pos(engine, "worktop")[2] + 0.0125 == pytest.approx(0.695, abs=1e-9)


@pytest.mark.parametrize("height", [0.695, 0.845, 0.995])
def test_height_places_the_top_face_exactly(tmp_path, height):
    engine = _built(tmp_path, {"height": height})
    assert _pos(engine, "worktop")[2] + 0.0125 == pytest.approx(height, abs=1e-9)


def test_raised_bench_extends_the_column_and_lifts_what_hangs_off_the_top(tmp_path):
    low, high = _built(tmp_path), _built(tmp_path, {"height": 0.995})
    delta = 0.995 - 0.695
    # The upper (telescopic) stage grows by the full stroke; the fixed lower stage does not move.
    assert _size(high, "column_upper_l")[2] - _size(low, "column_upper_l")[2] == pytest.approx(
        delta / 2, abs=1e-9
    )
    assert _pos(high, "column_lower_l")[2] == pytest.approx(_pos(low, "column_lower_l")[2])
    # Everything carried by the worktop rides with it: frame rails, cabinet, panel, monitor mount.
    for name in ("top_rail_l", "cabinet", "tool_panel", "monitor_plate", "power_strip"):
        assert _pos(high, name)[2] - _pos(low, name)[2] == pytest.approx(delta, abs=1e-9)
    # The uprights and the overhead frame are floor-referenced -- they stay put.
    for name in ("upright_l", "overhead_back", "luminaire"):
        assert _pos(high, name)[2] == pytest.approx(_pos(low, name)[2], abs=1e-9)


def test_raised_bench_keeps_the_tool_panel_clear_of_the_light_frame(tmp_path):
    engine = _built(tmp_path, {"height": 0.995})
    panel_top = _pos(engine, "tool_panel")[2] + _size(engine, "tool_panel")[2]
    luminaire_bottom = _pos(engine, "luminaire")[2] - _size(engine, "luminaire")[2]
    assert panel_top < luminaire_bottom


def test_bench_stands_on_the_floor(tmp_path):
    # min z == 0 (the foot pads), so a pose of (x y z) drops the bench exactly there.
    engine = _built(tmp_path, {"height": 0.995})
    m = engine.ctx.model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "workbench")
    # Box and cylinder alike put the half-height in size[2] resp. size[1] -- the only two types used.
    bottoms = [
        m.geom_pos[i][2]
        - (m.geom_size[i][2] if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[i][1])
        for i in range(m.ngeom)
        if m.geom_bodyid[i] == bid
    ]
    assert min(bottoms) == pytest.approx(0.0, abs=1e-9)


def test_cabinet_side_and_omission(tmp_path):
    assert _pos(_built(tmp_path), "cabinet")[0] < 0  # 'left' is the operator's left, -X
    assert _pos(_built(tmp_path, {"cabinet": "right"}), "cabinet")[0] > 0
    bare = _built(tmp_path, {"cabinet": "none"})
    assert not _has(bare, "cabinet")
    assert not _has(bare, "drawer_0")
    assert _has(bare, "power_strip")  # the strip is not part of the cabinet


def test_superstructure_can_be_dropped(tmp_path):
    engine = _built(tmp_path, {"superstructure": False})
    for name in ("upright_l", "tool_panel", "monitor_plate", "overhead_back", "luminaire"):
        assert not _has(engine, name)
    assert _has(engine, "worktop")


def test_width_moves_the_frame_and_the_panel(tmp_path):
    engine = _built(tmp_path, {"width": 1.60})
    assert _size(engine, "worktop")[0] == pytest.approx(0.80, abs=1e-9)
    assert _pos(engine, "upright_l")[0] == pytest.approx(-(0.80 - 0.15), abs=1e-9)
    assert _size(engine, "tool_panel")[0] == pytest.approx(0.80 - 0.012, abs=1e-9)


def test_structure_collides_but_trim_does_not(tmp_path):
    engine = _built(tmp_path)
    m = engine.ctx.model
    for name in ("worktop", "column_lower_l", "upright_l", "cabinet", "tool_panel"):
        assert m.geom_contype[_gid(engine, name)] != 0, name
    for name in ("drawer_0", "drawer_handle_0", "power_strip", "monitor_arm_1", "luminaire"):
        assert m.geom_contype[_gid(engine, name)] == 0, name


def test_validate_config_rejects_bad_geometry():
    def errs(cfg):
        return WorkbenchPlugin(cfg).validate_config(cfg)

    assert errs({"height": 0.60})  # below the column's stroke
    assert errs({"height": 1.20})  # above it
    assert errs({"height": "low"})  # not a number
    assert errs({"width": 0})  # not > 0
    assert errs({"width": 1.0})  # no room for a cabinet between the columns
    assert errs({"depth": 0.30})  # no room for the lift column
    assert errs({"cabinet": "middle"})  # not a side
    assert errs({"rpy": [0.0, 0.0]})  # wrong length
    assert not errs({"height": 0.995})  # the raised catalogue position
    assert not errs({"width": 1.0, "cabinet": "none"})  # a narrow bench without a cabinet
