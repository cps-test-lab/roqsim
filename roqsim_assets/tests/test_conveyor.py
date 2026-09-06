"""The reusable belt conveyor: it advances a riding package, its speed is live-controllable, and its
belt geometry rescales to a configured length/width. (The full sorting cell -- arms beside the belt --
is exercised by roqsim_tetrisort.)"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _belt_only(tmp_path, conv_extra=None):
    plugins = [
        {"conveyor": dict(conv_extra or {}), "name": "conveyor"},
    ]
    return load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path)


def _run(engine, steps):
    engine.setup()
    engine.reset()
    for _ in range(steps):
        engine.step()


def _package_x(engine):
    bid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "package")
    return float(engine.ctx.data.xpos[bid][0])


def _geom_size(engine, name):
    gid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return engine.ctx.model.geom_size[gid]


def _body_pos(engine, name):
    bid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return engine.ctx.model.body_pos[bid]


def _geom_pos(engine, name):
    """World position of a geom on a welded body (model-fixed, so no step is needed)."""
    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} not found"
    return m.body_pos[m.geom_bodyid[gid]] + m.geom_pos[gid]


def test_belt_advances_package(tmp_path):
    engine = Engine(_belt_only(tmp_path, {"speed": 0.2}))
    _run(engine, 5)
    x0 = _package_x(engine)
    for _ in range(1000):
        engine.step()
    x1 = _package_x(engine)
    # Belt drives -x (axis "-1 0 0"); the package should move toward the discharge end.
    assert x1 < x0 - 0.05


def test_speed_handle_reverses_belt(tmp_path):
    engine = Engine(_belt_only(tmp_path, {"speed": 0.2}))
    engine.setup()
    engine.reset()
    handle = engine.ctx.blackboard.require("conveyor:conveyor")
    handle.set_speed(-0.2)  # reverse toward the feed end (+x)
    for _ in range(5):
        engine.step()
    x0 = _package_x(engine)
    for _ in range(1000):
        engine.step()
    assert _package_x(engine) > x0 + 0.05


def test_belt_resizes_to_configured_length_and_width(tmp_path):
    # A custom 3.5 m x 0.8 m belt: half-extents L=1.75, W=0.4. Belt surfaces recompute exactly,
    # the drive slab keeps its fixed overhang, and rollers sit at the new belt ends.
    engine = Engine(_belt_only(tmp_path, {"length": 3.5, "width": 0.8}))
    engine.setup()
    engine.reset()
    L, W = 1.75, 0.4
    assert np.allclose(_geom_size(engine, "belt_visual"), [L, W, 0.0275])
    assert np.allclose(_geom_size(engine, "belt_surface"), [L + 0.029, W, 0.0275])
    # Roller half-length spans the width (radius fixed) and the body sits at the belt end.
    assert np.allclose(_geom_size(engine, "roller_a_geom"), [0.0275, W, 0.0])
    assert np.allclose(_body_pos(engine, "roller_a"), [-L, 0.0, 0.9425])
    assert np.allclose(_body_pos(engine, "roller_b"), [L, 0.0, 0.9425])
    # Package default start pose tracks the feed end (L - 0.221) and rides the resized belt.
    x0 = _package_x(engine)
    assert x0 == pytest.approx(L - 0.221, abs=1e-6)
    for _ in range(1000):
        engine.step()
    assert _package_x(engine) < x0 - 0.05


def test_placed_belt_keeps_its_package(tmp_path):
    # package_pose is model-local but the package rides a FREE joint, whose qpos is a world pose --
    # so a belt spawned away from the origin must still reset its package onto its own belt, not
    # onto the floor at the world origin.
    pos, yaw = [13.584, 5.779, 0.0], 0.8936
    engine = Engine(_belt_only(tmp_path, {"pos": pos, "rpy": [0.0, 0.0, yaw], "length": 2.0}))
    engine.setup()
    engine.reset()
    belt = _geom_pos(engine, "belt_visual")  # belt centre, world
    pkg = engine.ctx.data.xpos[
        mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "package")
    ]
    assert pkg[2] == pytest.approx(0.996, abs=1e-6)  # on the belt surface, not on the floor
    # Offset from belt centre is purely along the belt's own +x (the feed end), 0.779 m out.
    delta = np.array(pkg[:2]) - np.array(belt[:2])
    along = delta @ np.array([np.cos(yaw), np.sin(yaw)])
    across = delta @ np.array([-np.sin(yaw), np.cos(yaw)])
    assert along == pytest.approx(1.0 - 0.221, abs=1e-6)
    assert across == pytest.approx(0.0, abs=1e-6)
    # And it still rides: the belt carries it back toward the discharge end.
    for _ in range(1000):
        engine.step()
    moved = engine.ctx.data.xpos[
        mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "package")
    ]
    assert (np.array(moved[:2]) - np.array(belt[:2])) @ np.array(
        [np.cos(yaw), np.sin(yaw)]
    ) < along - 0.05


def test_belt_ships_no_table(tmp_path):
    # The belt is a benchtop unit: the industrial table it used to bundle is a separate prop now,
    # so a bare conveyor must contribute no table geometry (and the feet must still expect a
    # ~0.76 m top under them, which is what makes the split poses in the worlds line up).
    engine = Engine(_belt_only(tmp_path))
    engine.setup()
    m = engine.ctx.model
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
    assert not {n for n in names if n and ("platform" in n or "plat_leg" in n)}
    foot = _geom_pos(engine, "conv_leg_1")
    assert foot[2] - 0.0775 == pytest.approx(0.76, abs=1e-6)


def test_industrial_table_top_carries_the_belt(tmp_path):
    # The split-out table must present its top exactly where the belt's feet land, so
    # `spawn_model industrial_table` + `conveyor` at the same z reproduces the old bundled cell.
    plugins = [
        {
            "spawn_model": {
                "model": "industrial_table",
                "pose": {"position": {"x": -0.13, "y": 0.6, "z": 0.0}},
            },
            "name": "bench",
        },
        {"conveyor": {}, "name": "conveyor"},
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    top = _geom_pos(engine, "industrial_table_top")
    assert top[2] + _geom_size(engine, "industrial_table_top")[2] == pytest.approx(0.76, abs=1e-6)
    # Table top centred under the belt (belt at model y=0.6, table spawned at y=0.6 - its own centre).
    assert top[0] == pytest.approx(-0.13, abs=1e-6)
    assert top[1] == pytest.approx(0.6, abs=1e-6)


def test_default_size_matches_base_model(tmp_path):
    # Omitting length/width must leave the base geometry byte-for-byte (no resize path taken).
    engine = Engine(_belt_only(tmp_path))
    engine.setup()
    engine.reset()
    assert np.allclose(_geom_size(engine, "belt_visual"), [1.221, 0.29, 0.0275])
    assert np.allclose(_body_pos(engine, "roller_a"), [-1.221, 0.0, 0.9425])
