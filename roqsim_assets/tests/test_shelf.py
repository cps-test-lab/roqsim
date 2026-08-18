"""The parametric chipboard shelf: it builds from box geoms so the number of layers (and the
board dimensions) are config -- ``layers: 8`` yields an eight-board shelf, ``depth: 0.40`` the
half-depth footprint -- unlike the fixed 5-board baked mesh model."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_assets.plugins.shelf import ShelfPlugin


def _shelf(tmp_path, extra=None):
    plugins = [{"shelf": {"name": "shelf", **(extra or {})}}]
    return load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path)


def _built(tmp_path, extra=None):
    engine = Engine(_shelf(tmp_path, extra))
    engine.setup()
    return engine


def _count(engine, stem):
    """Number of geoms named ``<stem>_0``, ``<stem>_1``, ... present in the compiled model."""
    n = 0
    while mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, f"{stem}_{n}") >= 0:
        n += 1
    return n


def _geom_size(engine, name):
    gid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} not found"
    return engine.ctx.model.geom_size[gid]


def _board_z(engine, i):
    gid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, f"board_{i}")
    return float(engine.ctx.model.geom_pos[gid][2])


def test_default_five_boards(tmp_path):
    # Defaults reproduce the baked free_chipboard_shelf: 5 boards, ~0.80 x 1.51 m, + 4 uprights.
    engine = _built(tmp_path)
    assert _count(engine, "board") == 5
    assert _count(engine, "leg") == 4
    assert np.allclose(_geom_size(engine, "board_0"), [0.40, 0.755, 0.01])


def test_layers_eight_evenly_spaced(tmp_path):
    engine = _built(tmp_path, {"layers": 8})
    assert _count(engine, "board") == 8
    zs = [_board_z(engine, i) for i in range(8)]
    # Strictly increasing, evenly spaced, spanning from the base-board height to just under 2.0 m.
    diffs = np.diff(zs)
    assert np.allclose(diffs, diffs[0])
    assert zs[0] == pytest.approx(0.08, abs=1e-6)
    assert zs[-1] == pytest.approx(2.00 - 0.02 / 2, abs=1e-6)


def test_depth_halves_board(tmp_path):
    # depth: 0.40 reproduces the half-depth variant -> board X half-extent 0.20 (vs 0.40 default).
    engine = _built(tmp_path, {"depth": 0.40})
    assert _geom_size(engine, "board_0")[0] == pytest.approx(0.20, abs=1e-6)


def test_validate_config_rejects_bad_geometry():
    def errs(cfg):
        return ShelfPlugin(cfg).validate_config(cfg)

    assert errs({"layers": 1})  # too few
    assert errs({"layers": 3.5})  # non-integer
    assert errs({"width": 0})  # not > 0
    assert errs({"thickness": 0.5, "layers": 8, "height": 2.0})  # boards don't fit
    assert errs({"rpy": [0.0, 0.0]})  # wrong length
    assert not errs({"layers": 8, "depth": 0.40})  # a valid custom shelf
