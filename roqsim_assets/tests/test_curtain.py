"""The curtain: visual scenery of the stated size, matte, colliding with nothing."""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_assets.plugins.curtain import CurtainPlugin


def _built(tmp_path, extra=None):
    plugins = [{"curtain": dict(extra or {}), "name": "backdrop"}]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    return engine


def _cloth(engine):
    m = engine.ctx.model
    return [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("cloth_")]


def _bounds(engine, gids):
    m, d = engine.ctx.model, engine.ctx.data
    corners = []
    for g in gids:
        R, h = d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(d.geom_xpos[g] + R @ (h * [sx, sy, sz]))
    corners = np.array(corners)
    return corners.min(0), corners.max(0)


def test_the_cloth_spans_the_width_from_the_hem_to_the_bar(tmp_path):
    engine = _built(tmp_path, {"width": 2.4, "height": 2.1, "hem": 0.05, "pleats": 12})
    lo, hi = _bounds(engine, _cloth(engine))
    assert len(_cloth(engine)) == 24
    assert hi[0] - lo[0] == pytest.approx(2.4, abs=0.01)
    assert lo[2] == pytest.approx(0.05, abs=1e-9)
    assert hi[2] == pytest.approx(2.1 - 0.03, abs=1e-9)
    assert 0.05 < hi[1] - lo[1] < 0.08  # the folds' depth, and the cloth's thickness
    engine.shutdown()


def test_nothing_of_it_collides_and_the_cloth_is_matte(tmp_path):
    engine = _built(tmp_path)
    m = engine.ctx.model
    body = m.body("curtain").id
    ours = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
    assert len(ours) > 2 * 24  # the cloth, the bar and its clamps
    for g in ours:
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, m.geom(g).name
    mat = m.material("curtain_cloth")
    assert mat.specular == 0.0 and mat.shininess == 0.0 and mat.reflectance == 0.0
    assert max(mat.rgba[:3]) < 0.05
    engine.shutdown()


def test_the_pose_places_it(tmp_path):
    engine = _built(tmp_path, {"pose": {"position": {"x": 0.5, "y": 1.0}}, "width": 2.0})
    lo, hi = _bounds(engine, _cloth(engine))
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.5, abs=0.01)
    assert (lo[1] + hi[1]) / 2 == pytest.approx(1.0, abs=0.01)
    engine.shutdown()


def test_a_hem_above_the_bar_is_refused():
    config = {"height": 1.0, "hem": 1.2}
    assert any("'hem' must be under" in e for e in CurtainPlugin(config).validate_config(config))
