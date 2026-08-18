"""The three ceiling fittings -- acoustic panels, a duct run, an LED batten.

What matters about them is where they end up: everything must sit **above head height**, since that
is the whole contract with the core ``ceiling`` plugin (it opens a roof by deleting geoms whose whole
AABB clears a height, with no name list). The panel field's other job is not to overhang the room it
was measured against.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _built(tmp_path, plugins):
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    return engine


def _geoms(engine, prefix):
    model = engine.ctx.model
    out = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name and name.startswith(prefix):
            out[name] = g
    return out


def _pos(engine, gid):
    return np.asarray(engine.ctx.data.geom_xpos[gid], dtype=float)


# --- acoustic panels ------------------------------------------------------------------------


def _panels(tmp_path, **extra):
    cfg = {"name": "panels", "prefix": "p_", "area": [0.0, 0.0, 10.0, 6.0], "z": 3.5, **extra}
    return _built(tmp_path, [{"ceiling_panels": cfg}])


def test_panels_hang_below_the_soffit_and_stay_inside_the_area(tmp_path):
    engine = _panels(tmp_path, panel=[2.0, 1.0], pitch=[3.0, 2.0], drop=0.04, thickness=0.04)
    geoms = _panels_of(engine)
    assert geoms, "a 10 x 6 m area must fit some 2 x 1 m panels"
    for gid in geoms.values():
        pos = _pos(engine, gid)
        assert math.isclose(
            pos[2], 3.5 - 0.04 - 0.02, abs_tol=1e-6
        )  # soffit - drop - half thickness
        assert 0.0 <= pos[0] - 1.0 and pos[0] + 1.0 <= 10.0  # panel half-width inside the area
        assert 0.0 <= pos[1] - 0.5 and pos[1] + 0.5 <= 6.0


def test_panels_are_visual_only(tmp_path):
    engine = _panels(tmp_path)
    model = engine.ctx.model
    for gid in _panels_of(engine).values():
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


def test_a_panel_too_big_for_the_area_yields_none(tmp_path):
    # No panel fits: the field is empty rather than one panel overhanging the walls.
    engine = _panels(tmp_path, panel=[12.0, 1.0], pitch=[13.0, 2.0])
    assert not _panels_of(engine)


def test_rejects_a_pitch_tighter_than_the_panel(tmp_path):
    from roqsim_assets.plugins.ceiling_panels import CeilingPanelsPlugin

    errors = CeilingPanelsPlugin({"area": [0, 0, 4, 4]}).validate_config(
        {"area": [0, 0, 4, 4], "panel": [2.0, 1.0], "pitch": [1.0, 2.0]}
    )
    assert any("overlap" in e for e in errors)


def test_area_is_required(tmp_path):
    from roqsim_assets.plugins.ceiling_panels import CeilingPanelsPlugin

    assert any("'area' is required" in e for e in CeilingPanelsPlugin({}).validate_config({}))


def _panels_of(engine):
    return {n: g for n, g in _geoms(engine, "p_").items() if "panel" in n}


# --- duct -----------------------------------------------------------------------------------


def test_duct_runs_between_its_endpoints_with_drops_hanging_below(tmp_path):
    engine = _built(
        tmp_path,
        [
            {
                "duct": {
                    "name": "duct",
                    "prefix": "d_",
                    "start": [1.0, 2.0],
                    "end": [9.0, 2.0],
                    "z": 3.2,
                    "radius": 0.15,
                    "branches": [2.0, 6.0],
                    "branch_length": 0.4,
                }
            }
        ],
    )
    geoms = _geoms(engine, "d_")
    run = next(g for n, g in geoms.items() if n.endswith("run"))
    assert np.allclose(_pos(engine, run), [5.0, 2.0, 3.2], atol=1e-6)  # midpoint of the run
    assert math.isclose(engine.ctx.model.geom_size[run][1] * 2, 8.0, abs_tol=1e-6)  # its length

    drops = sorted(g for n, g in geoms.items() if "drop" in n)
    assert len(drops) == 2
    for gid, t in zip(drops, (2.0, 6.0), strict=True):
        pos = _pos(engine, gid)
        assert math.isclose(pos[0], 1.0 + t, abs_tol=1e-6)
        assert math.isclose(pos[2], 3.2 - 0.2, abs_tol=1e-6)  # centre of a 0.4 m drop
    diffusers = [g for n, g in geoms.items() if "diffuser" in n]
    assert len(diffusers) == 2
    # Everything the run emits still clears a 2.6 m ceiling cut -- the contract with `ceiling`.
    assert min(_pos(engine, g)[2] for g in geoms.values()) > 2.6


def test_duct_rejects_a_branch_past_the_end(tmp_path):
    from roqsim_assets.plugins.duct import DuctPlugin

    cfg = {"start": [0.0, 0.0], "end": [3.0, 0.0], "branches": [1.0, 4.0]}
    assert any("outside the 3.00 m run" in e for e in DuctPlugin(cfg).validate_config(cfg))


# --- strip light ----------------------------------------------------------------------------


def test_batten_is_a_fixture_and_emit_adds_a_light(tmp_path):
    # The default empty room already carries its own light, so what is tested is the difference.
    plain = _built(
        tmp_path, [{"strip_light": {"name": "s", "prefix": "s_", "pos": [2.0, 3.0, 3.5]}}]
    )
    fixture = next(g for n, g in _geoms(plain, "s_").items() if n.endswith("fixture"))
    assert _pos(plain, fixture)[2] < 3.5  # hangs below its mounting height
    assert _pos(plain, fixture)[2] > 2.6

    lit = _built(
        tmp_path,
        [{"strip_light": {"name": "s", "prefix": "s_", "pos": [2.0, 3.0, 3.5], "emit": True}}],
    )
    assert lit.ctx.model.nlight == plain.ctx.model.nlight + 1
    added = lit.ctx.model.light_pos[lit.ctx.model.nlight - 1]
    assert np.allclose(added[:2], [2.0, 3.0])
    assert added[2] < 3.5  # under the fixture, not inside it


def test_batten_rejects_a_2d_position(tmp_path):
    from roqsim_assets.plugins.strip_light import StripLightPlugin

    errors = StripLightPlugin({}).validate_config({"pos": [1.0, 2.0]})
    assert any("mounting height" in e for e in errors)


# --- the contract with the core `ceiling` plugin ---------------------------------------------


def test_opening_the_roof_removes_every_fitting_but_keeps_its_light(tmp_path):
    """``ceiling`` last in the list deletes all three fittings by height -- and only the geoms.

    The light a batten emits is added to the worldbody, so a top-down view of the opened building is
    still lit. This is the ordering the fittings' docstrings promise, so it is tested end to end.
    """
    fittings = [
        {"ceiling_panels": {"name": "p", "prefix": "p_", "area": [0.0, 0.0, 8.0, 6.0], "z": 3.5}},
        {"duct": {"name": "d", "prefix": "d_", "start": [1.0, 1.0], "end": [7.0, 1.0], "z": 3.2}},
        {"strip_light": {"name": "s", "prefix": "s_", "pos": [4.0, 3.0, 3.5], "emit": True}},
    ]
    closed = _built(tmp_path, fittings)
    opened = _built(tmp_path, [*fittings, {"ceiling": {"enabled": False, "above_z": 2.6}}])

    for prefix in ("p_", "d_", "s_"):
        assert _geoms(closed, prefix), prefix
        assert not _geoms(opened, prefix), f"{prefix} survived the roof opening"
    assert opened.ctx.model.nlight == closed.ctx.model.nlight
    assert opened.ctx.model.ngeom < closed.ctx.model.ngeom
