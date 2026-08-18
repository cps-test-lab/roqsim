"""The palm_tree prop: it compiles, it stands on the floor, and its bunch sites are where asked.

The last one is the point of the tests rather than a formality: a benchmark plans to a fruit bunch by
reading `bunch_world_pos` off the entity, so a site that does not sit where the config says would send
the arm to empty air with everything else looking correct.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

BUNCHES = [
    {"pos": [-0.10, -0.11, 0.89], "radius": 0.075, "fruits": 10},
    {"pos": [-0.08, 0.10, 0.97], "radius": 0.070, "fruits": 9},
]


def _world(tmp_path, **extra):
    cfg = {
        "palm_tree": {
            "name": "palm",
            "prefix": "palm_",
            "pos": [0.5, 0.2, 0.0],
            "bunches": BUNCHES,
            **extra,
        }
    }
    return load_config_from_dict({"sim": {}, "plugins": [cfg]}, base_dir=tmp_path)


def _built(tmp_path, **extra):
    engine = Engine(_world(tmp_path, **extra))
    engine.setup()
    return engine


def test_compiles_and_stands_on_the_floor(tmp_path):
    """min z == 0 at the declared pose, so a world places the tree without knowing the pot's height."""
    engine = _built(tmp_path, pot_height=0.16)
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    # The pot's underside is the tree's lowest point. Checked on that geom rather than by scanning the
    # whole model: the default world contributes a floor PLANE, whose bounding radius is enormous, so a
    # min-over-all-geoms test measures the floor and passes or fails for the wrong reason.
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm_pot")
    bottom = d.geom_xpos[gid][2] - m.geom_size[gid][1]  # cylinder: size = (radius, half-height)
    assert bottom == pytest.approx(0.0, abs=1e-9)
    # ... and the trunk starts inside the pot rather than at its rim, so a depth sensor sees no seam.
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm_trunk")
    assert d.geom_xpos[tid][2] - m.geom_size[tid][1] == pytest.approx(0.0, abs=1e-9)


def test_bunch_sites_and_entity_agree_with_the_config(tmp_path):
    engine = _built(tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)

    palm = engine.ctx.entities.get("palm")
    assert palm is not None, (
        "the plugin must register its entity, or a task cannot find the bunches"
    )
    assert palm.meta["bunch_sites"] == ["palm_bunch_0", "palm_bunch_1"]

    for i, bunch in enumerate(BUNCHES):
        want = [0.5 + bunch["pos"][0], 0.2 + bunch["pos"][1], bunch["pos"][2]]
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"palm_bunch_{i}")
        assert sid >= 0, f"missing site palm_bunch_{i}"
        # The site in the model and the position the entity advertises must be the same point --
        # a consumer may use either, and they must not be able to disagree.
        np.testing.assert_allclose(d.site_xpos[sid], want, atol=1e-9)
        np.testing.assert_allclose(palm.meta["bunch_world_pos"][i], want, atol=1e-9)


def test_fruit_and_fronds_collide(tmp_path):
    """The crown and the bunches are the obstacles; a non-colliding one is a silently easier scene."""
    engine = _built(tmp_path)
    m = engine.ctx.model
    for kind in ("frond_0_in", "frond_0_out", "bunch_0_fruit_0", "trunk", "pot"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"palm_{kind}")
        assert gid >= 0, f"missing geom palm_{kind}"
        assert m.geom_contype[gid] != 0 and m.geom_conaffinity[gid] != 0, (
            f"palm_{kind} does not collide"
        )


def test_frond_count_and_tiers(tmp_path):
    engine = _built(tmp_path, fronds=8, frond_tier=0.1, crown_height=1.0)
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    inner = [n for n in names if n.endswith("_in")]
    assert len(inner) == 8, "one inner segment per frond"
    # Two staggered tiers, which is what leaves gaps a planner can pass through -- a single flat disc
    # would be one planar barrier instead.
    heights = sorted({round(float(d.geom_xpos[names.index(n)][2]), 3) for n in inner})
    assert len(heights) == 2, f"expected two frond tiers, got heights {heights}"
    assert heights[1] - heights[0] == pytest.approx(0.1, abs=1e-6)


def test_rejects_a_crown_above_the_pole(tmp_path):
    """A floating crown reads as a modelling slip, so it is refused rather than built."""
    from roqsim_assets.plugins.palm_tree import PalmTreePlugin

    cfg = {"trunk_height": 1.0, "crown_height": 1.5, "bunches": BUNCHES}
    errors = PalmTreePlugin(cfg).validate_config(cfg)
    assert any("crown_height" in e for e in errors), errors


def test_rejects_a_malformed_bunch(tmp_path):
    from roqsim_assets.plugins.palm_tree import PalmTreePlugin

    cfg = {"bunches": [{"pos": [0.0, 0.0]}]}
    errors = PalmTreePlugin(cfg).validate_config(cfg)
    assert any("bunches[0]" in e for e in errors), errors


def test_yaw_rotates_the_crown(tmp_path):
    """`rpy` must actually turn the tree: an experiment aims the gap between fronds with it."""
    a = _built(tmp_path)
    b = _built(tmp_path, rpy=[0.0, 0.0, math.pi / 8])
    for engine in (a, b):
        mujoco.mj_forward(engine.ctx.model, engine.ctx.data)

    def frond_xy(engine):
        m, d = engine.ctx.model, engine.ctx.data
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm_frond_0_in")
        return d.geom_xpos[gid][:2] - np.array([0.5, 0.2])

    assert not np.allclose(frond_xy(a), frond_xy(b), atol=1e-6)
    # Rotation, not translation: the frond stays the same distance from the trunk axis.
    assert np.linalg.norm(frond_xy(a)) == pytest.approx(
        float(np.linalg.norm(frond_xy(b))), abs=1e-9
    )
