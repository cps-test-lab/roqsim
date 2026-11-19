"""The profile cart: its top, its slots and its footprint are what the config says.

The slots are collision geometry -- a cube slid across the top meets each opening -- so the tests read
the compiled geoms, not the config: the top's outline and height, the openings at the pitch, the
posts at the footprint, and the trim that does not collide.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_assets.plugins.profile_cart import ProfileCartPlugin


def _built(tmp_path, extra=None):
    plugins = [{"profile_cart": dict(extra or {}), "name": "cart"}]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    return engine


def _geoms(engine, prefix):
    m = engine.ctx.model
    return [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(prefix)]


def _aabb(engine, gids):
    """World-frame bounds of a set of box geoms (all axis-aligned in the cart)."""
    m, d = engine.ctx.model, engine.ctx.data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gids:
        half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
        lo = np.minimum(lo, d.geom_xpos[g] - half)
        hi = np.maximum(hi, d.geom_xpos[g] + half)
    return lo, hi


def _openings(engine, axis, z):
    """The gaps between the lands at height *z*, along *axis*: [(start, end), ...] in order."""
    m, d = engine.ctx.model, engine.ctx.data
    spans = []
    for g in _geoms(engine, "top_land_"):
        half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
        if d.geom_xpos[g][2] - half[2] <= z <= d.geom_xpos[g][2] + half[2]:
            spans.append((d.geom_xpos[g][axis] - half[axis], d.geom_xpos[g][axis] + half[axis]))
    spans.sort()
    return [(a[1], b[0]) for a, b in zip(spans, spans[1:], strict=False)]


CELL = {
    "top": {"length": 1.12, "depth": 0.70, "height": 0.85, "thickness": 0.024},
    "slots": {"pitch": 0.04, "width": 0.008, "depth": 0.010, "head_width": 0.016},
    "footprint": [1.10, 0.68],
}


@pytest.fixture
def cart(tmp_path):
    engine = _built(tmp_path, CELL)
    yield engine
    engine.shutdown()


def test_the_top_has_the_configured_size_and_height(cart):
    lo, hi = _aabb(cart, _geoms(cart, "top_"))
    assert hi - lo == pytest.approx([1.12, 0.70, 0.024], abs=1e-9)
    assert hi[2] == pytest.approx(0.85, abs=1e-9)
    assert lo[:2] == pytest.approx([-0.56, -0.35], abs=1e-9)


def test_the_slots_open_at_the_pitch_and_width_across_the_length(cart):
    gaps = _openings(cart, axis=0, z=0.85 - 0.001)
    assert len(gaps) == 28  # round(1.12 / 0.04)
    centres = np.array([(a + b) / 2 for a, b in gaps])
    assert np.diff(centres) == pytest.approx(np.full(27, 0.04), abs=1e-9)
    assert centres.mean() == pytest.approx(0.0, abs=1e-9)
    assert [b - a for a, b in gaps] == pytest.approx([0.008] * 28, abs=1e-9)
    # under the lips the T's head is wider, and the slot's floor is its depth down
    heads = _aabb(cart, _geoms(cart, "top_core_"))
    assert heads[1][2] == pytest.approx(0.85 - 0.004, abs=1e-9)
    slab = _aabb(cart, _geoms(cart, "top_slab"))
    assert slab[1][2] == pytest.approx(0.85 - 0.010, abs=1e-9)


def test_the_slots_collide(cart):
    m = cart.ctx.model
    for g in _geoms(cart, "top_"):
        assert m.geom_contype[g] and m.geom_conaffinity[g], m.geom(g).name


def test_the_slots_turn_with_direction(tmp_path):
    engine = _built(tmp_path, {**CELL, "slots": {**CELL["slots"], "direction": "x"}})
    gaps = _openings(engine, axis=1, z=0.85 - 0.001)
    assert len(gaps) == round(0.70 / 0.04)
    engine.shutdown()


def test_the_footprint_is_the_posts_outline_and_the_cart_stands_on_the_floor(cart):
    lo, hi = _aabb(cart, _geoms(cart, "post_"))
    assert (hi - lo)[:2] == pytest.approx([1.10, 0.68], abs=1e-9)
    assert (hi + lo)[:2] == pytest.approx([0.0, 0.0], abs=1e-9)
    m, d = cart.ctx.model, cart.ctx.data
    wheels = _geoms(cart, "castor_wheel_")
    assert len(wheels) == 4
    bottoms = [d.geom_xpos[g][2] - m.geom_size[g][0] for g in wheels]
    assert bottoms == pytest.approx([0.0] * 4, abs=1e-9)


def test_the_footprint_defaults_to_the_top(tmp_path):
    engine = _built(tmp_path, {"top": CELL["top"]})
    lo, hi = _aabb(engine, _geoms(engine, "post_"))
    assert (hi - lo)[:2] == pytest.approx([1.12, 0.70], abs=1e-9)
    engine.shutdown()


def test_the_trim_is_visual(cart):
    m = cart.ctx.model
    for prefix in ("edge_cover_", "front_panel", "castor_fork_", "castor_brake_"):
        gids = _geoms(cart, prefix)
        assert gids, prefix
        for g in gids:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, m.geom(g).name


def test_the_pose_places_the_footprint_centre(tmp_path):
    engine = _built(
        tmp_path,
        {**CELL, "pose": {"position": {"x": 1.0, "y": -2.0}, "orientation": {"yaw": 1.5707963}}},
    )
    lo, hi = _aabb(engine, _geoms(engine, "top_"))
    assert (hi + lo)[:2] / 2 == pytest.approx([1.0, -2.0], abs=1e-6)
    assert (hi - lo)[:2] == pytest.approx([0.70, 1.12], abs=1e-6)  # turned a quarter
    engine.shutdown()


def test_the_entity_records_the_top_and_the_slots(cart):
    meta = cart.ctx.entities.get("cart").meta
    assert meta["top"] == {"length": 1.12, "depth": 0.70, "height": 0.85}
    assert meta["slots"]["pitch"] == 0.04 and len(meta["slots"]["centres"]) == 28
    assert meta["footprint"] == [1.10, 0.68]


@pytest.mark.parametrize(
    "bad, says",
    [
        ({"slots": {"width": 0.02, "head_width": 0.016}}, "width < head_width < pitch"),
        ({"slots": {"depth": 0.03}}, "neck < depth"),
        ({"footprint": [1.3, 0.7]}, "stick out"),
        ({"shelf": {"height": 0.9}}, "shelf.height"),
        ({"drawer": {"height": 0.6}}, "drawer does not fit"),
        ({"shelf": {"enabled": False}}, "enable the shelf"),
        ({"slots": {"count": 40}}, "do not fit"),
        ({"shelf": {"height": 0.79}}, "front panel's 0.02 m frame does not fit"),
        (
            {"front_panel": {"carried_by": "drawer"}, "drawer": {"enabled": False}},
            "needs the drawer",
        ),
    ],
)
def test_impossible_carts_are_refused(bad, says):
    config = {**CELL, **bad}
    for key in ("top", "slots"):
        if key in bad:
            config[key] = {**CELL[key], **bad[key]}
    errors = ProfileCartPlugin(config).validate_config(config)
    assert any(says in e for e in errors), errors


def test_a_panel_on_the_drawer_comes_out_with_it_and_one_on_the_frame_stays(tmp_path):
    def panel_y(on):
        engine = _built(
            tmp_path, {**CELL, "drawer": {"open": 0.25}, "front_panel": {"carried_by": on}}
        )
        lo, hi = _aabb(engine, _geoms(engine, "front_panel"))
        engine.shutdown()
        return (lo[1] + hi[1]) / 2

    assert panel_y("frame") - panel_y("drawer") == pytest.approx(0.25, abs=1e-9)


def test_the_panel_keys_read_from_yaml_as_written(tmp_path):
    # YAML 1.1 reads a bare `on`/`off`/`yes` key as a boolean; the panel's keys are not such words.
    import yaml

    world = yaml.safe_load(
        "sim: {}\ncomponents:\n"
        "  - profile_cart: {drawer: {open: 0.1}, front_panel: {carried_by: drawer}}\n"
        "    name: cart\n"
    )
    engine = Engine(load_config_from_dict(world, base_dir=tmp_path))
    engine.setup()
    engine.shutdown()
