# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""``baked_lighting`` at load, against a synthetic bake (no Blender): a missing or stale bake draws the
world unbaked and says so, ``require`` stops it, a fitting one is composed into the model -- the light
times the model's own colour divided by the rig -- with the geoms still colliding and met by rays."""

from __future__ import annotations

import json
import math
import textwrap

import mujoco
import numpy as np
import pytest
from test_bake_export import room

from roqsim import conditions
from roqsim.bake.analysis import analyse_spec
from roqsim.bake.settings import PRESETS, rig_light
from roqsim.bake.store import EntityMaps, bake_meta, encode_entity, read_meta, write
from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim.presence import WINDOW_GEOM_GROUP


def grid_uv(n: int) -> np.ndarray:
    """A lightmap layout of *n* triangles, each in a cell of its own grid, a margin around it."""
    g = math.ceil(math.sqrt(n))
    cell = 1.0 / g
    base = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85]]) * cell
    i = np.arange(n)
    off = np.stack([(i % g) * cell, (i // g) * cell], 1)
    return (off[:, None, :] + base[None]).astype(np.float32)


def write_world(tmp_path, xml=None, **component):
    (tmp_path / "room.xml").write_text(xml or room())
    comp = json.dumps(component or {})
    (tmp_path / "room.yaml").write_text(
        textwrap.dedent(
            f"""
            sim: {{world: room.xml}}
            components:
              - baked_lighting: {comp}
            """
        ).lstrip()
    )
    return tmp_path / "room.yaml"


def synthetic_bake(world, light=1.0, exposure=1.0):
    """A bake of *world* whose light is *light* everywhere, laid out on a grid, written where the
    world's component reads it."""
    cfg = load_config(str(world))
    spec = mujoco.MjSpec.from_file(str(world.parent / "room.xml"))
    analysis, _ = analyse_spec(spec, PRESETS["default"])
    files, entries = {}, {}
    for name, ent in analysis.export.entities.items():
        s = ent.size
        maps = EntityMaps(
            light=np.full((s, s, 3), light, np.float32),
            indirect=np.zeros((s // 4, s // 4), np.float32),
            ao=np.ones((s // 4, s // 4), np.float32),
            uv=grid_uv(len(ent.pos)),
            geom=ent.geom,
            flip=np.zeros(len(ent.pos), bool),
        )
        f, entry = encode_entity(name, maps)
        entry["class"] = ent.cls
        files.update(f)
        entries[name] = entry
    meta = bake_meta(analysis, entries, exposure, {"baked_at": "test", "device": "none"})
    root = world.parent / "room.bake"
    write(root, meta, files)
    assert cfg is not None
    return root


def loaded(world, **kw):
    engine = Engine(load_config(str(world), kw or None), preview=True)
    engine.setup()
    return engine


def codes(engine):
    return [c["code"] for c in engine.ctx.conditions]


def test_a_missing_bake_draws_the_world_unbaked_and_says_how_to_bake_it(tmp_path):
    world = write_world(tmp_path)
    engine = loaded(world)
    try:
        (c,) = engine.ctx.conditions
        assert c["code"] == "bake_missing" and c["level"] == "warning"
        assert c["fix"] == f"roqsim bake {world.resolve()}"
        assert str(tmp_path / "room.bake") in c["message"]
        m = engine.ctx.model
        assert not any(m.geom(i).name.endswith("__baked") for i in range(m.ngeom))
        assert conditions.of_model(m) == [c]  # and the model carries it to every record
    finally:
        engine.shutdown()


def test_require_stops_a_world_whose_bake_is_missing(tmp_path):
    world = write_world(tmp_path, require=True)
    with pytest.raises(PluginError, match="roqsim bake"):
        loaded(world)


def test_a_fitting_bake_is_composed(tmp_path):
    world = write_world(tmp_path)
    synthetic_bake(world, light=1.0, exposure=1.0)
    engine = loaded(world)
    try:
        assert codes(engine) == ["bake_used"]
        m = engine.ctx.model
        wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall_e")
        visual = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall_e__baked")
        # The geom itself still collides and is met by rays; it is no longer drawn by default.
        assert m.geom_group[wall] == 3 and m.geom_contype[wall] == 1
        # Its bake is drawn in the window group: drawn, passed by rays, colliding with nothing.
        assert m.geom_group[visual] == WINDOW_GEOM_GROUP and m.geom_contype[visual] == 0
        assert m.geom_conaffinity[visual] == 0
        # The run-time rig replaces the world's lights: a key and three fills, no headlight.
        assert m.nlight == 4 and m.vis.headlight.active == 0
        assert m.light_castshadow[0] and not any(m.light_castshadow[1:])
        # Every baked geom and the light source throw no shadow of their own.
        assert not m.geom_castshadow[visual]
        assert not m.geom_castshadow[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "light_panel")]
        assert m.geom_castshadow[wall]
    finally:
        engine.shutdown()


def test_the_composed_texture_is_the_bake_divided_by_the_rig(tmp_path):
    """Light 1 at exposure 1 on the floor (grey 0.8, facing up, where the rig's light is 1): MuJoCo
    multiplies the texel by 1 and must draw the floor's own grey."""
    world = write_world(tmp_path)
    synthetic_bake(world, light=1.0, exposure=1.0)
    engine = loaded(world)
    try:
        m = engine.ctx.model
        tex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "baked_floor")
        w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
        a = int(m.tex_adr[tex])
        data = np.asarray(m.tex_data[a : a + w * h * 3]).reshape(h, w, 3)
        texels = data[data.sum(-1) > 0]
        assert rig_light(np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0)
        assert np.median(texels[:, 0]) == pytest.approx(round(0.8 * 255), abs=1)
        # A wall faces sideways, where the rig's light is about 0.75: its texels are divided by it.
        wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "baked_shell")
        w, h, a = int(m.tex_width[wall]), int(m.tex_height[wall]), int(m.tex_adr[wall])
        shell = np.asarray(m.tex_data[a : a + w * h * 3]).reshape(h, w, 3)
        assert np.max(shell) > round(0.6 * 255) + 5  # grey 0.6 brightened by the division
    finally:
        engine.shutdown()


def test_a_stale_bake_draws_unbaked_and_says_what_changed(tmp_path):
    world = write_world(tmp_path)
    synthetic_bake(world)
    (tmp_path / "room.xml").write_text(room(wall_x=2.3))
    engine = loaded(world)
    try:
        (c,) = engine.ctx.conditions
        assert c["code"] == "bake_stale" and "wall_e" in c["message"]
        assert c["details"]["changed"] == ["wall_e"]
        m = engine.ctx.model
        assert not any(m.geom(i).name.endswith("__baked") for i in range(m.ngeom))
    finally:
        engine.shutdown()


def test_require_stops_a_world_whose_bake_is_stale(tmp_path):
    world = write_world(tmp_path, require=True)
    synthetic_bake(world)
    (tmp_path / "room.xml").write_text(room(wall_x=2.3))
    with pytest.raises(PluginError, match="stale"):
        loaded(world)


def test_other_settings_make_it_stale(tmp_path):
    world = write_world(tmp_path, samples=128)
    synthetic_bake(world)  # baked at the preset's 64
    engine = loaded(world)
    try:
        (c,) = engine.ctx.conditions
        assert c["code"] == "bake_stale" and "samples" in c["message"]
    finally:
        engine.shutdown()


def test_a_removed_ceiling_draws_the_rest_baked(tmp_path):
    world = write_world(tmp_path)
    synthetic_bake(world)
    # The ceiling plugin removes it in `build`, before baked_lighting sees the world.
    world.write_text(world.read_text() + "  - ceiling: {keep: false, above_z: 2.495}\n")
    engine = loaded(world)
    try:
        assert codes(engine) == ["bake_partial", "bake_used"]
        partial = engine.ctx.conditions[0]
        assert partial["level"] == "info" and partial["details"]["missing"] == ["ceiling"]
    finally:
        engine.shutdown()


def test_a_robot_in_the_world_keeps_its_bake(tmp_path):
    world = write_world(tmp_path)
    synthetic_bake(world)
    robot = '<body name="bot" pos="0 1 0.2"><freejoint/><geom name="bot" type="box" size="0.2 0.2 0.1"/></body>'
    (tmp_path / "room.xml").write_text(room(extra=robot))
    engine = loaded(world)
    try:
        assert codes(engine) == ["bake_used"]
    finally:
        engine.shutdown()


def test_the_bake_is_kept_beside_the_mjcf_and_reads_back(tmp_path):
    world = write_world(tmp_path)
    root = synthetic_bake(world)
    assert root == tmp_path / "room.bake"
    meta = read_meta(root)
    assert meta["format"] == 1 and set(meta["entities"]) == {"floor", "shell", "P"}
    for entry in meta["entities"].values():
        for f in entry["files"]:
            assert (root / f).is_file()


def test_a_world_without_an_mjcf_must_name_the_path(tmp_path):
    (tmp_path / "w.yaml").write_text("sim: {world: empty_room}\ncomponents:\n  - baked_lighting: {}\n")
    with pytest.raises(PluginError, match="path:"):
        loaded(tmp_path / "w.yaml")


def test_moving_shadows_off_leaves_the_key_without_a_shadow(tmp_path):
    world = write_world(tmp_path, moving_shadows=False)
    synthetic_bake(world)
    engine = loaded(world)
    try:
        assert not any(engine.ctx.model.light_castshadow)
    finally:
        engine.shutdown()


def test_the_config_is_checked():
    from roqsim.plugins.baked_lighting import BakedLightingPlugin

    errors = BakedLightingPlugin({}).config_errors(
        {"quality": "ultra", "lights": "sun", "texels_per_m": {"walls": 3}, "emitters": [{"x": 1}]}
    )
    text = " ".join(errors)
    assert "quality" in text and "lights" in text and "texels_per_m.walls" in text and "emitters[0]" in text
