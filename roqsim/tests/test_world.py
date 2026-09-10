"""World definitions: the default empty room, and the floorplan/provides_world override."""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.world import DEFAULT_WORLD, available_worlds, build_world, world_file


def _geom_names(model) -> list[str]:
    return [model.geom(i).name for i in range(model.ngeom)]


def test_empty_room_is_registered_and_default():
    assert DEFAULT_WORLD == "empty_room"
    assert "empty_room" in available_worlds()


def test_no_scene_plugin_gets_default_empty_room():
    # A plugin-less world still stands on a lit floor, now enclosed by perimeter walls.
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": []}))
    engine.setup()
    names = _geom_names(engine.ctx.model)
    assert names.count("floor") == 1
    assert engine.ctx.model.nlight == 1
    assert {"wall_px", "wall_nx", "wall_py", "wall_ny"} <= set(names)  # empty_room is walled
    engine.shutdown()


def test_unknown_world_raises():
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(KeyError, match="unknown sim.world 'nope'"):
        build_world(spec, "nope")


def test_world_file_distinguishes_builtins_from_paths(tmp_path):
    # Built-in names and None are not file refs.
    assert world_file(None, tmp_path) is None
    assert world_file("empty_room", tmp_path) is None
    # A path/.xml value resolves relative to base_dir; missing file fails fast.
    xml = tmp_path / "scene.xml"
    xml.write_text("<mujoco><worldbody/></mujoco>")
    assert world_file("scene.xml", tmp_path) == str(xml)
    assert world_file(str(xml), tmp_path) == str(xml)  # absolute
    with pytest.raises(FileNotFoundError, match="sim.world file not found"):
        world_file("missing/scene.xml", tmp_path)


def test_sim_world_loads_an_mjcf_file(tmp_path):
    # sim.world pointing at an MJCF file loads it as the base scene (a baked world).
    (tmp_path / "scene.xml").write_text(
        "<mujoco><worldbody>"
        "<geom name='floor' type='plane' size='1 1 .05'/>"
        "<geom name='baked_box' type='box' size='.2 .2 .2' pos='0 0 .2'/>"
        "</worldbody></mujoco>"
    )
    cfg = load_config_from_dict({"sim": {"world": "scene.xml"}, "plugins": []}, base_dir=tmp_path)
    engine = Engine(cfg, plugins=[])
    engine.setup()
    names = _geom_names(engine.ctx.model)
    assert (
        "baked_box" in names and names.count("floor") == 1
    )  # the file is the world, no empty_room
    engine.shutdown()


class _ScenePlugin(Plugin):
    """A minimal stand-in for floorplan: provides its own ground + light."""

    provides_world = True

    def build(self, spec, ctx):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05])
        spec.worldbody.add_light()


def test_provides_world_plugin_fills_the_slot_instead_of_the_definition():
    """The plugin builds the ground, so no world definition is built underneath it."""
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    engine = Engine(cfg, plugins=[_ScenePlugin()])
    engine.setup()

    assert _geom_names(engine.ctx.model).count("floor") == 1
    engine.shutdown()


def test_a_world_stated_twice_is_refused():
    """``sim.world`` beside a plugin that builds its own ground: the author said it twice.

    Stacking them compiles -- a second plane coplanar with the first, a second light that only
    brightens it -- so the scene looks like a room and is not the one the document describes.
    Whichever of the two is dropped is the author's call, so neither can be picked here.
    """
    cfg = load_config_from_dict({"sim": {"world": "empty_room"}, "plugins": []})
    engine = Engine(cfg, plugins=[_ScenePlugin()])

    with pytest.raises(PluginError) as caught:
        engine.setup()

    message = str(caught.value)
    assert "empty_room" in message and "_ScenePlugin" in message


def test_two_scene_plugins_providing_the_world_are_refused():
    """The collision with no ``sim.world`` in it, and the one nothing warned about before.

    Two ground-providing plugins in one document -- a floorplan and a terrain, say -- is the same
    mistake without the config key that makes it visible.
    """
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    engine = Engine(cfg, plugins=[_ScenePlugin(name="indoors"), _ScenePlugin(name="outdoors")])

    with pytest.raises(PluginError) as caught:
        engine.setup()

    message = str(caught.value)
    assert "indoors" in message and "outdoors" in message


def test_a_baked_scene_beside_such_a_plugin_is_refused_too():
    """The path spelling of ``sim.world``, which was the worse half of the old behaviour.

    A file world is *loaded* as the base spec before any plugin builds, so a warning saying it was
    ignored described something that had already happened.
    """
    cfg = load_config_from_dict({"sim": {"world": "scene.xml"}, "plugins": []})
    engine = Engine(cfg, plugins=[_ScenePlugin()])

    with pytest.raises(PluginError, match="scene.xml"):
        engine.setup()
