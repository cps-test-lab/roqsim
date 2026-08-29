"""A world resolves a model file beside itself, from whatever directory it is loaded from.

``spawn_model`` resolved its ``model:`` with no ``base_dir``, so a filesystem path fell back to the
CWD -- and a world referencing a sibling asset then loaded only from its own directory. That is not
a niche case: the working directory is not the document's directory in general, and need not be the
same twice. Measured, on one world and one reference:

    from the world's own directory    ``roqsim sim world/cell.yaml``            -> resolved
    by absolute path from elsewhere   ``roqsim sim /abs/proj/world/cell.yaml``  -> NOT FOUND
    copied under a different root     the document mirrored by a staging tool   -> NOT FOUND

The mechanism was already there and simply unused: ``Config.base_dir`` is the document's directory
and ``resolve_model`` has taken a ``base_dir`` all along -- ``spawn_arm`` passes one for its
end-effector. This makes the world's own directory the anchor, which is the only one that does not
depend on who is doing the loading.
"""

from __future__ import annotations

import textwrap

import pytest

from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim.plugin import Plugin

BIN_MJCF = """\
<mujoco model="sibling_prop">
  <worldbody>
    <body name="sibling_prop">
      <geom name="sibling_prop_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def world(tmp_path):
    """A world whose prop is a file beside it, named without any directory."""
    (tmp_path / "world").mkdir()
    (tmp_path / "world" / "sibling_prop.xml").write_text(BIN_MJCF, encoding="utf-8")
    (tmp_path / "world" / "cell.yaml").write_text(
        textwrap.dedent("""
        components:
          - spawn_model: {model: sibling_prop.xml, pos: [0.0, 0.0, 0.0]}
            name: prop
    """),
        encoding="utf-8",
    )
    return tmp_path / "world" / "cell.yaml"


def _compile(path):
    engine = Engine(load_config(str(path)))
    engine.setup()
    return engine.ctx.model


def test_a_sibling_model_resolves_from_an_unrelated_cwd(world, monkeypatch, tmp_path):
    """THE regression. The run's CWD is not the world's directory, and never was."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    model = _compile(world)
    import mujoco

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sibling_prop") >= 0


def test_validation_resolves_it_too_not_just_the_build(world, monkeypatch, tmp_path):
    """The failure was reported by `validate_config`, before `build` ever ran -- so fixing only
    the build path would have changed nothing a caller could see."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(str(world))
    from roqsim.config import instantiate_plugins

    plugins = instantiate_plugins(cfg)  # raises PluginError if validation rejects the model
    assert any(type(p).__name__ == "SpawnModelPlugin" for p in plugins)


def test_the_cwd_is_not_a_second_anchor(world, monkeypatch, tmp_path):
    """There is exactly one anchor, so a file under the working directory is never consulted.

    While the CWD was a fallback, a world naming a sibling that did not exist picked up an unrelated
    file of the same name from wherever the process happened to start -- and compiled it. Measured:
    the same document and the same string gave a 0.5 m box from one directory and "not found" from
    another, with nothing downstream able to tell which it got.

    Pinned as ABSENCE rather than precedence: precedence would still leave a world that resolves
    only because of where it was run from.
    """
    decoy = tmp_path / "decoy"
    (decoy / "world").mkdir(parents=True)
    (decoy / "world" / "ghost.xml").write_text(BIN_MJCF, encoding="utf-8")
    (decoy / "world" / "cell.yaml").write_text(textwrap.dedent("""
        components:
          - spawn_model: {model: world/ghost.xml, pos: [0.0, 0.0, 0.0]}
            name: prop
    """), encoding="utf-8")
    monkeypatch.chdir(decoy)

    # The world is `decoy/world/cell.yaml`, so its sibling is `decoy/world/world/ghost.xml` --
    # which does not exist. `decoy/world/ghost.xml` does, and used to be found from this CWD.
    from roqsim.plugin import PluginError

    with pytest.raises(PluginError, match="ghost.xml"):
        _compile(decoy / "world" / "cell.yaml")


def test_a_bundled_model_name_is_unaffected(tmp_path, monkeypatch):
    """A bare NAME is a provider lookup, not a path, and must not start depending on a directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cell.yaml").write_text(
        textwrap.dedent("""
        components:
          - spawn_model: {model: graspable_box, pos: [0.0, 0.0, 0.0]}
            name: parcel
    """),
        encoding="utf-8",
    )
    import mujoco

    model = _compile(tmp_path / "cell.yaml")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "graspable_box") >= 0


def test_a_plugin_built_outside_a_document_still_has_a_base_dir():
    """A test or a driver constructs a plugin directly; it must get the CWD, not an AttributeError."""
    from pathlib import Path

    assert Plugin({}, name="x").base_dir == Path.cwd()
