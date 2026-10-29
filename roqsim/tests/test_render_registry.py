# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer is an extension: it is resolved like a plugin, and a world names it in its ``sim`` block.

What a user gets wrong is said in their terms: an unknown renderer lists the installed ones and the
package that would add it, a mistyped settings key names the key, the block and the valid keys.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from roqsim import renderers
from roqsim.config import load_config_from_dict, overrides_from_dotlist
from roqsim.plugin import PluginError
from roqsim.registry import resolve_renderer
from roqsim.render_api import Frame, Renderer, RendererCapabilities, Setting


class Flat(Renderer):
    """Fills every view with one colour; what a renderer needs to be, and nothing more."""

    capabilities = RendererCapabilities(images=frozenset({"rgb"}))
    SETTINGS = {"colour": Setting([0, 0, 0], "the colour every pixel gets")}

    def render(self, snapshot, requests):
        return [Frame(r.id, snapshot.sim_time, snapshot.step) for r in requests]


class NotARenderer:
    pass


REF = f"{__name__}:Flat"


def test_module_class_resolves():
    assert resolve_renderer(REF) is Flat


def test_file_class_resolves_beside_the_world(tmp_path):
    (tmp_path / "mine.py").write_text(
        textwrap.dedent(
            """
            from roqsim.render_api import Renderer

            class Mine(Renderer):
                pass
            """
        )
    )
    cls = resolve_renderer("mine.py:Mine", base_dir=tmp_path)
    assert cls.__name__ == "Mine" and issubclass(cls, Renderer)


def test_a_class_that_is_no_renderer_is_refused():
    with pytest.raises(PluginError, match="Renderer subclass"):
        resolve_renderer(f"{__name__}:NotARenderer")


def test_an_unknown_name_lists_the_installed_and_names_the_package():
    with pytest.raises(PluginError) as err:
        resolve_renderer("nope_xyz")
    text = str(err.value)
    assert "unknown renderer 'nope_xyz'" in text
    assert "Installed:" in text
    assert "roqsim_render_nope_xyz" in text


def _world(sim: dict) -> dict:
    return {"sim": sim, "components": []}


def test_a_world_naming_an_unknown_renderer_fails_at_load():
    with pytest.raises(PluginError, match="unknown renderer 'nope_xyz'"):
        load_config_from_dict(_world({"renderer": "nope_xyz"}))


def test_a_mistyped_settings_key_names_key_block_and_valid_keys():
    with pytest.raises(PluginError) as err:
        load_config_from_dict(_world({"renderer": REF, "renderers": {REF: {"color": [1, 0, 0]}}}))
    text = str(err.value)
    assert f"sim.renderers.{REF}" in text
    assert "'color'" in text
    assert "colour" in text


def test_the_render_mode_is_one_of_two():
    with pytest.raises(PluginError, match="render.mode"):
        load_config_from_dict(_world({"render": {"mode": "later"}}))
    with pytest.raises(PluginError, match="sim.render"):
        load_config_from_dict(_world({"render": {"modes": "sync"}}))
    cfg = load_config_from_dict(_world({"render": {"mode": "async"}}))
    assert cfg.render["mode"] == "async"


def test_defaults_when_a_world_says_nothing():
    cfg = load_config_from_dict(_world({}))
    assert cfg.renderer == "mujoco"
    assert cfg.render == {"mode": "sync", "workers": 1, "timeout_s": 60.0}
    assert cfg.renderers == {}


def test_the_choice_rides_the_set_override_path():
    cfg = load_config_from_dict(
        _world({}), overrides=overrides_from_dotlist([f"sim.renderer={REF}"])
    )
    assert cfg.renderer == REF


def test_describe_lists_capabilities_and_settings(capsys):
    out = renderers.describe_renderer(REF)
    assert out["capabilities"]["images"] == ["rgb"]
    assert out["settings"]["colour"] == {
        "default": [0, 0, 0],
        "meaning": "the colour every pixel gets",
    }
    assert out["settings_block"] == f"sim.renderers.{REF}"
    with pytest.raises(SystemExit) as code:
        renderers.main(["describe", "nope_xyz"])
    assert code.value.code == 2
    assert "unknown renderer" in json.loads(capsys.readouterr().out)["error"]
