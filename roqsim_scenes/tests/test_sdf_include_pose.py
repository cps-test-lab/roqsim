"""An ``<include>``'s ``<pose>`` replaces the included model's own top-level pose.

That is SDF's rule (libsdformat copies the include pose over the model's): the pose inside a
``model.sdf`` is its author's default placement, and composing the two moved every such model by
that default -- silently, so the imported scene disagreed with Gazebo's.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from lxml import etree

from roqsim_scenes.cli.sdf_to_scene import Importer

_MODEL = """<?xml version="1.0"?>
<sdf version="1.8"><model name="crate"><pose>1 0 0 0 0 0</pose>
  <link name="body"/>
</model></sdf>
"""


def _placed_link(tmp_path, include_xml):
    (tmp_path / "crate").mkdir()
    (tmp_path / "crate" / "model.sdf").write_text(_MODEL)
    imp = Importer(
        SimpleNamespace(cache=str(tmp_path / "cache"), model_path=[], collision_only=False)
    )
    imp._model_dir = lambda uri: tmp_path / "crate"
    seen = []
    imp._link = lambda link, world, name: seen.append(world)
    imp._include(etree.fromstring(include_xml), np.eye(4))
    (world,) = seen
    return world[:3, 3]


def test_the_include_pose_replaces_the_models_own(tmp_path):
    at = _placed_link(
        tmp_path, "<include><uri>model://crate</uri><pose>2 0 0.5 0 0 0</pose></include>"
    )
    np.testing.assert_allclose(at, [2.0, 0.0, 0.5])


def test_without_an_include_pose_the_models_own_applies(tmp_path):
    at = _placed_link(tmp_path, "<include><uri>model://crate</uri></include>")
    np.testing.assert_allclose(at, [1.0, 0.0, 0.0])
