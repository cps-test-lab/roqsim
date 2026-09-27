"""An SDF ``<pose>``'s attributes change what its numbers mean, so each is read or refused.

``degrees="true"`` and ``rotation_format="quat_xyzw"`` are read. ``relative_to`` a frame other than
the parent, and ``placement_frame``, are refused: reading the numbers alone would place the geometry
somewhere else without a word.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from lxml import etree

from roqsim_scenes.cli import fuel_fetch
from roqsim_scenes.cli.sdf_to_scene import Importer, _pose_of

_QUARTER = math.sqrt(0.5)


def _pose(xml):
    return _pose_of(etree.fromstring(f"<link name='l'>{xml}</link>"))


def _yawed(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "xml",
    [
        "<pose>1 2 3 0 0 1.5707963267948966</pose>",
        "<pose degrees='true'>1 2 3 0 0 90</pose>",
        f"<pose rotation_format='quat_xyzw'>1 2 3 0 0 {_QUARTER} {_QUARTER}</pose>",
        "<pose relative_to='__model__'>1 2 3 0 0 1.5707963267948966</pose>",
    ],
)
def test_every_spelling_of_a_quarter_turn_reads_the_same(xml):
    m = _pose(xml)
    np.testing.assert_allclose(m[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(m[:3, :3], _yawed(math.pi / 2), atol=1e-12)


def test_a_pose_relative_to_another_frame_is_refused():
    with pytest.raises(fuel_fetch.FuelError, match="relative_to='base'"):
        _pose("<pose relative_to='base'>1 0 0 0 0 0</pose>")


def test_a_quat_pose_with_six_values_is_refused():
    with pytest.raises(fuel_fetch.FuelError, match="7 values"):
        _pose("<pose rotation_format='quat_xyzw'>1 0 0 0 0 0</pose>")


def test_a_placement_frame_is_refused(tmp_path):
    imp = Importer(
        SimpleNamespace(cache=str(tmp_path / "cache"), model_path=[], collision_only=False)
    )
    inc = etree.fromstring(
        "<include><uri>model://crate</uri><placement_frame>base</placement_frame></include>"
    )
    with pytest.raises(fuel_fetch.FuelError, match="placement_frame"):
        imp._include(inc, np.eye(4))
