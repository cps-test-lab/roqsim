"""A capsule is mapped out to the tips of its end caps, not only along its cylinder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roqsim_scenes.cli import scene_to_map

# A rail lying along x: a capsule from -1 to 1 with radius 0.1, so it reaches x = +-1.1.
MJCF = """
<mujoco>
  <worldbody>
    <geom type="capsule" fromto="-1 0 0.1 1 0 0.1" size="0.1"/>
  </worldbody>
</mujoco>
"""


def test_a_lying_capsule_reaches_its_tips(tmp_path: Path):
    (tmp_path / "rail.xml").write_text(MJCF)
    world = tmp_path / "world.yaml"
    world.write_text(f"sim:\n  world: {tmp_path / 'rail.xml'}\n")
    _, lo, hi, _ = scene_to_map._load_world(str(world))
    np.testing.assert_allclose([lo[0], hi[0]], [-1.1, 1.1], atol=1e-9)
    assert hi[1] == pytest.approx(0.1)
