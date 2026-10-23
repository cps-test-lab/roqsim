"""A see-through geom does not throw a shadow, and an opaque one still does.

MuJoCo's shadow map is binary: drawn into it, a film at 10 % opacity blocks a light as completely as
a steel plate, so whatever is behind it renders dark. ``FrameRenderer`` keeps transparent geoms out of
the shadow pass (``rendering.clear_geoms_cast_no_shadow``). Measured here on the floor under a slab
held between it and a shadow-casting light.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.rendering import FrameRenderer, has_clear_geoms

_SCENE = """<mujoco>
  <visual><quality shadowsize="2048"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true" diffuse="0.9 0.9 0.9" ambient="0.1 0.1 0.1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="box" size="0.4 0.4 0.005" pos="0 0 0.5" rgba="0.9 0.9 1.0 {alpha}"/>
    <camera name="down" pos="0.9 0 1.2" xyaxes="0 1 0 -0.8 0 0.6"/>
  </worldbody>
</mujoco>"""


def _floor_under_slab(alpha: float) -> float:
    model = mujoco.MjModel.from_xml_string(_SCENE.format(alpha=alpha))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = FrameRenderer(model, 160, 120, camera="down").render(data)
    # The floor in the slab's shadow but seen past the slab's edge, below the image's middle.
    return float(np.mean(frame[100:118, 60:100]))


def test_transparent_geoms_are_found():
    assert has_clear_geoms(mujoco.MjModel.from_xml_string(_SCENE.format(alpha=0.2)))
    assert not has_clear_geoms(mujoco.MjModel.from_xml_string(_SCENE.format(alpha=1.0)))


def test_a_clear_slab_leaves_the_floor_lit_where_an_opaque_one_shades_it():
    shaded = _floor_under_slab(1.0)
    clear = _floor_under_slab(0.15)
    nothing = _floor_under_slab(0.0)
    assert shaded < nothing - 20, "the opaque slab should shade the floor"
    assert clear == pytest.approx(nothing, abs=5), "a clear slab must not throw a full shadow"
