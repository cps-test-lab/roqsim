"""A round geom lying on its side is a wall along its whole length.

A capsule or cylinder placed with ``fromto`` along the floor (a rail, a pipe, a beam) spans its
length in x/y, so its footprint is taken about its own axis in the world. Drawn in its local xy plane
it would be a collinear sliver at its centre, which the hull test discards. On a mocap body the same
geom is a moving obstacle's disc, measured about the body's origin, so it covers the whole length too.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim_nav.obstacles import dynamic_obstacle_bodies, wall_polygons

_SCENE = """
<mujoco>
  <worldbody>
    <geom type="plane" size="10 10 .1"/>
    <geom name="wall" type="{kind}" fromto="-2 3 .5  2 3 .5" size=".05"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("kind", ["capsule", "cylinder"])
def test_a_lying_round_geom_is_a_wall_along_its_length(kind):
    model = mujoco.MjModel.from_xml_string(_SCENE.format(kind=kind))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    polys = wall_polygons(model, data)
    assert len(polys) == 1, polys
    xs = np.array(polys[0])[:, 0]
    ys = np.array(polys[0])[:, 1]
    assert xs.min() <= -1.95 and xs.max() >= 1.95
    assert 2.9 <= ys.min() and ys.max() <= 3.1


def test_an_upright_cylinder_keeps_its_disc():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="cylinder" pos="1 1 .5" size=".3 .5"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    (poly,) = wall_polygons(model, data)
    radii = np.hypot(*(np.array(poly) - [1.0, 1.0]).T)
    assert np.allclose(radii, 0.3, atol=1e-6)


_MOCAP = """
<mujoco>
  <worldbody>
    <body name="prop" mocap="true" pos="1 1 0">
      <geom type="{kind}" fromto="-1 0 .5  1 0 .5" size=".1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("kind", ["capsule", "cylinder"])
def test_a_lying_round_mocap_geom_is_a_disc_as_long_as_it(kind):
    model = mujoco.MjModel.from_xml_string(_MOCAP.format(kind=kind))
    ((_, radius),) = dynamic_obstacle_bodies(model, exclude_mocapids=set())
    assert radius >= 1.1 - 1e-9


def test_an_offset_mocap_geom_is_measured_from_its_body():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body mocap="true"><geom type="sphere" pos=".5 0 .3" size=".2"/>'
        "</body></worldbody></mujoco>"
    )
    ((_, radius),) = dynamic_obstacle_bodies(model, exclude_mocapids=set())
    assert radius == pytest.approx(0.7)


def test_an_upright_mocap_cylinder_keeps_its_radius():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body mocap="true"><geom type="cylinder" size=".3 .5"/>'
        "</body></worldbody></mujoco>"
    )
    ((_, radius),) = dynamic_obstacle_bodies(model, exclude_mocapids=set())
    assert radius == pytest.approx(0.3)
