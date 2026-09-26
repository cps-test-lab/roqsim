"""A round geom lying on its side is a wall along its whole length.

A capsule or cylinder placed with ``fromto`` along the floor (a rail, a pipe, a beam) spans its
length in x/y. Drawn in its local xy plane its footprint was a sliver at its centre, collinear, and
the hull test discarded it -- the planner then routed straight through a 4 m obstacle.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim_nav.obstacles import wall_polygons

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
