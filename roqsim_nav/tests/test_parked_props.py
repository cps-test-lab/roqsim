"""A free prop nobody is driving is a wall while it stands still.

The grid holds static geometry because a raster cannot represent something that moves. A crate with
a free joint that has been sitting in a doorway since the world was built is not "something that
moves" in any useful sense, and planning through it only to stop in front of it wastes the route.

The distinction that keeps this safe is what is NOT in the set: the robot under test also stands
still at spawn, and it must never be baked into the grid -- it is stationary because nothing is
driving it *yet*. See ``NavigatorPlugin._parked_props``.
"""

from __future__ import annotations

import mujoco

from roqsim_nav.grid import build_grid, grid_key

_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 .05"/>
    <geom name="wall_n" type="box" pos="0  3 1" size="3 .1 1"/>
    <geom name="wall_s" type="box" pos="0 -3 1" size="3 .1 1"/>
    <body name="crate" pos="0 0 .3">
      <freejoint/>
      <geom name="crate_g" type="box" size=".4 .4 .3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model_and_data():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _crate_root(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    return int(model.body_weldid[bid])


def test_a_free_prop_is_not_a_wall_by_default():
    model, data = _model_and_data()
    grid = build_grid(model, data, z_lo=0.05, z_hi=0.6)
    assert grid is not None
    assert not grid.occupied[grid.world_to_cell(0.0, 0.0)], "a free body is traffic unless named"


def test_a_parked_free_prop_is_a_wall_when_named():
    model, data = _model_and_data()
    grid = build_grid(model, data, z_lo=0.05, z_hi=0.6, resting_roots=(_crate_root(model),))
    assert grid.occupied[grid.world_to_cell(0.0, 0.0)], "a parked crate should be planned around"


def test_a_moving_prop_is_never_a_wall_even_when_named():
    """The test is per call, so the same prop is a wall in one grid and not in the next."""
    model, data = _model_and_data()
    data.qvel[:3] = [1.0, 0.0, 0.0]  # sliding across the floor
    mujoco.mj_forward(model, data)
    grid = build_grid(model, data, z_lo=0.05, z_hi=0.6, resting_roots=(_crate_root(model),))
    assert not grid.occupied[grid.world_to_cell(0.0, 0.0)]


def test_the_set_is_part_of_the_grids_identity():
    """Two movers may only share a raster when they agree about what is in it."""
    assert grid_key(0.05, 0.1, 1.8, ()) != grid_key(0.05, 0.1, 1.8, (3,))
    assert grid_key(0.05, 0.1, 1.8, (3, 5)) == grid_key(0.05, 0.1, 1.8, (5, 3))
