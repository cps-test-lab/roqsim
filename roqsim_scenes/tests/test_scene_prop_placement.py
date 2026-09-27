"""``--prop PATH,X,Y,YAW`` puts the prop's footprint centre at (X, Y) whatever the yaw (radians).

An OBJ whose footprint is not centred on its own origin turns about that origin, so the placement has
to subtract the footprint centre rotated by the yaw, not the unrotated one.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim_scenes.cli.scene_to_mjcf import _add_prop

# A 0.4 x 0.2 x 0.3 box whose footprint centre sits at (1, 0) in its own file, base at z = 0.5.
_CORNERS = [(x, y, z) for x in (0.8, 1.2) for y in (-0.1, 0.1) for z in (0.5, 0.8)]
_FACES = "f 1 2 4 3\nf 5 7 8 6\nf 1 5 6 2\nf 3 4 8 7\nf 1 3 7 5\nf 2 6 8 4\n"


def _crate(tmp_path):
    obj = tmp_path / "crate.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in _CORNERS) + _FACES)
    return obj


def _world_vertices(tmp_path, prop):
    spec = mujoco.MjSpec()
    name = _add_prop(spec, prop.format(obj=_crate(tmp_path)))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    m = model.geom_dataid[g]
    verts = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    return verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, 0.65])
def test_the_footprint_centre_lands_where_asked(tmp_path, yaw):
    world = _world_vertices(tmp_path, f"{{obj}},10,20,{yaw}")
    centre = (world.min(axis=0) + world.max(axis=0)) / 2
    np.testing.assert_allclose(centre[:2], [10.0, 20.0], atol=1e-6)
    assert world[:, 2].min() == pytest.approx(0.0, abs=1e-6)


def test_yaw_is_radians(tmp_path):
    world = _world_vertices(tmp_path, f"{{obj}},0,0,{math.pi / 2}")
    # A quarter turn: the crate's 0.4 m side now lies along world y.
    np.testing.assert_allclose(world.max(axis=0)[:2] - world.min(axis=0)[:2], [0.2, 0.4], atol=1e-6)


@pytest.mark.parametrize(
    ("args", "field"), [("1,2,ninety", "YAW"), ("x,2", "X"), ("1,y,0", "Y"), ("1", "PATH,X,Y")]
)
def test_a_malformed_prop_is_refused_by_name(tmp_path, args, field):
    with pytest.raises(SystemExit, match=field):
        _add_prop(mujoco.MjSpec(), f"{_crate(tmp_path)},{args}")
