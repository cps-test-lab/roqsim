"""``--prop PATH,X,Y,YAW`` puts the prop's footprint centre at (X, Y) whatever the yaw.

An OBJ whose footprint is not centred on its own origin turns about that origin, so subtracting the
unrotated centre only works at yaw 0; at 90 deg the prop landed off (X, Y) by its turned offset.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim_scenes.cli.scene_to_mjcf import _add_prop

# A 0.4 x 0.2 x 0.3 box whose footprint centre sits at (1, 0) in its own file, base at z = 0.5.
_CORNERS = [(x, y, z) for x in (0.8, 1.2) for y in (-0.1, 0.1) for z in (0.5, 0.8)]
_FACES = "f 1 2 4 3\nf 5 7 8 6\nf 1 5 6 2\nf 3 4 8 7\nf 1 3 7 5\nf 2 6 8 4\n"


@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 37.0])
def test_the_footprint_centre_lands_where_asked(tmp_path, yaw_deg):
    obj = tmp_path / "crate.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in _CORNERS) + _FACES)
    spec = mujoco.MjSpec()
    name = _add_prop(spec, f"{obj},10,20,{yaw_deg}")
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    m = model.geom_dataid[g]
    verts = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    centre = (world.min(axis=0) + world.max(axis=0)) / 2
    np.testing.assert_allclose(centre[:2], [10.0, 20.0], atol=1e-6)
    assert world[:, 2].min() == pytest.approx(0.0, abs=1e-6)
