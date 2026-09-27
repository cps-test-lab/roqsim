"""A capsule geom's surface sits where MuJoCo has it: centred on the geom, along its z axis.

``roqsim assets collision`` measures a prop's colliders against its mesh, so a capsule surface
built half a cylinder-length low reports overreach and coverage that are not there.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from roqsim_assets.cli.collision import geom_surface  # noqa: E402


def test_a_capsule_surface_is_centred_on_its_geom():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="capsule" pos="0 0 1" size=".1 .5"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mesh = geom_surface(trimesh, model, data, 0)
    np.testing.assert_allclose(mesh.bounds[:, 2], [0.4, 1.6], atol=1e-6)
    np.testing.assert_allclose(mesh.bounds[:, :2], [[-0.1, -0.1], [0.1, 0.1]], atol=1e-3)
