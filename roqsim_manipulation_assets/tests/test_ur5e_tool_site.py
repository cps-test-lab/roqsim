# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Where the UR5e's ``tool_site`` sits against the FT adapter stack it is named after.

A tool hung from ``tool_site`` is placed by that site alone, so the site must lie on the stack's
outer face for the tool to be seated on the sensor. The face is measured from the compiled meshes
rather than restated, so a re-authored adapter moves the expectation with it.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

import roqsim_manipulation_assets

#: Sub-millimetre: the stack face is a flat mesh plane, and a tool seated on it touches it.
FACE_TOL_M = 5e-4


def _model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml = Path(roqsim_manipulation_assets.__file__).parent / "models" / "ur5e" / "ur5e.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _stack_extent_along_tool_z(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """Min and max of every mesh vertex on ``tool0``, along the tool body's own z axis."""
    tool = model.body("tool0").id
    rot, origin = data.xmat[tool].reshape(3, 3), data.xpos[tool]
    lo, hi = np.inf, -np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != tool or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = data.geom_xmat[g].reshape(3, 3) @ model.mesh_vert[start : start + count].T
        local_z = (rot.T @ (world.T + data.geom_xpos[g] - origin).T)[2]
        lo, hi = min(lo, local_z.min()), max(hi, local_z.max())
    assert np.isfinite(hi), "tool0 carries no mesh geoms -- the adapter stack is gone"
    return float(lo), float(hi)


def test_tool_site_is_on_the_adapter_stacks_outer_face():
    model, data = _model()
    _, face = _stack_extent_along_tool_z(model, data)
    site_z = float(model.site("tool_site").pos[2])
    assert abs(site_z - face) < FACE_TOL_M, (
        f"tool_site is at tool-frame z = {site_z * 1e3:.1f} mm but the adapter stack ends at "
        f"{face * 1e3:.1f} mm: a tool attached there is not seated on the FT sensor."
    )


def test_fts_site_is_inside_the_stack_and_before_the_tool():
    """The measurement cut sits within the sensor, so a seated tool still hangs below it."""
    model, data = _model()
    lo, face = _stack_extent_along_tool_z(model, data)
    fts_z = float(model.site("fts_site").pos[2])
    assert lo < fts_z < float(model.site("tool_site").pos[2]) <= face + FACE_TOL_M
