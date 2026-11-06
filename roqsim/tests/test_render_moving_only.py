# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A camera may ask for the moving bodies alone (``bodies: moving``), and MuJoCo's renderer draws them.

A renderer that brings its own static world -- a captured splat -- composites the robots into it by
depth, so it needs them drawn with nothing static: no floor, no wall, no static prop. The reference
is the same world with every static geom moved to a geom group MuJoCo does not draw, so the model,
its extent, its lights and its planes are those of the world drawn in full, and the only difference
is what is in the scene.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.render_api import CameraSpec, RendererCapabilities
from roqsim.rendering import RenderService, moving_bodies

#: A floor, a wall and a static post, a free crate, a mocap marker, and a body welded to the world
#: through a child (static too). ``{static_group}`` is the group of every static geom.
XML = """
<mujoco>
  <visual><global offwidth="160" offheight="120"/><quality offsamples="0"/></visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 .1" rgba=".5 .5 .5 1" group="{static_group}"/>
    <body name="wall" pos="-1 0 .5">
      <geom type="box" size=".05 1 .5" rgba=".2 .2 .8 1" group="{static_group}"/>
    </body>
    <body name="post_base" pos="0 -.6 0">
      <body name="post"><geom type="cylinder" size=".05 .4" pos="0 0 .4" group="{static_group}"/></body>
    </body>
    <body name="crate" pos="0 0 .3">
      <freejoint/>
      <geom type="box" size=".2 .2 .2" rgba=".9 .3 .2 1"/>
    </body>
    <body name="marker" mocap="true" pos=".2 .5 .4">
      <geom type="sphere" size=".1" rgba=".2 .9 .3 1"/>
    </body>
    <camera name="persp" pos="2 0 1" xyaxes="0 1 0 -0.4 0 1" fovy="55"/>
  </worldbody>
</mujoco>
"""


def _model(static_group: int) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(XML.replace("{static_group}", str(static_group)))
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.1, -0.2, 0.35]
    mujoco.mj_forward(model, data)
    return model, data


def _draw(model, data, bodies: str):
    service = RenderService(model)
    view = service.view(
        "cam", camera="persp", width=160, height=120, images=("rgb", "depth"), bodies=bodies
    )
    service.check()
    try:
        # A first frame may differ from later ones on some GPUs (see test_render_service_pixels).
        service.render_now([(view, ("rgb", "depth"))], data)
        (frame,) = service.render_now([(view, ("rgb", "depth"))], data)
        return frame.rgb.copy(), frame.depth.copy()
    finally:
        service.close()


def test_the_moving_bodies_alone_are_the_world_without_its_static_geoms():
    full_model, data = _model(static_group=0)
    rgb, depth = _draw(full_model, data, "moving")
    hidden_model, hidden_data = _model(static_group=3)  # group 3 is not drawn by default
    ref_rgb, ref_depth = _draw(hidden_model, hidden_data, "all")
    np.testing.assert_array_equal(rgb, ref_rgb)
    np.testing.assert_array_equal(depth, ref_depth)
    # Nothing static was drawn, so most of the picture is empty and its depth is inf.
    assert np.isinf(depth).mean() > 0.5
    # And the full world draws what it drew: the floor everywhere below the horizon.
    full_rgb, full_depth = _draw(full_model, data, "all")
    assert np.isfinite(full_depth).mean() > np.isfinite(depth).mean()
    assert not np.array_equal(full_rgb, rgb)


def test_the_moving_bodies_are_those_moving_bodies_marks():
    """MuJoCo's dynamic category is roqsim's ``moving``: free and mocap bodies, nothing welded."""
    model, data = _model(static_group=0)
    scene = mujoco.MjvScene(model, 100)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
        scene,
    )
    drawn = {int(model.geom_bodyid[g.objid]) for g in scene.geoms[: scene.ngeom]}
    assert drawn == set(np.flatnonzero(moving_bodies(model)).tolist())


def test_a_renderer_that_cannot_draw_them_alone_says_so():
    spec = CameraSpec(id="cam", width=4, height=4, bodies="moving")
    assert spec.missing_from(RendererCapabilities()) == ["the moving bodies alone"]
    assert spec.missing_from(RendererCapabilities(moving_only=True)) == []


def test_an_unknown_choice_of_bodies_is_refused():
    model, _ = _model(static_group=0)
    service = RenderService(model)
    with pytest.raises(ValueError, match="bodies 'static'; one of all, moving"):
        service.view("cam", camera="persp", width=8, height=8, bodies="static")
