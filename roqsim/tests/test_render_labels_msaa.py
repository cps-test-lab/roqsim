# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo's renderer labels a pixel with the geom drawn at its centre, also under MSAA.

The labels are id colours, and a multisampled buffer averages a pixel's samples when it is read:
at an edge two geoms' ids would blend into a third geom's, or past the last one, which was a wrong
label or an index out of range. The label pass draws single-sampled instead.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from roqsim.rendering import FrameRenderer

#: Three hundred spheres packed edge to edge, so that ids above 255 meet at many edges.
XML = (
    '<mujoco><visual><quality offsamples="4"/></visual><worldbody><light pos="0 0 3"/>'
    + "".join(
        f'<geom type="sphere" pos="{(i % 17) * 0.11 - 0.9} {(i // 17) * 0.11 - 0.9} 0" size=".06"/>'
        for i in range(300)
    )
    + '<camera name="c" pos="0 0 2.2"/></worldbody></mujoco>'
)


def _labels(model, data):
    frame = FrameRenderer(model, 200, 200, camera="c")
    try:
        return frame.render_segmentation(data)
    finally:
        frame.close()


def test_labels_under_msaa_are_the_single_sampled_labels():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    single = copy.copy(model)
    single.vis.quality.offsamples = 0
    labels = _labels(model, data)
    np.testing.assert_array_equal(labels, _labels(single, data))
    geom = labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    assert geom.any() and (labels[geom, 0] < model.ngeom).all()
    assert model.vis.quality.offsamples == 4, "the model is not written"
