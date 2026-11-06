# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer that is garbage-collected must not take a live renderer's GL objects with it.

Freeing a render context is a run of ``glDelete*`` calls that land in whatever context is current;
freed after its GL context (as ``mujoco.Renderer.close`` does), they hit another renderer's objects
of the same names. The symptom is a segmentation pass that reads back pixel values no geom has.
"""

from __future__ import annotations

import gc
import os

import mujoco
import numpy as np
import pytest

from roqsim.rendering import FrameRenderer

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .3"><geom type="box" size=".2 .2 .2" rgba="1 0 0 1"/></body>
    <camera name="cam" pos="1.5 0 1" xyaxes="0 1 0 -0.5 0 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_data(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _renderer(model) -> FrameRenderer:
    try:
        return FrameRenderer(model, 64, 64, camera=0)
    except Exception as err:  # noqa: BLE001 - the backend check raises its own type
        pytest.skip(f"no usable offscreen GL here: {err}")


def _ids(frame: FrameRenderer, data) -> np.ndarray:
    return frame.render_segmentation(data)[..., 0]


def test_a_collected_renderer_leaves_a_live_one_intact(model_data):
    model, data = model_data
    old = _renderer(model)
    old.render(data)
    live = _renderer(model)
    live.render(data)
    before = _ids(live, data)
    assert before.max() >= 0 and (before >= 0).any()

    del old
    gc.collect()

    after = _ids(live, data)  # raised IndexError, or read garbage, before the teardown fix
    np.testing.assert_array_equal(after, before)
    live.render(data)
    live.close()


def test_close_is_idempotent_and_survives_a_second_call(model_data):
    model, data = model_data
    frame = _renderer(model)
    frame.render(data)
    frame.close()
    frame.close()
    with pytest.raises(AttributeError):
        frame.render(data)  # a closed renderer says so rather than drawing nothing
