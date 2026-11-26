# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A posed camera: drawn at the pose and intrinsics its request carries, with no camera of the model.

A renderer declares whether it draws one (``RendererCapabilities.posed``); one that does not is
refused, naming what it lacks, and nothing falls back to another view.
"""

from __future__ import annotations

from roqsim.render_api import CameraSpec, RendererCapabilities


def _posed(**kw) -> CameraSpec:
    return CameraSpec(id="v", width=64, height=48, intrinsics=(50.0, 50.0, 31.5, 23.5), **kw)


def test_a_camera_with_intrinsics_and_no_model_camera_is_posed():
    assert _posed().posed
    # A model camera, or one without intrinsics, is not.
    assert not _posed(mujoco_camera=0).posed
    assert not CameraSpec(id="v", width=4, height=4).posed


def test_a_renderer_that_does_not_draw_a_posed_camera_says_so():
    assert _posed().missing_from(RendererCapabilities()) == ["a camera at a given pose"]
    assert _posed().missing_from(RendererCapabilities(posed=True)) == []


def test_the_capability_is_listed():
    assert RendererCapabilities().as_dict()["posed"] is False
    assert RendererCapabilities(posed=True).as_dict()["posed"] is True
