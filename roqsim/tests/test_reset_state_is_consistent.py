# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""What a plugin writes into the initial state at reset is what the derived quantities describe.

A plugin's ``on_reset`` may set a pose or a command; the site poses, sensor data and contacts that
MuJoCo derives from them are only recomputed by a forward pass. Without one after the reset hooks,
anything read before the first step described the state as it was before the plugins ran.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

#: Where the plugin puts the slider at every reset.
RESET_Q = 0.3


class _SliderSetAtReset(Plugin):
    """A body on a slide joint, with a site on it, placed at :data:`RESET_Q` by ``on_reset``."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        body = spec.worldbody.add_body(name="slider", pos=[0, 0, 1])
        body.add_joint(name="slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=1.0)
        body.add_site(name="marker")

    def on_reset(self, ctx: SimContext) -> None:
        ctx.data.qpos[ctx.model.jnt_qposadr[ctx.model.joint("slide").id]] = RESET_Q


def test_a_pose_set_at_reset_is_the_pose_the_sites_report_before_the_first_step():
    engine = Engine(
        load_config_from_dict({"sim": {}, "components": [{f"{__name__}:_SliderSetAtReset": {}}]})
    )
    engine.setup()
    engine.reset()
    marker = engine.ctx.data.site("marker").xpos
    assert marker[0] == pytest.approx(RESET_Q)
    engine.shutdown()
