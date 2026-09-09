# Copyright (C) 2025 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""``place_body`` places a ``motion: driven`` body, and the solver leaves it where it was put.

Until this, the only body the write knew was a free one, so "teleportable" implied "owned by the
solver". A trial that reveals an obstacle on cue therefore had to make it a free body, and a free
body answers a placement that intersects other geometry by leaving the scene at speed -- while an
obstacle the robot reaches is shoved off the placement the experiment selected, so the run measures a
layout nobody chose.

A mocap body has no degrees of freedom. It takes the pose, keeps it, and still collides -- which is
what a placed obstacle is, and what the other simulator's static obstacle model already did.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.placement import place_body

SCENE = """
<mujoco model="driven_placement">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="driven_prop" pos="40 40 0.5" mocap="true">
      <geom name="driven" type="box" size="0.25 0.25 0.5"/>
    </body>
    <body name="welded_prop" pos="1 1 0.5">
      <geom name="welded" type="box" size="0.25 0.25 0.5"/>
    </body>
    <body name="free_prop" pos="-3 0 0.5">
      <freejoint name="free_prop_joint"/>
      <geom name="free" type="box" size="0.25 0.25 0.5" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Entity:
    def __init__(self, body, meta=None):
        self.body = body
        self.meta = meta or {}


def _ctx():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return type("Ctx", (), {"model": model, "data": data})()


def _xpos(ctx, body):
    return np.array(ctx.data.xpos[mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)])


def test_a_driven_body_is_placed_without_a_base_joint():
    """No ``base_joint`` in its meta, and placeable regardless: the BODY carries the answer.

    A mocap prop registers exactly the meta a welded one does, so a write that asks meta whether an
    entity can be placed would refuse the one kind of body that most wants to be.
    """
    ctx = _ctx()
    assert place_body(
        ctx, _Entity("driven_prop"), (2.0, -1.0, 0.5), (1.0, 0.0, 0.0, 0.0)
    )
    assert np.allclose(_xpos(ctx, "driven_prop"), [2.0, -1.0, 0.5], atol=1e-9)


def test_a_driven_body_stays_where_it_was_put_even_intersecting_another():
    """The behaviour the whole change is for.

    Placed inside the welded prop, a free body is launched by the first contact resolution. This
    one is not a body the solver integrates, so the penetration is a rendering and contact fact and
    nothing else -- the obstacle is where the experiment said it is, for the whole trial.
    """
    ctx = _ctx()
    inside = _xpos(ctx, "welded_prop") + np.array([0.10, 0.0, 0.0])
    assert place_body(
        ctx, _Entity("driven_prop"), tuple(inside), (1.0, 0.0, 0.0, 0.0)
    )
    for _ in range(500):
        mujoco.mj_step(ctx.model, ctx.data)
    assert np.allclose(_xpos(ctx, "driven_prop"), inside, atol=1e-9)


def test_a_welded_body_is_still_refused():
    """Welded scenery has neither a mocap body nor a free joint, and the refusal is the diagnostic."""
    ctx = _ctx()
    assert not place_body(
        ctx, _Entity("welded_prop"), (2.0, -1.0, 0.5), (1.0, 0.0, 0.0, 0.0)
    )


def test_a_twist_asked_of_a_driven_body_does_not_fail_the_placement():
    """A mocap body has no DOF to carry a velocity, and the POSE was applied in full.

    Refusing here would report a completed placement as failed, which is the one answer a trial
    cannot act on.
    """
    ctx = _ctx()
    assert place_body(
        ctx,
        _Entity("driven_prop"),
        (2.0, -1.0, 0.5),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert np.allclose(_xpos(ctx, "driven_prop"), [2.0, -1.0, 0.5], atol=1e-9)


def test_a_free_body_placed_inside_another_is_ejected():
    """The contrast that makes ``driven`` the right mode, stated rather than assumed.

    This is not a defect in the solver -- it is what a free body IS. Which is exactly why an
    obstacle whose pose the experiment selected must not be one: the same overlap that this test
    measures as metres of travel reached a run as an obstacle visibly leaving the scene, and would
    have reached a quieter one as a layout the search never chose.
    """
    ctx = _ctx()
    entity = _Entity("free_prop", {"base_joint": "free_prop_joint"})
    inside = _xpos(ctx, "welded_prop") + np.array([0.10, 0.0, 0.0])
    assert place_body(ctx, entity, tuple(inside), (1.0, 0.0, 0.0, 0.0))
    for _ in range(500):
        mujoco.mj_step(ctx.model, ctx.data)
    assert np.linalg.norm(_xpos(ctx, "free_prop")[:2] - inside[:2]) > 0.2
