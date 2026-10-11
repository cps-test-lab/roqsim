"""Entity placement and presence, as every route reaches them (roqsim.entity_control)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim import entity_control
from roqsim.context import Entity, SimContext
from roqsim.kinematics import body_twist

SCENE = """
<mujoco>
  <worldbody>
    <body name="robot" pos="1 2 0.1"><freejoint name="robot_free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
    <body name="fixed" pos="3 3 0.1"><geom type="box" size="0.1 0.1 0.1"/></body>
    <body name="carried" pos="5 5 0.1" mocap="true"><geom type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def ctx():
    ctx = SimContext(config={})
    ctx.model = mujoco.MjModel.from_xml_string(SCENE)
    ctx.data = mujoco.MjData(ctx.model)
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.entities.add(Entity("robot", "robot", body="robot", meta={"base_joint": "robot_free"}))
    ctx.entities.add(Entity("prop", "object", body="fixed"))
    ctx.entities.add(Entity("walker", "object", body="carried"))
    return ctx


def test_set_state_places_a_free_body_at_rest(ctx):
    done = entity_control.set_state(ctx, "robot", [0.5, 0.5, 0.3], [1, 0, 0, 0])
    assert done == "placed at [0.5, 0.5, 0.3]"
    np.testing.assert_allclose(ctx.data.xpos[ctx.model.body("robot").id], [0.5, 0.5, 0.3])
    assert not ctx.data.qvel.any()


def test_welded_scenery_and_unknown_names_are_refused(ctx):
    with pytest.raises(entity_control.EntityRefused, match="welded scenery"):
        entity_control.set_state(ctx, "prop", [0, 0, 1], [1, 0, 0, 0])
    with pytest.raises(entity_control.UnknownEntity, match="no entity called 'ghost'"):
        entity_control.set_state(ctx, "ghost", [0, 0, 1], [1, 0, 0, 0])


def test_presence_flips_once_and_refuses_the_state_it_is_in(ctx):
    with pytest.raises(entity_control.EntityRefused, match="already present"):
        entity_control.set_presence(ctx, "robot", True)
    assert entity_control.set_presence(ctx, "robot", False) == "absent"
    assert (
        entity_control.set_presence(ctx, "robot", True, [0, 0, 0.5]) == "present at [0.0, 0.0, 0.5]"
    )


def test_a_velocity_is_set_in_the_world_frame_whatever_the_orientation(ctx):
    """MuJoCo keeps a free joint's angular velocity in the body frame; the request is world frame,
    as the pose reader reports it. A body turned 90 degrees about x makes the two differ."""
    half = np.sqrt(0.5)
    entity_control.set_state(
        ctx, "robot", [0, 0, 1], [half, half, 0, 0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]
    )
    twist = body_twist(ctx.model, ctx.data, ctx.model.body("robot").id)
    np.testing.assert_allclose(twist.linear, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(twist.angular, [0.0, 0.0, 2.0], atol=1e-9)


def test_a_velocity_for_a_body_that_cannot_carry_one_is_refused(ctx):
    """A mocap body takes the pose and has no DOF for the velocity, so placing it 'moving' would
    report a motion that never happens."""
    with pytest.raises(entity_control.EntityRefused, match="cannot carry the velocity"):
        entity_control.set_state(ctx, "walker", [5, 5, 0.1], [1, 0, 0, 0], [1.0, 0.0, 0.0])
    assert entity_control.set_state(ctx, "walker", [6, 5, 0.1], [1, 0, 0, 0]).startswith("placed")
