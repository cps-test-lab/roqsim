"""joint_state_publisher: every hinge and slide joint of an entity in ONE message.

The property a consumer relies on is completeness in a single message: a driven wheel and a
passive suspension travel arrive together, named without the spawn prefix, and a quaternion
joint is left out rather than mangled into a scalar. Checked on a synthetic body with a driven
hinge, a passive slide and a ball joint.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.joint_state_publisher import JointStatePublisherPlugin

SCENE = """
<mujoco model="jsp_test">
  <option timestep="0.001"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="r_base_link" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.1" mass="5"/>
      <body name="r_wheel" pos="0 0.25 0">
        <joint name="r_wheel_joint" type="hinge" axis="0 1 0" armature="0.01"/>
        <geom type="cylinder" size="0.05 0.02" mass="0.2"/>
      </body>
      <body name="r_suspension" pos="0 -0.25 0">
        <joint name="r_drop_joint" type="slide" axis="0 0 1" range="0 0.03" stiffness="450" springref="0.03" damping="50"/>
        <geom type="box" size="0.02 0.02 0.02" mass="1.0"/>
      </body>
      <body name="r_gimbal" pos="0.3 0 0">
        <joint name="r_gimbal_joint" type="ball"/>
        <geom type="sphere" size="0.02" mass="0.01"/>
      </body>
    </body>
    <body name="other" pos="2 0 0.5">
      <joint name="other_joint" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="r_wheel_motor" joint="r_wheel_joint" kv="2"/>
  </actuator>
</mujoco>
"""


def _plugin(**cfg):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(
            name="robot", kind="robot", body="r_base_link", meta={"prefix": "r_", "namespace": ""}
        )
    )
    plugin = JointStatePublisherPlugin(dict(cfg), entity="robot")
    assert plugin.validate_config(dict(cfg)) == []
    plugin.configure(ctx)
    return ctx, plugin


def test_every_scalar_joint_of_the_entity_and_nothing_else():
    """Driven and passive together; the ball joint and the other body's joint left out; names
    without the spawn prefix."""
    ctx, plugin = _plugin()
    names, pos, vel, eff = plugin.read_joint_states()
    assert names == ["wheel_joint", "drop_joint"]
    assert len(pos) == len(vel) == len(eff) == 2


def test_the_state_is_the_joints_state():
    ctx, plugin = _plugin()
    ctx.data.ctrl[0] = 5.0  # spin the wheel
    for _ in range(200):
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
    names, pos, vel, eff = plugin.read_joint_states()
    assert vel[0] == pytest.approx(5.0, abs=0.5)
    assert abs(pos[0]) > 0.1
    assert eff[0] != 0.0, "a driven joint reports the actuator's generalised force"
    assert eff[1] == 0.0, "a passive joint reports no actuator effort"


def test_joints_restricts_and_orders_the_message():
    ctx, plugin = _plugin(joints=["drop_joint"])
    assert plugin.read_joint_states()[0] == ["drop_joint"]


def test_one_endpoint_named_joint_states():
    ctx, plugin = _plugin(rate_hz=62.0)
    (ep,) = ctx.interface.all()
    assert ep.name == "joint_states"
    assert ep.rate_hz == 62.0
    assert ep.backend["ros2"]["type"] == "sensor_msgs.msg.JointState"
    assert ep.backend["ros2"]["topic"] == "joint_states"


@pytest.mark.parametrize(
    "joints, message",
    [
        (["nope"], "not found"),
        (["other_joint"], "not found"),
        (["gimbal_joint"], "hinge or slide"),
    ],
)
def test_a_named_joint_that_cannot_be_published_fails_loudly(joints, message):
    with pytest.raises(RuntimeError, match=message):
        _plugin(joints=joints)


def test_declared_at_the_top_of_a_document_it_is_refused():
    assert JointStatePublisherPlugin.requires_owner is True


@pytest.mark.parametrize("bad", [{"rate_hz": 0}, {"joints": "wheel_joint"}])
def test_bad_config_is_reported(bad):
    assert JointStatePublisherPlugin(bad, entity="robot").validate_config(bad) != []
