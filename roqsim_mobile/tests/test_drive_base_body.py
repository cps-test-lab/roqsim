"""A drive reads its wheels in the body the robot registered, never in one found by its name.

The wheel roll signs are the wheel axes expressed in the base. The robot here is rooted at
``chassis``, and beside it stands a prop that happens to be called ``base_link``, turned round: a
drive that looked the base up by that name would read every wheel backwards.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim_mobile.plugins.diff_drive import DiffDrivePlugin
from roqsim_mobile.plugins.omni_drive import OmniDrivePlugin

SCENE = """
<mujoco model="drive_base_body">
  <worldbody>
    <body name="chassis" pos="0 0 0.1">
      <freejoint name="base_free"/>
      <geom type="box" size="0.2 0.15 0.04" mass="5"/>
      <body name="left_wheel_link" pos="0 0.18 0">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" mass="0.2"/>
      </body>
      <body name="right_wheel_link" pos="0 -0.18 0">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" mass="0.2"/>
      </body>
    </body>
    <body name="base_link" pos="3 0 0.5" euler="0 0 180">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_wheel_motor" joint="left_wheel_joint" kv="1"/>
    <velocity name="right_wheel_motor" joint="right_wheel_joint" kv="1"/>
    <motor name="base_vx" joint="left_wheel_joint"/>
    <motor name="base_vy" joint="left_wheel_joint"/>
    <motor name="base_wz" joint="left_wheel_joint"/>
  </actuator>
</mujoco>
"""

#: Two wheels are enough to read signs off; configure does not count them.
OMNI = {
    "wheel_radius": 0.05,
    "wheel_separation": 0.36,
    "axis_separation": 0.1,
    "wheels": ["left_wheel", "right_wheel"],
}


def _ctx(body="chassis"):
    ctx = SimContext(config={})
    ctx.model = mujoco.MjModel.from_xml_string(SCENE)
    ctx.data = mujoco.MjData(ctx.model)
    ctx.entities.add(
        Entity(name="robot", kind="robot", body=body, meta={"prefix": "", "namespace": ""})
    )
    return ctx


def test_diff_drive_reads_the_wheels_in_the_registered_root():
    plugin = DiffDrivePlugin({}, entity="robot")
    plugin.configure(_ctx())
    assert plugin.base_body == "chassis"
    assert (plugin._sign_l, plugin._sign_r) == ([1.0], [1.0])


def test_omni_drive_reads_the_wheels_in_the_registered_root():
    plugin = OmniDrivePlugin(dict(OMNI), entity="robot")
    plugin.configure(_ctx())
    assert plugin.base_body == "chassis"
    assert list(plugin._wsign) == [1.0, 1.0]


def test_base_body_is_a_path_within_the_robot():
    """`base_link` within the robot is not the prop of that name, and naming it is refused."""
    plugin = DiffDrivePlugin({"base_body": "base_link"}, entity="robot")
    with pytest.raises(RuntimeError, match="no frame 'robot/base_link'"):
        plugin.configure(_ctx())


@pytest.mark.parametrize(("cls", "config"), [(DiffDrivePlugin, {}), (OmniDrivePlugin, OMNI)])
def test_a_robot_that_registered_no_body_is_refused(cls, config):
    plugin = cls(dict(config), entity="robot")
    with pytest.raises(RuntimeError, match="entity 'robot' registered no body"):
        plugin.configure(_ctx(body=None))
