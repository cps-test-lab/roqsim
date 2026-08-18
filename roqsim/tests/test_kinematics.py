"""``roqsim.kinematics``: the world-frame twist of a body, and the two ways to get it wrong.

Every case here spins or slides about a **tilted** axis. A planar test -- a wheeled robot on a floor
-- cannot fail: its angular velocity is pure z and its linear velocity pure xy, so swapping the two
halves of MuJoCo's 6-vector, or reading the com-frame ``cvel`` instead of the body-frame velocity,
both still look like plausible numbers.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.kinematics import body_twist

# A hinge about a tilted axis, with the spinning body's origin OFFSET from the axis, so the linear
# and angular parts are both non-trivial in all three components and differ from each other.
_TILTED_XML = """
<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="hub" pos="0 0 1">
      <joint name="spin" type="hinge" axis="0 0.6 0.8"/>
      <geom type="sphere" size=".05" mass="1"/>
      <body name="rim" pos="0.5 0 0">
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_SLIDE_XML = """
<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="b" pos="0 0 0">
      <joint name="s" type="slide" axis="0.6 0 0.8"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _spun(xml, joint_vel):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[0] = joint_vel
    mujoco.mj_forward(model, data)
    return model, data


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_angular_velocity_is_the_tilted_axis_times_the_rate():
    """omega must point along the joint's own axis, not along z."""
    model, data = _spun(_TILTED_XML, 2.0)
    twist = body_twist(model, data, _bid(model, "hub"))
    assert np.allclose(twist.angular, np.array([0.0, 0.6, 0.8]) * 2.0, atol=1e-9)
    # The hub sits on the axis, so it only spins -- this is what pins the slot order: a swapped
    # unpacking would report the axis as a LINEAR velocity here.
    assert np.allclose(twist.linear, [0.0, 0.0, 0.0], atol=1e-9)


def test_linear_velocity_is_omega_cross_r_at_the_body_origin():
    """The rim's linear velocity is the rigid-body one at ITS OWN origin, not at the subtree com."""
    model, data = _spun(_TILTED_XML, 2.0)
    rim = _bid(model, "rim")
    twist = body_twist(model, data, rim)
    omega = np.array([0.0, 0.6, 0.8]) * 2.0
    r = np.asarray(data.xpos[rim]) - np.asarray(data.xpos[_bid(model, "hub")])
    assert np.allclose(twist.linear, np.cross(omega, r), atol=1e-9)
    assert np.allclose(twist.angular, omega, atol=1e-9)


def test_it_is_not_cvel_which_is_measured_at_the_subtree_com():
    """The distinction the docstring claims is real and this world exhibits it.

    ``data.cvel``'s linear part is the velocity at the subtree centre of mass, which for the rim is
    a different point than its own origin -- so the two disagree by omega x r. Asserting they DIFFER
    is what stops someone 'simplifying' body_twist into a cvel read.
    """
    model, data = _spun(_TILTED_XML, 2.0)
    rim = _bid(model, "rim")
    assert not np.allclose(body_twist(model, data, rim).linear, data.cvel[rim][3:], atol=1e-6)


def test_pure_translation_along_a_tilted_axis_has_no_rotation():
    model, data = _spun(_SLIDE_XML, 3.0)
    twist = body_twist(model, data, _bid(model, "b"))
    assert np.allclose(twist.linear, np.array([0.6, 0.0, 0.8]) * 3.0, atol=1e-9)
    assert np.allclose(twist.angular, [0.0, 0.0, 0.0], atol=1e-9)


def test_a_body_at_rest_has_no_twist():
    model, data = _spun(_TILTED_XML, 0.0)
    twist = body_twist(model, data, _bid(model, "rim"))
    assert np.allclose(twist.linear + twist.angular, [0.0] * 6, atol=1e-12)


def test_the_named_fields_survive_a_round_trip_through_mujocos_slot_order():
    """MuJoCo packs rotational first; Twist exposes linear first. Pin the mapping explicitly."""
    model, data = _spun(_TILTED_XML, 1.5)
    rim = _bid(model, "rim")
    raw = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, rim, raw, 0)
    twist = body_twist(model, data, rim)
    assert np.allclose(twist.angular, raw[:3], atol=1e-12)
    assert np.allclose(twist.linear, raw[3:], atol=1e-12)
    # and they are genuinely different vectors here, so the assertion above has teeth
    assert not np.allclose(raw[:3], raw[3:], atol=1e-3)
    assert math.isfinite(twist.linear[0])
