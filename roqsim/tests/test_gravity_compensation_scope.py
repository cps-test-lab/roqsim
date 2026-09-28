# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Which bodies a drive is holding up, and which ones are holding the robot up.

``body_gravcomp`` cancels a body's weight. On an arm bolted to a bench that is what its motors
do, so every link gets it. On a machine that stands on the ground it is only true of the parts
that hang: cancel the base's weight and the robot presses on the floor with less than it weighs,
or nothing at all -- and it does not fall over, so no test that watches for a fall would see it.

The line is drawn from the actuator table: a joint driven by ``position`` or ``impedance`` is a
drive holding what is below it. ``velocity`` is left out because that is how a wheel is driven,
and a wheel carries the robot rather than being carried.
"""

from __future__ import annotations

import mujoco

from roqsim.actuators import apply_gravity_compensation
from roqsim.actuators import resolve as resolve_actuators

#: A machine with both kinds of body: a free base standing on a velocity-driven wheel, and a
#: two-link arm bolted to that base and driven by position servos.
_ROVER = """
<mujoco model="rover">
  <worldbody>
    <body name="base">
      <freejoint name="base_free"/>
      <geom name="chassis" type="box" size=".3 .2 .1" mass="20"/>
      <body name="wheel" pos=".2 0 -.1">
        <joint name="wheel_joint" type="hinge" axis="0 1 0"/>
        <geom name="tyre" type="sphere" size=".1" mass="2"/>
      </body>
      <body name="shoulder" pos="0 0 .1">
        <joint name="arm_1" type="hinge" axis="0 1 0"/>
        <geom name="upper" type="capsule" fromto="0 0 0 0 0 .3" size=".03" mass="1"/>
        <body name="forearm" pos="0 0 .3">
          <joint name="arm_2" type="hinge" axis="0 1 0"/>
          <geom name="lower" type="capsule" fromto="0 0 0 0 0 .3" size=".03" mass="1"/>
          <body name="tool" pos="0 0 .3">
            <geom name="tip" type="sphere" size=".04" mass=".2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="wheel_motor" joint="wheel_joint" kv="10"/>
    <position name="arm_1_motor" joint="arm_1" kp="100"/>
    <position name="arm_2_motor" joint="arm_2" kp="100"/>
  </actuator>
</mujoco>
"""


def _compensated(rows=None) -> set:
    """The names of the bodies :func:`apply_gravity_compensation` marks."""
    spec = mujoco.MjSpec.from_string(_ROVER)
    table = resolve_actuators(spec, None, model_name="rover") if rows is None else rows
    apply_gravity_compensation(spec, table)
    return {body.name for body in spec.bodies if body.name != "world" and body.gravcomp}


def test_the_arm_a_drive_holds_up_is_compensated():
    """Every body below a position-driven joint, including the tool welded past the last one.

    The tool has no joint of its own; its weight still lands on the arm's motors, and an arm
    compensated only as far as its last joint would sag by exactly the tool's weight.
    """
    assert _compensated() >= {"shoulder", "forearm", "tool"}


def test_the_base_and_its_wheels_are_not():
    """The load path to the ground, which is the half that must keep its weight.

    The base hangs off nothing and the wheel carries the robot. Compensated, the machine would
    stand on the floor pressing with less than it weighs -- upright, and wrong.
    """
    assert _compensated().isdisjoint({"base", "wheel"})


def test_without_a_table_the_whole_mechanism_is_compensated():
    """The explicit whole-mechanism form, for a caller that asks for it by name.

    Its only correct use is something that hangs: an arm whose controller is handed its own
    gravity term. It is offered separately rather than as a default for exactly that reason.
    """
    spec = mujoco.MjSpec.from_string(_ROVER)
    apply_gravity_compensation(spec)

    assert {b.name for b in spec.bodies if b.name != "world" and b.gravcomp} == {
        "base",
        "wheel",
        "shoulder",
        "forearm",
        "tool",
    }


def test_a_torque_driven_arm_is_left_alone():
    """``effort`` names a drive that is handed no gravity term, so neither is the model.

    Supplying it here would make a controller that omits gravity indistinguishable from one that
    has it -- which is the comparison such an experiment exists to make.
    """
    spec = mujoco.MjSpec.from_string(_ROVER)
    table = resolve_actuators(
        spec,
        {
            "each": {
                "arm_1_motor": {"control": "effort", "ctrlrange": [-50.0, 50.0]},
                "arm_2_motor": {"control": "effort", "ctrlrange": [-50.0, 50.0]},
            }
        },
        model_name="rover",
    )
    apply_gravity_compensation(spec, table)

    assert not {b.name for b in spec.bodies if b.name != "world" and b.gravcomp}


#: The rover's arm again, carrying on its tool a pendulum on an unactuated ball joint, a two-finger
#: gripper whose second finger follows the first through a joint equality, and a four-bar whose
#: passive joints a ``connect`` closes into a loop with a driven one.
_ARM_WITH_TOOLS = """
<mujoco model="tools">
  <worldbody>
    <body name="shoulder">
      <joint name="arm_1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".03" mass="1"/>
      <body name="tool" pos="0 0 .3">
        <geom type="sphere" size=".04" mass=".2"/>
        <body name="pendulum" pos=".1 0 0">
          <joint name="swing" type="ball"/>
          <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".01" mass=".1"/>
          <body name="bob" pos="0 0 -.3">
            <geom type="sphere" size=".03" mass=".2"/>
          </body>
        </body>
        <body name="finger_a" pos="0 .05 0">
          <joint name="finger_a" type="slide" axis="0 1 0"/>
          <geom type="box" size=".01 .01 .03" mass=".05"/>
        </body>
        <body name="finger_b" pos="0 -.05 0">
          <joint name="finger_b" type="slide" axis="0 -1 0"/>
          <geom type="box" size=".01 .01 .03" mass=".05"/>
        </body>
        <body name="crank" pos="-.1 0 0">
          <joint name="crank" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -.1" size=".01" mass=".05"/>
          <body name="coupler" pos="0 0 -.1">
            <joint name="coupler" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 -.1 0 0" size=".01" mass=".05"/>
          </body>
        </body>
        <body name="rocker" pos="-.2 0 0">
          <joint name="rocker" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -.1" size=".01" mass=".05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="finger_a" joint2="finger_b"/>
    <connect body1="coupler" body2="rocker" anchor="-.1 0 0"/>
  </equality>
  <actuator>
    <position name="arm_1_motor" joint="arm_1" kp="100"/>
    <position name="finger_motor" joint="finger_a" kp="100"/>
    <position name="crank_motor" joint="crank" kp="100"/>
  </actuator>
</mujoco>
"""


def _tools_compensated(whole_mechanism: bool) -> set:
    spec = mujoco.MjSpec.from_string(_ARM_WITH_TOOLS)
    if whole_mechanism:
        apply_gravity_compensation(spec)
    else:
        apply_gravity_compensation(spec, resolve_actuators(spec, None, model_name="tools"))
    return {body.name for body in spec.bodies if body.name != "world" and body.gravcomp}


def test_a_pendulum_on_the_tool_keeps_its_weight():
    """Nothing acts on its ball joint, so no drive holds its pose: it must hang and swing.

    Compensated, it would float at whatever angle it was spawned in -- still, and wrong. Its weight
    still reaches the arm through the joint, as a load the arm's drives carry.
    """
    assert _tools_compensated(whole_mechanism=False).isdisjoint({"pendulum", "bob"})
    assert _tools_compensated(whole_mechanism=True).isdisjoint({"pendulum", "bob"})


def test_a_coupled_finger_and_a_closed_linkage_are_still_held():
    """A joint an equality couples to a driven one, or one a loop closes through, is not free.

    A gripper's follower finger and the passive links of its linkage are held by the actuator
    through those constraints, so they stay compensated like the tool they belong to.
    """
    for whole_mechanism in (False, True):
        assert _tools_compensated(whole_mechanism) >= {
            "shoulder",
            "tool",
            "finger_a",
            "finger_b",
            "crank",
            "coupler",
            "rocker",
        }
