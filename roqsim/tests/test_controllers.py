# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Switching controllers the way ros2_control switches them.

The rules here are not roqsim's to choose: a scenario that switches in simulation has to be the
same scenario that switches on the robot, so every semantic below is the upstream one. The
tempting deviations are each a case where the simulation would accept what the arm refuses.
"""

from __future__ import annotations

import pytest

from roqsim.controllers import (
    ACTIVE,
    AUTO,
    BEST_EFFORT,
    FORCE_AUTO,
    INACTIVE,
    STRICT,
    Controller,
    ControllerRegistry,
)


def _registry():
    """A trajectory controller and a Cartesian one competing for the same joints, plus two
    broadcasters that claim nothing -- the shape of a real arm."""
    r = ControllerRegistry()
    r.register(
        Controller(
            name="joint_trajectory_controller",
            type="joint_trajectory_controller/JointTrajectoryController",
            claims=("shoulder_pan_joint/position", "elbow_joint/position"),
            state=ACTIVE,
        )
    )
    r.register(
        Controller(
            name="cartesian_compliance_controller",
            type="cartesian_controllers/CartesianComplianceController",
            claims=("shoulder_pan_joint/position", "elbow_joint/position"),
            state=INACTIVE,
        )
    )
    r.register(
        Controller(
            name="joint_state_broadcaster",
            type="joint_state_broadcaster/JointStateBroadcaster",
            reads=("shoulder_pan_joint/position", "elbow_joint/position"),
            state=ACTIVE,
        )
    )
    r.register(
        Controller(
            name="force_torque_sensor_broadcaster",
            type="force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster",
            reads=("tcp_fts_sensor/force.x",),
            state=ACTIVE,
        )
    )
    return r


# -- what a claim is -----------------------------------------------------------------------------


def test_only_command_interfaces_are_claimed_so_a_broadcaster_blocks_nothing():
    """A broadcaster reads the same joint the trajectory controller drives, and on every real robot
    both are active at once. Arbitrating STATE interfaces would refuse that."""
    r = _registry()
    assert r.claimed() == {
        "shoulder_pan_joint/position": "joint_trajectory_controller",
        "elbow_joint/position": "joint_trajectory_controller",
    }
    assert r.blockers("joint_state_broadcaster") == []


def test_an_inactive_controller_claims_nothing():
    """`claimed_interfaces` is populated only while active -- MoveIt's ros2_control manager derives
    a controller's joints from it, so filling it in while inactive would let a scenario select a
    controller the real robot would not offer."""
    r = _registry()
    cartesian = r.get("cartesian_compliance_controller")
    assert cartesian.claims, "it does claim something when it runs"
    assert cartesian.claimed_interfaces == ()

    r.switch(
        activate=["cartesian_compliance_controller"],
        deactivate=["joint_trajectory_controller"],
        strictness=STRICT,
    )
    assert cartesian.claimed_interfaces == cartesian.claims


# -- strictness ----------------------------------------------------------------------------------


def test_a_hand_over_names_both_sides_and_is_atomic():
    """The call a scenario makes: out and in, in one request."""
    r = _registry()
    ok, message = r.switch(
        activate=["cartesian_compliance_controller"],
        deactivate=["joint_trajectory_controller"],
        strictness=STRICT,
    )
    assert ok, message
    assert r.get("joint_trajectory_controller").state == INACTIVE
    assert r.get("cartesian_compliance_controller").state == ACTIVE


def test_strict_changes_nothing_when_anything_is_impossible():
    r = _registry()
    ok, message = r.switch(activate=["cartesian_compliance_controller"], strictness=STRICT)
    assert not ok
    assert "held by" in message and "nothing was changed" in message
    assert r.get("cartesian_compliance_controller").state == INACTIVE
    assert r.get("joint_trajectory_controller").state == ACTIVE, "the incumbent is untouched"


def test_best_effort_never_deactivates_a_controller_nobody_named():
    """The deviation that would cost the most: auto-deactivating under BEST_EFFORT. Upstream skips
    what it cannot do; a scenario written against an auto-deactivating simulation would hand over
    cleanly here and leave two controllers fighting on the arm."""
    r = _registry()
    ok, message = r.switch(activate=["cartesian_compliance_controller"], strictness=BEST_EFFORT)
    assert r.get("joint_trajectory_controller").state == ACTIVE
    assert r.get("cartesian_compliance_controller").state == INACTIVE
    assert not ok or "held by" in message


def test_force_auto_deactivates_what_blocks_the_activation():
    """The one strictness that DOES arbitrate, and the only one -- its own documentation states the
    mutually exclusive joint interface rule."""
    r = _registry()
    ok, _ = r.switch(activate=["cartesian_compliance_controller"], strictness=FORCE_AUTO)
    assert ok
    assert r.get("joint_trajectory_controller").state == INACTIVE
    assert r.get("cartesian_compliance_controller").state == ACTIVE


def test_an_unset_strictness_is_best_effort_not_a_refusal():
    """0 is not a valid value and is exactly what a default-constructed request carries, so a
    scenario that omits the field must not be refused for it."""
    r = _registry()
    ok, _ = r.switch(deactivate=["joint_trajectory_controller"], strictness=0)
    assert ok
    assert r.get("joint_trajectory_controller").state == INACTIVE


def test_a_controller_the_world_never_declared_cannot_be_switched_to():
    """The world file is the parameter file: `spawner` fails the same way against the real robot
    for a controller that is not in its params."""
    r = _registry()
    ok, message = r.switch(activate=["a_controller_nobody_declared"], strictness=STRICT)
    assert not ok
    assert "a_controller_nobody_declared" in message


@pytest.mark.parametrize("strictness", [BEST_EFFORT, STRICT, AUTO, FORCE_AUTO])
def test_a_switch_that_asks_for_nothing_impossible_succeeds_under_every_strictness(strictness):
    r = _registry()
    ok, _ = r.switch(
        activate=["cartesian_compliance_controller"],
        deactivate=["joint_trajectory_controller"],
        strictness=strictness,
    )
    assert ok


# -- the record ----------------------------------------------------------------------------------


def test_a_transition_is_stamped_so_a_run_can_say_when_the_hand_over_happened():
    """Otherwise the instant has to be inferred from when the motion changed, which is the thing
    being measured."""
    r = _registry()
    r.switch(
        activate=["cartesian_compliance_controller"],
        deactivate=["joint_trajectory_controller"],
        strictness=STRICT,
        sim_time=12.5,
    )
    assert r.get("cartesian_compliance_controller").transitions == [(12.5, INACTIVE, ACTIVE)]
    assert r.get("joint_trajectory_controller").transitions == [(12.5, ACTIVE, INACTIVE)]


def test_switching_a_controller_to_the_state_it_is_in_records_nothing():
    r = _registry()
    r.switch(activate=["joint_state_broadcaster"], strictness=STRICT, sim_time=3.0)
    assert r.get("joint_state_broadcaster").transitions == []


def test_the_switch_drives_the_controller_itself():
    """The registry holds the lifecycle state; the plugin holds its own flag. They move together,
    so neither can be read as the truth on its own."""
    seen = []
    r = ControllerRegistry()
    r.register(
        Controller(name="c", type="t", claims=("j/position",), state=INACTIVE, apply=seen.append)
    )
    r.switch(activate=["c"], strictness=STRICT)
    r.switch(deactivate=["c"], strictness=STRICT)
    assert seen == [True, False]


def test_two_controllers_cannot_share_a_name():
    r = _registry()
    with pytest.raises(RuntimeError, match="both called"):
        r.register(Controller(name="joint_trajectory_controller", type="other"))


# -- more than one robot -------------------------------------------------------------------------


def _two_arms():
    """Two arms in ONE namespace, each with a controller of the same name.

    roqsim allows this and disambiguates by entity; real ros2_control cannot have it, and
    `roqsim export moveit` already refuses it with advice on how to name them. The registry has to
    DESCRIBE that world rather than refuse it a second time in worse words -- and it must not let
    one arm's claims block the other's, since joint names here are unprefixed and identical.
    """
    r = ControllerRegistry()
    for arm in ("left", "right"):
        r.register(
            Controller(
                name="arm_controller",
                type="joint_trajectory_controller/JointTrajectoryController",
                claims=("shoulder_pan_joint/position",),
                state=ACTIVE,
                owner=arm,
            )
        )
    return r


def test_two_arms_may_each_have_a_controller_of_the_same_name():
    r = _two_arms()
    assert len(r.all()) == 2
    assert {c.owner for c in r.all()} == {"left", "right"}


def test_one_arms_claims_do_not_block_the_other_arms():
    """Joint names are unprefixed, so both arms claim `shoulder_pan_joint/position`. Arbitrating
    those against each other would refuse a world that works."""
    r = _two_arms()
    assert r.blockers("arm_controller") == []


def test_the_same_name_twice_on_ONE_robot_is_still_refused():
    r = ControllerRegistry()
    r.register(Controller(name="arm_controller", type="t", owner="left"))
    with pytest.raises(RuntimeError, match="both called"):
        r.register(Controller(name="arm_controller", type="other", owner="left"))


def test_controllers_are_listed_per_namespace():
    """One manager per robot namespace, as a multi-robot ros2_control deployment has."""
    r = ControllerRegistry()
    r.register(Controller(name="c", type="t", namespace="alice", owner="alice"))
    r.register(Controller(name="c", type="t", namespace="bob", owner="bob"))
    assert [c.namespace for c in r.all("alice")] == ["alice"]
    assert len(r.all()) == 2


def test_a_switch_names_a_controller_within_one_robots_manager():
    r = ControllerRegistry()
    r.register(Controller(name="c", type="t", namespace="alice", owner="alice", state=INACTIVE))
    r.register(Controller(name="c", type="t", namespace="bob", owner="bob", state=INACTIVE))

    ok, _ = r.switch(activate=["c"], namespace="alice", strictness=STRICT)

    assert ok
    assert r.get("c", "alice").state == ACTIVE
    assert r.get("c", "bob").state == INACTIVE, "bob's manager was not asked"
