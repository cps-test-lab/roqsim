# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The Cartesian controller's identity, its command surface, and the arithmetic behind both.

A contact task is driven from outside: a scenario switches to this controller, tells it what to press
with and where to be, and layers a task-space motion on the running force loop. That needs three
things to hold, and each failed quietly before -- a commanded frame that changed nothing, a masked
axis that came back saturated, and a speed cap that capped no speed.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_manipulation.plugins.cartesian_admittance import (
    CartesianAdmittancePlugin,
    _type_from_law,
)
from roqsim_sensors.plugins.force_torque import WrenchReader

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _law(*, stiffness=None, axes=None, wrench=(0.0, 0.0, 0.0), pos=None, mat=None, w_d=None):
    """The law alone, without a world: the question is what it commands for a given reading."""
    plugin = CartesianAdmittancePlugin.__new__(CartesianAdmittancePlugin)
    plugin.M = np.ones(6)
    plugin.D = np.zeros(6)  # no damping, so one step reads the forcing term directly
    plugin.C = np.array(stiffness if stiffness is not None else np.zeros(6), float)
    plugin.w_d = np.array(w_d if w_d is not None else [0, 0, -10, 0, 0, 0], float)
    plugin.axes = np.array(axes if axes is not None else np.ones(6), float)
    plugin._twist = np.zeros(6)
    plugin._uses_stiffness = bool(np.any(plugin.C))
    plugin._goal_pos = None
    plugin._goal_mat = None
    plugin._rest_pos = np.zeros(3)
    plugin._rest_mat = np.eye(3)
    plugin.v_lin, plugin.v_ang = 1e9, 1e9  # clamping is tested on its own
    plugin._ft = WrenchReader(
        name="ft",
        frame="world",
        read=lambda: (np.array(wrench, float), np.zeros(3)),
        measures="environment_on_tool",
    )
    plugin.read_pose = lambda: (
        np.array(pos if pos is not None else np.zeros(3), float),
        np.array(mat if mat is not None else np.eye(3), float),
    )
    return plugin


# -- the superposition: a commanded frame and a commanded wrench are live at once ---------------


def test_a_commanded_target_frame_becomes_the_equilibrium():
    """Before this, `target_frame` reached `_goal_pos`, which only the motion law read -- so under a
    force law the topic accepted messages and changed nothing at all."""
    law = _law(stiffness=[100, 100, 0, 0, 0, 0], pos=[0.0, 0.0, 0.0])
    assert law._deflection()[0] == pytest.approx(0.0)

    law.set_goal(np.array([-0.5, 0.0, 0.0]), None)
    # The tool is now 0.5 m on +x of where it has been told to be.
    assert law._deflection()[0] == pytest.approx(0.5)


def test_a_zero_stiffness_axis_stays_under_force_control_while_the_others_track():
    """The property a contact task needs: press on one axis, track a frame on the others, at once.

    z has no stiffness, so its command must come from the wrench alone and be untouched by a frame
    commanded far away; x has stiffness and must be pulled back toward the frame.
    """
    law = _law(stiffness=[100.0, 100.0, 0.0, 0.0, 0.0, 0.0], wrench=(0.0, 0.0, 0.0))
    z_before = law._wrench_twist(0.01)[2]

    law = _law(stiffness=[100.0, 100.0, 0.0, 0.0, 0.0, 0.0], wrench=(0.0, 0.0, 0.0))
    law.set_goal(np.array([-0.5, 0.0, +10.0]), None)  # absurd z, to catch any leak onto that axis
    twist = law._wrench_twist(0.01)

    assert twist[2] == pytest.approx(z_before), "a stiffness-free axis must ignore the frame"
    assert twist[0] < 0.0, "a stiff axis must be pulled back toward the commanded frame"


def test_rotational_stiffness_opposes_a_rotational_deflection():
    """Holding orientation was expressible only by masking the rotational axes -- an open-loop hold
    that never corrected the drift the DLS solve accumulates."""
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
    law = _law(stiffness=[0, 0, 0, 0, 0, 50.0], mat=turned)
    assert law._deflection()[5] == pytest.approx(np.pi / 2, abs=1e-6)
    assert law._wrench_twist(0.01)[5] < 0.0, "must rotate back toward the equilibrium"


# -- the two latent bugs -------------------------------------------------------------------------


def test_a_masked_axis_does_not_wind_up_behind_the_mask():
    """The mask used to be applied to the RESULT, leaving the integrator free to run to the clamp
    behind it -- so enabling an axis mid-run dumped a saturated velocity into the arm in one step."""
    law = _law(axes=[1, 1, 0, 1, 1, 1], w_d=[0, 0, -50, 0, 0, 0])
    for _ in range(200):
        law._wrench_twist(0.01)
    assert law._twist[2] == pytest.approx(0.0), "a masked axis must carry no integrator state"

    law.axes = np.ones(6)
    assert abs(law._wrench_twist(0.01)[2]) < 1.0, "unmasking must not release a wound-up velocity"


def test_the_translational_clamp_limits_magnitude_and_keeps_direction():
    """Per-axis clipping capped each component at the limit, so it allowed sqrt(3) times it in
    magnitude -- and turned the commanded direction, which for a task-space path points it somewhere
    nobody asked for."""
    law = _law()
    law.v_lin, law.v_ang = 0.1, 1.0
    out = law._clamp(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    assert np.linalg.norm(out[:3]) == pytest.approx(0.1)
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2]), "direction must survive"


def test_the_clamp_leaves_a_twist_inside_the_limit_alone():
    law = _law()
    law.v_lin, law.v_ang = 0.1, 1.0
    inside = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    assert law._clamp(inside) == pytest.approx(inside)


# -- identity ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("law", "stiffness", "expected"),
    [
        ("position", [0] * 6, "cartesian_motion_controller"),
        ("admittance", [0] * 6, "cartesian_force_controller"),
        ("admittance", [100, 100, 0, 0, 0, 0], "cartesian_compliance_controller"),
    ],
)
def test_the_legacy_law_key_derives_a_controller_type(law, stiffness, expected):
    """`law` is the older spelling and worlds still carry it; it must land on the identity that
    behaves the way it always did."""
    assert _type_from_law(law, stiffness) == expected


def test_an_explicit_controller_type_wins_over_the_law_key():
    plugin = CartesianAdmittancePlugin(
        {"controller_type": "cartesian_motion_controller", "law": "admittance"}, entity="arm"
    )
    assert plugin.controller_type == "cartesian_motion_controller"
    assert plugin._needs_ft is False


def test_a_controller_type_names_its_topics():
    plugin = CartesianAdmittancePlugin(
        {"controller_type": "cartesian_force_controller"}, entity="a"
    )
    assert plugin.controller_name == "cartesian_force_controller"
    named = CartesianAdmittancePlugin({"controller_name": "left_arm_force"}, entity="a")
    assert named.controller_name == "left_arm_force"


@pytest.mark.parametrize(("config", "active"), [({}, True), ({"initial_state": "inactive"}, False)])
def test_initial_state_defaults_to_active(config, active):
    """Default active, so a world that never switches behaves as it always has; `inactive` is what
    ros2_control's `spawner --inactive` leaves behind."""
    assert CartesianAdmittancePlugin(config, entity="arm")._active is active


@pytest.mark.parametrize(
    "config",
    [{"controller_type": "nonsense"}, {"initial_state": "paused"}],
)
def test_a_bad_identity_is_refused_with_the_alternatives_named(config):
    errors = CartesianAdmittancePlugin({}, entity="arm").validate_config(config)
    assert errors, f"{config} should not validate"


# -- in a world ----------------------------------------------------------------------------------


def _world(tmp_path, cart=None):
    return load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {"arm_controller": {}},
                        {
                            "force_torque": {"site": "fts_site", "frame": "world"},
                            "name": "ft",
                        },
                        {
                            "cartesian_admittance": {
                                "site": "tool_site",
                                "ft": "ft",
                                **(cart or {}),
                            },
                            "name": "compliance",
                        },
                    ],
                }
            ],
        },
        base_dir=tmp_path,
    )


def test_the_command_endpoints_are_declared_under_the_controller_name(tmp_path):
    """The names FZI's cartesian_controllers use, so a node driving this drives the real one."""
    engine = Engine(_world(tmp_path, {"controller_type": "cartesian_compliance_controller"}))
    engine.setup()
    eps = {e.name: e for e in engine.ctx.interface.all()}

    frame_ep = eps["target_frame"]
    assert frame_ep.direction == "in"
    assert frame_ep.backend["ros2"]["type"] == "geometry_msgs.msg.PoseStamped"
    assert frame_ep.backend["ros2"]["topic"] == "cartesian_compliance_controller/target_frame"

    wrench_ep = eps["target_wrench"]
    assert wrench_ep.direction == "in"
    assert wrench_ep.backend["ros2"]["type"] == "geometry_msgs.msg.WrenchStamped"

    assert eps["current_pose"].direction == "out"


def test_a_commanded_wrench_reaches_the_law(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    ep = next(e for e in engine.ctx.interface.all() if e.name == "target_wrench")

    ep.write(((1.0, 2.0, -8.0), (0.0, 0.0, 0.5)))

    plugin = engine.ctx.blackboard.require("cartesian:ur5e")
    assert plugin.controller_name
    law = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))
    assert law.w_d == pytest.approx([1.0, 2.0, -8.0, 0.0, 0.0, 0.5])


def test_a_commanded_frame_reaches_the_law(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    ep = next(e for e in engine.ctx.interface.all() if e.name == "target_frame")

    ep.write(((0.4, 0.1, 0.3), (1.0, 0.0, 0.0, 0.0)))

    law = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))
    assert law._goal_pos == pytest.approx([0.4, 0.1, 0.3])
    assert law._goal_mat is not None


def test_current_pose_reads_back_as_position_and_quaternion(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    ep = next(e for e in engine.ctx.interface.all() if e.name == "current_pose")

    position, quat = ep.read()
    assert len(position) == 3 and len(quat) == 4
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)


def test_activating_anchors_on_the_arm_state_now(tmp_path):
    """The hand-over instant is only well defined if the controller starts from where the arm IS.
    The equilibrium used to be captured once per episode, so a controller activated later pulled the
    tool back to wherever the episode began."""
    engine = Engine(_world(tmp_path, {"initial_state": "inactive"}))
    engine.setup()
    engine.reset()
    law = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))

    started, _ = law.read_pose()
    # Move the arm while the controller is inactive, the way a trajectory controller would.
    arm = engine.ctx.blackboard.require("arm:ur5e")
    arm.set_targets(arm.joint_names, [0.4] * len(arm.joint_names))
    for _ in range(400):
        engine.step()
    moved, _ = law.read_pose()
    assert np.linalg.norm(moved - started) > 0.05, (
        "the arm must actually have moved for this to mean anything"
    )

    law.set_active(True)
    assert law._rest_pos == pytest.approx(moved, abs=1e-9)
    assert law._twist == pytest.approx(np.zeros(6))
