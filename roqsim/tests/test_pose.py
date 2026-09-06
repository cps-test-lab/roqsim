# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Reading a ``pose:`` written the way ``SpawnEntity.srv`` states one.

The shape is a ``geometry_msgs/PoseStamped`` because the same pose reaches an entity two ways -- a
world declaring where it starts, and a spawn call placing it during a trial. What these pin is the
handful of values that are *wrong in a way that looks right*, since those are what a single shared
shape exists to prevent.
"""

from __future__ import annotations

import math

import pytest

from roqsim.pose import PoseError, parse_pose, rpy_to_quat, yaw_of

IDENTITY = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _quat(yaw):
    return {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0), "w": math.cos(yaw / 2.0)}


def test_a_pose_reads_as_position_and_quaternion():
    pos, quat = parse_pose({"position": {"x": 1.5, "y": 2.0, "z": 0.25}, "orientation": _quat(0.5)})
    assert pos == [1.5, 2.0, 0.25]
    assert yaw_of(quat) == pytest.approx(0.5)


def test_the_message_can_be_pasted_at_either_level():
    """A caller copying the service request has a ``header``/``pose``; one copying the pose does
    not. Accepting only the inner one would make the outer a silent 'no key' failure."""
    inner = {"position": {"x": 1.0, "y": 2.0}, "orientation": IDENTITY}
    assert parse_pose(inner) == parse_pose({"header": {"frame_id": "world"}, "pose": inner})


def test_an_omitted_z_is_unstated_rather_than_zero():
    """A wheeled model is authored with its base at the origin and its wheels below it, so z=0
    buries it by a wheel radius. The document can say 'unstated'; the message cannot."""
    pos, _ = parse_pose({"position": {"x": 1.0, "y": 2.0}, "orientation": IDENTITY})
    assert pos == [1.0, 2.0, None]


def test_a_stated_zero_z_means_the_floor():
    pos, _ = parse_pose({"position": {"x": 1.0, "y": 2.0, "z": 0.0}, "orientation": IDENTITY})
    assert pos == [1.0, 2.0, 0.0]


def test_an_absent_orientation_is_the_identity():
    """What the message's own defaults give: geometry_msgs/Quaternion declares w=1."""
    _, quat = parse_pose({"position": {"x": 0.0, "y": 0.0}})
    assert quat == [1.0, 0.0, 0.0, 0.0]


def test_a_heading_may_be_written_as_a_yaw():
    """The spelling a person reaches for. Nobody hand-authors a quaternion for a robot facing
    along a wall, and a world is written by hand."""
    _, quat = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": 1.0}})
    assert yaw_of(quat) == pytest.approx(1.0)


def test_the_two_spellings_of_one_rotation_agree():
    """Whichever a document uses, the entity ends up at the same orientation."""
    by_euler = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": 0.75}})[1]
    by_quat = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": _quat(0.75)})[1]
    assert by_euler == pytest.approx(by_quat)


def test_unset_euler_components_are_zero():
    """A yaw alone is a heading, not a rotation with two thirds of it missing."""
    _, quat = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": 0.4}})
    assert quat == pytest.approx(rpy_to_quat(0.0, 0.0, 0.4))


def test_roll_and_pitch_are_read_too():
    _, quat = parse_pose(
        {"position": {"x": 0.0, "y": 0.0}, "orientation": {"roll": 0.1, "pitch": 0.2, "yaw": 0.3}}
    )
    assert quat == pytest.approx(rpy_to_quat(0.1, 0.2, 0.3))


def test_the_two_spellings_cannot_be_mixed():
    """No reading has both applying, so a value carrying each is a document mid-edit."""
    with pytest.raises(PoseError, match="mixes a quaternion"):
        parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": 1.0, "w": 1.0}})


def test_an_unset_w_is_one_not_zero():
    """geometry_msgs/Quaternion declares w=1, and that default is what keeps a partly-written
    quaternion a rotation instead of a zero-length one."""
    _, quat = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {"x": 0.0}})
    assert quat == [1.0, 0.0, 0.0, 0.0]


def test_an_empty_orientation_is_the_identity():
    _, quat = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {}})
    assert quat == [1.0, 0.0, 0.0, 0.0]


def test_a_zero_length_quaternion_is_refused():
    with pytest.raises(PoseError, match="not a rotation"):
        parse_pose(
            {"position": {"x": 0.0, "y": 0.0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 0}}
        )


def test_a_quaternion_is_normalised():
    """MuJoCo reads a free joint's quaternion as a unit one, so a near-unit value would otherwise
    scale the body's orientation."""
    _, quat = parse_pose(
        {"position": {"x": 0.0, "y": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 2.0, "w": 2.0}}
    )
    assert math.sqrt(sum(c * c for c in quat)) == pytest.approx(1.0)
    assert yaw_of(quat) == pytest.approx(math.pi / 2)


def test_a_frame_this_simulator_does_not_know_is_refused():
    """The same answer the spawn service gives: there is no other frame to state a pose in."""
    with pytest.raises(PoseError, match="does not know"):
        parse_pose({"header": {"frame_id": "map"}, "pose": {"position": {"x": 0.0, "y": 0.0}}})


def test_a_missing_coordinate_is_refused():
    with pytest.raises(PoseError, match="'pose.position.y' is required"):
        parse_pose({"position": {"x": 1.0}, "orientation": IDENTITY})


def test_a_stray_key_names_itself():
    with pytest.raises(PoseError, match=r"no key\(s\) \['rpy'\]"):
        parse_pose({"position": {"x": 0.0, "y": 0.0}, "rpy": [0, 0, 0]})


def test_a_stray_orientation_key_names_both_spellings():
    with pytest.raises(PoseError, match="quaternion .x/y/z/w. or roll/pitch/yaw"):
        parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": {"heading": 1.0}})


def test_a_non_numeric_coordinate_is_refused():
    with pytest.raises(PoseError, match="must be a number"):
        parse_pose({"position": {"x": "1.0", "y": 0.0}, "orientation": IDENTITY})


def test_yaw_of_the_identity_is_zero():
    assert yaw_of([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


@pytest.mark.parametrize("yaw", [-3.0, -1.57, 0.0, 0.75, 3.0])
def test_yaw_survives_the_round_trip(yaw):
    _, quat = parse_pose({"position": {"x": 0.0, "y": 0.0}, "orientation": _quat(yaw)})
    assert yaw_of(quat) == pytest.approx(yaw)
