"""A walker's hands on points: the reach pose puts the wrists there, one hand or both, bends only
as far as the reaching hands must, and keeps the legs where they stood."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roqsim.pose import yaw_to_quat
from roqsim_walker.humanoid import DEFAULT_SKELETON, JOINT_NAMES, forward_kinematics
from roqsim_walker.reach import reach_pose

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
ROOT = np.array([1.0, 2.0, DEFAULT_SKELETON.root_height])


def _pose(yaw, targets):
    """The reach pose for ``targets`` (left, right; either may be None) over the rest pose."""
    rest = {name: IDENTITY for name in JOINT_NAMES}
    local = {name: rest[name] for name in ("pelvis", "spine", "hip_l", "hip_r")}
    left, right = targets
    over = reach_pose(DEFAULT_SKELETON, ROOT, yaw_to_quat(yaw), local, {"l": left, "r": right})
    posed = forward_kinematics(ROOT, yaw, {**rest, **over})
    return posed, forward_kinematics(ROOT, yaw, rest), over


def _ahead(yaw, x, y, z):
    c, s = math.cos(yaw), math.sin(yaw)
    return ROOT[:2] + np.array([c * x - s * y, s * x + c * y]), z


@pytest.mark.parametrize("height", [1.15, 0.95, 0.75])
@pytest.mark.parametrize("yaw", [0.0, 2.0])
def test_the_wrists_reach_the_points(yaw, height):
    targets = []
    for side in (0.3, -0.3):
        xy, z = _ahead(yaw, 0.5, side, height)
        targets.append(np.array([xy[0], xy[1], z]))
    poses, rest, _ = _pose(yaw, targets)
    assert np.linalg.norm(poses["wrist_l"][0] - targets[0]) < 0.005
    assert np.linalg.norm(poses["wrist_r"][0] - targets[1]) < 0.005
    # The legs stand as they did, near enough: the hips swing back a little as the body bends at them.
    for name in ("knee_l", "ankle_l", "knee_r", "ankle_r"):
        assert np.linalg.norm(poses[name][0] - rest[name][0]) < 0.05, name


def test_the_body_bends_only_as_far_as_the_points_need():
    near = [np.array([*_ahead(0.0, 0.35, s, 1.15)[0], 1.15]) for s in (0.3, -0.3)]
    low = [np.array([*_ahead(0.0, 0.5, s, 0.75)[0], 0.75]) for s in (0.3, -0.3)]
    *_, upright = _pose(0.0, near)
    *_, bent = _pose(0.0, low)
    assert np.allclose(upright["pelvis"], IDENTITY) and np.allclose(upright["spine"], IDENTITY)
    assert not np.allclose(bent["pelvis"], IDENTITY)


def test_a_point_out_of_reach_is_reached_for():
    far = [np.array([*_ahead(0.0, 1.5, s, 1.2)[0], 1.2]) for s in (0.3, -0.3)]
    poses, rest, _ = _pose(0.0, far)
    # Arms out toward it, as far as they go.
    assert poses["wrist_l"][0][0] > rest["wrist_l"][0][0] + 0.4


def test_the_bend_follows_a_point_coming_up_smoothly():
    from roqsim_walker.reach import needed_bend

    local = {name: IDENTITY for name in ("pelvis", "spine", "hip_l", "hip_r")}
    bends = []
    for z in np.linspace(0.70, 1.15, 46):  # a box lifted 45 cm, a centimetre at a time
        left, right = (np.array([*_ahead(0.0, 0.55, s, z)[0], z]) for s in (0.3, -0.3))
        targets = {"l": left, "r": right}
        bends.append(needed_bend(DEFAULT_SKELETON, ROOT, yaw_to_quat(0.0), local, targets))
    steps = np.abs(np.diff(bends))
    assert bends[0] > bends[-1]
    assert steps.max() < math.radians(2.0), math.degrees(steps.max())


def test_one_hand_reaches_and_the_other_arm_is_left_to_the_gait():
    point = np.array([*_ahead(0.0, 0.5, -0.3, 0.8)[0], 0.8])
    poses, _, over = _pose(0.0, (None, point))
    assert np.linalg.norm(poses["wrist_r"][0] - point) < 0.005
    assert not {"shoulder_l", "elbow_l", "wrist_l"} & set(over)
    assert not np.allclose(over["pelvis"], IDENTITY)


def test_only_the_reaching_hands_bend_the_body():
    from roqsim_walker.reach import needed_bend

    local = {name: IDENTITY for name in ("pelvis", "spine", "hip_l", "hip_r")}
    low = np.array([*_ahead(0.0, 0.5, 0.3, 0.7)[0], 0.7])
    near = np.array([*_ahead(0.0, 0.35, -0.3, 1.15)[0], 1.15])
    heading = yaw_to_quat(0.0)

    def bend(targets):
        return needed_bend(DEFAULT_SKELETON, ROOT, heading, local, targets)

    assert bend({"r": near}) == 0.0
    assert bend({"l": None, "r": None}) == 0.0
    assert bend({"l": low, "r": near}) == bend({"l": low}) > 0.0


def test_an_arm_it_does_not_have_is_refused():
    local = {name: IDENTITY for name in ("pelvis", "spine", "hip_l", "hip_r")}
    with pytest.raises(ValueError, match="no arm"):
        reach_pose(DEFAULT_SKELETON, ROOT, yaw_to_quat(0.0), local, {"x": np.zeros(3)})
