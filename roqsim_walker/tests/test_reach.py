"""A walker's hands on two points: the reach pose puts the wrists there, bends only as far as it
must, keeps the legs where they stood, and a walker eases into a hold and out of it."""

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
    rest = {name: IDENTITY for name in JOINT_NAMES}
    local = {name: rest[name] for name in ("pelvis", "spine", "hip_l", "hip_r")}
    over = reach_pose(DEFAULT_SKELETON, ROOT, yaw_to_quat(yaw), local, targets)
    return forward_kinematics(ROOT, yaw, {**rest, **over}), forward_kinematics(ROOT, yaw, rest), over


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
        targets = [np.array([*_ahead(0.0, 0.55, s, z)[0], z]) for s in (0.3, -0.3)]
        bends.append(needed_bend(DEFAULT_SKELETON, ROOT, yaw_to_quat(0.0), local, targets))
    steps = np.abs(np.diff(bends))
    assert bends[0] > bends[-1]
    assert steps.max() < math.radians(2.0), math.degrees(steps.max())
