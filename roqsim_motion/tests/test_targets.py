"""A world pose as a target: whether it states a heading, in every form a pose is written in."""

from __future__ import annotations

import pytest
from roqsim_motion.targets import make_target

POSITION = {"x": 0.5, "y": 0.0, "z": 0.75}
ORIENTATION = {"yaw": 0.25}


@pytest.mark.parametrize(
    ("spec", "heading"),
    [
        ({"position": POSITION}, False),
        ({"position": POSITION, "orientation": ORIENTATION}, True),
        ({"pose": {"position": POSITION}}, False),
        ({"pose": {"position": POSITION, "orientation": ORIENTATION}}, True),
        (
            {
                "header": {"frame_id": ""},
                "pose": {"position": POSITION, "orientation": ORIENTATION},
            },
            True,
        ),
    ],
)
def test_a_world_pose_states_a_heading_only_with_an_orientation(spec, heading):
    assert make_target(None, None, spec).heading is heading
