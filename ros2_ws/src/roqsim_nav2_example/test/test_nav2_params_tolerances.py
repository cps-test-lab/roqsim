# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""No consumer of ``map -> odom`` may be more impatient than the localizer that publishes it.

This is the one nav2 parameter relationship worth a test, because getting it wrong is **silent and
produces a passing trial with wrong data** -- which no amount of reading the results reveals.

``amcl`` broadcasts ``map -> odom`` only from a scan it has processed, so under load that transform
goes stale for as long as amcl is behind. Nav2's default ``transform_tolerance`` is 0.3 s, well
under that. A costmap whose lookup then fails does not raise: the controller's ``isGoalReached()``
transforms the goal from ``map`` into the costmap's frame and does not check, so a stale lookup
collapses the goal to a default-constructed pose at the frame origin -- which is where the robot
started. The first goal of a run reads as an arrival milliseconds after being accepted.

So the rule is a floor, not a preference, and it is asserted rather than left in a comment: a
comment is advice to whoever reads it, and the params file that needed this had none of the four
values set correctly. A new params file added beside this one is covered without anyone remembering.
"""

import pathlib

import pytest
import yaml

PARAMS_DIR = pathlib.Path(__file__).resolve().parent.parent / "params"

#: Frames whose staleness amcl governs. A block transforming only ``odom -> base`` is not covered:
#: that chain is published by the odometry source at its own rate and has nothing to do with amcl,
#: which is why ``collision_monitor`` legitimately keeps a tighter tolerance.
AMCL_GOVERNED_FRAMES = {"map", "odom"}


def _params_files():
    return sorted(p for p in PARAMS_DIR.glob("*.yaml") if "amcl" in p.read_text())


def _blocks(document):
    """``(label, mapping)`` for every block that could carry a ``transform_tolerance``."""
    for node, body in (document or {}).items():
        params = (body or {}).get("ros__parameters") if isinstance(body, dict) else None
        if not isinstance(params, dict):
            continue
        yield node, params
        for key, nested in params.items():
            if isinstance(nested, dict):
                yield f"{node}.{key}", nested


@pytest.mark.parametrize("path", _params_files(), ids=lambda p: p.name)
def test_no_map_frame_consumer_is_more_impatient_than_amcl(path):
    document = yaml.safe_load(path.read_text())
    amcl = document["amcl"]["ros__parameters"]["transform_tolerance"]

    too_tight = [
        (label, block["transform_tolerance"])
        for label, block in _blocks(document)
        if block.get("global_frame") in AMCL_GOVERNED_FRAMES
        and block.get("transform_tolerance") is not None
        and block["transform_tolerance"] < amcl
    ]

    assert not too_tight, (
        f"{path.name}: these transform map <-> odom but give up sooner than amcl ({amcl}s) "
        f"promises its transform stays valid: {too_tight}. A stale lookup here is silent and "
        f"reads as an arrival at the frame origin.")


@pytest.mark.parametrize("path", _params_files(), ids=lambda p: p.name)
def test_every_map_frame_consumer_states_a_tolerance(path):
    """Omitting it is the same bug as setting it too low, and harder to see: nav2's built-in
    default is 0.3 s, so a block that says nothing has already chosen a value below amcl's."""
    document = yaml.safe_load(path.read_text())

    silent = [label for label, block in _blocks(document)
              if block.get("global_frame") in AMCL_GOVERNED_FRAMES
              and "transform_tolerance" not in block]

    assert not silent, (
        f"{path.name}: {silent} transform map <-> odom without stating a transform_tolerance, so "
        f"they inherit nav2's 0.3s default -- below what amcl promises. State it explicitly.")
