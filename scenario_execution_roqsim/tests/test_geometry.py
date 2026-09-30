# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""How far, and whether inside: the arithmetic of ``entity_near`` and ``entity_in_region``."""

from __future__ import annotations

import math

import pytest

from scenario_execution_roqsim.geometry import (
    DISTANCE_MODES,
    Region,
    RegionError,
    point_of,
    separation,
)


@pytest.mark.parametrize(
    ("mode", "a", "b", "expected"),
    [
        ("planar", (0, 0, 0), (3, 4, 0), 5.0),
        ("planar", (0, 0, 0), (3, 4, 12), 5.0),  # z is ignored
        ("spatial", (0, 0, 0), (3, 4, 12), 13.0),
        ("spatial", (1, 1, 1), (1, 1, 1), 0.0),
    ],
)
def test_separation(mode, a, b, expected):
    assert math.isclose(separation(mode, a, b), expected)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="planar, spatial"):
        separation("x", (0, 0, 0), (1, 0, 0))
    assert DISTANCE_MODES == ("planar", "spatial")


def test_a_position_argument_is_read_in_metres_and_defaults_to_zero():
    assert point_of({"x": 1.5, "y": -2.0}) == (1.5, -2.0, 0.0)
    assert point_of(None) == (0.0, 0.0, 0.0)


BOX = Region.from_points([(2, 3, 0), (0, 0, 0)])  # corners in any order
# An L: the notch at x > 1, y > 1 is outside.
L_SHAPE = Region.from_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)])


@pytest.mark.parametrize(
    ("region", "point", "inside"),
    [
        (BOX, (1, 1, 0), True),
        (BOX, (1, 1, 7), True),  # z is ignored
        (BOX, (0, 3, 0), True),  # a corner is on the boundary
        (BOX, (2.01, 1, 0), False),
        (BOX, (-0.01, 1, 0), False),
        (L_SHAPE, (0.5, 1.5, 0), True),
        (L_SHAPE, (1.5, 0.5, 0), True),
        (L_SHAPE, (1.5, 1.5, 0), False),  # the notch
        (L_SHAPE, (1.0, 1.5, 0), True),  # on an inner edge
        (L_SHAPE, (1.0, 1.0, 0), True),  # the inner corner
        (L_SHAPE, (3, 0.5, 0), False),
    ],
)
def test_contains(region, point, inside):
    assert region.contains(point) is inside


def test_two_points_are_a_box_and_more_a_polygon():
    assert BOX.kind == "box" and BOX.describe() == "box x 0..2, y 0..3"
    assert L_SHAPE.kind == "polygon" and L_SHAPE.describe() == "6-gon"


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([], "got 0"),
        ([(1, 1, 0)], "got 1"),
        ([(0, 0, 0), (0, 2, 0)], "must differ in both x and y"),
        ([(0, 0, 0), (1, 1, 0), (2, 2, 0)], "no area"),
    ],
)
def test_points_that_describe_no_region_are_refused(points, message):
    with pytest.raises(RegionError, match=message):
        Region.from_points(points)
