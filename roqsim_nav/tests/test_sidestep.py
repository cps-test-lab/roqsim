"""The default avoidance model: two movers meeting head-on get past each other.

This is the case the package exists to handle and the one that has no answer without a model at all:
two movers that only *stop* for each other stop nose to nose and stay there. So the assertions are
about the encounter resolving, not about any particular path through it.

Run against the model directly where the property is geometric, and through a world where it is
about the integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_nav.avoidance.sidestep import SidestepModel


def _model(**params) -> SidestepModel:
    model = SidestepModel()
    model.configure(None, params)
    return model


def _pair(model, gap=4.0, speed=1.0):
    """Two agents ``gap`` apart on the x axis, each heading at the other."""
    a = model.add_agent("a", radius=0.3, max_speed=2.0, yields=True, params={})
    b = model.add_agent("b", radius=0.3, max_speed=2.0, yields=True, params={})
    model.submit(a, (-gap / 2, 0.0), (speed, 0.0), (speed, 0.0))
    model.submit(b, (gap / 2, 0.0), (-speed, 0.0), (-speed, 0.0))
    model.solve(0.02)
    return a, b


def test_a_head_on_pair_is_pushed_to_opposite_sides():
    """The whole point: a symmetric encounter must have an asymmetric outcome.

    'Steer away from them' is symmetric and leaves the pair mirroring each other into the collision.
    'Steer to your own right' is not, so the two lateral pushes point opposite ways in the world.
    """
    model = _model()
    a, b = _pair(model)
    lat_a, lat_b = float(model.result(a)[1]), float(model.result(b)[1])
    assert lat_a != pytest.approx(0.0, abs=1e-6), "the encounter produced no sidestep at all"
    assert lat_a * lat_b < 0.0, "both were pushed the same way in the world -- they still collide"


def test_the_side_rule_is_configurable_and_mirrors():
    right = _model(side="right")
    a, _ = _pair(right)
    left = _model(side="left")
    c, _ = _pair(left)
    assert float(right.result(a)[1]) == pytest.approx(-float(left.result(c)[1]))


def test_an_unknown_side_is_refused():
    with pytest.raises(ValueError, match="'right' or 'left'"):
        _model(side="middle")


def test_speed_is_preserved_by_the_sidestep():
    """It changes where a mover is going, not how fast: the follower assumes a constant speed, and
    a push that added to it would read as accelerating out of a conflict."""
    model = _model()
    a, _ = _pair(model, speed=0.8)
    assert float(np.linalg.norm(model.result(a))) == pytest.approx(0.8, abs=1e-9)


def test_agents_that_are_separating_are_left_alone():
    model = _model()
    a = model.add_agent("a", radius=0.3, max_speed=2.0, yields=True, params={})
    model.add_agent("b", radius=0.3, max_speed=2.0, yields=True, params={})
    model.submit(a, (0.0, 0.0), (-1.0, 0.0), (-1.0, 0.0))  # driving away
    model.submit(1, (1.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    model.solve(0.02)
    assert model.result(a) == pytest.approx([-1.0, 0.0])


def test_a_non_yielding_agent_is_never_moved():
    """`yields=False` is the subject of the experiment: this model may not touch it."""
    model = _model()
    a = model.add_agent("a", radius=0.3, max_speed=2.0, yields=False, params={})
    model.add_agent("b", radius=0.3, max_speed=2.0, yields=True, params={})
    model.submit(a, (-1.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    model.submit(1, (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0))
    model.solve(0.02)
    assert model.result(a) == pytest.approx([1.0, 0.0])


def test_a_stopped_mover_is_not_steered():
    """Stopping is the caution probe's decision; overriding it here would undo it."""
    model = _model()
    a = model.add_agent("a", radius=0.3, max_speed=2.0, yields=True, params={})
    model.add_agent("b", radius=0.3, max_speed=2.0, yields=True, params={})
    model.submit(a, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    model.submit(1, (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0))
    model.solve(0.02)
    assert model.result(a) == pytest.approx([0.0, 0.0])


def test_it_is_deterministic():
    """An opponent whose avoidance varied run to run would not be apparatus."""
    first = _model()
    a, _ = _pair(first)
    second = _model()
    c, _ = _pair(second)
    assert np.array_equal(first.result(a), second.result(c))
