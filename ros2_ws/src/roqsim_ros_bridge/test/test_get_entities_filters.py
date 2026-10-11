"""``GetEntities`` applies the filters ``EntityFilters.msg`` defines, in its own terms.

``filter`` is a POSIX Extended regular expression, which Python's ``re`` reads differently in
places -- ``[[:digit:]]`` is a set of punctuation and letters followed by a literal ``]`` -- so a
pattern a ROS caller wrote for any other simulator would silently match nothing here. ``bounds`` of
``TYPE_SPHERE`` is the one bounds type every simulator must support; it is tested against the
entities' geometry in a compiled model, not against their origins.
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import pytest
from geometry_msgs.msg import Vector3
from simulation_interfaces.msg import Bounds, EntityFilters, Result
from simulation_interfaces.srv import GetEntities

import roqsim_ros_bridge.sim_interfaces as mod
from roqsim_ros_bridge.sim_interfaces import SimInterfacesPlugin, _posix_ere

# Each entity's body origin is at x = 0, and its geometry is offset along x, so a test on origins
# and a test on geometry disagree.
_XML = """
<mujoco>
  <worldbody>
    <body name="near_b" pos="0 0 0"><geom type="sphere" size="0.5" pos="3 0 0"/></body>
    <body name="far_b" pos="0 0 0"><geom type="box" size="0.5 0.5 0.5" pos="10 0 0"/></body>
    <body name="parent_b" pos="0 5 0">
      <body name="child_b" pos="0 0 0"><geom type="capsule" size="0.1 0.4" pos="3 0 0"/></body>
    </body>
  </worldbody>
</mujoco>
"""
_ENTITIES = {"near": "near_b", "far": "far_b", "arm": "parent_b", "marker": None}


def _plugin(names=("near", "far", "arm", "marker")):
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    entities = SimpleNamespace(
        names=lambda present_only=False: list(names),
        get=lambda name: SimpleNamespace(body=_ENTITIES[name]),
    )
    plugin = SimInterfacesPlugin()
    plugin._ctx = SimpleNamespace(entities=entities, model=model, data=data)
    return plugin


@pytest.fixture(autouse=True)
def _inline_physics(monkeypatch):
    monkeypatch.setattr(mod, "run_on_physics", lambda ctx, fn, timeout=None: (fn(ctx), True)[1])


def _get(filters, names=("near", "far", "arm", "marker")):
    req = GetEntities.Request(filters=filters)
    return _plugin(names)._get_entities(req, GetEntities.Response())


def _sphere(center, radius, n_points=2):
    points = [Vector3(x=center[0], y=center[1], z=center[2]), Vector3(x=radius)]
    return Bounds(type=Bounds.TYPE_SPHERE, points=points[:n_points])


# -- filter: POSIX Extended ------------------------------------------------------------------


def test_a_posix_character_class_matches_what_posix_says():
    resp = _get(EntityFilters(filter="^obstacle_[[:digit:]]+$"), ["obstacle_0", "obstacle_x", "a"])
    assert resp.result.result == Result.RESULT_OK
    assert list(resp.entities) == ["obstacle_0"]


def test_negated_and_combined_classes():
    names = ["box_A", "box_b", "box_1"]
    assert list(_get(EntityFilters(filter="box_[^[:lower:][:digit:]]"), names).entities) == [
        "box_A"
    ]
    assert list(_get(EntityFilters(filter="_[[:alnum:]]$"), names).entities) == names


def test_a_backslash_in_brackets_is_literal():
    assert _posix_ere(r"[\d]").search("\\")
    assert _posix_ere(r"[\d]").search("d")
    assert not _posix_ere(r"[\d]").search("5")


def test_the_match_is_unanchored_and_dollar_ends_the_name():
    assert _posix_ere("stacle").search("obstacle_0")
    assert not _posix_ere("a$").search("a\n")
    assert _posix_ere("a.b").search("a\nb")


def test_an_escaped_special_character_is_literal():
    assert list(_get(EntityFilters(filter=r"a\.b"), ["a.b", "axb"]).entities) == ["a.b"]


@pytest.mark.parametrize("pattern", [r"\d", r"(a)\1", "(?i)abc", "[[:digits:]]", "[z-a]", "[a"])
def test_what_is_not_a_posix_ere_is_refused(pattern):
    resp = _get(EntityFilters(filter=pattern), ["a"])
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "'filter'" in resp.result.error_message
    assert "POSIX Extended" in resp.result.error_message


# -- bounds: TYPE_SPHERE --------------------------------------------------------------------


def test_a_sphere_keeps_entities_whose_geometry_overlaps_it():
    # The sphere around (3, 0, 0) reaches near's geom; no entity origin lies inside it.
    resp = _get(EntityFilters(bounds=_sphere((3.0, 0.0, 0.0), 0.2)))
    assert resp.result.result == Result.RESULT_OK
    assert list(resp.entities) == ["near"]


def test_a_sphere_sees_a_geom_on_a_child_body():
    resp = _get(EntityFilters(bounds=_sphere((3.0, 5.0, 0.0), 0.05)))
    assert list(resp.entities) == ["arm"]


def test_a_sphere_touching_nothing_keeps_nothing():
    resp = _get(EntityFilters(bounds=_sphere((6.0, 0.0, 0.0), 1.0)))
    assert resp.result.result == Result.RESULT_OK
    assert list(resp.entities) == []


def test_a_sphere_and_a_name_filter_apply_together():
    big = _sphere((5.0, 2.0, 0.0), 10.0)
    assert list(_get(EntityFilters(bounds=big)).entities) == ["near", "far", "arm"]
    assert list(_get(EntityFilters(filter="^(near|arm)$", bounds=big)).entities) == [
        "near",
        "arm",
    ]


def test_empty_bounds_filter_nothing():
    resp = _get(EntityFilters(bounds=Bounds(type=Bounds.TYPE_EMPTY)))
    assert list(resp.entities) == ["near", "far", "arm", "marker"]


@pytest.mark.parametrize(
    "bounds",
    [
        _sphere((0.0, 0.0, 0.0), 1.0, n_points=1),
        _sphere((0.0, 0.0, 0.0), -1.0),
        Bounds(type=Bounds.TYPE_EMPTY, points=[Vector3()]),
    ],
)
def test_malformed_bounds_are_refused(bounds):
    resp = _get(EntityFilters(bounds=bounds))
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "'bounds'" in resp.result.error_message


@pytest.mark.parametrize("kind", [Bounds.TYPE_BOX, Bounds.TYPE_CONVEX_HULL])
def test_box_and_convex_hull_are_unsupported(kind):
    resp = _get(EntityFilters(bounds=Bounds(type=kind, points=[Vector3(), Vector3(x=1.0)])))
    assert resp.result.result == Result.RESULT_FEATURE_UNSUPPORTED
    assert "'bounds'" in resp.result.error_message
