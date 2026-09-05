"""``SpawnEntity`` honours the pose it is handed, or says why it cannot, in the service's own terms.

The failure this guards is not a crash. A spawn request carries ``initial_pose``, ``uri``, a
namespace and a frame, and a handler that reads only ``name`` answers ``RESULT_OK`` while the entity
stays where the model compiled it -- so a trial records a pose its body was never at, and every
other check passes. That is the class of wrong data the whole control plane is shaped against.

The contract is ``SpawnEntity.srv``'s, not one invented here, and two of its details decide the
tests below: ``initial_pose`` is stated unconditionally, so there is no "unset" for the handler to
detect; and ``Result.msg`` asks an implementation to answer with the service's extended codes where
one fits, rather than the generic ones.

Duck-typed over the request members, so a malformed pose is cheap to state.
"""

from __future__ import annotations

import math

import pytest
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import SpawnEntity

from roqsim_ros_bridge.sim_interfaces import (
    _already_at,
    _pose_of,
    _unsupported_spawn_request,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    # w=1 like geometry_msgs/Quaternion itself, which is why a default request states the identity.
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = w, x, y, z


class _Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position or _Vec()
        self.orientation = orientation if orientation is not None else _Quat()


class _Header:
    def __init__(self, frame_id=""):
        self.frame_id = frame_id


class _Stamped:
    def __init__(self, pose=None, frame_id=""):
        self.pose = pose or _Pose()
        self.header = _Header(frame_id)


class _Request:
    def __init__(
        self,
        name="obstacle_0",
        initial_pose=None,
        uri="",
        resource_string="",
        entity_namespace="",
        allow_renaming=False,
    ):
        self.name = name
        self.initial_pose = initial_pose or _Stamped()
        self.uri = uri
        self.resource_string = resource_string
        self.entity_namespace = entity_namespace
        self.allow_renaming = allow_renaming


def test_a_default_request_states_the_origin():
    """There is no "unset" to detect: geometry_msgs/Quaternion declares w=1.

    The heuristic this replaces read an all-zero quaternion as "no pose given" -- a value no real
    ROS request ever carries, so every plain spawn was read as a request for the origin anyway,
    silently. Stating it as the contract is what makes that visible.
    """
    assert _pose_of(_Request()) == ((0.0, 0.0, 0.0), IDENTITY)


def test_a_stated_pose_is_returned():
    yaw90 = _Quat(w=math.sqrt(0.5), z=math.sqrt(0.5))
    req = _Request(initial_pose=_Stamped(_Pose(_Vec(5.0, -4.0, 0.25), yaw90)))
    pos, quat = _pose_of(req)
    assert pos == (5.0, -4.0, 0.25)
    assert quat == pytest.approx((yaw90.w, 0.0, 0.0, yaw90.z))


def test_a_non_unit_quaternion_is_normalised():
    """MuJoCo reads a free joint's quaternion as a unit one, so a near-unit value would scale it."""
    _, quat = _pose_of(_Request(initial_pose=_Stamped(_Pose(orientation=_Quat(w=2.0)))))
    assert quat == pytest.approx(IDENTITY)


def test_a_zero_length_quaternion_is_reported_not_repaired():
    """The one orientation that cannot be a rotation, and the service has a code that says so."""
    req = _Request(initial_pose=_Stamped(_Pose(orientation=_Quat(w=0.0))))
    assert _pose_of(req) is None


def test_geometry_is_refused_with_the_services_own_code():
    for req in (_Request(uri="model://box"), _Request(resource_string="<sdf/>")):
        code, message = _unsupported_spawn_request(req)
        assert code == SpawnEntity.Response.UNSUPPORTED_FORMAT
        assert "uri" in message


def test_a_namespace_is_refused_rather_than_ignored():
    """A name is settled at compile, so a namespaced request asks for an entity that cannot exist."""
    code, message = _unsupported_spawn_request(_Request(entity_namespace="robot_2"))
    assert code == Result.RESULT_FEATURE_UNSUPPORTED
    assert "entity_namespace" in message


def test_renaming_is_refused_rather_than_ignored():
    """Spawning selects by name, so an existing name is the requirement, not a collision."""
    code, message = _unsupported_spawn_request(_Request(allow_renaming=True))
    assert code == Result.RESULT_FEATURE_UNSUPPORTED
    assert "allow_renaming" in message


def test_an_unknown_frame_is_refused():
    code, message = _unsupported_spawn_request(_Request(initial_pose=_Stamped(frame_id="map")))
    assert code == SpawnEntity.Response.INVALID_POSE
    assert "map" in message


@pytest.mark.parametrize("frame_id", ["", "world"])
def test_the_world_frame_is_accepted_by_either_spelling(frame_id):
    """Empty is the service's default for the world frame; 'world' is the name it gives it."""
    assert _unsupported_spawn_request(_Request(initial_pose=_Stamped(frame_id=frame_id))) is None


def test_an_ordinary_request_is_supported():
    assert _unsupported_spawn_request(_Request()) is None


def test_already_at_accepts_the_pose_the_body_holds():
    state = {"pos": [1.0, 2.0, 0.25], "quat": list(IDENTITY)}
    assert _already_at(state, ((1.0, 2.0, 0.25), IDENTITY))


def test_already_at_accepts_the_negated_quaternion():
    """q and -q are one rotation; a component-wise test would call this a mismatch."""
    state = {"pos": [0.0, 0.0, 0.0], "quat": [-1.0, 0.0, 0.0, 0.0]}
    assert _already_at(state, ((0.0, 0.0, 0.0), IDENTITY))


def test_already_at_rejects_a_different_place():
    state = {"pos": [1.0, 2.0, 0.25], "quat": list(IDENTITY)}
    assert not _already_at(state, ((1.0, 2.5, 0.25), IDENTITY))


def test_already_at_rejects_a_body_it_cannot_read():
    """No reading is not a match: it would let an unplaceable entity report success."""
    assert not _already_at({}, ((0.0, 0.0, 0.0), IDENTITY))
