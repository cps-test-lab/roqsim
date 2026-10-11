"""A pose and a twist stated in an entity's frame are placed relative to that entity.

``SpawnEntity`` states ``initial_pose`` in a frame that "must be known to the simulator, e.g. of an
object spawned earlier", and ``EntityState`` states pose and twist in its header's frame. A present
entity's name, exactly as ``GetEntities`` lists it, names that entity's root body; every other frame
is refused with ``INVALID_POSE`` rather than read as the world frame, which would place the entity
somewhere nobody asked for under a ``RESULT_OK``.

The frame entity's pose is read on the physics thread where the write happens, so a frame that
moved between the request and the write is the one the pose is composed with.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import SetEntityState, SpawnEntity

import roqsim_ros_bridge.sim_interfaces as mod
from roqsim_ros_bridge.sim_interfaces import SimInterfacesPlugin

H = math.sqrt(0.5)
YAW_90 = (H, 0.0, 0.0, H)
ROLL_90 = (H, H, 0.0, 0.0)


class _World:
    """Entities by name, each with a world pose the stubbed physics thread reads."""

    def __init__(self):
        self.entities = {}
        self.poses = {}
        # Applied on the physics thread, before the posted command runs.
        self.before_apply = []

    def add(self, name, *, present=True, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        self.entities[name] = SimpleNamespace(
            name=name, body=f"{name}_body", meta={}, present=present
        )
        self.poses[f"{name}_body"] = (pos, quat)

    def names(self, present_only=False):
        return [n for n, e in self.entities.items() if e.present or not present_only]

    def get(self, name):
        return self.entities.get(name)


@pytest.fixture
def world(monkeypatch):
    w = _World()
    w.add("shelf", pos=(2.0, 1.0, 0.0), quat=YAW_90)
    w.add("box")
    w.add("spare", present=False)
    w.written = []
    w.reads = []

    def run_on_physics(ctx, fn, timeout=None):
        for change in w.before_apply:
            change()
        fn(ctx)
        return True

    def read_body(ctx, body):
        w.reads.append(body)
        pos, quat = w.poses[body]
        return {"pos": list(pos), "quat": list(quat), "lin": [0.0] * 3, "ang": [0.0] * 3}

    monkeypatch.setattr(mod, "run_on_physics", run_on_physics)
    monkeypatch.setattr(
        mod,
        "place_body",
        lambda ctx, e, pos, quat, vel=None: w.written.append((e.name, pos, quat, vel)) or True,
    )
    monkeypatch.setattr(mod, "carries_velocity", lambda ctx, e: True)
    monkeypatch.setattr(mod, "set_present", lambda ctx, e, present: setattr(e, "present", present))
    monkeypatch.setattr(SimInterfacesPlugin, "_read_body", staticmethod(read_body))
    return w


def _plugin(world):
    plugin = SimInterfacesPlugin()
    plugin._ctx = SimpleNamespace(entities=world)
    return plugin


def _spawn(world, frame_id, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), uri="spare"):
    req = SpawnEntity.Request(uri=uri)
    req.initial_pose.header.frame_id = frame_id
    p, o = req.initial_pose.pose.position, req.initial_pose.pose.orientation
    p.x, p.y, p.z = pos
    o.w, o.x, o.y, o.z = quat
    return _plugin(world)._spawn_entity(req, SpawnEntity.Response())


def _set(
    world, frame_id, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), lin=(0, 0, 0), ang=(0, 0, 0)
):
    req = SetEntityState.Request(entity="box")
    req.state.header.frame_id = frame_id
    p, o = req.state.pose.position, req.state.pose.orientation
    p.x, p.y, p.z = pos
    o.w, o.x, o.y, o.z = quat
    t = req.state.twist
    t.linear.x, t.linear.y, t.linear.z = map(float, lin)
    t.angular.x, t.angular.y, t.angular.z = map(float, ang)
    return _plugin(world)._set_entity_state(req, SetEntityState.Response())


def _same_rotation(a, b):
    return abs(sum(x * y for x, y in zip(a, b, strict=True))) == pytest.approx(1.0)


def test_spawn_composes_the_pose_with_a_rotated_translated_entity(world):
    """One metre ahead of a shelf at (2, 1) facing +y, turned a further 90 degrees."""
    resp = _spawn(world, "shelf", pos=(1.0, 0.0, 0.5), quat=YAW_90)
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    ((name, pos, quat, _),) = world.written
    assert name == "spare"
    assert pos == pytest.approx((2.0, 2.0, 0.5))
    assert _same_rotation(quat, (0.0, 0.0, 0.0, 1.0))
    assert world.entities["spare"].present


def test_set_state_composes_the_pose_with_a_rotated_translated_entity(world):
    resp = _set(world, "shelf", pos=(1.0, -0.5, 0.25), quat=YAW_90)
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    ((name, pos, quat, _),) = world.written
    assert name == "box"
    assert pos == pytest.approx((2.5, 2.0, 0.25))
    assert _same_rotation(quat, (0.0, 0.0, 0.0, 1.0))


def test_the_frame_is_read_where_the_pose_is_written(world):
    """A frame that moved after the request arrived is the one the pose is composed with."""

    def move_shelf():
        world.poses["shelf_body"] = ((5.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    world.before_apply.append(move_shelf)
    resp = _set(world, "shelf", pos=(1.0, 0.0, 0.0))
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    ((_, pos, _, _),) = world.written
    assert pos == pytest.approx((6.0, 0.0, 0.0))


def test_a_twist_in_an_entity_frame_is_rotated_into_the_world(world):
    """Expressed in the shelf's axes; the shelf's own velocity is not added."""
    resp = _set(world, "shelf", lin=(1.0, 0.0, 0.0), ang=(0.0, 0.0, 0.5))
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    ((_, _, _, vel),) = world.written
    assert vel == pytest.approx((0.0, 1.0, 0.0, 0.0, 0.0, 0.5))


def test_the_angular_part_is_rotated_too(world):
    world.add("tilted", pos=(0.0, 0.0, 1.0), quat=ROLL_90)
    resp = _set(world, "tilted", ang=(0.0, 0.0, 1.0))
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    ((_, _, _, vel),) = world.written
    assert vel == pytest.approx((0.0, 0.0, 0.0, 0.0, -1.0, 0.0))


@pytest.mark.parametrize(
    "frame_id",
    [
        "map",  # a TF frame, not an entity
        "spare",  # an entity, but absent: GetEntities does not list it
        "shelf/base_link",  # a body path below an entity
        "shelf_body",  # the body's own name rather than the entity's
    ],
)
def test_a_frame_that_is_not_a_present_entity_is_refused(world, frame_id):
    resp = _set(world, frame_id, pos=(1.0, 0.0, 0.0))
    assert resp.result.result == SetEntityState.Response.INVALID_POSE
    assert repr(frame_id) in resp.result.error_message
    assert "get_entities" in resp.result.error_message
    assert not world.written

    world.add("other", present=False)
    resp = _spawn(world, frame_id, pos=(1.0, 0.0, 0.0), uri="other")
    assert resp.result.result == SpawnEntity.Response.INVALID_POSE
    assert repr(frame_id) in resp.result.error_message
    assert not world.written
    assert not world.entities["other"].present


def test_an_entity_as_its_own_frame_is_refused(world):
    """The write moves that frame, so the pose would hold of neither its old nor its new place."""
    resp = _set(world, "box", pos=(1.0, 0.0, 0.0))
    assert resp.result.result == SetEntityState.Response.INVALID_POSE
    assert "itself" in resp.result.error_message
    assert not world.written


def test_a_frame_entity_deleted_before_the_write_is_refused(world):
    world.before_apply.append(lambda: setattr(world.entities["shelf"], "present", False))
    resp = _set(world, "shelf", pos=(1.0, 0.0, 0.0))
    assert resp.result.result == SetEntityState.Response.INVALID_POSE
    assert "'shelf'" in resp.result.error_message
    assert not world.written

    world.entities["shelf"].present = True
    resp = _spawn(world, "shelf", pos=(1.0, 0.0, 0.0))
    assert resp.result.result == SpawnEntity.Response.INVALID_POSE
    assert "'shelf'" in resp.result.error_message
    assert not world.written
    assert not world.entities["spare"].present


def test_the_world_frame_reads_no_entity(world):
    resp = _set(world, "world", pos=(1.0, 0.0, 0.0), lin=(1.0, 0.0, 0.0))
    assert resp.result.result == Result.RESULT_OK, resp.result.error_message
    ((_, pos, _, vel),) = world.written
    assert pos == (1.0, 0.0, 0.0)
    assert vel == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert not world.reads
