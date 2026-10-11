"""An absent entity is not found by any service but ``SpawnEntity``.

``simulation_interfaces`` identifies an entity as ``GetEntities`` or ``SpawnEntity`` returns it, and
an absent one is neither listed nor spawned: ``GetSpawnables`` offers it. So naming it is "no match
for input", ``RESULT_NOT_FOUND``, and the message says how to bring it in. Spawning it is how it
appears, so that stays a success.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import (
    DeleteEntity,
    GetEntityState,
    SetEntityState,
    SpawnEntity,
)

import roqsim_ros_bridge.sim_interfaces as mod
from roqsim_ros_bridge.sim_interfaces import SimInterfacesPlugin


@pytest.fixture
def plugin(monkeypatch):
    """A handler over one absent free-jointed entity, recording every physics write."""
    crate = SimpleNamespace(name="crate", kind="object", body="crate", meta={}, present=False)
    entities = SimpleNamespace(
        names=lambda present_only=False: [e.name for e in [crate] if e.present or not present_only],
        get=lambda name: crate if name == "crate" else None,
        all=lambda: [crate],
    )
    plugin = SimInterfacesPlugin()
    plugin._ctx = SimpleNamespace(entities=entities)
    plugin.writes = []
    plugin.crate = crate
    monkeypatch.setattr(mod, "run_on_physics", lambda ctx, fn, timeout=None: (fn(ctx), True)[1])
    monkeypatch.setattr(
        mod, "place_body", lambda ctx, ent, *a, **k: plugin.writes.append(("place", a)) or True
    )
    monkeypatch.setattr(mod, "carries_velocity", lambda ctx, ent: True)
    monkeypatch.setattr(
        mod,
        "set_present",
        lambda ctx, ent, present: (
            plugin.writes.append(("present", present)) or setattr(ent, "present", present)
        ),
    )
    plugin._read_body = lambda ctx, body: {
        "pos": [1.0, 2.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "lin": [0.0, 0.0, 0.0],
        "ang": [0.0, 0.0, 0.0],
    }
    return plugin


def _assert_absent(resp):
    assert resp.result.result == Result.RESULT_NOT_FOUND
    assert "'crate' is absent" in resp.result.error_message
    assert "spawn_entity" in resp.result.error_message


def test_deleting_an_absent_entity_is_not_found(plugin):
    resp = plugin._delete_entity(DeleteEntity.Request(entity="crate"), DeleteEntity.Response())
    _assert_absent(resp)
    assert not plugin.writes


def test_the_state_of_an_absent_entity_is_not_found(plugin):
    resp = plugin._get_entity_state(
        GetEntityState.Request(entity="crate"), GetEntityState.Response()
    )
    _assert_absent(resp)


def test_setting_the_state_of_an_absent_entity_is_not_found_and_places_nothing(plugin):
    req = SetEntityState.Request(entity="crate")
    req.state.pose.position.x = 3.0
    resp = plugin._set_entity_state(req, SetEntityState.Response())
    _assert_absent(resp)
    assert not plugin.writes


def test_spawning_an_absent_entity_still_brings_it_in(plugin):
    resp = plugin._spawn_entity(SpawnEntity.Request(uri="crate"), SpawnEntity.Response())
    assert resp.result.result == Result.RESULT_OK
    assert ("present", True) in plugin.writes


def test_once_spawned_it_is_found(plugin):
    plugin.crate.present = True
    resp = plugin._get_entity_state(
        GetEntityState.Request(entity="crate"), GetEntityState.Response()
    )
    assert resp.result.result == Result.RESULT_OK
    assert resp.state.pose.position.y == 2.0
    resp = plugin._delete_entity(DeleteEntity.Request(entity="crate"), DeleteEntity.Response())
    assert resp.result.result == Result.RESULT_OK
    assert plugin.writes == [("present", False)]
