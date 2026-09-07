"""``SetEntityState`` says WHY it could not place an entity, in the service's own terms.

The failure this guards produced no crash and no clue. A placement plugin compiles welded
scenery by default; a welded body carries no free joint; and the handler places an entity by
writing one. So a world that parks an obstacle out of the way and teleports it in on cue
failed on its first call, in every run, while the world compiled, the entity existed under
the name the trial used, and ``GetEntities`` listed it -- and the only thing said about it
was that a service call failed.

``SpawnEntity`` already answered this case in full. Both doors reach the same check, so both
say the same thing.
"""

from __future__ import annotations

from simulation_interfaces.msg import Result

from roqsim_ros_bridge.sim_interfaces import SimInterfacesPlugin


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = w, x, y, z


class _State:
    def __init__(self, pos):
        self.pose = type("P", (), {"position": _Vec(*pos), "orientation": _Quat()})()


class _Req:
    def __init__(self, entity, pos):
        self.entity = entity
        self.state = _State(pos)


class _Resp:
    result = None


class _Entity:
    body = "dynamic_0"
    meta: dict = {}


def _plugin(*, writable, entity=_Entity(), at=(40.0, 40.0, 0.03)):
    """A handler whose physics write succeeds or refuses, with everything else stubbed."""
    plugin = SimInterfacesPlugin.__new__(SimInterfacesPlugin)
    plugin._ctx = type("Ctx", (), {"entities": type("E", (), {"get": staticmethod(
        lambda name: entity)})()})()
    plugin._write_body = lambda ctx, ent, pos, quat: writable
    # `quat` too: _already_at compares both, and a state missing either is "not there".
    plugin._read_body = lambda ctx, body: {"pos": list(at), "quat": [1.0, 0.0, 0.0, 0.0]}
    return plugin


def _run(plugin, req):
    import roqsim_ros_bridge.sim_interfaces as mod
    original = mod.run_on_physics
    # The command runs inline: this test is about what the handler CONCLUDES, and threading a
    # real physics queue through it would test the queue instead.
    mod.run_on_physics = lambda ctx, fn, timeout=None: (fn(ctx), True)[1]
    try:
        return plugin._set_entity_state(req, _Resp())
    finally:
        mod.run_on_physics = original


def test_a_welded_entity_is_refused_by_naming_the_weld():
    resp = _run(_plugin(writable=False), _Req("dynamic_0", (1.0, 2.0, 0.03)))
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    # The name, the cause and the fix -- the three things the campaign log had none of.
    assert "dynamic_0" in resp.result.error_message
    assert "free joint" in resp.result.error_message
    assert "free: true" in resp.result.error_message


def test_a_welded_entity_asked_for_the_pose_it_holds_succeeds():
    """Only a MOVE is what a weld refuses. The caller asking where it already is got what it
    asked for, and answering FAILED there would refuse a world that states a pose twice."""
    resp = _run(_plugin(writable=False, at=(40.0, 40.0, 0.03)),
                _Req("dynamic_0", (40.0, 40.0, 0.03)))
    assert resp.result.result == Result.RESULT_OK


def test_a_free_entity_is_placed():
    resp = _run(_plugin(writable=True), _Req("dynamic_0", (1.0, 2.0, 0.03)))
    assert resp.result.result == Result.RESULT_OK


def test_an_unknown_entity_is_still_not_found():
    plugin = _plugin(writable=True, entity=None)
    resp = _run(plugin, _Req("nope", (1.0, 2.0, 0.03)))
    assert resp.result.result == Result.RESULT_NOT_FOUND
    assert "nope" in resp.result.error_message
