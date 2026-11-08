"""The hinged door: a machine whose leaf hangs on a hinge joint driven by a position actuator, opened
and closed by its programs and live-controllable. It unifies a passive door (holds its ``open``
angle) and an automatic door (exposes ROS I/O), and locates the hinge from the opening centre +
``hinge_side``."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim import entity_pose
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError


def _plugin_endpoints(engine):
    """The endpoints the door declares; core adds a pose endpoint per entity (owner :data:`roqsim.entity_pose.OWNER`)."""
    return [e for e in engine.ctx.interface.all() if e.owner != entity_pose.OWNER]


def _door(tmp_path, extra=None):
    plugins = [{"door": dict(extra or {}), "name": "door"}]
    return load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path)


def _built(tmp_path, extra=None):
    engine = Engine(_door(tmp_path, extra))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _id(engine, objtype, name):
    return mujoco.mj_name2id(engine.ctx.model, objtype, name)


def _hinge_angle(engine):
    jid = _id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    return float(engine.ctx.data.qpos[engine.ctx.model.jnt_qposadr[jid]])


def test_builds_hinge_actuator_and_leaf(tmp_path):
    engine = _built(tmp_path)
    assert _id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge") >= 0
    assert _id(engine, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_pos") >= 0
    assert _id(engine, mujoco.mjtObj.mjOBJ_GEOM, "leaf") >= 0
    assert engine.ctx.entities.get("door").kind == "door"


def test_leaf_sized_to_opening(tmp_path):
    # Leaf spans the width and is inset by the floor gap top and bottom: half-height = (2.1 - 2*0.01)/2.
    engine = _built(tmp_path, {"width": 1.2, "height": 2.1, "thickness": 0.05, "floor_gap": 0.01})
    gid = _id(engine, mujoco.mjtObj.mjOBJ_GEOM, "leaf")
    assert np.allclose(engine.ctx.model.geom_size[gid], [0.6, 0.025, 1.04])


def test_command_opens_the_door(tmp_path):
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.0})
    assert abs(_hinge_angle(engine)) < math.radians(2)  # starts closed
    engine.ctx.blackboard.require("machine:door").start("open")
    for _ in range(3000):
        engine.step()
    assert math.degrees(_hinge_angle(engine)) > 88  # reached ~fully open


def test_swing_sign_sets_open_direction(tmp_path):
    engine = _built(tmp_path, {"swing": -1, "max_angle": 90})
    engine.ctx.blackboard.require("machine:door").start("open")
    for _ in range(3000):
        engine.step()
    assert math.degrees(_hinge_angle(engine)) < -88  # opens to the other side


def test_hinge_side_places_leaf(tmp_path):
    # 'left' hinge -> leaf centre at +width/2 in the leaf body; 'right' -> -width/2.
    left = _built(tmp_path, {"hinge_side": "left", "width": 0.9})
    right = _built(tmp_path, {"hinge_side": "right", "width": 0.9})
    gl = left.ctx.model.geom_pos[_id(left, mujoco.mjtObj.mjOBJ_GEOM, "leaf")][0]
    gr = right.ctx.model.geom_pos[_id(right, mujoco.mjtObj.mjOBJ_GEOM, "leaf")][0]
    assert gl > 0 and gr < 0


def test_passive_door_holds_open_angle(tmp_path):
    # A half-open passive door: no ROS surface, and it stays put around its target.
    engine = _built(tmp_path, {"controllable": False, "open": 0.5, "max_angle": 90})
    for _ in range(2000):
        engine.step()
    assert 40 < math.degrees(_hinge_angle(engine)) < 50
    assert {e.name for e in _plugin_endpoints(engine)} == {"start", "status", "cancel"}
    # The openness reader is there for a passive door too.
    openness, _rate = engine.ctx.blackboard.require("machine:door:openness")()
    assert openness == pytest.approx(0.5, abs=0.03)
    # No ROS surface, but a scenario in the same process can still open it.
    door = engine.ctx.blackboard.require("machine:door")
    _finish(engine, door, door.start("open"))
    assert math.degrees(_hinge_angle(engine)) > 88


def test_controllable_declares_endpoints(tmp_path):
    engine = _built(tmp_path, {"controllable": True, "namespace": "foyer"})
    eps = {e.name: e for e in _plugin_endpoints(engine)}
    assert set(eps) == {"cmd", "state", "door", "start", "status", "cancel"}
    assert eps["cmd"].backend["ros2"]["type"] == "std_msgs.msg.Float64"
    assert eps["state"].direction == "out"
    action = eps["door"].backend["ros2"]
    assert action["action"] == "control_msgs.action.GripperCommand"
    # The action handler reads the door's own state, not a gripper's.
    assert action["state_key"] == "machine:door:openness"
    assert callable(engine.ctx.blackboard.get("machine:door:openness"))


def test_the_door_action_gives_a_slow_door_time_for_its_swing(tmp_path):
    """The action handler aborts a goal still moving when its time runs out; a door at 0.3 rad/s
    takes 7 s for its 120 degrees, so it states its own bound: the swing, a stall, and 5 s."""
    engine = _built(tmp_path, {"speed": 0.3, "stall_timeout": 3.0})
    action = engine.ctx.interface.find("door", "door").backend["ros2"]
    assert action["timeout"] == pytest.approx(math.radians(120) / 0.3 + 3.0 + 5.0)


def test_the_door_action_states_the_doors_arrival_band_and_an_abort_at_its_timeout(tmp_path):
    """The handler asks the producer: ``reached_goal`` within the band the door's programs end in,
    so ROS and the door's status agree, and an abort past the door's own bound, where an arm's
    gripper keeps its result."""
    engine = _built(tmp_path, {"speed": 0.3})
    action = engine.ctx.interface.find("door", "door").backend["ros2"]
    assert action["reach_tolerance"] == 0.02
    assert action["on_timeout"] == "abort"
    # The band is the one its programs end in (test_a_leaf_stopped_just_short_of_its_openness_...).
    assert action["reach_tolerance"] == _door_plugin(engine)._reach_tol


def test_mesh_leaf_model_loads_and_swings(tmp_path):
    # The textured leaf model (white panel + chrome handle) hangs on the hinge and opens on command.
    engine = _built(tmp_path, {"model": "door", "width": 0.9, "height": 2.0})
    assert _id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge") >= 0
    # panel + handle geoms come in under the leaf_ prefix.
    assert _id(engine, mujoco.mjtObj.mjOBJ_MESH, "leaf_door_panel") >= 0
    engine.ctx.blackboard.require("machine:door").start("open")
    for _ in range(3000):
        engine.step()
    assert math.degrees(_hinge_angle(engine)) > 85


def test_frame_welded_by_default_and_optional(tmp_path):
    # A static frame is welded around the opening by default; the leaf still hinges inside it.
    with_frame = _built(tmp_path, {"model": "door"})
    assert _id(with_frame, mujoco.mjtObj.mjOBJ_MESH, "frame_door_frame") >= 0
    fbody = _id(with_frame, mujoco.mjtObj.mjOBJ_BODY, "frame_frame")
    assert fbody >= 0 and with_frame.ctx.model.body_jntnum[fbody] == 0  # static (no joint)
    # frame: false leaves a bare opening.
    without = _built(tmp_path, {"model": "door", "frame": False})
    assert _id(without, mujoco.mjtObj.mjOBJ_MESH, "frame_door_frame") < 0


@pytest.mark.parametrize(
    ("config", "expect"),
    [
        ({"frame_model": "no_such_frame"}, "frame model 'no_such_frame' could not be resolved"),
        (
            {"frame_model": "no_such_frame", "leaf": False},
            "frame model 'no_such_frame' could not be resolved",
        ),
        ({"model": "no_such_leaf"}, "leaf model 'no_such_leaf' could not be resolved"),
    ],
)
def test_a_model_that_does_not_resolve_is_a_plugin_error(tmp_path, config, expect):
    """A misspelt frame or leaf model is refused as bad input, never left out: a missing frame would
    leave a bare opening, and a leafless door would register on a casing body that is not there."""
    with pytest.raises(PluginError, match=expect):
        _built(tmp_path, config)


def test_unresolved_models_are_config_errors_reported_together(tmp_path):
    """Validation, not build, refuses them, so `roqsim check` lists both with the world's other
    errors instead of the run stopping at whichever model build() meets first."""
    with pytest.raises(PluginError) as err:
        Engine(_door(tmp_path, {"frame_model": "no_such_frame", "model": "no_such_leaf"}))
    assert "plugin config validation failed" in str(err.value)
    assert "frame model 'no_such_frame' could not be resolved" in str(err.value)
    assert "leaf model 'no_such_leaf' could not be resolved" in str(err.value)


def _body_rgba(engine, body_name):
    """The rgba every geom of ``body_name`` renders with, via its material (or its own rgba).

    Goes through the geoms rather than material names because the engine dedups identical materials:
    once the casing and the leaf share a colour they also share one material, and the name a given
    model contributed may be the one that was merged away.
    """
    m = engine.ctx.model
    bid = _id(engine, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0, body_name
    out = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != bid:
            continue
        matid = m.geom_matid[gid]
        out.append(np.asarray(m.mat_rgba[matid] if matid >= 0 else m.geom_rgba[gid], dtype=float))
    assert out, body_name
    return out


def test_color_repaints_leaf_and_casing_but_not_decoration(tmp_path):
    # `color` paints the leaf panel and -- by default -- the casing with it. The chrome handle is
    # non-colliding decoration by this library's convention, so it keeps its own finish.
    grey = [0.62, 0.62, 0.64, 1.0]
    engine = _built(tmp_path, {"model": "door", "color": grey})
    panel, handle = _body_rgba(engine, "leaf_leaf")  # the mesh leaf body: panel geom, then handle
    assert np.allclose(panel, grey)
    assert not np.allclose(handle, grey)
    assert all(np.allclose(c, grey) for c in _body_rgba(engine, "frame_frame"))


def test_frame_color_paints_the_casing_alone(tmp_path):
    # What a glazed door needs: recoloured trim, untouched pane -- painting it would opaque the glass.
    grey = [0.62, 0.62, 0.64, 1.0]
    engine = _built(tmp_path, {"model": "door_glass", "frame_color": grey})
    assert all(np.allclose(c, grey) for c in _body_rgba(engine, "frame_frame"))
    assert min(c[3] for c in _body_rgba(engine, "leaf_leaf")) < 1.0  # the pane is still translucent


def test_color_applies_to_a_box_leaf(tmp_path):
    grey = [0.62, 0.62, 0.64, 1.0]
    engine = _built(tmp_path, {"color": grey[:3]})  # alpha optional
    assert all(np.allclose(c, grey) for c in _body_rgba(engine, "door_leaf"))


def test_leafless_door_is_a_cased_opening(tmp_path):
    # `leaf: false`: the casing is welded, nothing hangs in it, and no hinge/actuator DOF is added.
    engine = _built(tmp_path, {"leaf": False, "width": 1.5})
    assert _id(engine, mujoco.mjtObj.mjOBJ_MESH, "frame_door_frame") >= 0  # casing is there
    assert _id(engine, mujoco.mjtObj.mjOBJ_GEOM, "leaf") < 0  # no box leaf
    assert _id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge") < 0  # no DOF
    assert _id(engine, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_pos") < 0
    # still a door entity, so anything enumerating the building's doors finds it
    ent = engine.ctx.entities.get("door")
    assert ent.kind == "door" and ent.meta["leaf"] is False
    assert _plugin_endpoints(engine) == []  # nothing to command
    assert engine.ctx.blackboard.get("machine:door") is None
    for _ in range(50):
        engine.step()  # the tick hooks stay quiet


def test_leafless_door_keeps_its_casing_colour(tmp_path):
    grey = [0.62, 0.62, 0.64, 1.0]
    engine = _built(tmp_path, {"leaf": False, "frame_color": grey})
    assert all(np.allclose(c, grey) for c in _body_rgba(engine, "frame_frame"))


def test_leafless_without_frame_is_refused(tmp_path):
    from roqsim_assets.plugins.door import DoorPlugin

    bad = {"leaf": False, "frame": False}
    assert any("place nothing" in e for e in DoorPlugin(bad).validate_config(bad))


def test_blocked_door_pushes_gently_and_gives_up(tmp_path):
    # A 30 kg box parked in the swing arc: the door must never exceed its torque cap, and after
    # stalling it gives up (stops pressing) short of open.
    plugins = [
        {
            "door": {
                "width": 0.9,
                "height": 2.0,
                "hinge_side": "left",
                "max_torque": 15.0,
                "stall_timeout": 2.0,
            },
            "name": "door",
        },
        {
            # A parametric box, not a modelled prop: this test needs an obstacle of a known size in
            # the leaf's swing, and stating the size here beats inheriting it from some asset's
            # dimensions (which a mesh swap could then quietly change).
            "box": {
                "prefix": "obs_",
                "pose": {"position": {"x": 0.45, "y": 0.5}},
                "size": [0.45, 0.30, 0.45],
            },
            "name": "obs",
        },
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    aid = _id(engine, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_pos")
    door = engine.ctx.blackboard.require("machine:door")
    seq = door.start("open")
    peak = 0.0
    for _ in range(4000):
        engine.step()
        peak = max(peak, abs(float(engine.ctx.data.actuator_force[aid])))
    assert peak <= 15.0 + 1e-6  # never pushed harder than the cap
    status = door.status()
    # Blocked short of open, it gave up: the program's own failure, and the run went on.
    assert status.seq == seq and status.finished and status.error.startswith("stalled at openness")
    openness = engine.ctx.blackboard.get("machine:door:openness")()[0]
    assert openness < 0.9
    assert door.value("hinge_pos") == pytest.approx(openness * math.radians(120), abs=1e-3)
    stalled = [e for e in engine.ctx.events.all() if e["event"] == "stalled"]
    assert len(stalled) == 1 and stalled[0]["openness"] == pytest.approx(openness, abs=0.01)


def test_validate_config_flags_bad_input(tmp_path):
    from roqsim_assets.plugins.door import DoorPlugin

    bad = {"width": -1, "open": 2.0, "hinge_side": "up", "swing": 0}
    errors = DoorPlugin(bad).validate_config(bad)
    assert any("width" in e for e in errors)
    assert any("open" in e for e in errors)
    assert any("hinge_side" in e for e in errors)
    assert any("swing" in e for e in errors)


def _finish(engine, door, seq, limit=20.0):
    t0 = engine.ctx.sim_time
    while not (door.status().seq == seq and door.status().finished):
        assert engine.ctx.sim_time - t0 < limit, door.status()
        engine.step()
    return door.status()


def test_open_and_close_ramp_at_the_stated_speed_and_end_when_the_leaf_first_arrives(tmp_path):
    engine = _built(tmp_path, {"max_angle": 90, "speed": 0.5})
    door = engine.ctx.blackboard.require("machine:door")
    openness = engine.ctx.blackboard.require("machine:door:openness")
    t0 = engine.ctx.sim_time
    seq = door.start("open")
    there = None
    while not (door.status().seq == seq and door.status().finished):
        engine.step()
        if there is None and openness()[0] >= 0.98:
            there = engine.ctx.sim_time - t0
    assert door.status().error == ""
    # 90 degrees at 0.5 rad/s is 3.14 s; the leaf trails the ramp by about kv * speed / kp.
    assert 3.1 < there < 4.2
    assert engine.ctx.sim_time - t0 - there < 0.01, "it ends where the leaf first arrives"
    events = [e["event"] for e in engine.ctx.events.all() if e["source"] == "door"]
    assert events == ["program_start", "program_end"]
    assert _finish(engine, door, door.start("close")).error == ""
    assert abs(math.degrees(_hinge_angle(engine))) < 2


def test_a_start_while_it_moves_is_refused_and_the_command_port_takes_over(tmp_path):
    engine = _built(tmp_path, {"max_angle": 90, "speed": 0.5})
    door = engine.ctx.blackboard.require("machine:door")
    first = door.start("open")
    for _ in range(100):
        engine.step()
    with pytest.raises(Exception, match="busy running 'open'"):
        door.start("close")
    cmd = engine.ctx.interface.find("door", "cmd")
    cmd.write({"data": 0.0})
    engine.step()
    assert door.status().seq == first + 1 and door.status().program == "move_to"
    ends = [e for e in engine.ctx.events.all() if e["event"] == "program_end"]
    assert ends[-1]["seq"] == first and ends[-1]["error"] == "cancelled"
    # The same value again changes nothing: no new start.
    cmd.write({"data": 0.0})
    engine.step()
    assert door.status().seq == first + 1
    assert _finish(engine, door, first + 1).error == ""
    assert abs(math.degrees(_hinge_angle(engine))) < 2


def test_an_idle_door_returns_from_a_push_shorter_than_its_stall_timeout(tmp_path):
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.5})
    door = engine.ctx.blackboard.require("machine:door")
    m, d = engine.ctx.model, engine.ctx.data
    dof = m.jnt_dofadr[_id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge")]
    for _ in range(500):
        d.qfrc_applied[dof] = 5.0  # a steady push, below the 15 N*m cap
        engine.step()
    d.qfrc_applied[dof] = 0.0
    for _ in range(1500):
        engine.step()
    assert 40 < math.degrees(_hinge_angle(engine)) < 50, "an idle door returns to its target"
    assert door.status().seq == 0 and not door.busy


def test_a_slow_unobstructed_door_opens_without_counting_its_ramp_as_a_stall(tmp_path):
    # 0.04 rad/s over the default 120 degrees is 0.019 openness/s, below stall_speed (0.02) all the
    # way, for 52 s: the leaf keeps up with its target, so it is not held back.
    engine = _built(tmp_path, {"speed": 0.04})
    door = engine.ctx.blackboard.require("machine:door")
    status = _finish(engine, door, door.start("open"), limit=70.0)
    assert status.error == ""
    assert math.degrees(_hinge_angle(engine)) > 118
    assert not [e for e in engine.ctx.events.all() if e["event"] == "stalled"]


def test_an_idle_door_held_off_its_target_longer_than_its_stall_timeout_gives_way(tmp_path):
    """A robot holding an automatic door open past its target: after stall_timeout the door takes
    where it stands as its target and stops pressing back, rather than pushing at max_torque for as
    long as it is held. Held back for less, it returns (the test above)."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.5, "stall_timeout": 1.0})
    door = engine.ctx.blackboard.require("machine:door")
    m, d = engine.ctx.model, engine.ctx.data
    jid = _id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    dof, aid = m.jnt_dofadr[jid], _id(engine, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_pos")
    held = math.radians(60)
    for _ in range(1500):  # 3 s held at 60 degrees, as a stiff hand holds it; its target is 45
        d.qfrc_applied[dof] = 400.0 * (held - _hinge_angle(engine)) - 300.0 * float(d.qvel[dof])
        engine.step()
    assert abs(float(d.actuator_force[aid])) < 2.0, (
        "it no longer presses back (held, it pressed about 9 N*m)"
    )
    d.qfrc_applied[dof] = 0.0
    for _ in range(1500):
        engine.step()
    gave = [e for e in engine.ctx.events.all() if e["event"] == "gave_way"]
    assert len(gave) == 1, "it gave way once, where the hand held it"
    where = math.degrees(gave[0]["openness"] * math.radians(90))
    assert 55 < where < 61
    assert math.degrees(_hinge_angle(engine)) == pytest.approx(where, abs=1.5), "it stays there"
    assert door.status().seq == 0 and not door.busy


def _door_with_a_post(tmp_path, degrees, door=None):
    """A default door with a 0.1 m post on the leaf's arc, 0.75 m out from the hinge, ``degrees``
    round from closed."""
    door = {"stall_timeout": 1.0, **(door or {})}
    probe = _built(tmp_path, door)
    hinge = probe.ctx.data.xpos[_id(probe, mujoco.mjtObj.mjOBJ_BODY, "door_leaf")].copy()
    probe.shutdown()
    angle = math.radians(degrees)
    post = {
        "prefix": "obs_",
        "pose": {
            "position": {
                "x": float(hinge[0] + 0.75 * math.cos(angle)),
                "y": float(hinge[1] + 0.75 * math.sin(angle)),
            }
        },
        "size": [0.1, 0.1, 0.45],
    }
    plugins = [{"door": door, "name": "door"}, {"box": post, "name": "post"}]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize("degrees", [64.5, 65.5])
def test_a_leaf_stopped_just_short_of_its_openness_ends_its_program_there(tmp_path, degrees):
    """Within reach of the target but short of the servo's own tolerance, held by a post: neither
    held back nor at the target. It has arrived, and the program must end, or the door stays busy
    and a caller waiting on it waits for ever."""
    engine = _door_with_a_post(tmp_path, degrees)
    door = engine.ctx.blackboard.require("machine:door")
    status = _finish(engine, door, door.start("move_to", openness=0.5), limit=10.0)
    assert status.error == ""
    where = engine.ctx.blackboard.require("machine:door:openness")()[0]
    assert 0.48 <= where < 0.499, "the post holds it, inside reach of 0.5"
    assert not door.busy
    door.start("close")  # and it takes the next start


def test_a_repeated_command_retries_after_a_stall(tmp_path):
    """The same value again is no new target while the door moves there or stands there; after a
    stall it is a retry."""
    engine = _door_with_a_post(tmp_path, 40.0)
    door = engine.ctx.blackboard.require("machine:door")
    cmd = engine.ctx.interface.find("door", "cmd")
    cmd.write({"data": 1.0})
    engine.step()
    first = door.status().seq
    assert _finish(engine, door, first).error.startswith("stalled")
    cmd.write({"data": 1.0})
    engine.step()
    assert door.status().seq == first + 1 and door.busy, "the stalled move is tried again"


def test_the_command_port_reads_a_door_homed_by_hinge_target(tmp_path):
    """``home: {hinge_pos: ...}`` puts the door open; the first command to close it is a new target."""
    engine = _built(
        tmp_path, {"max_angle": 90, "open": 0.0, "home": {"hinge_pos": math.radians(90)}}
    )
    door = engine.ctx.blackboard.require("machine:door")
    assert math.degrees(_hinge_angle(engine)) == pytest.approx(90, abs=1)
    engine.ctx.interface.find("door", "cmd").write({"data": 0.0})
    engine.step()
    assert door.busy, "closing is not the door's home, so it moves"
    assert _finish(engine, door, door.status().seq).error == ""
    assert abs(math.degrees(_hinge_angle(engine))) < 2


def _hold(engine, degrees, steps):
    """Hold the leaf at ``degrees`` for ``steps``, as a stiff hand holds it."""
    m, d = engine.ctx.model, engine.ctx.data
    dof = m.jnt_dofadr[_id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge")]
    held = math.radians(degrees)
    for _ in range(steps):
        d.qfrc_applied[dof] = 400.0 * (held - _hinge_angle(engine)) - 300.0 * float(d.qvel[dof])
        engine.step()
    d.qfrc_applied[dof] = 0.0


def test_a_door_that_gave_way_returns_when_commanded_back_to_where_it_was(tmp_path):
    """Giving way makes where it stands the door's target, so the old target is a new one again."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.5, "stall_timeout": 1.0})
    door = engine.ctx.blackboard.require("machine:door")
    _hold(engine, 60, 1500)
    assert [e["event"] for e in engine.ctx.events.all() if e["event"] == "gave_way"] == ["gave_way"]
    engine.ctx.interface.find("door", "cmd").write({"data": 0.5})
    engine.step()
    assert door.busy
    assert _finish(engine, door, door.status().seq).error == ""
    assert math.degrees(_hinge_angle(engine)) == pytest.approx(45, abs=2)


def test_a_door_held_while_it_runs_a_program_stalls_and_does_not_give_way(tmp_path):
    """Giving way is for an idle door: a program counts its own stall, from its own start."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.5, "stall_timeout": 1.0})
    door = engine.ctx.blackboard.require("machine:door")
    # Held while idle until it has been held back for about 0.6 s of its 1 s.
    _hold(engine, 60, 1150)
    seq = door.start("move_to", openness=0.2)
    m, d = engine.ctx.model, engine.ctx.data
    dof = m.jnt_dofadr[_id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge")]
    held = math.radians(60)
    while not (door.status().seq == seq and door.status().finished):
        assert engine.ctx.sim_time < 10.0, door.status()
        d.qfrc_applied[dof] = 400.0 * (held - _hinge_angle(engine)) - 300.0 * float(d.qvel[dof])
        engine.step()
    assert door.status().error.startswith("stalled")
    assert not [e for e in engine.ctx.events.all() if e["event"] == "gave_way"]


def test_a_door_under_manual_control_does_not_give_way(tmp_path):
    """Under manual control something else drives the hinge: the door does not take where it is
    held as its own target."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.0, "stall_timeout": 1.0})
    engine.ctx.manual_control = True
    _hold(engine, 60, 2500)  # 5 s, where an idle door gives way within 3
    assert not [e for e in engine.ctx.events.all() if e["event"] == "gave_way"]


@pytest.mark.parametrize(
    "config, program, args",
    [
        pytest.param({"width": 1.6}, "open", {}, id="width-1.6-open"),
        pytest.param(
            {"width": 1.2, "stall_timeout": 1.0}, "move_to", {"openness": 0.75}, id="width-1.2-0.75"
        ),
    ],
)
def test_an_unobstructed_wide_door_arrives_without_a_stall(tmp_path, config, program, args):
    """A heavier leaf swings past its target and back; slowing at a turn is not being held back."""
    engine = _built(tmp_path, config)
    door = engine.ctx.blackboard.require("machine:door")
    assert _finish(engine, door, door.start(program, **args)).error == ""
    assert not [e for e in engine.ctx.events.all() if e["event"] == "stalled"]


def test_a_move_ends_only_once_its_ramp_has_reached_the_target(tmp_path):
    """A leaf pushed to the openness before its ramp gets there has not arrived: the door's own
    target is still on its way."""
    engine = _built(tmp_path, {"max_angle": 90, "speed": 0.3, "stall_timeout": 10.0})
    door = engine.ctx.blackboard.require("machine:door")
    _hold(engine, 60, 1500)  # it stands short of 60 degrees, where hand and servo balance
    where = _hinge_angle(engine)
    seq = door.start("move_to", openness=where / math.radians(90))
    _hold(engine, 60, 5)
    status = _finish(engine, door, seq)
    assert status.error == ""
    assert door.value("hinge_pos") == pytest.approx(where, abs=1e-6), "its target got there too"


def test_short_pushes_do_not_add_up_to_a_give_way(tmp_path):
    """Each push shorter than stall_timeout, together longer: the door returns after each."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.5, "stall_timeout": 1.0})
    for _ in range(3):
        _hold(engine, 60, 1150)  # held back for about 0.6 s of the 1 s
        for _ in range(1500):
            engine.step()
    assert not [e for e in engine.ctx.events.all() if e["event"] == "gave_way"]
    assert math.degrees(_hinge_angle(engine)) == pytest.approx(45, abs=3)


def _door_plugin(engine):
    return next(p for p in engine.plugins if p.address == "door")


@pytest.mark.parametrize("start", [("open", {}), ("move_to", {"openness": 0.7})])
def test_a_command_back_right_after_a_handle_start_is_not_taken_for_a_repeat(tmp_path, start):
    """A handle's start is applied a step after it is made, and the door compares a ``cmd`` with what
    it was commanded last: commanded open, then back to closed before the start was applied, it
    closes, rather than dropping the ``0.0`` as the target it stood at and opening."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.0})
    door = engine.ctx.blackboard.require("machine:door")
    program, args = start
    first = door.start(program, **args)
    # The stream's value, applied on the physics thread before the posted start is: what a cmd
    # arriving between the start and its step does.
    _door_plugin(engine).cmd(0.0)
    engine.step()
    assert door.status().seq > first and door.status().program == "move_to"
    started = [e["seq"] for e in engine.ctx.events.all() if e["event"] == "program_start"]
    assert first not in started, "the handle's start was replaced before it ran"
    assert _finish(engine, door, door.status().seq).error == ""
    for _ in range(500):
        engine.step()
    assert abs(math.degrees(_hinge_angle(engine))) < 2


def test_a_repeated_command_after_a_handle_start_is_still_a_repeat(tmp_path):
    """Recorded at acceptance, the handle's target is what a ``cmd`` of the same value repeats."""
    engine = _built(tmp_path, {"max_angle": 90, "open": 0.0})
    door = engine.ctx.blackboard.require("machine:door")
    first = door.start("open")
    _door_plugin(engine).cmd(1.0)
    engine.step()
    assert door.status().seq == first and door.status().program == "open"
    assert _finish(engine, door, first).error == ""
