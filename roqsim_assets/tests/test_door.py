"""The hinged door: a leaf on a hinge joint driven by a position actuator, its openness
live-controllable. It unifies a passive door (holds its ``open`` angle) and an automatic door
(exposes ROS I/O), and locates the hinge from the opening centre + ``hinge_side``."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _door(tmp_path, extra=None):
    plugins = [{"door": dict(extra or {}), "name": "door"}]
    return load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path)


def _built(tmp_path, extra=None):
    engine = Engine(_door(tmp_path, extra))
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
    engine.ctx.blackboard.require("door:door").set_openness(1.0)
    for _ in range(3000):
        engine.step()
    assert math.degrees(_hinge_angle(engine)) > 88  # reached ~fully open


def test_swing_sign_sets_open_direction(tmp_path):
    engine = _built(tmp_path, {"swing": -1, "max_angle": 90})
    engine.ctx.blackboard.require("door:door").set_openness(1.0)
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
    assert engine.ctx.interface.all() == []


def test_controllable_declares_endpoints(tmp_path):
    engine = _built(tmp_path, {"controllable": True, "namespace": "foyer"})
    eps = {e.name: e for e in engine.ctx.interface.all()}
    assert set(eps) == {"cmd", "state", "door"}
    assert eps["cmd"].backend["ros2"]["type"] == "std_msgs.msg.Float64"
    assert eps["state"].direction == "out"
    action = eps["door"].backend["ros2"]
    assert action["action"] == "control_msgs.action.GripperCommand"
    # The action handler reads the door's own state, not a gripper's.
    assert action["state_key"] == "door:door:state"
    assert callable(engine.ctx.blackboard.get("door:door:state"))


def test_mesh_leaf_model_loads_and_swings(tmp_path):
    # The textured leaf model (white panel + chrome handle) hangs on the hinge and opens on command.
    engine = _built(tmp_path, {"model": "door", "width": 0.9, "height": 2.0})
    assert _id(engine, mujoco.mjtObj.mjOBJ_JOINT, "hinge") >= 0
    # panel + handle geoms come in under the leaf_ prefix.
    assert _id(engine, mujoco.mjtObj.mjOBJ_MESH, "leaf_door_panel") >= 0
    engine.ctx.blackboard.require("door:door").set_openness(1.0)
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
    assert engine.ctx.interface.all() == []  # nothing to command
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
                "pos": [0.45, 0.5],
                "size": [0.45, 0.30, 0.45],
            },
            "name": "obs",
        },
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    aid = _id(engine, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_pos")
    engine.ctx.blackboard.require("door:door").set_openness(1.0)
    peak = 0.0
    for _ in range(4000):
        engine.step()
        peak = max(peak, abs(float(engine.ctx.data.actuator_force[aid])))
    door = engine.ctx.blackboard.require("door:door")
    assert peak <= 15.0 + 1e-6  # never pushed harder than the cap
    assert door.get_openness() == 1.0  # still commanded open...
    assert door.read_state()[0] < 0.9  # ...but blocked short of it (gave up)


def test_validate_config_flags_bad_input(tmp_path):
    from roqsim_assets.plugins.door import DoorPlugin

    bad = {"width": -1, "open": 2.0, "hinge_side": "up", "swing": 0}
    errors = DoorPlugin(bad).validate_config(bad)
    assert any("width" in e for e in errors)
    assert any("open" in e for e in errors)
    assert any("hinge_side" in e for e in errors)
    assert any("swing" in e for e in errors)
