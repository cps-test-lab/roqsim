"""A walker's motion layers and its look: a clip played over the gait on some joints, and the head
turned toward a target -- for a walker with hands off, through its handle and its commands."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim_motion import quat as Q
from roqsim_walker.blueprint import BlueprintError, resolve_clip, resolve_walker
from roqsim_walker.humanoid import JOINT_NAMES, joints_of, quat_rotate
from roqsim_walker.motion import Clip, ClipError

#: The right arm raised sideways to the shoulder's height: a rotation about +x of the arm hanging down.
RAISE = -math.pi / 2


def _arm_clip(path, frames=31, fps=30.0, joint="shoulder_r", angle=RAISE):
    """A clip of ``frames`` that holds one joint turned about +x; everything else at rest."""
    rot = np.tile([1.0, 0.0, 0.0, 0.0], (frames, len(JOINT_NAMES), 1))
    rot[:, JOINT_NAMES.index(joint)] = [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]
    Clip(rot, np.zeros(frames), fps=fps, stride_len=1.0, loop=False).save(path)
    return str(path)


def _world(layers=None, others=()):
    walker = {"walker": "MaleVisitorWalk", "skin": False}
    if layers is not None:
        walker["layers"] = layers
    return load_config_from_dict(
        {
            "sim": {"pacing": "asap"},
            "components": [{"walker": walker, "name": "ped"}, *others],
        }
    )


@pytest.fixture(scope="module")
def _engine():
    other = {
        "walker": {
            "walker": "MaleVisitorWalk",
            "skin": False,
            "pose": {"position": {"x": 2.0, "y": 2.0}},
        },
        "name": "other",
    }
    engine = Engine(_world(others=[other]))
    engine.setup()
    yield engine
    engine.shutdown()


@pytest.fixture
def sim(_engine):
    _engine.reset()
    return _engine


def _run(engine, seconds):
    for _ in range(max(1, int(seconds / engine.ctx.dt))):
        engine.step()


def _part(engine, name, part):
    m, d = engine.ctx.model, engine.ctx.data
    mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{part}")]
    return d.mocap_pos[mid].copy(), d.mocap_quat[mid].copy()


def _wrist_z(engine, side):
    return _part(engine, "ped", f"wrist_{side}")[0][2]


# -- joints, clips ---------------------------------------------------------------------------------
def test_joint_groups_and_names_mix_and_an_unknown_one_is_named():
    assert joints_of(["arm_r", "head"]) == ("head", "shoulder_r", "elbow_r", "wrist_r")
    assert joints_of("upper")[0] == "spine" and "pelvis" not in joints_of("upper")
    assert set(joints_of("all")) == set(JOINT_NAMES)
    with pytest.raises(ValueError, match="no joint or joint group 'tail'"):
        joints_of(["tail"])


def test_a_clip_that_cannot_be_read_says_why(tmp_path):
    with pytest.raises(ClipError, match="cannot be read"):
        Clip.load(tmp_path / "missing.npz")
    np.savez(
        tmp_path / "flat.npz",
        joint_rot=np.zeros((3, 4)),
        root_z=np.zeros(3),
        fps=30.0,
        stride_len=1.0,
        loop=False,
    )
    with pytest.raises(ClipError, match="joint_rot is"):
        Clip.load(tmp_path / "flat.npz")


def test_a_clip_is_found_by_name_in_the_walkers_set_or_refused_naming_those_there():
    assert resolve_clip("MaleVisitorWalk", "walk").endswith("anims/adult/walk.npz")
    with pytest.raises(BlueprintError, match=r"no clip 'wave'.*the clips there are .*walk"):
        resolve_clip("MaleVisitorWalk", "wave")


def test_a_locomotion_clip_the_world_names_must_exist(tmp_path):
    with pytest.raises(BlueprintError, match="no clip file"):
        resolve_walker("MaleVisitorWalk", motion={"walk": str(tmp_path / "nope.npz")})
    with pytest.raises(BlueprintError, match="'sit', which is no locomotion clip"):
        resolve_walker("MaleVisitorWalk", motion={"sit": "walk"})


# -- layers ------------------------------------------------------------------------------------------
def test_a_clip_played_on_one_arm_moves_that_arm_only_and_ends(sim, tmp_path):
    clip = _arm_clip(tmp_path / "raise.npz")
    handle = sim.ctx.blackboard.get("walker:ped")
    _run(sim, 0.2)
    left0 = _wrist_z(sim, "l")
    layer = handle.play(clip, joints=["arm_r"], fade=0.1)
    _run(sim, 0.5)
    assert handle.playing()[0]["joints"] == ["shoulder_r", "elbow_r", "wrist_r"]
    shoulder = _part(sim, "ped", "shoulder_r")[0][2]
    assert abs(_wrist_z(sim, "r") - shoulder) < 0.05  # raised to the shoulder's height
    assert abs(_wrist_z(sim, "l") - left0) < 0.03  # the other arm keeps the gait's
    _run(sim, 0.7)
    assert handle.clip_ended(layer) and handle.playing() == []
    assert abs(_wrist_z(sim, "r") - _wrist_z(sim, "l")) < 0.03  # back to the gait


def test_a_looping_clip_plays_until_stopped_and_a_weight_blends_it(sim, tmp_path):
    clip = _arm_clip(tmp_path / "raise.npz")
    handle = sim.ctx.blackboard.get("walker:ped")
    _run(sim, 0.2)
    hang = _wrist_z(sim, "r")
    half = handle.play(clip, joints="arm_r", loop=True, weight=0.5, fade=0.0)
    _run(sim, 2.0)
    shoulder = _part(sim, "ped", "shoulder_r")[0][2]
    assert hang + 0.05 < _wrist_z(sim, "r") < shoulder - 0.05  # half way
    assert not handle.clip_ended(half)
    assert handle.stop_clip(clip) == [half]
    _run(sim, 0.5)
    assert handle.clip_ended(half)


def test_play_refuses_what_it_cannot_play(sim, tmp_path):
    handle = sim.ctx.blackboard.get("walker:ped")
    with pytest.raises(ValueError, match="no clip 'wave'"):
        handle.play("wave")
    clip = _arm_clip(tmp_path / "raise.npz")
    with pytest.raises(ValueError, match="'weight' is from 0 to 1"):
        handle.play(clip, weight=1.5)
    with pytest.raises(ValueError, match="no joint or joint group"):
        handle.play(clip, joints=["wing"])


def test_the_play_command_reaches_the_walker_as_any_command_does(sim, tmp_path):
    clip = _arm_clip(tmp_path / "raise.npz")
    future = sim.ctx.interface.find("ped", "play").write({"clip": clip, "joints": ["arm_r"]})
    sim.step()
    layer = future.result(0)
    (playing,) = sim.ctx.interface.find("ped", "clips").read()
    assert playing["id"] == layer and playing["clip"] == clip


def test_a_world_states_the_clips_a_walker_plays_from_the_start(tmp_path):
    clip = _arm_clip(tmp_path / "raise.npz")
    engine = Engine(_world(layers=[{"clip": clip, "joints": ["arm_r"], "fade": 0.0}]))
    engine.setup()
    try:
        for _ in range(2):  # every episode
            engine.reset()
            _run(engine, 0.3)
            shoulder = _part(engine, "ped", "shoulder_r")[0][2]
            assert abs(_wrist_z(engine, "r") - shoulder) < 0.05
            assert engine.ctx.blackboard.get("walker:ped").playing()[0]["loop"] is True
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "layer, said",
    [
        ({"clip": "wave"}, "no clip 'wave'"),
        ({"clip": "walk", "joints": ["wing"]}, "no joint or joint group 'wing'"),
        ({"clip": "walk", "weight": 2}, "'weight' is from 0 to 1"),
        ({"clip": "walk", "volume": 1}, "'volume' not read"),
        ({"joints": ["arm_r"]}, "a mapping with a 'clip'"),
    ],
)
def test_a_layer_the_walker_cannot_play_is_refused_when_the_world_loads(layer, said):
    with pytest.raises(PluginError, match=said):
        Engine(_world(layers=[layer])).setup()


# -- the look ------------------------------------------------------------------------------------
def _head_yaw(engine, name="ped"):
    _, q = _part(engine, name, "head")
    f = quat_rotate(q, [1.0, 0.0, 0.0])
    return math.atan2(f[1], f[0])


def test_the_head_turns_to_a_point_at_a_bounded_rate_and_back(sim):
    handle = sim.ctx.blackboard.get("walker:ped")
    _run(sim, 0.2)
    handle.look_at([3.0, 3.0, 1.6])  # 45 degrees to its left
    _run(sim, 0.1)
    assert _head_yaw(sim) < math.radians(30)  # turning, not snapped
    _run(sim, 1.5)
    assert abs(_head_yaw(sim) - math.radians(45)) < math.radians(4)
    handle.look_at(None)
    _run(sim, 1.5)
    assert abs(_head_yaw(sim)) < math.radians(4)


def test_the_head_follows_an_entity_by_name_within_its_limits(sim):
    handle = sim.ctx.blackboard.get("walker:ped")
    sim.ctx.interface.find("ped", "look_at").write({"target": "other"})
    _run(sim, 1.5)
    assert abs(_head_yaw(sim) - math.radians(45)) < math.radians(5)
    handle.look_at([0.0, 5.0, 1.6])  # straight to its left: past the head's 70 degrees
    _run(sim, 1.5)
    _, spine = _part(sim, "ped", "spine")
    _, head = _part(sim, "ped", "head")
    on_spine = quat_rotate(Q.mul(Q.inv(spine), head), [1.0, 0.0, 0.0])
    assert abs(math.atan2(on_spine[1], on_spine[0]) - math.radians(70)) < math.radians(1)
    with pytest.raises(ValueError, match="unknown target 'nobody'"):
        handle.look_at("nobody")


def test_the_walker_handle_is_the_body_and_routes_go_to_the_navigator(sim):
    handle = sim.ctx.blackboard.get("walker:ped")
    assert not hasattr(handle, "send_route")
    assert sim.ctx.blackboard.get("nav:ped:handle") is not None


def test_a_required_clip_the_blueprint_lacks_is_taken_from_motion(tmp_path, monkeypatch):
    """A blueprint whose anim set has no ``short`` is refused, naming ``motion:``, and the world that
    gives it there loads, its layers looked up in the walker's own set."""
    from roqsim_walker import blueprint as bp
    from roqsim_walker.plugins.walker import WalkerPlugin

    shipped = bp.resolve_walker("MaleVisitorWalk")
    people = tmp_path / "people"
    people.mkdir()
    (people / "MaleVisitorWalk").symlink_to(
        bp.os.path.dirname(shipped["mesh"]), target_is_directory=True
    )
    anims = tmp_path / "anims" / "adult"
    anims.mkdir(parents=True)
    for kind in ("idle", "walk", "run"):
        (anims / f"{kind}.npz").symlink_to(shipped["motion"][kind])
    monkeypatch.setattr(bp, "_asset_roots", lambda base_dir=None: [str(tmp_path)])
    with pytest.raises(BlueprintError, match=r"no short clip .* give it with motion: \{short:"):
        bp.resolve_walker("MaleVisitorWalk")
    config = {"walker": "MaleVisitorWalk", "motion": {"short": "walk"}, "layers": [{"clip": "run"}]}
    assert WalkerPlugin(config, name="ped").validate_config(config) == []
    assert "no short clip" in " ".join(
        WalkerPlugin({"walker": "MaleVisitorWalk"}, name="ped").validate_config(
            {"walker": "MaleVisitorWalk"}
        )
    )
