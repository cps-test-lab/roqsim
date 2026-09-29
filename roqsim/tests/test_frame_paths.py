"""Frames by path (roqsim.paths, roqsim.frames) and the pose_publisher that publishes them.

The world is a TurtleBot 4 named ``tb4`` -- bodies, sites, the ``frames:`` its manifest declares,
and the OAK-D nested in it with its device frame chain -- and a crate prop whose body and site share
the name ``lid``.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim import entity_pose
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.frames import frame_pose, resolve_frame
from roqsim.paths import Offer, PathError, resolve
from roqsim.plugin import PluginError
from roqsim.types import Transform

pytest.importorskip("roqsim_mobile", reason="the turtlebot4 model lives in roqsim_mobile")

_CRATE = textwrap.dedent(
    """\
    <mujoco model="crate">
      <worldbody>
        <body name="crate">
          <geom type="box" size="0.1 0.1 0.1" mass="1"/>
          <site name="lid" pos="0 0 0.1"/>
          <body name="lid" pos="0 0 0.1">
            <geom type="box" size="0.1 0.1 0.01" mass="0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
)

_POSES = [
    {"frame": "tb4"},
    {"frame": "crate/crate", "relative_to": "tb4/base_link"},
    {
        "frame": "tb4/oakd/oakd_rgb_camera_optical_frame",
        "relative_to": "tb4/oakd_camera_bracket",
        "child": "camera",
    },
]


def _engine(tmp_path: Path, extra=(), crate_components=()) -> Engine:
    crate = tmp_path / "crate.xml"
    crate.write_text(_CRATE)
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {"spawn_robot": {"model": "turtlebot4"}, "name": "tb4"},
            {
                "spawn_model": {
                    "model": str(crate),
                    "pose": {
                        "position": {"x": 1.0, "y": 0.5, "z": 0.1},
                        "orientation": {"yaw": math.pi / 3},
                    },
                },
                "name": "crate",
                "components": list(crate_components),
            },
            *extra,
        ],
    }
    overrides = {"components": {"tb4.oakd.oakd_camera": {"enabled": False}}}  # no GL needed
    engine = Engine(load_config_from_dict(world, base_dir=Path("."), overrides=overrides))
    engine.ctx.seed = 0
    return engine


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    publisher = {
        "pose_publisher": {
            "poses": _POSES,
            "rate_hz": 10,
            "world_frame": "world",
            "topics": {"poses": "truth"},
        },
        "name": "gt",
    }
    nested = {
        "pose_publisher": {"poses": [{"frame": "/tb4/mouse", "relative_to": "."}]},
        "name": "watch",
    }
    engine = _engine(tmp_path_factory.mktemp("crate"), [publisher], [nested])
    engine.setup()
    engine.reset()
    (cmd,) = [e for e in engine.ctx.interface.all() if e.name == "cmd_vel"]
    cmd.write({"vx": 0.2, "wz": 0.4})
    for _ in range(300):
        engine.step()
    yield engine
    engine.shutdown()


def _id(model, kind, name):
    found = mujoco.mj_name2id(model, kind, name)
    assert found >= 0, name
    return found


# -- the resolver --------------------------------------------------------------------------------------
def test_each_kind_of_frame_resolves_by_its_tf_name(world):
    m = world.ctx.model
    body, site = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE
    expected = {
        "tb4": ("root", "tb4", _id(m, body, "base_link")),
        "tb4/base_link": ("body", "base_link", _id(m, body, "base_link")),
        "tb4/mouse": ("site", "mouse", _id(m, site, "mouse")),
        # declared in the TurtleBot 4 manifest's `frames:`
        "tb4/oakd_camera_bracket": (
            "frame",
            "oakd_camera_bracket",
            _id(m, site, "oakd_camera_bracket"),
        ),
        # the nested OAK-D: its root, its body and its device frame chain, without the oakd_ prefix
        "tb4/oakd": ("root", "tb4.oakd", _id(m, body, "oakd_mount")),
        "tb4/oakd/mount": ("body", "mount", _id(m, body, "oakd_mount")),
        "tb4/oakd/oakd_rgb_camera_optical_frame": (
            "frame",
            "oakd_rgb_camera_optical_frame",
            _id(m, site, "oakd_oakd_rgb_camera_optical_frame"),
        ),
        "crate/crate": ("body", "crate", _id(m, body, "crate")),
    }
    for path, (kind, name, index) in expected.items():
        frame = resolve_frame(world.ctx, path)
        assert (frame.path, frame.kind, frame.name, frame.index) == (path, kind, name, index), path


def test_a_path_within_an_entity_is_relative_to_it(world):
    assert resolve_frame(world.ctx, ".", within="tb4").path == "tb4"
    assert resolve_frame(world.ctx, "mouse", within="tb4").path == "tb4/mouse"
    assert resolve_frame(world.ctx, "oakd/mount", within="tb4").path == "tb4/oakd/mount"
    assert resolve_frame(world.ctx, "/crate/crate", within="tb4").path == "crate/crate"


def test_a_body_and_a_site_of_one_name_are_refused_naming_both(world):
    with pytest.raises(PathError) as err:
        resolve_frame(world.ctx, "crate/lid")
    assert err.value.reason == "ambiguous"
    assert "crate/lid (body)" in str(err.value) and "crate/lid (site)" in str(err.value)


def test_an_unknown_path_names_the_nearest_and_what_its_component_offers(world):
    with pytest.raises(PathError) as err:
        resolve_frame(world.ctx, "tb4/mouze")
    assert err.value.suggestion == "tb4/mouse"
    assert err.value.component == "tb4"
    assert "Did you mean 'tb4/mouse'?" in str(err.value)
    assert "tb4/base_link" in err.value.offered and "tb4/ir_omni" in err.value.offered

    with pytest.raises(PathError) as err:
        resolve_frame(world.ctx, "tb4/oakd/oakd_rgb_camera_optical_fram")
    assert err.value.component == "tb4/oakd"
    assert err.value.suggestion == "tb4/oakd/oakd_rgb_camera_optical_frame"
    assert "tb4/oakd/oakd_link" in err.value.offered

    with pytest.raises(PathError, match="The components offering one: crate, tb4"):
        resolve_frame(world.ctx, "robot/base_link")


def test_a_frame_and_an_endpoint_of_one_path_never_collide():
    offers = [
        Offer("tb4", "imu", "frame", "site", target="the site"),
        Offer("tb4", "imu", "out", "out", target="the endpoint"),
    ]
    assert resolve(offers, "tb4/imu", "frame").target == "the site"
    assert resolve(offers, "tb4/imu", "out").target == "the endpoint"
    with pytest.raises(PathError, match="no command 'tb4/imu'"):
        resolve(offers, "tb4/imu", "in", noun="command")


# -- poses ---------------------------------------------------------------------------------------------
def test_a_world_pose_is_the_core_pose_data(world):
    ctx = world.ctx
    d = ctx.data
    root = frame_pose(ctx, "tb4")
    core = ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name("tb4")).read()
    assert (root.parent, root.child) == ("world", "tb4")
    assert np.array_equal(root.translation, core.position)
    assert np.array_equal(root.rotation, core.orientation)
    bid = _id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    body = frame_pose(ctx, "crate/crate")
    assert np.array_equal(body.translation, d.xpos[bid])
    assert np.array_equal(body.rotation, d.xquat[bid])
    site = frame_pose(ctx, "tb4/mouse")
    assert np.array_equal(
        site.translation, d.site_xpos[_id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, "mouse")]
    )


def test_a_relative_pose_is_the_reference_frames_inverse_times_the_frames(world):
    ctx = world.ctx
    m = ctx.model
    # A site on the base, relative to the base: its fixed mount, wherever the robot drove.
    mouse = frame_pose(ctx, "tb4/mouse", relative_to="tb4/base_link")
    sid = _id(m, mujoco.mjtObj.mjOBJ_SITE, "mouse")
    assert (mouse.parent, mouse.child) == ("base_link", "mouse")
    assert np.allclose(mouse.translation, m.site_pos[sid], rtol=0, atol=1e-12)
    assert abs(float(np.dot(mouse.rotation, m.site_quat[sid]))) == pytest.approx(1.0, abs=1e-12)
    # Across two entities: composing the reference's world pose with the relative one gives back
    # the frame's world pose.
    rel = frame_pose(ctx, "crate/crate", relative_to="tb4")
    base, crate = frame_pose(ctx, "tb4"), frame_pose(ctx, "crate/crate")
    pos = np.empty(3)
    mujoco.mju_rotVecQuat(pos, rel.translation, base.rotation)
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, base.rotation, rel.rotation)
    assert np.allclose(pos + base.translation, crate.translation, rtol=0, atol=1e-12)
    assert abs(float(np.dot(quat, crate.rotation))) == pytest.approx(1.0, abs=1e-12)


# -- pose_publisher --------------------------------------------------------------------------------------
def _published(engine, producer):
    return {e.name: e for e in engine.ctx.interface.all() if e.producer == producer}


def test_pose_publisher_puts_each_pose_on_its_topic_under_its_parent_frame(world):
    eps = _published(world, "gt")
    assert set(eps) == {"poses/tb4", "poses/crate", "poses/camera"}
    parents = {
        "poses/tb4": "world",
        "poses/crate": "base_link",
        "poses/camera": "oakd_camera_bracket",
    }
    for name, ep in eps.items():
        assert ep.owner == "gt" and ep.payload_type.cls is Transform
        assert ep.backend == {"ros2": {"topic": "truth", "frame_id": parents[name]}}
        assert (ep.rate_hz, ep.lazy) == (10.0, False)
    for entry, name in zip(_POSES, ("poses/tb4", "poses/crate", "poses/camera"), strict=True):
        value = eps[name].read()
        expected = frame_pose(world.ctx, entry["frame"], entry.get("relative_to"))
        assert (value.parent, value.child) == (parents[name], name.removeprefix("poses/"))
        assert np.array_equal(value.translation, expected.translation)
        assert np.array_equal(value.rotation, expected.rotation)


def test_nested_under_an_entity_its_paths_start_there(world):
    (ep,) = _published(world, "crate.watch").values()
    assert ep.owner == "crate" and ep.name == "poses/mouse"
    assert ep.backend["ros2"]["frame_id"] == "crate"
    value = ep.read()
    expected = frame_pose(world.ctx, "tb4/mouse", relative_to="crate")
    assert (value.parent, value.child) == ("crate", "mouse")
    assert np.array_equal(value.translation, expected.translation)


def test_an_unknown_frame_is_refused_when_the_world_is_set_up(tmp_path):
    bad = {"pose_publisher": {"poses": [{"frame": "tb4/mouze"}]}, "name": "gt"}
    engine = _engine(tmp_path, [bad])
    with pytest.raises(PluginError, match=r"poses\[0\]\.frame .*Did you mean 'tb4/mouse'\?"):
        engine.setup()


def test_two_poses_under_one_child_name_are_refused(tmp_path):
    twice = {
        "pose_publisher": {
            "poses": [{"frame": "tb4/mouse"}, {"frame": "crate/crate", "child": "mouse"}]
        },
        "name": "gt",
    }
    engine = _engine(tmp_path, [twice])
    with pytest.raises(PluginError, match="publishes 'mouse' a second time"):
        engine.setup()
