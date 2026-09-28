"""The Create 3 ground-truth streams: the topics, frames, rate and numbers its stack reads.

``REFERENCE`` holds, for this scene after this drive, what each stream carries on the wire: the owner,
topic, parent frame, child frame, rate, laziness, translation and rotation of its one transform. The
TurtleBot 4 publishes its world pose under ``turtlebot4`` and its mouse and IR receiver relative to
``base_link``; the dock publishes its world pose under ``standard_dock`` and its halo emitter relative
to ``std_dock_link``.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

import roqsim  # noqa: F401, I001
from roqsim import entity_pose
from roqsim.config import load_config_from_dict
from roqsim.endpoint import qos_profile
from roqsim.engine import Engine
from roqsim_mobile.plugins.create3_pose_publisher import FrameTransform

_ROBOT, _DOCK = "_internal/sim_ground_truth_pose", "_internal/sim_ground_truth_dock_pose"

# fmt: off
REFERENCE = [
    ("robot", _ROBOT, "map", "turtlebot4",
     [-0.09858653510103818, -0.007658704386760387, -0.0052622439962866675],
     [0.9924586467817256, 0.0006478517522578294, -0.005340766553845099, 0.12246179375180938]),
    ("robot", _ROBOT, "base_link", "mouse",
     [0.10149999999999999, 0.08699999999999994, 0.009199999999999996],
     [0.9238797538373535, 0.0, 0.0, -0.3826828980362601]),
    ("robot", _ROBOT, "base_link", "ir_omni",
     [0.15299999999999994, 0.0, 0.09919999999999995],
     [1.0, 0.0, 0.0, 0.0]),
    ("standard_dock", _DOCK, "map", "standard_dock",
     [0.15700987906423636, -5.849544358563105e-09, -0.001161932861300741],
     [-2.268345594954499e-07, -0.00631871373373533, -1.0645125521785886e-05, 0.9999800366724241]),
    ("standard_dock", _DOCK, "std_dock_link", "halo_link",
     [-0.06, 0.0, 0.0904],
     [1.0, 0.0, 0.0, 0.0]),
]
# fmt: on


@pytest.fixture(scope="module")
def scene():
    """The robot and its dock as the Create 3 world places them, driven for 0.8 s."""
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {"spawn_robot": {"model": "turtlebot4"}, "name": "robot"},
            {
                "spawn_model": {
                    "model": "create3_dock",
                    "pose": {
                        "position": {"x": 0.157, "y": 0.0, "z": 0.0},
                        "orientation": {"yaw": math.pi},
                    },
                },
                "name": "standard_dock",
                "components": [
                    {
                        "create3_pose_publisher": {
                            "topic": _DOCK,
                            "frame": "standard_dock",
                            "sites": ["halo_link"],
                            "rate_hz": 62,
                        },
                        "name": "gt",
                    }
                ],
            },
        ],
    }
    overrides = {"components": {"robot.oakd.oakd_camera": {"enabled": False}}}  # no GL needed
    engine = Engine(load_config_from_dict(world, base_dir=Path("."), overrides=overrides))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    for _ in range(200):
        engine.step()
    (cmd,) = [e for e in engine.ctx.interface.all() if e.name == "cmd_vel"]
    cmd.write({"vx": -0.2, "wz": 0.5})
    for _ in range(400):
        engine.step()
    yield engine
    engine.shutdown()


def _streams(engine):
    return [
        e
        for e in engine.ctx.interface.all()
        if (e.backend.get("ros2") or {}).get("topic") in (_ROBOT, _DOCK)
    ]


def test_each_stream_is_the_reference_on_the_wire(scene):
    streams = _streams(scene)
    assert len(streams) == len(REFERENCE), "one stream per frame, each a one-transform message"
    got = {}
    for ep in streams:
        value = ep.read()
        got[value.child_frame_id] = (ep, value)
    for owner, topic, parent, child, pos, quat in REFERENCE:
        ep, value = got[child]
        assert ep.owner == owner and ep.namespace == ""
        assert ep.backend == {"ros2": {"topic": topic, "frame_id": parent}}
        assert ep.payload_type.cls is FrameTransform
        assert (ep.rate_hz, ep.lazy) == (62.0, True)
        assert np.allclose(value.translation, pos, rtol=0, atol=1e-12), child
        assert np.allclose(value.rotation, quat, rtol=0, atol=1e-12), child


def test_ros_carries_each_stream_as_a_one_transform_tf_message(scene):
    """What the adapter receives: a ``TFMessage`` of one transform on the stream's topic, with the
    parent and child frames and the numbers of the reference, at the default QoS."""
    pytest.importorskip("tf2_msgs")
    typemap = pytest.importorskip("roqsim_ros_bridge.typemap")
    from builtin_interfaces.msg import Time
    from tf2_msgs.msg import TFMessage

    got = {}
    for ep in _streams(scene):
        binding = typemap.resolve(ep)
        assert binding.hints["type"] == "tf2_msgs.msg.TFMessage"
        assert binding.hints["qos"] == qos_profile("default")
        msg = TFMessage()
        binding.fill(msg, ep.read(), Time(sec=1), binding.hints)
        (tf,) = msg.transforms
        got[tf.child_frame_id] = (binding.hints["topic"], tf)
    assert set(got) == {row[3] for row in REFERENCE}
    for _, topic, parent, child, pos, quat in REFERENCE:
        got_topic, tf = got[child]
        assert (got_topic, tf.header.frame_id, tf.header.stamp.sec) == (topic, parent, 1)
        t, q = tf.transform.translation, tf.transform.rotation
        assert np.allclose([t.x, t.y, t.z], pos, rtol=0, atol=1e-12), child
        assert np.allclose([q.w, q.x, q.y, q.z], quat, rtol=0, atol=1e-12), child


def test_the_world_pose_is_the_core_pose_endpoints(scene):
    core = scene.ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name("robot")).read()
    (ep,) = [e for e in _streams(scene) if e.read().child_frame_id == "turtlebot4"]
    value = ep.read()
    pos, quat = value.translation, value.rotation
    assert np.array_equal(pos, core.position) and np.array_equal(quat, core.orientation)
    bid = mujoco.mj_name2id(scene.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert np.array_equal(pos, scene.ctx.data.xpos[bid])


def test_a_missing_site_is_refused():
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {"model": "turtlebot3_waffle"},
                "name": "robot",
                "components": [{"create3_pose_publisher": {"sites": ["mouse"]}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    with pytest.raises(RuntimeError, match="site 'mouse' not found"):
        engine.setup()
