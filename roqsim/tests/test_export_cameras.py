"""Sensor cameras in a scene export: a glTF perspective camera on the body the sensor is mounted on.

The walk's cameras and the glTF file's are checked against MuJoCo itself: the camera's world pose,
composed from its body's pose and the mount pose, is ``cam_xpos``/``cam_xmat``, and what the camera
looks at lies down its -z, the direction a glTF camera looks.
"""

from __future__ import annotations

import logging
import math

import mujoco
import numpy as np
import pytest
from test_export_gltf import _W, Glb
from test_scene_content import _quat_to_mat, _rest_mat

from roqsim.export_gltf import export_gltf as write_glb
from roqsim.scene_content import SensorCamera, walk

LOG = logging.getLogger("t")

# A camera on a hinged link, mounted off the link's origin and turned, looking at a box.
_MJCF = """
<mujoco>
  <worldbody>
    <geom type="plane" size="3 3 0.1"/>
    <geom name="target" type="box" pos="2 0.5 0.4" size="0.2 0.2 0.2"/>
    <body name="base" pos="0.1 -0.2 0.3" euler="0 0 20">
      <joint name="pan" type="hinge" axis="0 0 1" pos="0.05 0 0"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="mount" pos="0.2 0 0.15">
        <camera name="eye" pos="0.03 0.01 0.02" mode="fixed" fovy="50" resolution="64 48"
                xyaxes="0 -1 0 0 0 1"/>
        <camera name="follower" mode="trackcom" pos="0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _model(pan: float = 0.0):
    model = mujoco.MjSpec.from_string(_MJCF).compile()
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("pan").id]] = pan
    mujoco.mj_forward(model, data)
    return model, data


def _sensor(model, camera="eye", kind="rgb", width=64, height=48, fx=None, fy=None):
    cam = model.camera(camera).id
    f = height / (2.0 * math.tan(math.radians(float(model.cam_fovy[cam])) / 2.0))
    return SensorCamera("robot.eye", cam, kind, width, height, fx or f, fy or f)


def _cam_world(model, data, camera="eye"):
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[model.camera(camera).id].reshape(3, 3)
    T[:3, 3] = data.cam_xpos[model.camera(camera).id]
    return T


def _cameras(model, data, sensors=None):
    """The walk's cameras of ``model`` at ``data``, with the default sensor unless told otherwise."""
    content = walk(model, data, LOG, [_sensor(model)] if sensors is None else sensors)
    return content, content.cameras


def _camera_world(content, cam):
    """The camera's world matrix: its body's pose at the exported state, then the mount pose."""
    return _rest_mat(*content.world_poses[cam.body]) @ _rest_mat(cam.pos, cam.quat)


@pytest.mark.parametrize("pan", [0.0, 0.6, -1.1])
def test_the_camera_sits_where_mujoco_puts_it(pan):
    model, data = _model(pan)
    content, (cam,) = _cameras(model, data)
    assert cam.name == "robot.eye" and cam.kind == "rgb"
    assert content.bodies[cam.body].name == "mount"
    np.testing.assert_allclose(_camera_world(content, cam), _cam_world(model, data), atol=1e-9)


def test_the_camera_looks_down_its_minus_z_with_y_up():
    """The glTF convention, checked on what the camera sees rather than on a matrix."""
    model, data = _model()
    content, (cam,) = _cameras(model, data)
    world = _camera_world(content, cam)
    # xyaxes "0 -1 0  0 0 1": image right is world -y, image up is world +z, so the view is +x.
    np.testing.assert_allclose(
        world[:3, :3] @ [0, 0, -1],
        [math.cos(math.radians(20)), math.sin(math.radians(20)), 0],
        atol=1e-9,
    )
    np.testing.assert_allclose(world[:3, :3] @ [0, 1, 0], [0, 0, 1], atol=1e-9)


def test_the_projection_is_the_published_images():
    model, data = _model()
    _content, (cam,) = _cameras(model, data)
    assert cam.yfov == pytest.approx(math.radians(50))
    assert cam.aspect_ratio == pytest.approx(64 / 48)
    assert (cam.width, cam.height) == (64, 48)
    extent = float(model.stat.extent)
    assert cam.znear == pytest.approx(float(model.vis.map.znear) * extent)
    assert cam.zfar == pytest.approx(float(model.vis.map.zfar) * extent)


def test_a_sensor_that_states_its_planes_is_exported_with_them():
    """A sensor renders with its own clip planes, so its exported projection carries those."""
    model, data = _model()
    sensor = _sensor(model)
    sensor.znear, sensor.zfar = 0.04, 7.5
    _content, (cam,) = _cameras(model, data, [sensor])
    assert (cam.znear, cam.zfar) == (0.04, 7.5)


def test_non_square_pixels_widen_the_aspect_ratio():
    """aspectRatio is the ratio of the half-angle tangents, so fx < fy (wider pixels) widens it."""
    model, data = _model()
    _content, (cam,) = _cameras(model, data, [_sensor(model, fx=40.0, fy=50.0)])
    assert cam.yfov == pytest.approx(2 * math.atan(48 / 100))
    assert cam.aspect_ratio == pytest.approx((64 / 80) / (48 / 100))


def test_a_camera_not_fixed_to_its_body_is_left_out_and_named(caplog):
    model, data = _model()
    with caplog.at_level("WARNING"):
        _content, cameras = _cameras(model, data, [_sensor(model, "follower")])
    assert cameras == []
    assert "robot.eye" in caplog.text and "not fixed" in caplog.text


def test_no_sensors_means_no_cameras():
    model, data = _model()
    assert _cameras(model, data, [])[1] == []


@pytest.mark.parametrize("pan", [0.0, 0.6])
def test_the_gltf_camera_node_is_a_child_of_its_body_and_sits_where_mujoco_puts_it(tmp_path, pan):
    model, data = _model(pan)
    out = tmp_path / "scene.glb"
    write_glb(model, data, out, log=LOG, max_tex_dim=0, cameras=[_sensor(model)])
    glb = Glb(out)
    (n,) = [
        i for i, node in enumerate(glb.nodes) if "camera" in node and "sensor" in node["extras"]
    ]
    node = glb.nodes[n]
    assert node["name"] == "robot.eye"
    assert node["extras"] == {"sensor": "robot.eye", "kind": "rgb", "width": 64, "height": 48}
    assert glb.parents()[n] == glb.body_nodes()[model.body("mount").id]
    camera = glb.json["cameras"][node["camera"]]
    assert camera["type"] == "perspective"
    assert camera["perspective"]["yfov"] == pytest.approx(math.radians(50))
    assert camera["perspective"]["aspectRatio"] == pytest.approx(64 / 48)
    # In the scene's Y-up frame the node is the MuJoCo camera behind the world node's turn.
    np.testing.assert_allclose(glb.globals()[n], _W @ _cam_world(model, data), atol=1e-6)


def test_the_gltf_camera_is_the_walks(tmp_path):
    model, data = _model(0.4)
    _content, (cam,) = _cameras(model, data)
    out = tmp_path / "scene.glb"
    gltf = write_glb(model, data, out, log=LOG, max_tex_dim=0, cameras=[_sensor(model)])
    (camera,) = [c for c in gltf["cameras"] if c.get("name") == "robot.eye"]
    assert camera["perspective"] == pytest.approx(
        {"yfov": cam.yfov, "aspectRatio": cam.aspect_ratio, "znear": cam.znear, "zfar": cam.zfar}
    )
    (node,) = [n for n in gltf["nodes"] if n.get("extras", {}).get("sensor") == "robot.eye"]
    np.testing.assert_allclose(node["translation"], cam.pos)
    q = node["rotation"]
    np.testing.assert_allclose(
        _quat_to_mat([q[3], q[0], q[1], q[2]]), _quat_to_mat(cam.quat), atol=1e-12
    )
