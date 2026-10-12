"""A camera plugin declares the image it publishes as a scene camera, which a scene export places."""

from __future__ import annotations

import logging
import math

import pytest
import yaml
from test_cameras import _world

from roqsim.engine import Engine
from roqsim.export_web import export_scene
from roqsim.scene_source import compile_world, declared_cameras

OAKD = "roqsim_sensors.plugins.oakd_camera:OakDCameraPlugin"
SEGMENTATION = "roqsim_sensors.plugins.segmentation_camera:SegmentationCameraPlugin"


def _declared(plugin_ref: str, **config):
    engine = Engine(_world(plugin_ref, **config))
    engine.setup()
    try:
        return engine, declared_cameras(engine)
    finally:
        engine.shutdown()


def test_a_colour_camera_declares_its_image_with_the_intrinsics_it_publishes():
    engine, (cam,) = _declared(OAKD)
    plugin = engine.plugins[-1]
    assert cam.sensor == plugin.address
    assert cam.camera == engine.ctx.model.camera("cam").id
    assert cam.kind == "rgb"
    assert (cam.width, cam.height) == (64, 48)
    f = 48 / (2 * math.tan(math.radians(45) / 2))
    assert (cam.fx, cam.fy) == (pytest.approx(f), pytest.approx(f))


def test_a_configured_size_and_fov_are_the_declared_ones():
    _engine, (cam,) = _declared(OAKD, width=128, height=64, fovy=60)
    assert (cam.width, cam.height) == (128, 64)
    assert cam.fy == pytest.approx(64 / (2 * math.tan(math.radians(60) / 2)))


def test_a_depth_camera_without_colour_declares_its_depth_image():
    _engine, (cam,) = _declared(OAKD, color=False)
    assert cam.kind == "depth"


def test_a_label_camera_declares_nothing():
    """Its colour stream is off by default, and labels are not an image a viewer renders."""
    classes = [{"class_id": 1, "name": "box", "bodies": ["*"]}]
    assert _declared(SEGMENTATION, classes=classes)[1] == []


def test_a_world_export_places_its_sensor_camera(tmp_path):
    world = tmp_path / "world.yaml"
    world.write_text(
        yaml.safe_dump(
            {
                "sim": {},
                "components": [
                    {"test_cameras:_CameraScene": {}},
                    {OAKD: {"camera": "cam", "depth_camera": "cam"}},
                ],
            }
        )
    )
    log = logging.getLogger("t")
    src = compile_world(world, set(), {}, log)
    scene = export_scene(src.model, src.data, tmp_path / "out", log, cameras=src.cameras)
    (cam,) = scene["cameras"]
    assert cam["kind"] == "rgb" and (cam["width"], cam["height"]) == (64, 48)
    assert cam["perspective"]["yfov"] == pytest.approx(math.radians(45))
    assert scene["bodies"][cam["body"]]["name"] == "world"
