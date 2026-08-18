"""The framing helpers in :mod:`roqsim.rendering` -- ``autoframe``, ``preview_camera``, and the
occlusion-aware ``focus_camera``.

Pure geometry (MjModel/MjData + ``mj_forward``); no GL context needed, so these run headless.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from roqsim.rendering import (
    _line_of_sight_clear,
    autoframe,
    bounding_sphere,
    default_free_camera,
    dolly,
    eye_position,
    focus_camera,
    look_in_place,
    preview_camera,
    walk_delta,
)

# A small prop (a 0.25 x 0.25 x 0.5 m box standing on the floor at the origin) inside a big room: a
# ground plane and a wall 8 m away. Autoframing the prop must ignore the room entirely.
_XML = """
<mujoco>
  <visual><global fovy="45"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <geom name="wall" type="box" pos="8 0 1" size="0.1 8 1"/>
    <body name="prop" pos="0 0 0">
      <geom type="box" size="0.25 0.25 0.5" pos="0 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_data():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_autoframe_zooms_onto_the_prop_not_the_room(model_data):
    model, data = model_data
    cam = autoframe(model, data, [_bid(model, "prop")], aspect=1.0)
    # Centred on the box (z-mid 0.5), and far closer than framing the whole 20 m room would put it.
    assert list(cam.lookat) == pytest.approx([0.0, 0.0, 0.5], abs=1e-3)
    assert cam.distance == pytest.approx(1.76, abs=0.1)
    assert cam.distance < default_free_camera(model).distance / 3


def test_autoframe_distance_fills_the_field_of_view(model_data):
    model, data = model_data
    cam = autoframe(model, data, [_bid(model, "prop")], aspect=1.0)
    # The bounding sphere (box half-diagonal ~0.61 m) should subtend ~the full vertical FOV, with the
    # 1.1 margin: half-angle == asin(radius/distance) ~= fovy/2 / 1.1.
    radius = math.sqrt(0.25**2 + 0.25**2 + 0.5**2)
    half_angle = math.asin(radius / cam.distance)
    assert half_angle == pytest.approx(math.radians(45) / 2 / 1.1, rel=0.05)


def test_autoframe_narrow_aspect_backs_off(model_data):
    model, data = model_data
    wide = autoframe(model, data, [_bid(model, "prop")], aspect=1.5)
    tall = autoframe(model, data, [_bid(model, "prop")], aspect=0.5)
    # A tall (narrow) window is horizontally tighter, so the camera must sit further back to fit.
    assert tall.distance > wide.distance


def test_autoframe_empty_falls_back_to_default(model_data):
    model, data = model_data
    cam = autoframe(model, data, [], aspect=1.0)
    assert cam.distance == pytest.approx(default_free_camera(model).distance)


def test_preview_camera_frames_the_named_entities(model_data):
    model, data = model_data
    entities = [
        SimpleNamespace(body="prop"),
        SimpleNamespace(body="nonexistent"),  # unresolved name -> skipped
        SimpleNamespace(body=None),            # no body -> skipped
    ]
    # Same result as autoframing the resolved body directly -- the room's wall never enters the frame.
    cam = preview_camera(model, data, entities, aspect=1.0)
    assert list(cam.lookat) == pytest.approx([0.0, 0.0, 0.5], abs=1e-3)
    assert cam.distance == pytest.approx(1.76, abs=0.1)


def test_preview_camera_no_resolvable_entities_falls_back(model_data):
    model, data = model_data
    cam = preview_camera(model, data, [SimpleNamespace(body=None)], aspect=1.0)
    assert cam.distance == pytest.approx(default_free_camera(model).distance)


def test_bounding_sphere_encloses_the_prop_and_none_when_empty(model_data):
    model, data = model_data
    center, radius = bounding_sphere(model, data, [_bid(model, "prop")])
    assert list(center) == pytest.approx([0.0, 0.0, 0.5], abs=1e-3)
    assert radius == pytest.approx(math.sqrt(0.25**2 + 0.25**2 + 0.5**2), rel=0.05)
    assert bounding_sphere(model, data, []) is None


def _focus_los_clear(model, data, cam):
    """Re-run the line-of-sight predicate on a returned camera (its eye -> the prop)."""
    center, radius = bounding_sphere(model, data, [_bid(model, "prop")])
    return _line_of_sight_clear(model, data, eye_position(cam), center, radius)


# The prop's default orbit eye sits on the -y side (azimuth 90, elevation -45); each scene below
# differs only in what stands between that eye and the prop.
_PROP = '<body name="prop" pos="0 0 0"><geom type="box" size="0.25 0.25 0.5" pos="0 0 0.5"/></body>'


def _scene(walls: str, prop: str = _PROP) -> tuple:
    xml = (
        '<mujoco><visual><global fovy="45"/></visual><worldbody>'
        '<geom name="floor" type="plane" size="10 10 0.1"/>'
        f"{walls}{prop}</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_focus_camera_keeps_the_default_angle_when_unoccluded():
    # A wall 8 m away never crosses the default -y viewpoint, so focus keeps today's orbit angle.
    model, data = _scene('<geom name="wall" type="box" pos="8 0 1" size="0.1 8 1"/>')
    default = default_free_camera(model)
    cam = focus_camera(model, data, [_bid(model, "prop")], aspect=1.0)
    assert cam.azimuth == pytest.approx(default.azimuth)
    assert cam.elevation == pytest.approx(default.elevation)
    assert list(cam.lookat) == pytest.approx([0.0, 0.0, 0.5], abs=1e-3)  # same framing as autoframe
    assert _focus_los_clear(model, data, cam)


def test_focus_camera_rotates_off_an_occluding_wall():
    # A wall on the -y side blocks the default viewpoint; focus must turn to a clear line of sight.
    model, data = _scene('<geom name="wall" type="box" pos="0 -0.8 1.3" size="2 0.1 2"/>')
    default = default_free_camera(model)
    cam = focus_camera(model, data, [_bid(model, "prop")], aspect=1.0)
    assert cam.azimuth != pytest.approx(default.azimuth)  # it actually moved
    assert list(cam.lookat) == pytest.approx([0.0, 0.0, 0.5], abs=1e-3)  # framing preserved
    assert _focus_los_clear(model, data, cam)


def test_focus_camera_falls_back_to_default_framing_when_fully_enclosed():
    # A closed box (4 walls + ceiling) leaves no unoccluded angle: focus returns the default framing
    # rather than raising, so the human can still orbit.
    box = (
        '<geom type="box" pos="0 -0.6 0.6" size="0.6 0.05 0.6"/>'
        '<geom type="box" pos="0 0.6 0.6" size="0.6 0.05 0.6"/>'
        '<geom type="box" pos="-0.6 0 0.6" size="0.05 0.6 0.6"/>'
        '<geom type="box" pos="0.6 0 0.6" size="0.05 0.6 0.6"/>'
        '<geom type="box" pos="0 0 1.2" size="0.6 0.6 0.05"/>'
    )
    model, data = _scene(box, prop='<body name="prop" pos="0 0 0">'
                              '<geom type="box" size="0.25 0.25 0.5" pos="0 0 0.3"/></body>')
    default = default_free_camera(model)
    cam = focus_camera(model, data, [_bid(model, "prop")], aspect=1.0)
    assert cam.azimuth == pytest.approx(default.azimuth)
    assert cam.elevation == pytest.approx(default.elevation)


def test_focus_camera_empty_falls_back_to_default():
    model, data = _scene("")
    cam = focus_camera(model, data, [], aspect=1.0)
    assert cam.distance == pytest.approx(default_free_camera(model).distance)


def test_walk_delta_follows_camera_heading():
    # azimuth 0 = looking along +x: W walks +x, D strafes -y (right hand of the view), E rises.
    assert list(np.round(walk_delta(0.0, {"w"}, 1.0), 6)) == [1.0, 0.0, 0.0]
    assert list(np.round(walk_delta(0.0, {"s"}, 1.0), 6)) == [-1.0, 0.0, 0.0]
    assert list(np.round(walk_delta(0.0, {"d"}, 1.0), 6)) == [0.0, -1.0, 0.0]
    assert list(np.round(walk_delta(0.0, {"a"}, 1.0), 6)) == [0.0, 1.0, 0.0]
    assert list(np.round(walk_delta(0.0, {"e"}, 1.0), 6)) == [0.0, 0.0, 1.0]
    assert list(np.round(walk_delta(0.0, {"q"}, 1.0), 6)) == [0.0, 0.0, -1.0]
    # turned 90 deg, forward is +y
    assert list(np.round(walk_delta(90.0, {"w"}, 1.0), 6)) == [0.0, 1.0, 0.0]


def test_walk_delta_flies_along_the_look_direction():
    # looking 30 deg down, W descends with the view (spectator-camera feel), A/D stay level
    down = walk_delta(0.0, {"w"}, 1.0, -30.0)
    assert down[2] == pytest.approx(-0.5) and down[0] == pytest.approx(math.cos(math.radians(30)))
    assert walk_delta(0.0, {"d"}, 1.0, -30.0)[2] == 0.0  # strafing never leaves the horizontal


def test_walk_delta_normalises_diagonals():
    diagonal = walk_delta(0.0, {"w", "d"}, 1.0)
    assert round(float(np.linalg.norm(diagonal)), 6) == 1.0  # not faster than a single key
    assert list(np.round(walk_delta(0.0, {"w", "s"}, 1.0), 6)) == [0.0, 0.0, 0.0]  # cancel out


def test_walk_delta_ignores_unknown_and_empty():
    assert list(walk_delta(0.0, set(), 1.0)) == [0.0, 0.0, 0.0]
    assert list(walk_delta(0.0, {"x", "return"}, 1.0)) == [0.0, 0.0, 0.0]
    assert list(np.round(walk_delta(0.0, {"w", "x"}, 2.5), 6)) == [2.5, 0.0, 0.0]


def _free_cam(**kw):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.0, 0.0, 1.0]
    cam.distance = 4.0
    cam.azimuth = 0.0
    cam.elevation = 0.0
    for k, v in kw.items():
        setattr(cam, k, v)
    return cam


def test_look_in_place_pivots_on_the_eye_not_the_lookat():
    cam = _free_cam()
    eye = eye_position(cam).copy()
    look_in_place(cam, 0.25, 0.0)  # a quarter-height drag = 45 deg at the default sensitivity
    assert cam.azimuth == pytest.approx(-45.0)
    # the head turned: the eye did not move, and lookat swung to the new heading at the same radius
    assert eye_position(cam) == pytest.approx(eye)
    assert np.linalg.norm(np.asarray(cam.lookat) - eye) == pytest.approx(cam.distance)


def test_look_in_place_clamps_at_the_poles():
    cam = _free_cam()
    look_in_place(cam, 0.0, -10.0)  # a drag far past straight up
    assert cam.elevation == pytest.approx(89.9)
    look_in_place(cam, 0.0, 10.0)
    assert cam.elevation == pytest.approx(-89.9)


def test_dolly_flies_along_the_view_and_keeps_the_orbit_radius():
    cam = _free_cam(azimuth=90.0)  # looking along +y, level
    dolly(cam, 2.0)
    assert list(np.round(cam.lookat, 6)) == [1.0, 2.0, 1.0]
    assert cam.distance == pytest.approx(4.0)  # the pivot travels with the eye, it does not shrink
    dolly(cam, -2.0)
    assert list(np.round(cam.lookat, 6)) == [1.0, 0.0, 1.0]
