"""A camera path: keyframes in, a camera at any time out, and what the document may not say."""

from __future__ import annotations

import json
import math

import mujoco
import pytest

from roqsim.camera_path import CameraPath, CameraPathError, Moment, parse_moment
from roqsim.rendering import eye_position, orbit_from_eye


def _path(keyframes, **kw):
    return CameraPath.from_doc({"keyframes": keyframes, **kw})


# -- moments -------------------------------------------------------------------------------------


def test_a_number_is_seconds():
    assert parse_moment("12.5") == 12.5
    assert parse_moment(3) == 3.0


def test_a_moment_is_named_with_an_offset():
    assert parse_moment("onset") == Moment("onset", 0.0)
    assert parse_moment("onset+2.5") == Moment("onset", 2.5)
    assert parse_moment("t_goal - 1") == Moment("t_goal", -1.0)


def test_garbage_is_not_a_moment():
    with pytest.raises(CameraPathError, match="neither a time"):
        parse_moment("2 seconds in")


def test_an_unknown_moment_lists_the_known_ones():
    with pytest.raises(CameraPathError, match="It knows: onset, t_goal"):
        Moment("t_stop").resolve({"onset": 1.0, "t_goal": 5.0})


# -- interpolation --------------------------------------------------------------------------------


def test_holds_before_the_first_and_after_the_last_keyframe():
    path = _path([{"t": 2, "distance": 1.0}, {"t": 4, "distance": 3.0}])
    assert path.at(0)["distance"] == 1.0
    assert path.at(3)["distance"] == 2.0
    assert path.at(9)["distance"] == 3.0


def test_each_key_is_its_own_track():
    """A key stated in some keyframes only interpolates between those, and an unstated key is absent."""
    path = _path(
        [
            {"t": 0, "distance": 1.0, "elevation": -10},
            {"t": 10, "distance": 3.0},
            {"t": 20, "elevation": -50},
        ]
    )
    at5 = path.at(5)
    assert at5["distance"] == 2.0
    assert at5["elevation"] == -20.0  # halfway from -10 (t=0) to -50 (t=20) is 5/20 of the way
    assert "azimuth" not in at5 and "lookat" not in at5
    assert path.keys == {"distance", "elevation"}


def test_azimuth_takes_the_shortest_arc():
    path = _path([{"t": 0, "azimuth": 350}, {"t": 10, "azimuth": 10}])
    assert path.at(5)["azimuth"] == pytest.approx(360.0)  # through 0, not through 180


def test_wrap_false_lets_azimuth_sweep_a_full_orbit():
    path = _path([{"t": 0, "azimuth": 180}, {"t": 10, "azimuth": 540}], wrap=False)
    assert path.at(5)["azimuth"] == pytest.approx(360.0)


def test_smoothstep_starts_and_ends_on_the_keyframes_and_eases_between():
    path = _path([{"t": 0, "elevation": 0.0}, {"t": 1, "elevation": 1.0}], ease="smoothstep")
    assert path.at(0)["elevation"] == 0.0 and path.at(1)["elevation"] == 1.0
    assert path.at(0.5)["elevation"] == pytest.approx(0.5)
    assert path.at(0.25)["elevation"] < 0.25  # slower off the start than linear


def test_lookat_interpolates_as_a_vector():
    path = _path([{"t": 0, "lookat": [0, 0, 0]}, {"t": 2, "lookat": [2, 4, 6]}])
    assert path.at(1)["lookat"] == [1.0, 2.0, 3.0]


# -- eye / target ----------------------------------------------------------------------------------


def test_eye_and_target_round_trip_through_the_orbit_keys():
    eye, target = [3.0, -2.0, 2.0], [0.5, 0.5, 0.3]
    lookat, distance, azimuth, elevation = orbit_from_eye(eye, target)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    assert eye_position(cam) == pytest.approx(eye, abs=1e-9)


def test_eye_target_keyframe_lowers_to_orbit_keys():
    path = _path([{"t": 0, "eye": [1, 0, 1], "target": [0, 0, 0]}])
    got = path.at(0)
    assert set(got) == {"lookat", "distance", "azimuth", "elevation"}
    assert got["distance"] == pytest.approx(math.sqrt(2))
    assert got["elevation"] == pytest.approx(-45.0)


def test_eye_without_target_is_refused():
    with pytest.raises(CameraPathError, match="eye and target go together"):
        _path([{"t": 0, "eye": [1, 0, 1]}])


def test_eye_target_beside_orbit_keys_is_refused():
    with pytest.raises(CameraPathError, match="drop distance"):
        _path([{"t": 0, "eye": [1, 0, 1], "target": [0, 0, 0], "distance": 2}])


# -- anchoring ------------------------------------------------------------------------------------


def test_named_moments_anchor_against_what_the_render_knows():
    path = _path([{"t": "onset", "azimuth": 0}, {"t": "onset+10", "azimuth": 90}])
    assert not path.anchored and path.moments == {"onset"}
    with pytest.raises(CameraPathError, match="must be resolved first"):
        path.at(1.0)
    anchored = path.anchor({"onset": 4.0})
    assert anchored.span == (4.0, 14.0)
    assert anchored.at(9.0)["azimuth"] == pytest.approx(45.0)


def test_keyframes_out_of_order_are_refused():
    with pytest.raises(CameraPathError, match="time order"):
        _path([{"t": 5, "azimuth": 0}, {"t": 1, "azimuth": 1}])


# -- the document -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc, message",
    [
        ({"keyframes": []}, "non-empty"),
        ({"keyframes": [{"t": 0, "zoom": 2}]}, "unknown key"),
        ({"keyframes": [{"azimuth": 2}]}, "no 't'"),
        ({"keyframes": [{"t": 0}]}, "no camera key"),
        ({"keyframes": [{"t": 0, "distance": -1}]}, "positive"),
        ({"keyframes": [{"t": 0, "lookat": [1, 2]}]}, "three numbers"),
        ({"keyframes": [{"t": 0, "azimuth": "ninety"}]}, "expected a number"),
        ({"keyframes": [{"t": 0, "azimuth": 1}], "ease": "bounce"}, "ease"),
        ({"keyframes": [{"t": 0, "azimuth": 1}], "loop": True}, "unknown key"),
    ],
)
def test_a_bad_document_says_what_is_wrong(doc, message):
    with pytest.raises(CameraPathError, match=message):
        CameraPath.from_doc(doc)


def test_a_bare_list_is_keyframes():
    assert CameraPath.from_doc([{"t": 0, "azimuth": 1}]).keys == {"azimuth"}


def test_from_arg_takes_inline_json_or_a_file(tmp_path):
    inline = CameraPath.from_arg(json.dumps({"keyframes": [{"t": 1, "distance": 2}]}))
    assert inline.at(0)["distance"] == 2.0
    file = tmp_path / "p.yaml"
    file.write_text("ease: smoothstep\nkeyframes:\n  - {t: onset, azimuth: 30}\n")
    from_file = CameraPath.from_arg(str(file))
    assert from_file.ease == "smoothstep" and from_file.moments == {"onset"}
    with pytest.raises(CameraPathError, match="no such file"):
        CameraPath.from_arg(str(tmp_path / "missing.yaml"))


def test_describe_reports_the_resolved_keyframes():
    path = _path([{"t": "onset", "azimuth": 0}]).anchor({"onset": 2.0})
    described = path.describe()
    assert described["keyframes"] == 1 and described["keys"] == ["azimuth"]
    assert described["resolved"] == [{"t": 2.0, "azimuth": 0.0}]


# -- apply ----------------------------------------------------------------------------------------


def test_apply_writes_the_free_camera():
    cam = mujoco.MjvCamera()
    _path([{"t": 0, "lookat": [1, 2, 3], "distance": 4, "azimuth": 5, "elevation": -6}]).apply(
        cam, 0
    )
    assert list(cam.lookat) == [1.0, 2.0, 3.0]
    assert (cam.distance, cam.azimuth, cam.elevation) == (4.0, 5.0, -6.0)


def test_apply_leaves_lookat_to_a_tracking_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.lookat[:] = [9, 9, 9]
    _path([{"t": 0, "lookat": [1, 2, 3], "distance": 4}]).apply(cam, 0)
    assert list(cam.lookat) == [9.0, 9.0, 9.0] and cam.distance == 4.0


def test_apply_animates_the_offset_under_follow_heading():
    """The azimuth goes through the tracking controller, never the raw struct (see TrackingCamera)."""

    class _Tracking:
        follow_heading = True
        azimuth_offset = 0.0

    cam = mujoco.MjvCamera()
    cam.azimuth = 45.0
    tracking = _Tracking()
    _path([{"t": 0, "azimuth": 180}]).apply(cam, 0, tracking=tracking)
    assert tracking.azimuth_offset == 180.0 and cam.azimuth == 45.0
