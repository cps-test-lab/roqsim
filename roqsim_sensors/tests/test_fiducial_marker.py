"""fiducial_marker: config validation, marker generation, body attachment, and a render->detect
round-trip proving a tag placed in the scene actually decodes in a rendered camera image."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

cv2 = pytest.importorskip("cv2")

from roqsim_sensors.plugins.fiducial_marker import (  # noqa: E402
    FiducialMarkerPlugin,
    _dict_name,
)

_MARKER = "roqsim_sensors.plugins.fiducial_marker:FiducialMarkerPlugin"


def _detect_ids(rgb: np.ndarray, family: str) -> list[int]:
    """Ids decoded from an RGB image, handling both the pre- and post-4.7 cv2.aruco APIs."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _dict_name(family)))
    try:
        detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
        _, ids, _ = detector.detectMarkers(gray)
    except AttributeError:  # OpenCV < 4.7
        _, ids, _ = cv2.aruco.detectMarkers(gray, dic)
    return [] if ids is None else ids.flatten().tolist()


# -- generation (no GL needed) --------------------------------------------------------------


@pytest.mark.parametrize("family,mid", [("apriltag_36h11", 7), ("aruco_4x4_50", 3)])
def test_generated_marker_decodes_directly(family, mid):
    plugin = FiducialMarkerPlugin({"family": family, "id": mid, "pose": [0, 0, 0]})
    rgb = plugin._render_marker()
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    assert _detect_ids(rgb, family) == [mid]


def test_dict_name_maps_friendly_families():
    assert _dict_name("apriltag_36h11") == "DICT_APRILTAG_36h11"
    assert _dict_name("aruco_4x4_50") == "DICT_4X4_50"
    assert _dict_name("DICT_5X5_100") == "DICT_5X5_100"


# -- config validation ----------------------------------------------------------------------


def test_validate_config_rejects_bad_input():
    p = FiducialMarkerPlugin()
    assert p.validate_config({"family": "apriltag_36h11", "id": 0}) == [
        "provide exactly one of 'pose' (world) or 'attach_to' (body)"
    ]
    assert "out of range" in " ".join(
        p.validate_config({"family": "apriltag_36h11", "id": 99999, "pose": [0, 0, 0]})
    )
    assert "unknown fiducial family" in " ".join(
        p.validate_config({"family": "nope_1x1", "pose": [0, 0, 0]})
    )
    assert p.validate_config({"family": "aruco_4x4_50", "id": 0, "pose": [0, 0, 0]}) == []


# -- body attachment (compiles; no GL) ------------------------------------------------------


class _BodyScene(Plugin):
    """A single free body named ``rob_link`` to weld a marker onto."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        b = spec.worldbody.add_body(name="rob_link", pos=[0, 0, 0.5])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05])


def test_marker_attaches_to_named_body():
    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_BodyScene": {}},
                {
                    _MARKER: {
                        "family": "apriltag_36h11",
                        "id": 1,
                        "size": 0.04,
                        "attach_to": "link",
                        "prefix": "rob_",
                        "rel_pose": [0, 0, 0.06],
                    },
                    "name": "wristtag",
                },
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wristtag_geom")
    assert gid >= 0
    # the marker geom's parent body is the one we targeted
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rob_link")
    assert m.geom_bodyid[gid] == bid


def test_missing_body_raises():
    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {
                    _MARKER: {"family": "apriltag_36h11", "attach_to": "ghost", "prefix": "x_"},
                }
            ],
        }
    )
    with pytest.raises(Exception, match="not found"):
        Engine(cfg).setup()


# -- render -> detect round-trip (real mujoco.Renderer, needs a GL backend e.g. MUJOCO_GL=egl) --


class _MarkerCamScene(Plugin):
    """A downward-looking camera above the origin, over a grey backdrop."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_light(pos=[0, 0, 1], dir=[0, 0, -1], diffuse=[1, 1, 1])
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], rgba=[0.5, 0.5, 0.5, 1], pos=[0, 0, -0.02]
        )
        spec.worldbody.add_camera(
            name="cam", pos=[0, 0, 0.5], xyaxes=[1, 0, 0, 0, 1, 0], fovy=45, resolution=[640, 480]
        )


def _render_cam(engine: Engine) -> np.ndarray:
    m, d = engine.ctx.model, engine.ctx.data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    renderer = mujoco.Renderer(m, 480, 640)
    try:
        renderer.update_scene(d, camera=cam)
        return renderer.render().copy()
    finally:
        renderer.close()


def test_world_marker_renders_and_decodes():
    family, mid = "apriltag_36h11", 7
    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_MarkerCamScene": {}},
                {
                    _MARKER: {
                        "family": family,
                        "id": mid,
                        "size": 0.1,
                        "pose": [0, 0, 0.001],  # flat on the backdrop, facing +Z up at the camera
                    },
                },
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    engine.step()
    rgb = _render_cam(engine)
    assert rgb.max() > 0  # not a black frame
    assert mid in _detect_ids(rgb, family), "rendered marker did not decode (check vflip/mapping)"
