# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A views file: its conventions, how it is read and refused, and drawing it through one renderer."""

from __future__ import annotations

import hashlib
import json
import math

import mujoco
import numpy as np
import pytest
import yaml
from PIL import Image

from roqsim import views as V
from roqsim.render_api import IMAGES, RenderError
from roqsim.rendering import MujocoRenderer, RenderService, camera_intrinsics

#: One camera of the tests: 1.5 m up, looking along +x and 20 degrees down, rolled 5 degrees, with
#: non-square pixels and an off-centre principal point (pixel centres).
W, H = 120, 90
FX, FY, CX, CY = 100.0, 104.0, 61.25, 41.5
POSITION = np.array([-1.0, 0.2, 1.5])


def _c2w_mujoco() -> np.ndarray:
    """The camera's axes in the world as columns, MuJoCo's frame (x right, y up, -z forward)."""
    pitch, roll = math.radians(-20), math.radians(5)
    fwd = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
    right0 = np.cross(fwd, [0.0, 0.0, 1.0])
    right0 /= np.linalg.norm(right0)
    up0 = np.cross(right0, fwd)
    right = math.cos(roll) * right0 - math.sin(roll) * up0
    up = np.cross(right, fwd)
    return np.column_stack([right, up, -fwd])


def _quat(mat) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat).reshape(-1))
    return q.tolist()


def _entry(**pose) -> dict:
    return {"name": "v", "width": W, "height": H, "fx": FX, "fy": FY, **pose}


def _six_ways() -> dict[str, tuple]:
    """The same camera in every convention, as a file would write it."""
    c2w_mj = _c2w_mujoco()
    c2w_cv = c2w_mj @ np.diag([1.0, -1.0, -1.0])
    w2c_cv = c2w_cv.T
    t = (-w2c_cv @ POSITION).tolist()
    q_ros = _quat(c2w_cv)
    return {
        "colmap R position": (
            "colmap",
            _entry(cx=CX + 0.5, cy=CY + 0.5, R=w2c_cv.tolist(), position=POSITION.tolist()),
        ),
        "colmap quat t": ("colmap", _entry(cx=CX + 0.5, cy=CY + 0.5, quat=_quat(w2c_cv), t=t)),
        "opencv R t": ("opencv", _entry(cx=CX, cy=CY, R=w2c_cv.tolist(), t=t)),
        "ros": (
            "ros",
            _entry(
                cx=CX,
                cy=CY,
                quat={"x": q_ros[1], "y": q_ros[2], "z": q_ros[3], "w": q_ros[0]},
                position=dict(zip("xyz", POSITION.tolist(), strict=True)),
            ),
        ),
        "mujoco": ("mujoco", _entry(cx=CX, cy=CY, quat=_quat(c2w_mj), position=POSITION.tolist())),
        "keys": (
            {"axes": "mujoco", "rotation": "camera_to_world", "pixel_origin": "corner"},
            _entry(cx=CX + 0.5, cy=CY + 0.5, R=c2w_mj.tolist(), position=POSITION.tolist()),
        ),
    }


def _same(a: V.View, b: V.View) -> None:
    assert (a.width, a.height) == (b.width, b.height)
    np.testing.assert_allclose(a.intrinsics, b.intrinsics, atol=1e-9)
    np.testing.assert_allclose(a.pos, b.pos, atol=1e-9)
    np.testing.assert_allclose(a.rotation, b.rotation, atol=1e-9)


# -- the conventions -----------------------------------------------------------------------------


def test_one_camera_written_six_ways_is_one_view():
    ways = {k: V.View.from_pose(c, **e) for k, (c, e) in _six_ways().items()}
    ref = ways["mujoco"]
    np.testing.assert_allclose(ref.rotation, _c2w_mujoco(), atol=1e-12)
    for view in ways.values():
        _same(view, ref)


@pytest.mark.parametrize("preset", sorted(V.PRESETS))
def test_every_preset_is_its_three_keys(preset):
    by_name = V.read_convention(preset)
    by_keys = V.read_convention(V.PRESETS[preset].keys())
    assert by_keys.keys() == by_name.keys() and by_keys.preset == preset
    assert V.PRESETS["opencv"].keys() == {
        "axes": "opencv",
        "rotation": "world_to_camera",
        "pixel_origin": "centre",
    }


def test_the_opencv_preset_puts_a_point_where_opencv_does():
    cv2 = pytest.importorskip("cv2")
    rvec = np.array([0.3, -1.2, 0.4])
    tvec = np.array([0.2, -0.1, 2.5])
    k = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
    point = np.array([[0.4, -0.3, 0.6]])
    (uv,), _ = cv2.projectPoints(point, rvec, tvec, k, None)
    rot, _ = cv2.Rodrigues(rvec)
    view = V.View.from_pose("opencv", **_entry(cx=CX, cy=CY, R=rot.tolist(), t=tvec.tolist()))
    local = (point[0] - np.array(view.pos)) @ view.rotation  # MuJoCo's camera frame
    x, y, z = local[0], -local[1], -local[2]
    fx, fy, cx, cy = view.intrinsics
    np.testing.assert_allclose([fx * x / z + cx, fy * y / z + cy], uv.ravel(), atol=1e-6)


def test_a_file_reads_in_json_and_yaml_alike(tmp_path):
    conv, entry = _six_ways()["colmap R position"]
    doc = {"convention": conv, "near": 0.05, "far": 20.0, "views": [entry]}
    (tmp_path / "a.json").write_text(json.dumps(doc))
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(doc))
    a, b = V.load_views(tmp_path / "a.json"), V.load_views(tmp_path / "a.yaml")
    _same(a[0], b[0])
    assert a[0].near == 0.05 and a[0].far == 20.0 and a.convention.preset == "colmap"
    assert len(a.sha256) == 64


# -- refusals ------------------------------------------------------------------------------------


def _refused(doc, match, tmp_path):
    path = tmp_path / "views.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(V.ViewsError, match=match):
        V.load_views(path)


def _doc(conv="mujoco", **override):
    entry = _six_ways()["mujoco"][1]
    entry.update(override)
    entry = {k: v for k, v in entry.items() if v is not None}  # None: the key left out
    return {"convention": conv, "views": [entry]}


def test_a_file_without_a_convention_is_refused_naming_the_presets_and_the_command(tmp_path):
    doc = _doc()
    del doc["convention"]
    _refused(doc, r"no convention.*colmap, opencv, ros, mujoco.*roqsim views describe", tmp_path)


def test_an_unknown_preset_is_refused_naming_the_presets_the_keys_and_the_command(tmp_path):
    with pytest.raises(V.ViewsError) as err:
        V.read_convention("colmpa")
    text = str(err.value)
    for want in (*V.PRESETS, *V.KEYS, "did you mean 'colmap'", "roqsim views describe"):
        assert want in text


def test_a_convention_of_some_keys_or_a_wrong_value_is_refused(tmp_path):
    with pytest.raises(V.ViewsError, match="missing pixel_origin"):
        V.read_convention({"axes": "opencv", "rotation": "world_to_camera"})
    with pytest.raises(V.ViewsError, match="axes: 'opengl'"):
        V.read_convention(
            {"axes": "opengl", "rotation": "world_to_camera", "pixel_origin": "corner"}
        )


@pytest.mark.parametrize(
    ("conv", "override", "match"),
    [
        ("mujoco", {"pos": [0, 0, 1]}, r"unknown key 'pos' \(did you mean 'position'\?\)"),
        ("mujoco", {"near": 2.0, "far": 1.0}, "not beyond the near one"),
        ("mujoco", {"near": 0}, "'near' is a distance > 0"),
        ("mujoco", {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 2]], "quat": None}, "not a rotation"),
        ("mujoco", {"t": [0, 0, 1]}, "'t' is for a world_to_camera rotation"),
        ("mujoco", {"fx": -1.0}, "'fx' must be > 0"),
        ("mujoco", {"width": 0}, "'width' must be > 0"),
        ("mujoco", {"name": "a/b"}, "'name' is a file stem"),
        ("ros", {}, r"'quat' under ros is the mapping \{x, y, z, w\}"),
    ],
)
def test_a_view_that_cannot_be_read_as_written_is_refused_by_name(conv, override, match, tmp_path):
    _refused(_doc(conv, **override), match, tmp_path)


def test_a_name_used_twice_is_refused(tmp_path):
    doc = _doc()
    doc["views"].append(dict(doc["views"][0]))
    _refused(doc, "used twice", tmp_path)


# -- how it was read -----------------------------------------------------------------------------


def test_describe_says_where_the_camera_stands_and_looks_and_where_its_principal_point_lies():
    view = V.View.from_pose("mujoco", **_six_ways()["mujoco"][1])
    reading = view.describe()
    assert reading["position"] == pytest.approx(POSITION.tolist())
    assert reading["toward"] == "+x" and reading["elevation_deg"] == pytest.approx(-20.0)
    assert reading["roll_deg"] == pytest.approx(5.0)
    assert reading["principal_offset_px"] == {"right": 1.75, "below": -3.0}
    assert reading["warnings"] == []
    assert "toward +x, 20° down" in reading["text"] and "rolled 5° right" in reading["text"]
    assert "1.8 px right of and 3.0 px above the image centre" in reading["text"]


def test_a_camera_read_with_the_wrong_axes_reads_upside_down():
    conv, entry = _six_ways()["colmap R position"]
    wrong = V.View.from_pose({**V.PRESETS[conv].keys(), "axes": "mujoco"}, **entry)
    reading = wrong.describe()
    assert reading["toward"] == "-x"  # looking backwards
    assert any("upside down" in w for w in reading["warnings"])


def test_a_principal_point_far_from_the_centre_is_said():
    entry = _six_ways()["mujoco"][1]
    far = V.View.from_pose("mujoco", **{**entry, "cx": W * 0.9})
    out = V.View.from_pose("mujoco", **{**entry, "cx": W + 5.0})
    assert any("quarter" in w for w in far.describe()["warnings"])
    assert any("outside the image" in w for w in out.describe()["warnings"])


# -- drawing -------------------------------------------------------------------------------------

XML = """
<mujoco>
  <visual><quality offsamples="0"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".5 .6 .7 1"/>
    <geom name="box" type="box" pos="1.5 0 .3" size=".3 .3 .3" rgba="1 0 0 1"/>
    <camera name="cam" pos="-1 0.2 1.5" xyaxes="0 -1 0 .34 0 .94" fovy="50"/>
  </worldbody>
</mujoco>
"""


def _world():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_a_view_of_a_file_is_the_model_camera_s_picture():
    model, data = _world()
    intr = camera_intrinsics(model, 0, W, H, float(model.cam_fovy[0]))
    view = V.View.from_pose(
        "mujoco",
        **_entry(
            fx=intr[0],
            fy=intr[1],
            cx=intr[2],
            cy=intr[3],
            quat=_quat(np.asarray(data.cam_xmat[0]).reshape(3, 3)),
            position=np.asarray(data.cam_xpos[0]).tolist(),
        ),
    )
    with V.render_views((model, data), [view], images=IMAGES) as rendered:
        ((_, got),) = list(rendered)
    service = RenderService(model)
    ref_view = service.view("ref", camera=0, width=W, height=H, images=IMAGES, near=V.DEFAULT_NEAR)
    (ref,) = service.render_now([(ref_view, IMAGES)], data)
    service.close()
    for kind in IMAGES:
        assert np.array_equal(got.image(kind), ref.image(kind)), kind


def test_colour_is_drawn_unless_more_is_asked_for():
    model, data = _world()
    view = V.View.from_pose("mujoco", **_six_ways()["mujoco"][1])
    ((_, frame),) = list(V.render_views((model, data), [view]))
    assert frame.rgb is not None and frame.depth is None and frame.segmentation is None


def test_a_thousand_views_of_three_sizes_open_one_renderer(monkeypatch):
    opened = []
    original = MujocoRenderer.open
    monkeypatch.setattr(
        MujocoRenderer, "open", lambda self, scene: (opened.append(1), original(self, scene))[1]
    )
    model, data = _world()
    entry = _six_ways()["mujoco"][1]
    views = []
    for i in range(1000):
        w, h = ((40, 30), (64, 48), (32, 32))[i % 3]
        views.append(
            V.View.from_pose(
                "mujoco",
                **{**entry, "name": f"v{i}", "width": w, "height": h, "cx": w / 2, "cy": h / 2},
            )
        )
    rendered = V.render_views((model, data), views)
    sizes = [(v.name, f.rgb.shape) for v, f in rendered]
    assert len(sizes) == 1000 and opened == [1]
    assert all(shape == (v.height, v.width, 3) for (_, shape), v in zip(sizes, views, strict=True))
    assert len(rendered.batches()) == math.ceil(1000 / V.BATCH_VIEWS)


def test_an_image_the_renderer_does_not_give_is_refused_before_anything_is_drawn(tmp_path):
    path = tmp_path / "rgbonly.py"
    path.write_text(
        "from roqsim.render_api import Renderer, RendererCapabilities\n"
        "class RgbOnly(Renderer):\n"
        "    capabilities = RendererCapabilities(posed=True, principal_point=True,\n"
        "                                        non_square_pixels=True)\n"
    )
    model, data = _world()
    view = V.View.from_pose("mujoco", **_six_ways()["mujoco"][1])
    with pytest.raises(RenderError, match="does not give image 'depth'"):
        V.render_views((model, data), [view], images=("rgb", "depth"), renderer=f"{path}:RgbOnly")


def test_written_views_are_lossless_and_named_in_the_manifest(tmp_path):
    model, data = _world()
    view = V.View.from_pose("mujoco", **_six_ways()["mujoco"][1])
    rendered = V.render_views((model, data), [view], images=IMAGES)
    manifest = V.write_views(tmp_path, rendered, {"target": "test"})
    files = manifest["views"][0]["files"]
    rgb = np.array(Image.open(tmp_path / files["rgb"]["path"]))
    depth = np.load(tmp_path / files["depth"]["path"])
    seg = np.array(Image.open(tmp_path / files["segmentation"]["path"]))
    assert rgb.dtype == np.uint8 and rgb.shape == (H, W, 3)
    assert depth.dtype == np.float32 and np.isinf(depth).any()
    assert seg.dtype == np.uint16
    legend = {e["index"]: e for e in manifest["legend"]}
    assert set(np.unique(seg)) <= set(legend)
    assert {legend[i]["name"] for i in np.unique(seg) if i} == {"floor", "box"}
    assert manifest["depth"] == V.DEPTH_CONVENTION and manifest["images"] == list(IMAGES)
    assert json.loads((tmp_path / "manifest.json").read_text())["target"] == "test"
    for record in files.values():
        digest = hashlib.sha256((tmp_path / record["path"]).read_bytes()).hexdigest()
        assert digest == record["sha256"]
