"""The RealSense devices are mounted where the vendor macro's origin places them, and no camera moved.

Four claims, each against the vendor's published numbers rather than against the model:

* **The chain is the vendor's.** Each mount is ``camera_bottom_screw_frame``, the frame
  ``realsense2_description``'s macro attaches to its parent at the origin it is given, with
  ``camera_link`` at the macro's fixed offset from it. It publishes ``camera_bottom_screw_frame ->
  camera_link -> camera_color_frame -> camera_color_optical_frame`` and the depth frames with the
  offsets the vendor states. The MuJoCo camera colour is rendered from IS the colour optical frame,
  so a consumer that looks a pixel up in TF finds where it was taken.
* **Depth is rendered from the depth frame.** The depth camera IS the depth optical frame the
  published TF puts together, with the depth stream's data-sheet optics, and a surface's depth,
  reprojected and carried through that TF, lands on the surface.
* **The re-seat moved names, not cameras.** The retired ``d415``/``d435``/``d455`` models pre-rotated
  ``mount`` so a mount at ``rpy [0, 0, 0]`` looked along ``+y``. A mount of the old model at ``T_old``
  and one of the new at ``T_old * D`` (``D = Rq * T_link_mesh^-1 * T_screw_link^-1``, from the
  constants below) put the housing and the D435's and D455's cameras at the same world pose to float
  tolerance. So do the mounts ``vendor_mount.rewrite_mounts`` writes (what
  ``external/convert/build_realsense_devices.py --rewrite-mounts`` runs) and the demo world's. The
  retired D415's camera was not at its colour lens -- centred on the housing, 5 mm in front of the
  glass -- and sits at the vendor colour frame now; that one move is asserted as what it is.
* **The old names are refused**, naming the new one."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import ModelError, resolve_model

OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# realsense2_description @ realsense-ros 4.56.1: the `${name}_link_joint` origin (`camera_link` in the
# bottom screw frame), the `${name}_link` visual origin used with the mesh, and the
# `${name}_color_joint` origin, the last two in `${name}_link`. Every depth frame is `${name}_link`'s.
VENDOR = {
    "realsense_d415": {
        # (0, d415_cam_depth_py, d415_cam_depth_pz)
        "screw": (0.0, 0.020, 0.0115),
        # (d415_cam_mount_from_center_offset, -d415_cam_depth_py, 0)
        "mesh": (0.00987, -0.020, 0.0),
        "color": (0.0, 0.015, 0.0),
        "cam": "d415",
    },
    "realsense_d435": {
        # (d435_mesh_x_offset = 0.0149 - 0.1e-3 - 4.2e-3, d435_cam_depth_py, d435_cam_depth_pz)
        "screw": (0.0106, 0.0175, 0.0125),
        # (d435_zero_depth_to_glass + d435_glass_to_front, -d435_cam_depth_py, 0)
        "mesh": (0.0043, -0.0175, 0.0),
        "color": (0.0, 0.015, 0.0),
        "cam": "d435",
    },
    "realsense_d455": {
        # (d455_mesh_x_offset, d455_cam_depth_py, d455_cam_depth_pz)
        "screw": (0.01115, 0.0475, 0.0145),
        # (d455_zero_depth_to_glass + d455_glass_to_front, -d455_cam_depth_py, 0)
        "mesh": (0.00465, -0.0475, 0.0),
        "color": (0.0, -0.059, 0.0),
        "cam": "d455",
    },
}
#: The depth stream, from the Intel RealSense D400 Series data sheet (document 337029): depth FOV
#: 65 x 40 deg (D415), 87 x 58 deg (D435, D455); the vertical angle and the resolution rendered.
DEPTH_OPTICS = {
    "realsense_d415": (40.0, 848, 480),
    "realsense_d435": (58.0, 848, 480),
    "realsense_d455": (58.0, 848, 480),
}
MESH_RPY = (math.pi / 2, 0.0, math.pi / 2)  # every one of the three

# The retired models, as they were: a `mount` pre-rotated by this quaternion (mesh +z -> +y, mesh +y
# -> +z), holding the mesh in its own axes and the colour camera at this mesh-local pose.
RETIRED_QUAT = (0.0, 0.0, 0.70710678, 0.70710678)
RETIRED_CAMERA = {
    "realsense_d415": {"pos": (0.0, 0.0, 0.005), "xyaxes": (-1, 0, 0, 0, 1, 0)},
    "realsense_d435": {"pos": (0.0325, 0.0, -0.0043), "xyaxes": (-1, 0, 0, 0, 1, 0)},
    "realsense_d455": {"pos": (-0.0115, 0.0, -0.00465), "xyaxes": (-1, 0, 0, 0, 1, 0)},
}

POSES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([1.5, 0.0, 0.5], [0.0, 0.0, 1.5708]),
    ([16.069, 8.678, 3.019], [-0.611, 0.0, 2.3964]),
    ([-0.3, 2.0, 1.1], [0.4, -0.7, -2.9]),
]


def _rot(rpy) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(rpy, dtype=float), "XYZ")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _quat_rot(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q / np.linalg.norm(q))
    return mat.reshape(3, 3)


def _rpy(m: np.ndarray) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    return [math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])]


def _delta(model: str) -> tuple[np.ndarray, np.ndarray]:
    r_lm, t_lm = _rot(MESH_RPY), np.asarray(VENDOR[model]["mesh"])
    r = _quat_rot(RETIRED_QUAT) @ r_lm.T
    return r, -r @ (t_lm + np.asarray(VENDOR[model]["screw"]))


def _re_expressed(model, pos, rpy):
    r_d, t_d = _delta(model)
    r_old = _rot(rpy)
    return (np.asarray(pos) + r_old @ t_d).tolist(), _rpy(r_old @ r_d)


def _retired_mjcf(tmp_path, model) -> str:
    cam = RETIRED_CAMERA[model]
    short = VENDOR[model]["cam"]
    path = tmp_path / f"retired_{short}.xml"
    # The very mesh the model ships, so the two housings are one mesh placed twice (MuJoCo puts a
    # mesh geom's frame at the mesh's own centroid, which a stand-in shape would not share).
    mesh = resolve_model(model).path.parent / "meshes" / f"{short}.obj"
    path.write_text(
        f"""<mujoco>
  <asset><mesh name="{short}_mesh" file="{mesh}"/></asset>
  <worldbody>
    <body name="mount" quat="{" ".join(map(str, RETIRED_QUAT))}">
      <geom name="{short}_visual" type="mesh" mesh="{short}_mesh" contype="0" conaffinity="0"/>
      <camera name="{short}_color" pos="{" ".join(map(str, cam["pos"]))}"
              xyaxes="{" ".join(map(str, cam["xyaxes"]))}"/>
    </body>
  </worldbody>
</mujoco>"""
    )
    return str(path)


def _compiled(model, pos, rpy, prefix="cam_"):
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_sensor": {
                        "model": model,
                        "prefix": prefix,
                        "pos": list(pos),
                        "rpy": list(rpy),
                        "default_plugins": False,
                    },
                    "name": "cam",
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    return engine


def _cam(engine, name):
    m, d = engine.ctx.model, engine.ctx.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    assert cid >= 0, name
    return d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()


def _geom(engine, name):
    m, d = engine.ctx.model, engine.ctx.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, name
    return d.geom_xpos[gid].copy(), d.geom_xmat[gid].reshape(3, 3).copy()


# -- the re-seat moved names, not cameras -------------------------------------------------------


def _assert_where_the_retired_mount_put_it(tmp_path, model, retired, new, atol=1e-9):
    """A retired mount at *retired* and a mount of *model* at *new* (both ``(pos, rpy)``) put the
    housing and the camera at one world pose -- the D415's camera aside, which moved to its lens."""
    short = VENDOR[model]["cam"]
    old = _compiled(_retired_mjcf(tmp_path, model), *retired)
    now = _compiled(model, *new)
    # The housing: the retired model's mesh sat at its body origin, the new one's at the vendor
    # mesh-in-link pose; the same mesh placed the same way compiles to the same geom pose.
    old_h, new_h = _geom(old, f"cam_{short}_visual"), _geom(now, f"cam_{short}_visual")
    assert np.allclose(old_h[0], new_h[0], atol=atol)
    assert np.allclose(old_h[1], new_h[1], atol=atol)
    old_c, new_c = _cam(old, f"cam_{short}_color"), _cam(now, f"cam_{short}_color")
    assert np.allclose(old_c[1], new_c[1], atol=atol)  # every camera still points where it did
    if model == "realsense_d415":
        # The retired D415 centred its camera on the housing, 5 mm proud of the glass. It is at the
        # colour lens now: (0.035, 0, -0.00987) in mesh axes, 38 mm from where it was.
        moved = np.linalg.norm(new_c[0] - old_c[0])
        assert moved == pytest.approx(math.hypot(0.035, 0.005 + 0.00987), abs=atol)
    else:
        assert np.allclose(old_c[0], new_c[0], atol=atol)


@pytest.mark.parametrize("model", sorted(VENDOR))
@pytest.mark.parametrize("pos, rpy", POSES)
def test_a_re_expressed_mount_puts_the_housing_and_camera_where_the_retired_one_did(
    tmp_path, model, pos, rpy
):
    _assert_where_the_retired_mount_put_it(
        tmp_path, model, (pos, rpy), _re_expressed(model, pos, rpy)
    )


def _vendor_mount():
    """``external/convert/vendor_mount.py``, the rewrite the builder's ``--rewrite-mounts`` runs."""
    path = Path(__file__).resolve().parents[2] / "external" / "convert" / "vendor_mount.py"
    spec = importlib.util.spec_from_file_location("vendor_mount", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rewritten_mounts_keep_each_camera_where_it_was(tmp_path):
    """Block- and flow-style mounts of each retired model, rewritten, load where they were."""
    retired = {"realsense_d415": "d415", "realsense_d435": "d435", "realsense_d455": "d455"}
    mounts = [(model, pos, rpy) for model in sorted(VENDOR) for pos, rpy in POSES[1:3]]
    lines = ["components:"]
    for i, (model, pos, rpy) in enumerate(mounts):
        if i % 2:
            lines += [
                f"  - spawn_sensor: {{model: {retired[model]}, pos: {pos}, rpy: {rpy}}}",
                f"    name: m{i}",
            ]
        else:
            lines += [
                "  - spawn_sensor:",
                f"      model: {retired[model]}",
                f"      pos: {pos}  # kept",
                f"      rpy: {rpy}",
                f"    name: m{i}",
            ]
    world = tmp_path / "world.yaml"
    world.write_text("\n".join(lines) + "\n")
    deltas = {retired[m]: (m, _delta(m)) for m in VENDOR}
    assert _vendor_mount().rewrite_mounts(world, deltas) == len(mounts)
    assert "pos: [" in world.read_text() and "  # kept" in world.read_text()
    rewritten = yaml.safe_load(world.read_text())["components"]
    for (model, pos, rpy), entry in zip(mounts, rewritten, strict=True):
        spec = entry["spawn_sensor"]
        assert spec["model"] == model
        _assert_where_the_retired_mount_put_it(
            tmp_path, model, (pos, rpy), (spec["pos"], spec["rpy"])
        )


#: The demo world's RealSense mounts as they were written for the retired models.
RETIRED_DEMO_MOUNTS = {
    "realsense_d435": ([1.5, 0.0, 0.5], [0.0, 0.0, 1.5708]),
    "realsense_d415": ([0.0, -1.5, 0.5], [0.0, 0.0, 0.0]),
    "realsense_d455": ([0.0, 1.5, 0.5], [0.0, 0.0, 3.14159]),
}


def test_the_demo_world_keeps_each_realsense_where_the_retired_mount_put_it(tmp_path):
    demo = (
        Path(resolve_model("realsense_d435").path).parents[2] / "worlds" / "all_sensors_demo.yaml"
    )
    mounts = {
        c["spawn_sensor"]["model"]: c["spawn_sensor"]
        for c in yaml.safe_load(demo.read_text())["components"]
        if isinstance(c, dict) and "spawn_sensor" in c
    }
    for model, retired in RETIRED_DEMO_MOUNTS.items():
        spec = mounts[model]
        # The world writes a pose to ten decimals.
        _assert_where_the_retired_mount_put_it(
            tmp_path, model, retired, (spec["pos"], spec["rpy"]), atol=1e-8
        )


# -- the chain is the vendor's ------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(VENDOR))
def test_the_mount_publishes_the_vendor_chain(model):
    engine = _compiled(model, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    tfs = {(t["parent"], t["child"]): t for t in frames.backend["ros2"]["static_tf"]}
    assert set(tfs) >= {
        ("world", "camera_bottom_screw_frame"),
        ("camera_bottom_screw_frame", "camera_link"),
        ("camera_link", "camera_color_frame"),
        ("camera_color_frame", "camera_color_optical_frame"),
        ("camera_link", "camera_depth_frame"),
        ("camera_depth_frame", "camera_depth_optical_frame"),
    }
    # The mount IS the frame the macro's origin places; camera_link hangs off it at the vendor offset.
    assert np.allclose(tfs[("world", "camera_bottom_screw_frame")]["translation"], [0.0, 0.0, 1.0])
    assert np.allclose(tfs[("world", "camera_bottom_screw_frame")]["rotation"], [1, 0, 0, 0])
    link = tfs[("camera_bottom_screw_frame", "camera_link")]
    assert np.allclose(link["translation"], VENDOR[model]["screw"], atol=1e-9)
    assert np.allclose(np.abs(link["rotation"]), [1, 0, 0, 0], atol=1e-9)
    color = tfs[("camera_link", "camera_color_frame")]
    assert np.allclose(color["translation"], VENDOR[model]["color"], atol=1e-9)
    optical = tfs[("camera_color_frame", "camera_color_optical_frame")]
    want = np.zeros(4)
    mujoco.mju_euler2Quat(want, np.asarray(OPTICAL_RPY), "XYZ")
    assert np.allclose(np.abs(optical["rotation"]), np.abs(want), atol=1e-9)


@pytest.mark.parametrize("model", sorted(VENDOR))
@pytest.mark.parametrize("pos, rpy", POSES)
def test_camera_link_sits_at_the_vendor_offset_from_the_mount(model, pos, rpy):
    """The mount body is at the pose the world gives it, and the body everything hangs from --
    ``camera_link`` -- at the vendor ``<name>_link_joint`` offset from it, unrotated."""
    engine = _compiled(model, pos, rpy)
    m, d = engine.ctx.model, engine.ctx.data
    mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cam_mount")
    link = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cam_link")
    assert mount >= 0 and link >= 0
    r_mount = d.xmat[mount].reshape(3, 3)
    assert np.allclose(d.xpos[mount], pos, atol=1e-9)
    assert np.allclose(r_mount, _rot(rpy), atol=1e-9)
    assert np.allclose(
        r_mount.T @ (d.xpos[link] - d.xpos[mount]), VENDOR[model]["screw"], atol=1e-9
    )
    assert np.allclose(d.xmat[link].reshape(3, 3), r_mount, atol=1e-9)
    # The published camera_link is that body.
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    tf_pos, tf_rot = _tf_to_world(frames.backend["ros2"]["static_tf"], "camera_link")
    assert np.allclose(tf_pos, d.xpos[link], atol=1e-9)
    assert np.allclose(tf_rot, d.xmat[link].reshape(3, 3), atol=1e-9)


@pytest.mark.parametrize("model", sorted(VENDOR))
@pytest.mark.parametrize("pos, rpy", POSES)
def test_the_image_is_rendered_from_the_frame_it_is_stamped_in(model, pos, rpy):
    """The colour optical frame the chain builds and the MuJoCo camera are one pose."""
    engine = _compiled(model, pos, rpy)
    m, d = engine.ctx.model, engine.ctx.data
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cam_camera_color_optical_frame")
    assert site >= 0
    cam_pos, cam_mat = _cam(engine, f"cam_{VENDOR[model]['cam']}_color")
    assert np.allclose(d.site_xpos[site], cam_pos, atol=1e-9)
    # MuJoCo's camera looks down -z with +y up; the optical frame down +z with +y down.
    assert np.allclose(
        d.site_xmat[site].reshape(3, 3), cam_mat @ np.diag([1.0, -1.0, -1.0]), atol=1e-9
    )


def test_the_published_frames_are_the_ones_the_data_is_stamped_in():
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_sensor": {"model": "realsense_d435", "device_name": "head_camera"},
                    "name": "cam",
                    "components": [{"realsense_d435": {"depth": True}}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    eps = {e.name: e.backend["ros2"] for e in engine.ctx.interface.all() if e.owner == "cam"}
    children = {t["child"] for t in eps["frames"]["static_tf"]}
    assert eps["image"]["frame_id"] == "head_camera_color_optical_frame"
    assert eps["depth"]["frame_id"] == "head_camera_depth_optical_frame"
    assert eps["imu"]["frame_id"] == "head_camera_imu_optical_frame"
    assert {
        "head_camera_color_optical_frame",
        "head_camera_depth_optical_frame",
        "head_camera_imu_optical_frame",
    } <= children


# -- depth is rendered from the depth frame -----------------------------------------------------


@pytest.mark.parametrize("model", sorted(VENDOR))
@pytest.mark.parametrize("pos, rpy", POSES)
def test_depth_is_rendered_from_the_frame_it_is_stamped_in(model, pos, rpy):
    """The depth camera, the depth optical frame the chain builds and the one the published TF
    composes to are one pose."""
    engine = _compiled(model, pos, rpy)
    m, d = engine.ctx.model, engine.ctx.data
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cam_camera_depth_optical_frame")
    assert site >= 0
    cam_pos, cam_mat = _cam(engine, f"cam_{VENDOR[model]['cam']}_depth")
    # MuJoCo's camera looks down -z with +y up; the optical frame down +z with +y down.
    optical = cam_mat @ np.diag([1.0, -1.0, -1.0])
    assert np.allclose(d.site_xpos[site], cam_pos, atol=1e-9)
    assert np.allclose(d.site_xmat[site].reshape(3, 3), optical, atol=1e-9)
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    tf_pos, tf_rot = _tf_to_world(frames.backend["ros2"]["static_tf"], "camera_depth_optical_frame")
    assert np.allclose(tf_pos, cam_pos, atol=1e-9)
    assert np.allclose(tf_rot, optical, atol=1e-9)
    # And it is not the colour camera: the two sit a baseline offset apart.
    color_pos, _ = _cam(engine, f"cam_{VENDOR[model]['cam']}_color")
    assert np.linalg.norm(color_pos - cam_pos) == pytest.approx(
        np.linalg.norm(VENDOR[model]["color"]), abs=1e-9
    )


@pytest.mark.parametrize("model", sorted(VENDOR))
def test_the_depth_camera_has_the_depth_streams_optics(model):
    engine = _compiled(model, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    m = engine.ctx.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"cam_{VENDOR[model]['cam']}_depth")
    fovy, w, h = DEPTH_OPTICS[model]
    assert float(m.cam_fovy[cid]) == pytest.approx(fovy)
    assert tuple(int(v) for v in m.cam_resolution[cid]) == (w, h)


def test_the_depth_camera_is_a_stream_of_the_device_not_another_view():
    """Coverage and ``show_fov`` count a device once: its depth camera is set aside, its colour
    camera and a lone camera of another model are not."""
    from roqsim_sensors.plugins.camera_common import depth_stream_cameras

    engine = _compiled("realsense_d435", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    m = engine.ctx.model
    depth = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_d435_depth")
    assert depth_stream_cameras(m) == {depth}
    lone = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><camera name="x_depth"/></body></worldbody></mujoco>'
    )
    assert depth_stream_cameras(lone) == set()


def _tf_to_world(static_tf: list[dict], frame: str) -> tuple[np.ndarray, np.ndarray]:
    """``frame``'s pose in ``world``, composed from the published static transforms alone."""
    by_child = {t["child"]: t for t in static_tf}
    pos, rot = np.zeros(3), np.eye(3)
    while frame != "world":
        t = by_child[frame]
        r = _quat_rot(t["rotation"])
        pos, rot = r @ pos + np.asarray(t["translation"]), r @ rot
        frame = t["parent"]
    return pos, rot


@pytest.mark.parametrize("model", ["realsense_d435", "realsense_d455"])
def test_a_depth_return_reprojects_onto_the_surface_through_tf(model):
    """A wall whose edge is known: the cloud, carried to ``world`` through the published TF of
    ``camera_depth_optical_frame``, puts the wall's returns on its face and its edge on the edge.
    Rendered from the colour camera and stamped in the depth frame, the edge would land a baseline
    (15 mm, 59 mm) away."""
    edge_y, face_x = 0.05, 1.45
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "box": {
                        "pose": {"position": {"x": face_x + 0.05, "y": edge_y + 1.0, "z": 1.0}},
                        "size": [0.1, 2.0, 2.0],
                        "motion": "static",
                    },
                    "name": "wall",
                },
                {
                    "spawn_sensor": {"model": model, "pos": [0.0, 0.0, 1.0]},
                    "name": "cam",
                    "components": [{model: {"points": True}}],
                },
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    engine.step()
    eps = {e.name: e for e in engine.ctx.interface.all() if e.owner == "cam"}
    assert eps["points"].backend["ros2"]["frame_id"] == "camera_depth_optical_frame"
    pos, rot = _tf_to_world(
        eps["frames"].backend["ros2"]["static_tf"], "camera_depth_optical_frame"
    )
    points = eps["points"].read().points.astype(float) @ rot.T + pos
    # The wall's face: the returns within a few millimetres of its plane (its side face, seen past
    # the edge, and whatever lies beyond are further away).
    face = points[np.abs(points[:, 0] - face_x) < 5e-3]
    assert len(face) > 1000
    assert np.abs(face[:, 0] - face_x).max() < 1e-3
    # Within a pixel's footprint at 1.4 m (about 3 mm), well inside the 15 mm baseline.
    assert face[:, 1].min() == pytest.approx(edge_y, abs=5e-3)
    # Its camera_info is the depth stream's, not the colour one's.
    fovy, w, h = DEPTH_OPTICS[model]
    info = eps["depth_camera_info"].read()
    assert (info.width, info.height) == (w, h)
    assert info.fy == pytest.approx(h / (2 * math.tan(math.radians(fovy) / 2)))


# -- the old names are refused ------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new", [("d415", "realsense_d415"), ("d435", "realsense_d435"), ("d455", "realsense_d455")]
)
def test_a_retired_name_is_refused_naming_the_new_one(old, new):
    with pytest.raises(ModelError) as exc:
        resolve_model(old)
    assert str(exc.value) == (
        f"spawn_sensor: model {old!r} — renamed to {new!r} when its mount frame became the one its "
        f"vendor macro places (it was a display convention pointing the lens along +y). Update the "
        f"name, and re-express this mount's pos/rpy as the vendor macro's origin; see "
        f"roqsim_sensors/README.md."
    )
    with pytest.raises(ModelError, match=f"renamed to '{new}'"):
        resolve_model(f"roqsim_sensors:{old}")
