"""The RealSense devices are mounted by their vendor ``camera_link``, and the move changed no camera.

Three claims, each against the vendor's published numbers rather than against the model:

* **The chain is the vendor's.** Each mount publishes ``camera_link -> camera_color_frame ->
  camera_color_optical_frame`` and the depth frames with the offsets ``realsense2_description``
  states, and the MuJoCo camera the image is rendered from IS the colour optical frame, so a
  consumer that looks a pixel up in TF finds where it was taken.
* **The re-seat moved names, not cameras.** The retired ``d415``/``d435``/``d455`` models pre-rotated
  ``mount`` so a mount at ``rpy [0, 0, 0]`` looked along ``+y``. A mount of the old model at ``T_old``
  and one of the new at ``T_old * D`` (``D = Rq * T_link_mesh^-1``, from the constants below) put the
  housing and the D435's and D455's cameras at the same world pose to float tolerance. The retired
  D415's camera was not at its colour lens -- centred on the housing, 5 mm in front of the glass --
  and sits at the vendor colour frame now; that one move is asserted as what it is.
* **The old names are refused**, naming the new one.

Worlds are re-expressed with ``external/convert/build_realsense_devices.py --rewrite-mounts``, which
applies the same ``D``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import ModelError, resolve_model

OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# realsense2_description @ realsense-ros 4.56.1, the `${name}_link` visual origin used with the mesh,
# and the `${name}_color_joint` origin, both in `${name}_link`.
VENDOR = {
    # _d415.urdf.xacro: (d415_cam_mount_from_center_offset, -d415_cam_depth_py, 0)
    "realsense_d415": {"mesh": (0.00987, -0.020, 0.0), "color": (0.0, 0.015, 0.0), "cam": "d415"},
    # _d435.urdf.xacro: (d435_zero_depth_to_glass + d435_glass_to_front, -d435_cam_depth_py, 0)
    "realsense_d435": {"mesh": (0.0043, -0.0175, 0.0), "color": (0.0, 0.015, 0.0), "cam": "d435"},
    # _d455.urdf.xacro: (d455_zero_depth_to_glass + d455_glass_to_front, -d455_cam_depth_py, 0)
    "realsense_d455": {"mesh": (0.00465, -0.0475, 0.0), "color": (0.0, -0.059, 0.0), "cam": "d455"},
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
    return r, -r @ t_lm


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


@pytest.mark.parametrize("model", sorted(VENDOR))
@pytest.mark.parametrize("pos, rpy", POSES)
def test_a_re_expressed_mount_puts_the_housing_and_camera_where_the_retired_one_did(
    tmp_path, model, pos, rpy
):
    short = VENDOR[model]["cam"]
    old = _compiled(_retired_mjcf(tmp_path, model), pos, rpy)
    new = _compiled(model, *_re_expressed(model, pos, rpy))
    # The housing: the retired model's mesh sat at its body origin, the new one's at the vendor
    # mesh-in-link pose; the same mesh placed the same way compiles to the same geom pose.
    old_h, new_h = _geom(old, f"cam_{short}_visual"), _geom(new, f"cam_{short}_visual")
    assert np.allclose(old_h[0], new_h[0], atol=1e-9)
    assert np.allclose(old_h[1], new_h[1], atol=1e-9)
    old_c, new_c = _cam(old, f"cam_{short}_color"), _cam(new, f"cam_{short}_color")
    assert np.allclose(old_c[1], new_c[1], atol=1e-9)  # every camera still points where it did
    if model == "realsense_d415":
        # The retired D415 centred its camera on the housing, 5 mm proud of the glass. It is at the
        # colour lens now: (0.035, 0, -0.00987) in mesh axes, 38 mm from where it was.
        moved = np.linalg.norm(new_c[0] - old_c[0])
        assert moved == pytest.approx(math.hypot(0.035, 0.005 + 0.00987), abs=1e-9)
    else:
        assert np.allclose(old_c[0], new_c[0], atol=1e-9)


# -- the chain is the vendor's ------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(VENDOR))
def test_the_mount_publishes_the_vendor_chain(model):
    engine = _compiled(model, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    tfs = {(t["parent"], t["child"]): t for t in frames.backend["ros2"]["static_tf"]}
    assert set(tfs) >= {
        ("world", "camera_link"),
        ("camera_link", "camera_color_frame"),
        ("camera_color_frame", "camera_color_optical_frame"),
        ("camera_link", "camera_depth_frame"),
        ("camera_depth_frame", "camera_depth_optical_frame"),
    }
    assert np.allclose(tfs[("world", "camera_link")]["translation"], [0.0, 0.0, 1.0])
    color = tfs[("camera_link", "camera_color_frame")]
    assert np.allclose(color["translation"], VENDOR[model]["color"], atol=1e-9)
    optical = tfs[("camera_color_frame", "camera_color_optical_frame")]
    want = np.zeros(4)
    mujoco.mju_euler2Quat(want, np.asarray(OPTICAL_RPY), "XYZ")
    assert np.allclose(np.abs(optical["rotation"]), np.abs(want), atol=1e-9)


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


# -- the old names are refused ------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new", [("d415", "realsense_d415"), ("d435", "realsense_d435"), ("d455", "realsense_d455")]
)
def test_a_retired_name_is_refused_naming_the_new_one(old, new):
    with pytest.raises(ModelError) as exc:
        resolve_model(old)
    assert str(exc.value) == (
        f"spawn_sensor: model {old!r} — renamed to {new!r} when its mount frame became the vendor "
        f"link (it was a display convention pointing the lens along +y). Update the name, and "
        f"re-express this mount's pos/rpy against the vendor link; see roqsim_sensors/README.md."
    )
    with pytest.raises(ModelError, match=f"renamed to '{new}'"):
        resolve_model(f"roqsim_sensors:{old}")
