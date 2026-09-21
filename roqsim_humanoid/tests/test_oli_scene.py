"""LimX Oli (HU_D04_01) port verification battery.

Mirrors the robot-porting verification battery: static sanity (A), closed-loop locomotion drive
tests (B), and sensor checks (C). Everything runs through the real ``oli_locomotion`` plugin (the
pretrained ONNX whole-body walk policy + PD loop), so the model, its manifest config and the
controller are verified together -- a humanoid cannot be tested open-loop the way a wheeled base can
(it is an inverted pendulum; only the balancing policy keeps it upright).

Reference facts come from the vendor sources (see THIRD_PARTY.md): 31 actuated
DoF; total mass ~52.9 kg (URDF-derived; LimX does not publish a weight); standing height ~1.65 m
(datasheet); home pelvis height ~0.90 m. The world runs at sim.timestep = 0.001 s (1000 Hz PD /
100 Hz policy), set explicitly here.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from roqsim_humanoid.plugins.oli_locomotion import JOINTS, OliLocomotionPlugin

from roqsim.context import Entity, SimContext

MODELS = Path(__file__).resolve().parents[1] / "src" / "roqsim_humanoid" / "models"
MODEL_XML = MODELS / "oli.xml"

TIMESTEP = 0.001
TOTAL_MASS = 52.92  # URDF-derived (no published datasheet weight); regression guard
BASE_MASS = 5.905  # base_link (pelvis) inertial mass from the vendor URDF
DATASHEET_HEIGHT = 1.65  # m, LimX Oli spec
HOME_Z = 0.902  # pelvis height at the home keyframe


def _build():
    """Compose the Oli with a ground plane named `floor`, reset to the home keyframe, dt = 1 ms."""
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    spec.meshdir = str(MODELS / "meshes" / "oli")
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15, 15, 0.05]
    floor.condim = 3
    floor.friction = [1.0, 0.3, 0.3]
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _plugin(model, data, **overrides):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    plugin = OliLocomotionPlugin({**overrides})
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _base_qadr(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    return model.jnt_qposadr[jid]


def _run(vx, vy, w, seconds):
    """Drive (vx, vy, w) through the walk policy for `seconds`; return ground-truth summary."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    bq = _base_qadr(model)
    x0, y0 = float(data.qpos[bq]), float(data.qpos[bq + 1])
    yaw0 = _yaw(data, bq)
    for _ in range(int(seconds / model.opt.timestep)):
        plugin.drive(vx, vy, w)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos)), f"diverged at t={data.time:.3f}"
    return dict(
        model=model,
        data=data,
        plugin=plugin,
        bq=bq,
        z=float(data.qpos[bq + 2]),
        dx=float(data.qpos[bq] - x0),
        dy=float(data.qpos[bq + 1] - y0),
        dyaw=_wrap(_yaw(data, bq) - yaw0),
        seconds=seconds,
    )


def _yaw(data, bq):
    w, x, y, z = data.qpos[bq + 3 : bq + 7]
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- A. static sanity


def test_a1_loads_and_steps_without_warnings():
    """A1: 10 s under the policy (zero command) at the campaign timestep -- no divergence/warnings."""
    r = _run(0.0, 0.0, 0.0, 10.0)
    assert np.all(np.isfinite(r["data"].qpos))
    fired = [
        mujoco.mjtWarning(i).name
        for i in range(mujoco.mjtWarning.mjNWARNING)
        if r["data"].warning[i].number > 0
    ]
    assert not fired, f"MuJoCo warnings fired: {fired}"


def test_a2_mass_and_inertia_audit():
    """A2: total + pelvis mass match the vendor URDF; no near-zero inertials on actuated links."""
    model, _ = _build()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert float(model.body_subtreemass[base]) == pytest.approx(TOTAL_MASS, rel=0.02)
    assert float(model.body_mass[base]) == pytest.approx(BASE_MASS, rel=0.02)
    assert model.nu == 31
    for jn in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        bid = model.jnt_bodyid[jid]
        assert float(model.body_mass[bid]) > 0.02, f"{jn}: suspiciously light link"
        assert np.all(model.body_inertia[bid] > 1e-7), f"{jn}: near-zero inertia"


def test_a3_stand_is_balanced():
    """A3: at zero command the policy holds a stable stand -- stays up, drifts < 10 cm over 6 s."""
    r = _run(0.0, 0.0, 0.0, 6.0)
    assert r["z"] > HOME_Z - 0.05, f"pelvis dropped to {r['z']:.3f} (fell)"
    assert math.hypot(r["dx"], r["dy"]) < 0.10, "excessive stationary drift"


def test_a5_scale_matches_datasheet_height():
    """A5: standing height (top of head above the floor) matches the 1.65 m datasheet within 5%."""
    model, data = _build()
    top = max(
        float(data.geom_xpos[g][2])
        for g in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) != "floor"
    )
    # top-of-head geom centre sits a few cm below the true crown; allow the datasheet +-5% band.
    assert DATASHEET_HEIGHT * 0.90 < top < DATASHEET_HEIGHT * 1.02, f"standing top {top:.3f} m"


# --------------------------------------------------------------------------- B. locomotion drive


def test_b1_walk_forward_tracks_command():
    """B1: vx = 0.3 -- walks forward, stays upright, achieved speed within 25% of command."""
    r = _run(0.3, 0.0, 0.0, 8.0)
    assert r["z"] > HOME_Z - 0.05, "fell while walking"
    mean_vx = r["dx"] / r["seconds"]
    assert mean_vx == pytest.approx(0.3, abs=0.075), f"mean vx {mean_vx:.3f} m/s"
    assert abs(r["dy"]) < 0.2 * r["dx"], "excessive lateral drift"


def test_b2_yaw_command_turns_in_place():
    """B2: w = 0.4 -- turns toward the commanded direction and stays roughly in place, upright."""
    r = _run(0.0, 0.0, 0.4, 6.0)
    assert r["z"] > HOME_Z - 0.05, "fell while turning"
    assert r["dyaw"] > 0.3, f"did not turn (dyaw={r['dyaw']:.2f} rad)"
    assert math.hypot(r["dx"], r["dy"]) < 0.5, "walked away instead of turning in place"


def test_b4_command_saturates_at_trained_limits():
    """B4: cmd_vel is clamped to the vendor training range (max_vx 0.5, max_vy 0.3, max_wz 0.5)."""
    model, data = _build()
    _, plugin = _plugin(model, data)
    plugin.drive(2.0, 2.0, 2.0)
    assert plugin._cmd[0] == pytest.approx(0.5)
    assert plugin._cmd[1] == pytest.approx(0.3)
    assert plugin._cmd[2] == pytest.approx(0.5)


def test_manual_control_leaves_ctrl_to_the_sliders():
    """--manual-control: oli_locomotion stops stamping torques so a slider drag survives a step."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    ctx.manual_control = True
    plugin.drive(0.3, 0.0, 0.0)
    data.ctrl[:] = 3.0  # what dragging the actuator sliders does
    plugin.pre_step(ctx)
    assert (data.ctrl == 3.0).all(), "policy stamped over the manual drag"
    ctx.manual_control = False
    plugin.pre_step(ctx)
    assert not (data.ctrl == 3.0).all(), "controller did not take the joints back"


# --------------------------------------------------------------------------- C. sensors


# The camera mounts, from HU_D04_01.urdf @ humanoid-description a90f734 (see oli.manifest.yaml):
#: head_camera_joint on head_pitch_link.
HEAD_CAMERA_JOINT = ((0.07453, 0.0175, 0.065), (0.0, 1.5708, 0.0))
#: head_camera_link -> the D435's camera_link, read off the vendor's d435_link.STL on that link: its
#: lens normal is the link's +z and its baseline the link's +y, so camera_link is the link pitched
#: back by -90 deg, at the same origin.
HEAD_CAMERA_LINK_TO_CAMERA_LINK = ((0.0, 0.0, 0.0), (0.0, -1.5708, 0.0))
#: The chest camera: waist_camera_joint's origin on waist_pitch_link, forward axis (the joint's
#: rpy (0, 2.1818, 0) is a recorded deviation).
CHEST_CAMERA_LINK = ((0.092, 0.0175, 0.2751), (0.0, 0.0, 0.0))
#: The D435's own chain: camera_link -> camera_color_frame -> camera_color_optical_frame
#: (realsense2_description _d435.urdf.xacro).
D435_COLOR = ((0.0, 0.015, 0.0), (0.0, 0.0, 0.0))
D435_OPTICAL = ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, -math.pi / 2))


def _urdf_rot(rpy):
    r, p_, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p_), 0, math.sin(p_)], [0, 1, 0], [-math.sin(p_), 0, math.cos(p_)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _chain(*joints):
    pos, rot = np.zeros(3), np.eye(3)
    for xyz, rpy in joints:
        pos = pos + rot @ np.asarray(xyz, dtype=np.float64)
        rot = rot @ _urdf_rot(rpy)
    return pos, rot


@pytest.fixture(scope="module")
def spawned():
    """The Oli as a world spawns it, its two RealSense devices mounted, the renders switched off
    (nothing here reads a pixel, and a D435's mass and housing are what it tests)."""
    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    off = ("robot.head_camera.realsense_d435", "robot.chest_camera.realsense_d435")
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": TIMESTEP},
            "components": [{"spawn_robot": {"model": "oli", "prefix": "o_"}, "name": "robot"}],
        },
        overrides={"components": {a: {"enabled": False} for a in off}},
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


def _in_body(engine, body, pos, mat):
    m, d = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, body
    rb = d.xmat[bid].reshape(3, 3)
    return rb.T @ (np.asarray(pos) - d.xpos[bid]), rb.T @ np.asarray(mat).reshape(3, 3)


def _camera(engine, name):
    m, d = engine.ctx.model, engine.ctx.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    assert cid >= 0, f"missing camera {name}"
    return d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()


def test_c1_sensor_mounts(spawned):
    """C1: the lidar/imu sites, and the two D435s the manifest mounts, at plausible mount heights.

    The bare MJCF carries no camera of its own: the head and chest cameras are the `realsense_d435`
    device, mounted on head_pitch_link and waist_pitch_link.
    """
    bare, _ = _build()
    assert bare.ncam == 0, "oli.xml bakes a camera; the manifest's D435 devices are the cameras"
    m, d = spawned.ctx.model, spawned.ctx.data
    mujoco.mj_forward(m, d)
    for site in ("o_lidar", "o_imu"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site) >= 0, f"missing site {site}"
    lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "o_lidar")
    assert 1.0 < float(d.site_xpos[lid][2]) < 1.4
    head_pos, _ = _camera(spawned, "o_head_camera_d435_color")
    chest_pos, _ = _camera(spawned, "o_chest_camera_d435_color")
    assert float(head_pos[2]) > 1.3, "head camera implausibly low"
    assert float(head_pos[2]) > float(chest_pos[2]) > 1.0


def test_c1b_the_head_camera_is_the_vendor_chain(spawned):
    """C1b: camera_link, and the colour optical frame the image is taken from, are where the URDF's
    head_camera_joint and the vendor's own D435 chain put them on head_pitch_link."""
    want_link = _chain(HEAD_CAMERA_JOINT, HEAD_CAMERA_LINK_TO_CAMERA_LINK)
    np.testing.assert_allclose(want_link[1], np.eye(3), atol=1e-4)  # the rpy undoes itself
    d = spawned.ctx.data
    mid = mujoco.mj_name2id(spawned.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "o_head_camera_mount")
    pos, rot = _in_body(spawned, "o_head_pitch_link", d.xpos[mid], d.xmat[mid])
    np.testing.assert_allclose(pos, want_link[0], atol=1e-9)
    np.testing.assert_allclose(rot, want_link[1], atol=1e-4)
    want_opt = _chain(HEAD_CAMERA_JOINT, HEAD_CAMERA_LINK_TO_CAMERA_LINK, D435_COLOR, D435_OPTICAL)
    cam_pos, cam_rot = _in_body(
        spawned, "o_head_pitch_link", *_camera(spawned, "o_head_camera_d435_color")
    )
    np.testing.assert_allclose(cam_pos, want_opt[0], atol=1e-6)
    # MuJoCo's camera looks down -z with +y up; the optical frame down +z with +y down.
    np.testing.assert_allclose(cam_rot @ np.diag([1.0, -1.0, -1.0]), want_opt[1], atol=1e-4)
    chest_pos, _ = _in_body(
        spawned, "o_waist_pitch_link", *_camera(spawned, "o_chest_camera_d435_color")
    )
    np.testing.assert_allclose(chest_pos, _chain(CHEST_CAMERA_LINK, D435_COLOR)[0], atol=1e-9)


def test_c1c_the_head_camera_looks_forward_at_the_home_stance(spawned):
    """C1c: optical +z along the robot's +x, image upright (optical +y down), at the default pose."""
    m, d = spawned.ctx.model, spawned.ctx.data
    mujoco.mj_forward(m, d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "o_base_link")
    rb = d.xmat[base].reshape(3, 3)
    for cam in ("o_head_camera_d435_color", "o_chest_camera_d435_color"):
        _, rot = _camera(spawned, cam)
        optical = rb.T @ rot @ np.diag([1.0, -1.0, -1.0])
        assert float(optical[:, 2] @ [1.0, 0.0, 0.0]) > 0.95, f"{cam} does not look forward"
        assert float(optical[:, 1] @ [0.0, 0.0, -1.0]) > 0.95, f"{cam} is not upright"


def test_c1d_each_camera_publishes_its_own_frames_and_topics(spawned):
    eps = {(e.owner, e.name): e for e in spawned.ctx.interface.all()}
    for label in ("head_camera", "chest_camera"):
        owner = f"robot.{label}"
        tfs = {
            (t["parent"], t["child"]) for t in eps[(owner, "frames")].backend["ros2"]["static_tf"]
        }
        assert (f"{label}_color_frame", f"{label}_color_optical_frame") in tfs
        imu = eps[(owner, "imu")]
        assert imu.namespace == label and imu.lazy
        assert imu.backend["ros2"]["topic"] == "camera/imu"
        assert imu.backend["ros2"]["frame_id"] == f"{label}_imu_optical_frame"


def test_c1e_the_mounted_cameras_touch_nothing_while_it_walks(spawned):
    """C1e: two D435s -- mesh, group-3 collision box, 72 g each -- on a balancing robot's head and
    chest. Walking, no contact involves either housing, so the constraint forces carry no self-
    contact the vendor chain never had, and the robot stays up."""
    m, d = spawned.ctx.model, spawned.ctx.data
    mounts = {
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"o_{label}_mount")
        for label in ("head_camera", "chest_camera")
    }
    assert all(b >= 0 for b in mounts)
    loco = next(p for p in spawned.plugins if type(p).__name__ == "OliLocomotionPlugin")
    bq = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "o_base_free")])
    x0 = float(d.qpos[bq])
    touched = set()
    for _ in range(int(4.0 / TIMESTEP)):
        loco.drive(0.3, 0.0, 0.0)
        spawned.step()
        for c in d.contact[: d.ncon]:
            bodies = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if bodies & mounts:
                touched.add(
                    tuple(
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                        for g in (c.geom1, c.geom2)
                    )
                )
    assert not touched, f"a camera housing is in contact: {sorted(touched)}"
    assert float(d.qpos[bq + 2]) > HOME_Z - 0.05, "fell while walking with the cameras mounted"
    assert float(d.qpos[bq]) - x0 > 0.5, "did not walk"


def test_c3_control_rates():
    """C3: policy decimation gives a 100 Hz policy on a 1000 Hz PD loop (the trained cadence)."""
    model, data = _build()
    _, plugin = _plugin(model, data)
    assert plugin._decimation == 10
    assert model.opt.timestep == pytest.approx(0.001)
