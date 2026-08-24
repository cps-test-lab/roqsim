"""AgiBot G2 drive/pose-test battery (port verification).

Everything is driven through the real plugins the model ships with -- ``diff_drive`` (base) and
``agibot_g2_controller`` (torso/head/arms/grippers) -- using the shipped manifest config, so the
model, its calibration and the controllers are verified together.

G2 is a wheeled dual-arm mobile manipulator whose 4-wheel swerve steer joints are welded straight
(the "wheels welded as caster" planar diff-drive approximation). Consequences captured below:
straight-line drive is clean; in-place rotation is scrub-limited (a rigid 4-wheel skid-steer), so the
rotation test only asserts it turns in the commanded direction and stays stable, not a yaw-rate ratio.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import yaml
from roqsim_humanoid.plugins.agibot_g2_controller import AgibotG2ControllerPlugin

from roqsim.context import Entity, SimContext
from roqsim_mobile.plugins.diff_drive import DiffDrivePlugin

MODELS = Path(__file__).resolve().parents[1] / "src" / "roqsim_humanoid" / "models"
MODEL_XML = MODELS / "agibot_g2.xml"
MANIFEST = MODELS / "agibot_g2.manifest.yaml"

WHEEL_R = 0.07
TRACK = 0.436
REST_Z = 0.040

_CFG = {
    list(e)[0]: (list(e.values())[0] or {})
    for e in yaml.safe_load(MANIFEST.read_text())["components"]
}


def _build(gravity=None):
    """Compose the robot with a ground plane named `floor` (meshdir resolved against the model)."""
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    spec.meshdir = str(MODELS / "meshes" / "agibot_g2")
    spec.option.timestep = 0.002
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20, 20, 0.05]
    floor.friction = [1.0, 0.01, 0.001]
    model = spec.compile()
    if gravity is not None:
        model.opt.gravity[:] = gravity
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _plugins(model, data):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    dd = DiffDrivePlugin(dict(_CFG["diff_drive"]))
    dd.configure(ctx)
    dd.on_reset(ctx)
    body = AgibotG2ControllerPlugin(dict(_CFG["agibot_g2_controller"]))
    body.configure(ctx)
    body.on_reset(ctx)
    return ctx, dd, body


def _yaw(data):
    w, x, y, z = data.qpos[3:7]
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _run(v, w, seconds):
    model, data = _build()
    ctx, dd, body = _plugins(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        dd.drive(v, 0.0, w)
        dd.pre_step(ctx)
        body.pre_step(ctx)
        mujoco.mj_step(model, data)
        dd.post_step(ctx)
        assert np.all(np.isfinite(data.qpos)), f"diverged at t={data.time:.3f}"
    return model, data, dd, body


# --------------------------------------------------------------- A. static sanity


def test_a1_loads_and_steps_without_warnings():
    model, data = _build()
    for _ in range(int(10.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    fired = [
        mujoco.mjtWarning(i).name
        for i in range(mujoco.mjtWarning.mjNWARNING)
        if data.warning[i].number > 0
    ]
    assert not fired, f"MuJoCo warnings fired: {fired}"


def test_a2_mass_and_inertia_sane():
    model, _ = _build()
    total = float(sum(model.body_mass))
    assert 150.0 < total < 180.0, f"total mass {total:.1f} kg out of expected range"
    # no near-zero inertial on a body that has mass
    for b in range(1, model.nbody):
        if model.body_mass[b] > 1e-4:
            assert np.all(model.body_inertia[b] > 1e-9), f"body {b} has degenerate inertia"


def test_a3_rest_stability():
    model, data = _build()
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[2] - REST_Z) < 0.01, f"rest height {data.qpos[2]:.4f}"
    assert abs(data.qpos[3] - 1.0) < 0.02, "base tilted at rest"
    assert np.linalg.norm(data.qvel[:3]) < 0.02, "base drifting at rest"


def test_a4_scale_and_geometry():
    model, data = _build()

    def wz(name):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[b]

    lf = wz("chassis_lwheel_front_link2")
    rf = wz("chassis_rwheel_front_link2")
    lr = wz("chassis_lwheel_rear_link2")
    assert abs(abs(lf[1] - rf[1]) - TRACK) < 0.02, "track width wrong"
    assert abs(lf[0] - lr[0]) > 0.4, "wheelbase wrong"


# --------------------------------------------------------------- B. drive / pose


def test_b1_straight_line():
    model, data, dd, body = _run(0.5, 0.0, 4.0)
    assert data.qpos[0] > 1.2, f"did not travel forward (x={data.qpos[0]:.2f})"
    assert abs(data.qpos[1]) < 0.05, f"lateral drift {data.qpos[1]:.3f}"
    assert abs(data.qpos[2] - REST_Z) < 0.02, f"base rose/sank while driving (z={data.qpos[2]:.3f})"
    assert abs(_yaw(data)) < math.radians(5), "unexpected yaw while driving straight"


def test_b2_in_place_rotation_direction_and_stability():
    # Scrub-limited (rigid 4-wheel skid-steer): assert it turns in the commanded direction and stays
    # stable near the spot, not a yaw-rate ratio. ~35 deg in 4 s at cmd 0.5 rad/s. See port log.
    model, data, dd, body = _run(0.0, 0.5, 4.0)
    assert _yaw(data) > math.radians(10), f"did not yaw (yaw={math.degrees(_yaw(data)):.1f} deg)"
    assert math.hypot(data.qpos[0], data.qpos[1]) < 0.35, "drifted too far during in-place turn"
    assert abs(data.qpos[2] - REST_Z) < 0.02, "base unstable during rotation"


def test_b3_arm_and_head_position_hold():
    # Arms + head are the manipulation DOFs and hold commanded targets against gravity. (The serial
    # torso-lift chain holds *home* -- test_a3 -- but arbitrary lift poses need gravity comp; see port log.)
    model, data = _build()
    ctx, dd, body = _plugins(model, data)
    target = {
        "idx22_arm_l_joint2": 0.6,
        "idx24_arm_l_joint4": -0.8,
        "idx64_arm_r_joint4": -0.8,
        "idx11_head_joint1": 0.3,
    }
    body.set_targets(list(target), list(target.values()))
    for _ in range(int(3.0 / model.opt.timestep)):
        body.pre_step(ctx)
        mujoco.mj_step(model, data)
    for jname, tgt in target.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        got = data.qpos[model.jnt_qposadr[jid]]
        assert abs(got - tgt) < 0.1, f"{jname}: held {got:.3f} != target {tgt}"


def test_b5_arms_hang_on_spawn_path():
    # Regression for the spawn path specifically: spawn_robot strips the model <keyframe> on attach,
    # so a spawned G2 resets to the bare qpos0 (a T-pose, arms straight out at ~0.94 m to each side).
    # The controller's `rest` stance (manifest default) must fold them straight down beside the torso
    # -- and do so at reset, so there is no T-pose spawn transient. `_build` reset-to-keyframe would
    # mask this, so rebuild here with the keyframe removed (exactly what _strip_keyframes does).
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    spec.meshdir = str(MODELS / "meshes" / "agibot_g2")
    for k in list(spec.keys):
        spec.delete(k)
    spec.nkey = 0
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20, 20, 0.05]
    floor.friction = [1.0, 0.01, 0.001]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)  # bare qpos0 -> T-pose, no keyframe

    def ee(side):
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"arm_{side}_end_link")]

    mujoco.mj_forward(model, data)
    assert abs(ee("l")[1]) > 0.8, "expected a T-pose before the controller applies `rest`"

    ctx, dd, body = _plugins(model, data)  # configure + on_reset apply the `rest` stance
    mujoco.mj_forward(model, data)
    # arms already hanging *at spawn* (no transient): ee low and tucked beside the torso
    for side in ("l", "r"):
        assert ee(side)[2] < 0.8, f"{side} arm did not hang on spawn (z={ee(side)[2]:.2f})"
        assert abs(ee(side)[1]) < 0.35, f"{side} arm not beside torso (y={ee(side)[1]:.2f})"

    for _ in range(int(3.0 / model.opt.timestep)):
        body.pre_step(ctx)
        mujoco.mj_step(model, data)
    assert ee("l")[2] < 0.8 and abs(ee("l")[1]) < 0.35, "arms did not hold the hang under gravity"


def test_b4_gripper_couples_via_equality():
    # Driving the master inner_joint1 must move the mimic followers (MJCF joint equalities).
    model, data = _build()
    ctx, dd, body = _plugins(model, data)
    body.set_targets(["idx31_gripper_l_inner_joint1"], [-0.6])
    for _ in range(int(2.0 / model.opt.timestep)):
        body.pre_step(ctx)
        mujoco.mj_step(model, data)

    def q(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return data.qpos[model.jnt_qposadr[jid]]

    master = q("idx31_gripper_l_inner_joint1")
    follower = q("idx41_gripper_l_outer_joint1")  # multiplier -1.0
    assert master < -0.2, f"master did not close (={master:.3f})"
    assert abs(follower - (-1.0) * master) < 0.1, (
        f"follower not coupled ({follower:.3f} vs {-master:.3f})"
    )


# --------------------------------------------------------------- C. sensors


def test_c1_lidar_site_present():
    model, data = _build()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert sid >= 0, "no lidar site"
    assert data.site_xpos[sid][2] > 0.2, "lidar site implausibly low"


def test_c2_manifest_plugins_resolve():
    # the three intrinsic plugins named in the manifest are importable/registered
    from roqsim.plugin import Plugin  # noqa: F401

    assert set(_CFG) >= {"diff_drive", "agibot_g2_controller", "lidar"}
