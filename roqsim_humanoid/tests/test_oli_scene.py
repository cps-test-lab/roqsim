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


def test_c1_sensor_mounts():
    """C1: lidar/imu sites and the two depth cameras exist at plausible mount heights."""
    model, data = _build()
    for site in ("lidar", "imu"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        assert sid >= 0, f"missing site {site}"
    # lidar rides the torso ~1.2 m up at the home stance (base 0.90 + 0.30 site offset)
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert 1.0 < float(data.site_xpos[lid][2]) < 1.4
    for cam in ("head_camera", "chest_camera"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        assert cid >= 0, f"missing camera {cam}"
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    assert float(data.cam_xpos[head][2]) > 1.3, "head camera implausibly low"


def test_c3_control_rates():
    """C3: policy decimation gives a 100 Hz policy on a 1000 Hz PD loop (the trained cadence)."""
    model, data = _build()
    _, plugin = _plugin(model, data)
    assert plugin._decimation == 10
    assert model.opt.timestep == pytest.approx(0.001)
