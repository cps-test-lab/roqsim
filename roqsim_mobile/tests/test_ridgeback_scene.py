"""The Clearpath Ridgeback: the substrate's only holonomic wheeled base.

Every other robot in `roqsim_mobile` is differential or skid-steer and uses ``diff_drive``. The
Ridgeback's four mecanum wheels strafe, so it uses ``omni_drive`` -- the plugin written for PAL's
OMNI base and, until this port, used by nothing else in the package. ``test_strafes`` is the test
that matters: it is the one behaviour no other base here can produce, and the reason the platform
ledger recorded this port as adding no new capability.

``test_has_no_slip_factor`` guards the other half of that. A holonomic base does not turn by
scrubbing, so unlike husky_a200 / clearpath_jackal / rosbot / panther it must not acquire the ICR
compensation those four need.

The scanner is the ``hokuyo_ust`` device (a UST-10LX) where Clearpath's default Ridgeback
configuration mounts it; the tests at the end check it in a closed room.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named
from scan_mount_utils import (
    chain,
    endpoint,
    forward_range,
    lidar,
    pose_in_base,
    recast,
    robot_hits,
    spawn,
    static_tf,
)

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

#: From Clearpath's expanded r100 xacro @ b0f6d920, not measured from our model.
DESCRIPTION_MASS = 195.838
#: The hokuyo_ust device: 130 g, the UST-10LX specification's weight.
UST_MASS = 0.13
TOTAL_MASS = DESCRIPTION_MASS + UST_MASS
WHEEL_RADIUS = 0.0759
#: Clearpath's published figures for the Ridgeback.
MAX_LINEAR, MAX_ANGULAR = 1.1, 2.0


def _engine():
    engine = Engine(load_config_from_dict(
        {"sim": {"timestep": 0.002}, "components": [
            {"spawn_robot": {"model": "ridgeback", "prefix": "rb_"}, "name": "rb"}]},
        base_dir=Path(".")))
    # A test driving an Engine is the driver, and `ctx.seed` is driver-owned: the scanner's range
    # noise refuses to draw without one.
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _twist(engine):
    """The base's achieved twist in its OWN frame: (vx, vy, wz).

    Read from the free joint's DOFs rather than by integrating world poses. An earlier version of
    this measurement reset the engine inside the loop and reported both a wrong magnitude and a
    wrong yaw *sign* for a model that was correct all along.
    """
    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base_link")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rb_base_free")
    dof = model.jnt_dofadr[jid]
    rot = np.array(data.xmat[bid]).reshape(3, 3)
    body = rot.T @ np.array(data.qvel[dof:dof + 3])
    return float(body[0]), float(body[1]), float(data.qvel[dof + 5])


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-2)
    finally:
        engine.shutdown()


def test_manifest_is_expanded():
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:rb") is not None, "omni_drive did not attach"
        mounts = [p for p in engine.plugins if type(p).__name__ == "SpawnSensorPlugin"]
        assert [p.config["model"] for p in mounts] == ["hokuyo_ust"], "the UST-10LX did not mount"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
        assert any(type(p).__name__ == "OmniDrivePlugin" for p in engine.plugins), (
            "this base is holonomic and must use omni_drive, not diff_drive"
        )
    finally:
        engine.shutdown()


def test_rests_on_its_wheels():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base_link")
        for _ in range(1500):
            engine.step()
        # base_link rides at the wheel radius less the axle height (0.0759 - 0.05).
        assert float(data.xpos[bid][2]) == pytest.approx(0.0259, abs=0.005)
        assert np.abs(data.qvel).max() < 1e-3, "did not settle"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("command", [(0.8, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 1.5),
                                     (0.5, 0.5, 0.0)])
def test_strafes(command):
    """Holonomic: it must track vy as faithfully as vx, which no other base here can do."""
    engine = _engine()
    try:
        handle = engine.ctx.blackboard.get("robot:rb")
        for _ in range(500):
            engine.step()
        handle.drive(*command)
        for _ in range(1500):
            engine.step()
        achieved = _twist(engine)
        for axis, (want, got) in enumerate(zip(command, achieved, strict=True)):
            assert got == pytest.approx(want, abs=0.08), (
                f"axis {'xyw'[axis]}: commanded {want}, achieved {got:.3f}"
            )
    finally:
        engine.shutdown()


def test_has_no_slip_factor():
    """A holonomic base does not scrub, so it must not carry the skid-steer ICR compensation."""
    import yaml

    manifest = resolve_model("roqsim_mobile:ridgeback").path.parent / "ridgeback.manifest.yaml"
    components = yaml.safe_load(manifest.read_text())["components"]
    drive = next(c["omni_drive"] for c in components if "omni_drive" in c)
    assert "slip_factor" not in drive
    assert not any("diff_drive" in c for c in components), "this base is not differential"
    assert drive["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
    assert drive["max_linear_vel"] == pytest.approx(MAX_LINEAR)
    assert drive["max_angular_vel"] == pytest.approx(MAX_ANGULAR)


def test_wheels_are_upright_and_the_riser_survives():
    """Wheel axes on y, and every non-mesh visual present.

    The riser's box visual was silently dropped while this generator hand-picked mesh visuals --
    the same omission that left the Raspberry Pi Mouse's scanner floating. Both are why the shared
    `urdf_source.link_visuals` emits primitives as well as meshes.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(1000):
            engine.step()
        tyres = 0
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not name.endswith("_wheel_tyre"):
                continue
            tyres += 1
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            axis = np.array(data.xmat[model.geom_bodyid[g]]).reshape(3, 3) @ (
                rot.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            )
            assert abs(axis[1]) > 0.99, f"{name}: tyre axis is {axis}, not along y"
        assert tyres == 4

        boxes = [g for g in range(model.ngeom)
                 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_group[g] == 2]
        assert boxes, "the riser's box visual is missing -- only mesh visuals were emitted"
    finally:
        engine.shutdown()


# -- the scanner: a hokuyo_ust device where Clearpath's default configuration mounts it -----------

#: clearpath_config @ b2a64ba, sample/r100/r100_default.yaml:13-17: `hokuyo_ust` on `chassis_link` at
#: xyz (0.3922, 0, 0.1856), no rpy. clearpath_sensors_description @ b0f6d92, urdf/hokuyo_ust.urdf.xacro:
#: `lidar2d_0_link` at that origin, `lidar2d_0_laser` 0.0474 above it (:39-44), scan on
#: `$(arg namespace)/sensors/lidar2d_0/scan` in `lidar2d_0_laser` (:50-52).
MOUNT_XYZ, MOUNT_RPY = (0.3922, 0.0, 0.1856), (0.0, 0.0, 0.0)
LASER_XYZ, LASER_RPY = (0.0, 0.0, 0.0474), (0.0, 0.0, 0.0)
SCAN_FRAME = "lidar2d_0_laser"
SCAN_TOPIC = "sensors/lidar2d_0/scan"
LABEL = "lidar2d_0"
NAMESPACE = "r100_0000"


@pytest.fixture(scope="module")
def scan():
    engine = spawn("ridgeback", {LABEL: None}, owner="rb", prefix="rb_", namespace=NAMESPACE)
    yield engine
    engine.shutdown()


def test_the_scan_frame_is_the_vendor_chain(scan):
    """chassis_link coincides with base_link (r100.urdf.xacro:33-37), so the scan origin in base_link
    is the mount origin composed with the bracket-to-focal-point offset: z 0.2330."""
    want_pos, want_rot = chain((MOUNT_XYZ, MOUNT_RPY), (LASER_XYZ, LASER_RPY))
    assert np.allclose(want_pos, (0.3922, 0.0, 0.2330), atol=1e-9)
    for site in (f"rb_{LABEL}_scan", f"rb_{LABEL}_{SCAN_FRAME}"):
        pos, rot = pose_in_base(scan, site, "rb_")
        assert np.allclose(pos, want_pos, atol=1e-6), f"{site} at {pos}"
        assert np.allclose(rot, want_rot, atol=1e-6), f"{site} rotation {rot}"


def test_the_forward_ray_reads_the_wall(scan):
    published, true = forward_range(scan, lidar(scan, f"rb.{LABEL}"))
    assert published == pytest.approx(true, abs=1e-3), f"reads {published:.4f} m, wall at {true:.4f} m"


def test_the_scan_skips_its_own_mount_and_nothing_else(scan):
    model = scan.ctx.model
    scanner = lidar(scan, f"rb.{LABEL}")
    mount = named(model, mujoco.mjtObj.mjOBJ_BODY, f"rb_{LABEL}_mount")
    assert scanner._bodyexclude == mount, "the scanner excludes something other than its housing"
    _, hits = recast(scan, scanner)
    assert mount not in set(model.geom_bodyid[hits.geomid[hits.geomid >= 0]].tolist())


def test_no_ray_starts_inside_robot_geometry(scan):
    """The previous deck-mounted site at z 0.29 started every ray inside chassis_link's geometry."""
    scanner = lidar(scan, f"rb.{LABEL}")
    inside, _ = robot_hits(scan, scanner, "rb_")
    assert not inside, {body: len(d) for body, d in inside.items()}
    assert np.asarray(scanner.latest.ranges).min() > scanner.range_min, "a ray reads too close"


def test_the_scan_sees_no_part_of_the_robot(scan):
    """At z 0.2330 near the front edge, the 270 degree fan clears the chassis: no robot returns."""
    _, outside = robot_hits(scan, lidar(scan, f"rb.{LABEL}"), "rb_")
    assert outside == {}, sorted(outside)


def test_the_tf_chain_and_topic_are_clearpaths(scan):
    address = f"rb.{LABEL}"
    tf = static_tf(scan, address, NAMESPACE)
    assert [(t["parent"], t["child"]) for t in tf] == [("chassis_link", SCAN_FRAME)], tf
    want_pos, _ = chain((MOUNT_XYZ, MOUNT_RPY), (LASER_XYZ, LASER_RPY))
    assert np.allclose(tf[0]["translation"], want_pos, atol=1e-6)
    assert np.allclose(tf[0]["rotation"], (1.0, 0.0, 0.0, 0.0), atol=1e-9)
    hints = endpoint(scan, "scan", address).backend["ros2"]
    assert (hints["frame_id"], hints["topic"]) == (SCAN_FRAME, SCAN_TOPIC)
    assert "static_tf" not in hints, "the mount owns the chain; the scan publishes none"


def test_the_scanner_is_the_ust_10lx():
    """The scan is the Hokuyo UST-10LX's, as urg_node publishes it."""
    engine = spawn("ridgeback", {LABEL: None}, owner="rb", prefix="rb_", namespace=NAMESPACE)
    try:
        scanner = lidar(engine, f"rb.{LABEL}")
        assert scanner.num_rays == 1081  # steps 0..1080, the last on +135 deg
        assert (scanner.angle_min, scanner.angle_max) == pytest.approx((-2.35619449, 2.35619449))
        assert (scanner.range_min, scanner.range_max) == pytest.approx((0.06, 10.0))
        assert (scanner.detection_min, scanner.detection_max) == pytest.approx((0.021, 30.0))
        assert (scanner.too_close, scanner.no_return) == pytest.approx((0.004, 65.533))
        assert scanner.rate_hz == pytest.approx(40.0)
        assert scanner.config["range_stddev"] == pytest.approx(0.03)
    finally:
        engine.shutdown()
