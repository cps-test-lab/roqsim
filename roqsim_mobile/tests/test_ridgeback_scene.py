"""The Clearpath Ridgeback: the substrate's only holonomic wheeled base.

Every other robot in `roqsim_mobile` is differential or skid-steer and uses ``diff_drive``. The
Ridgeback's four mecanum wheels strafe, so it uses ``omni_drive`` -- the plugin written for PAL's
OMNI base and, until this port, used by nothing else in the package. ``test_strafes`` is the test
that matters: it is the one behaviour no other base here can produce, and the reason the platform
ledger recorded this port as adding no new capability.

``test_has_no_slip_factor`` guards the other half of that. A holonomic base does not turn by
scrubbing, so unlike husky_a200 / clearpath_jackal / rosbot / panther it must not acquire the ICR
compensation those four need.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

#: From Clearpath's expanded r100 xacro @ b0f6d920, not measured from our model.
TOTAL_MASS = 195.838
WHEEL_RADIUS = 0.0759
#: Clearpath's published figures for the Ridgeback.
MAX_LINEAR, MAX_ANGULAR = 1.1, 2.0


def _engine():
    engine = Engine(load_config_from_dict(
        {"sim": {"timestep": 0.002}, "components": [
            {"spawn_robot": {"model": "ridgeback", "prefix": "rb_"}, "name": "rb"}]},
        base_dir=Path(".")))
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
