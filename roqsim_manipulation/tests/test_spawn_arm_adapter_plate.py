"""An arm on an adapter plate: the plate stands under the base, its top face at the arm's pose.

The plate is the support's, not the arm's, so it meets everything but its support and the arm's base
-- what a plate bolted between a base and a table does. The FR3 is the arm: its first moving link
starts well above the base face, as on the robot, where an arm whose shoulder dips below its base
plane (the UR10e's collision capsule) would meet a plate under it.
"""

from __future__ import annotations

import mujoco
import pytest
from roqsim_manipulation.plugins.spawn_arm import SpawnArmPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.schema import validate

POSE = {"position": {"x": 0.3, "y": -0.2, "z": 0.815}, "orientation": {"yaw": 1.5707963267948966}}
PLATE = {"diameter": 0.21, "height": 0.015, "offset": [-0.04, 0.0]}


def _engine(tmp_path, plate=PLATE, extra=None):
    plugins = [
        {
            "spawn_arm": {"model": "fr3", "prefix": "a_", "pose": POSE, "adapter_plate": plate}
            | (extra or {}),
            "name": "arm",
        }
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    return engine


def test_the_plate_stands_under_the_base_with_its_top_face_at_the_pose(tmp_path):
    engine = _engine(tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "a_adapter_plate")
    assert g >= 0
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert m.geom_size[g][:2] == pytest.approx([0.105, 0.0075])
    # its offset is in the base frame: yawed a quarter turn, base -x is world -y
    assert d.geom_xpos[g] == pytest.approx([0.3, -0.2 - 0.04, 0.815 - 0.0075], abs=1e-9)
    assert d.geom_xpos[g][2] + m.geom_size[g][1] == pytest.approx(0.815, abs=1e-9)
    engine.shutdown()


def test_the_plate_collides_but_not_with_the_base_it_carries(tmp_path):
    engine = _engine(tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "a_adapter_plate")
    assert m.geom_contype[g] and m.geom_conaffinity[g]
    # at reset, in the arm's home: the base sits on the plate, no link reaches it
    mujoco.mj_forward(m, d)
    touching = {int(c.geom1) for c in d.contact[: d.ncon]} | {
        int(c.geom2) for c in d.contact[: d.ncon]
    }
    assert g not in touching
    engine.shutdown()


def test_a_pedestal_under_a_plate_ends_at_the_plate(tmp_path):
    engine = _engine(tmp_path, extra={"pedestal": True})
    m = engine.ctx.model
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "a_pedestal")
    top = m.geom_pos[g][2] + m.geom_size[g][2]
    assert top == pytest.approx(POSE["position"]["z"] - PLATE["height"])
    engine.shutdown()


def test_without_a_plate_there_is_none(tmp_path):
    plugins = [{"spawn_arm": {"model": "fr3", "prefix": "a_", "pose": POSE}, "name": "arm"}]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, "a_adapter_plate") < 0
    engine.shutdown()


@pytest.mark.parametrize(
    "plate, extra, says",
    [
        ({"diameter": 0.0, "height": 0.015}, {}, "'adapter_plate.diameter' must be > 0"),
        ({"diameter": 0.2, "height": 0.0}, {}, "'adapter_plate.height' must be > 0"),
        (PLATE, {"rail": {"axis": [1, 0, 0]}}, "not with 'rail'"),
    ],
)
def test_a_plate_that_cannot_be_is_refused(plate, extra, says):
    config = {"model": "fr3", "prefix": "a_", "pose": POSE, "adapter_plate": plate} | extra
    errors = SpawnArmPlugin(config).validate_config(config)
    assert any(says in e for e in errors), errors


def test_a_plate_needs_its_size():
    config = {"model": "fr3", "prefix": "a_", "adapter_plate": {"diameter": 0.2}}
    errors = validate(SpawnArmPlugin.CONFIG_SCHEMA, config, strict_keys=True)
    assert any("'adapter_plate.height' is required" in e for e in errors), errors
