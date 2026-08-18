"""Object detector sanity checks on a synthetic scene -- no dependency on a robot package.

The property that matters is the one the plugin exists for: the reported pose is relative to the
ROBOT, computed from ground truth, and carries no error the config did not ask for. A detector that
quietly reports world coordinates, or that walks TF and picks up a localization error on the way,
would look identical at the topic level and put a gripper 80 mm off the object.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.object_detector import ObjectDetectorPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin


class _Scene(Plugin):
    """A 'robot' body at (1, 2, 0) yawed 90 degrees, and a target body at (3, 2, 0.5)."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(
            name="base_footprint", pos=[1, 2, 0], quat=[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
        )
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
        target = spec.worldbody.add_body(name="target", pos=[3, 2, 0.5])
        target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.012, 0.045])


def _engine(**config) -> Engine:
    config.setdefault("objects", [{"body": "target", "class_id": "parcel", "size": [0.04, 0.024, 0.09]}])
    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_Scene": {}},
                {"roqsim_sensors.plugins.object_detector:ObjectDetectorPlugin": config},
            ],
        }
    )
    eng = Engine(cfg)
    eng.setup()
    # Populate xpos/xmat: the detector reads body world poses, and without a forward they are zero,
    # which reads as "every object is exactly at the robot" rather than as an error.
    mujoco.mj_forward(eng.ctx.model, eng.ctx.data)
    return eng


def _read(engine: Engine):
    ep = next(e for e in engine.ctx.interface.all() if e.name == "detections")
    return ep.read()


def test_pose_is_relative_to_the_robot_not_the_world():
    """The whole point. The target is 2 m ahead of the robot in world +x, but the robot is yawed 90
    degrees, so in its own frame the target is 2 m to its RIGHT (-y) and 0.5 m up."""
    detections = _read(_engine())
    assert len(detections) == 1
    class_id, pose, size, score = detections[0]
    assert class_id == "parcel"
    assert score == pytest.approx(1.0)
    assert size == (0.04, 0.024, 0.09)
    assert pose[0] == pytest.approx(0.0, abs=1e-9)
    assert pose[1] == pytest.approx(-2.0, abs=1e-9)
    assert pose[2] == pytest.approx(0.5, abs=1e-9)


def test_zero_noise_is_exact():
    """With the error model off the detector must add nothing of its own: a grasp budget is
    millimetres, so a detector that rounds is a detector that misses."""
    a = _read(_engine())[0][1]
    b = _read(_engine())[0][1]
    assert a == pytest.approx(b, abs=1e-12)


def test_bias_is_systematic_and_noise_is_not():
    biased = _read(_engine(position_bias=[0.01, 0.0, 0.0]))[0][1]
    clean = _read(_engine())[0][1]
    assert biased[0] - clean[0] == pytest.approx(0.01, abs=1e-9)

    noisy = _read(_engine(position_stddev=0.05))[0][1]
    assert noisy[:3] != pytest.approx(clean[:3], abs=1e-9)


def test_noise_is_reproducible_for_the_same_seed_and_time():
    """Counter-based via ctx.rng_for: the same (seed, sim_time, name) reproduces the same draw, which
    is what lets a campaign replay a run and lets a noise sweep be a controlled factor."""
    first = _read(_engine(position_stddev=0.05))[0][1]
    second = _read(_engine(position_stddev=0.05))[0][1]
    assert first == pytest.approx(second, abs=1e-12)


def test_max_range_gates_detection():
    assert _read(_engine(max_range=1.0)) == []
    assert len(_read(_engine(max_range=5.0))) == 1


def test_full_dropout_reports_nothing():
    """An empty reading is valid and means 'not detected this cycle' -- a consumer must not read it
    as a pose at the origin."""
    assert _read(_engine(dropout_percent=100.0)) == []


def test_unknown_body_fails_loudly_at_configure():
    """Not silently undetected: a typo would otherwise look exactly like an occlusion."""
    with pytest.raises(RuntimeError, match="nonexistent"):
        _engine(objects=[{"body": "nonexistent", "class_id": "parcel"}])


def test_validate_config_rejects_bad_values():
    errors = ObjectDetectorPlugin().validate_config(
        {"rate_hz": 0, "position_stddev": -1, "dropout_percent": 150, "objects": []}
    )
    assert len(errors) == 4


def test_endpoint_declares_the_detector_contract():
    """Topic, type and frame are the contract a real detector has to satisfy to replace this."""
    eng = _engine(frame="base_footprint")
    ep = next(e for e in eng.ctx.interface.all() if e.name == "detections")
    hints = ep.backend["ros2"]
    assert hints["type"] == "vision_msgs.msg.Detection3DArray"
    assert hints["topic"] == "detections"
    assert hints["frame_id"] == "base_footprint"


def test_orientation_noise_perturbs_the_reported_rotation():
    clean = _read(_engine())[0][1]
    rotated = _read(_engine(orientation_stddev=0.2))[0][1]
    assert np.abs(np.array(rotated[3:]) - np.array(clean[3:])).max() > 1e-6
