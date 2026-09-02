# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The PiRacer: the first base here that cannot turn on the spot, and must not pretend to.

``test_a_stationary_car_cannot_turn`` is the reason this model exists. Every other mobile base in
this package answers a yaw-rate command at zero speed by rotating; a car answers it by doing
nothing, because curvature is ``w / v`` and a stationary car has none. A substrate that quietly
rotated here would hide exactly the planner failure a car-like experiment is run to provoke.

The second finding pinned here is that the *split* is real, not just the average steer angle. On a
curve the inner wheel must stand at a larger angle than the outer one -- that is what the linkage
the mechanism is named after does mechanically, and getting the mean right while both wheels share
an angle is a different vehicle. The tolerances below are tight enough to fail if the steering
servos go soft: at the gain first tried, both wheels landed ~0.025 rad short of the geometry.

The masses are OURS, not the description's. Upstream gives the chassis 1000 kg and each wheel 50 kg
with a uniform 0.1 inertia tensor -- about 1203 kg for a 24 cm car, which are Gazebo solver hacks
rather than measurements. Restoring them would give this toy the yaw inertia of a van.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mobile_scene_utils import named

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: Geometry, straight from the source description's joint origins and collision primitives.
WHEELBASE = 0.155       # front axle 0.179 - rear axle 0.024
KINGPIN_TRACK = 0.120   # 2 x 0.060; what the steer split is computed from
WHEEL_RADIUS = 0.034
#: base_link is the chassis origin, which the wheels hold this far off the ground.
REST_HEIGHT = WHEEL_RADIUS - 0.015

#: See the module docstring: assumed pending measurement of the physical car, and the manifest and
#: port log say so. A test that hard-codes them is how a placeholder becomes permanent, so this is
#: deliberately one named constant per assumed quantity rather than a magic number in an assert.
TOTAL_MASS = 1.5
WHEEL_MASS = 0.045


def _engine(prefix="p_"):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": "piracer", "prefix": prefix}, "name": "p"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _yaw(data, bid):
    q = data.xquat[bid]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _drive(engine, v, w, seconds=3.0, settle=0.5):
    """Settle the car, then hold (v, w) and report what the base actually did."""
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
    handle = engine.ctx.blackboard.get("robot:p")
    for _ in range(int(settle / 0.002)):
        engine.step()
    start, yaw0 = data.xpos[bid].copy(), _yaw(data, bid)
    for _ in range(int(seconds / 0.002)):
        handle.drive(v, 0.0, w)
        engine.step()
    moved = float(np.hypot(*(data.xpos[bid][:2] - start[:2])))
    return moved, _yaw(data, bid) - yaw0


def _steer_angles(engine):
    model, data = engine.ctx.model, engine.ctx.data
    return tuple(
        float(data.qpos[model.jnt_qposadr[named(model, mujoco.mjtObj.mjOBJ_JOINT, f"p_{j}")]])
        for j in ("left_steer_joint", "right_steer_joint")
    )


def test_a_stationary_car_cannot_turn():
    """v=0 with a yaw rate must move the car NOWHERE -- the point of the whole model."""
    engine = _engine()
    moved, turned = _drive(engine, 0.0, 1.0)
    assert moved == pytest.approx(0.0, abs=1e-4)
    assert turned == pytest.approx(0.0, abs=1e-4)


def test_the_inner_wheel_steers_harder_than_the_outer():
    """The Ackermann split, against the geometry rather than against itself.

    A commanded (v, w) is a radius R = v/w at the rear axle; the two front wheels then stand at
    ``atan(L / (R -/+ track/2))``, which differ because the inner wheel follows the tighter circle.
    """
    engine = _engine()
    v, w = 0.6, 0.8
    _drive(engine, v, w)
    left, right = _steer_angles(engine)
    radius = v / w
    inner = np.arctan(WHEELBASE / (radius - KINGPIN_TRACK / 2))
    outer = np.arctan(WHEELBASE / (radius + KINGPIN_TRACK / 2))
    assert left > right, "left is the inner wheel of a left turn and must stand at the larger angle"
    assert left == pytest.approx(inner, abs=0.005)
    assert right == pytest.approx(outer, abs=0.005)


def test_a_straight_run_keeps_the_rack_centred_and_tracks_the_speed():
    engine = _engine()
    moved, turned = _drive(engine, 0.6, 0.0)
    assert turned == pytest.approx(0.0, abs=0.01)
    # 98% of commanded: the driven tyres slip a little, which is left in rather than tuned away.
    assert moved / 3.0 == pytest.approx(0.6, rel=0.05)
    left, right = _steer_angles(engine)
    assert abs(left) < 0.01 and abs(right) < 0.01


def test_the_car_rests_on_four_tyres_and_the_chassis_clears_the_floor():
    """The description as authored puts the car 19 mm THROUGH the floor; this is the fix, pinned.

    Also that the chassis box is not a bearing surface: the front tyres sit inside the chassis
    footprint, and MuJoCo filters contacts only against a body's DIRECT parent -- so without an
    explicit exclude each front tyre grinds 4 mm into the chassis for the whole run, silently.
    """
    engine = _engine()
    model, data = engine.ctx.model, engine.ctx.data
    for _ in range(2000):
        engine.step()
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
    assert data.xpos[bid][2] == pytest.approx(REST_HEIGHT, abs=0.001)

    touching = set()
    for contact in data.contact[: data.ncon]:
        pair = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
        }
        touching |= pair
    for wheel in ("front_left", "front_right", "rear_left", "rear_right"):
        assert f"p_{wheel}_tire_geom" in touching, f"{wheel} is not carrying the car"
    assert "p_chassis_geom" not in touching, "the chassis must not touch anything at rest"


def test_the_masses_are_ours_and_not_the_descriptions():
    engine = _engine()
    model = engine.ctx.model
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
    assert model.body_subtreemass[bid] == pytest.approx(TOTAL_MASS, abs=0.01)
    for wheel in ("front_left", "front_right", "rear_left", "rear_right"):
        wid = named(model, mujoco.mjtObj.mjOBJ_BODY, f"p_{wheel}_tire")
        assert model.body_mass[wid] == pytest.approx(WHEEL_MASS, abs=1e-4)


def test_only_the_rear_wheels_are_driven():
    """The front wheels spin free, as they do on the car and as the upstream control block says."""
    engine = _engine()
    model = engine.ctx.model
    actuated = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[a, 0])
        for a in range(model.nu)
    }
    assert actuated == {
        "p_left_steer_joint", "p_right_steer_joint",
        "p_rear_left_tire_joint", "p_rear_right_tire_joint",
    }
