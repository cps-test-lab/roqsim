# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``tricycle_drive`` on a complete vehicle: a three-wheel AGV driven on the floor, with a scanner.

``test_tricycle_drive.py`` checks the plugin's contract on two small trucks; this file drives a whole
vehicle, ``fixtures/tricycle_agv.xml``, through the physics. No tricycle is bundled, so the fixture
is spawned by path and its components are stated here, where a bundled model's manifest would state
them. Each constant is the fixture's value, and each is an ASSUMPTION there, since it is no product.

What it pins:

* the vehicle rests on its three tyres, not on its chassis;
* it holds a commanded speed forwards and backwards, and a commanded curve at the steering angle the
  tricycle relation gives;
* at the lock it turns on the physical minimum radius ``steer_offset / tan(max_steer_angle)``, the
  number a Nav2 ``minimum_turning_radius`` is chosen above;
* ``cmd_vel`` with ``v = 0`` moves nothing;
* odometry agrees with ground truth over a curve;
* the scanner's rays meet no part of the vehicle.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named
from scan_mount_utils import assert_mounts, forward_range, lidar, robot_hits, spawn

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

MODEL = str(Path(__file__).parent / "fixtures" / "tricycle_agv.xml")
STEER_OFFSET = 0.60  # m, the steered wheel's x in base_link
MAX_STEER = 1.3  # rad, the steering lock
#: The physical minimum turning radius of base_link, at the lock.
MIN_RADIUS = STEER_OFFSET / np.tan(MAX_STEER)
MOUNTS = {"scan_front": ("sick_tim571", "laser_link", (0.66, 0.0, 0.43), (0.0, 0.0, 0.0), "scan")}
DT = 0.002

#: What a manifest would bring. The steered wheel is also the driven one (``drive: steer_wheel``), as
#: on a tugger or a pallet truck.
COMPONENTS = [
    {
        "tricycle_drive": {
            "drive": "steer_wheel",
            "steer_offset": STEER_OFFSET,  # the MJCF's steer_link x
            "steer_wheel_radius": 0.09,  # the front tyre ellipsoid
            "max_steer_angle": MAX_STEER,  # the MJCF's steer_joint range
            "max_linear_vel": 1.0,  # ASSUMPTION: m/s, an indoor AGV's pace
            # ASSUMPTION: no tread above 1.2 m/s. The steered wheel rolls at v / cos(delta), 3.7 v
            # at the lock, so near the lock this cap is what slows the vehicle through the turn.
            "max_wheel_speed": 1.2,
            "steer_rate": 2.0,  # ASSUMPTION: rad/s
            "accel_limit": 0.8,  # ASSUMPTION: m/s^2
            "cmd_vel_timeout": 0.5,
            "steer_actuator": "steer_motor",
            "steer_joint": "steer_joint",
            "drive_actuators": ["front_wheel_motor"],
            "drive_joints": ["front_wheel_joint"],
            "passive_joints": ["rear_left_wheel_joint", "rear_right_wheel_joint"],
        }
    },
    # A SICK TiM571 device on the deck's front edge, with the scan plane at z = 0.43 m: above the
    # deck (0.36 m), so no ray of the 270 degree field meets the vehicle.
    {
        "spawn_sensor": {
            "model": "sick_tim571",
            "parent_frame": "base_link",
            "pose": {"position": {"x": 0.66, "z": 0.43}},
            "frame_id": "laser_link",
        },
        "name": "scan_front",
        "components": [{"lidar": {"topics": {"scan": "scan"}}}],
    },
]


def _engine():
    world = {
        "sim": {"timestep": DT},
        "components": [
            {
                "spawn_robot": {"model": MODEL, "prefix": "t_"},
                "name": "t",
                "components": COMPONENTS,
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0  # a test driving an Engine is the driver, and the seed is driver-owned
    engine.setup()
    engine.reset()
    for _ in range(int(0.5 / DT)):  # settle onto the tyres
        engine.step()
    return engine


def _base(engine):
    return named(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "t_base_link")


def _pose(engine):
    d, bid = engine.ctx.data, _base(engine)
    q = d.xquat[bid]
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return float(d.xpos[bid][0]), float(d.xpos[bid][1]), yaw


def _steer(engine):
    m, d = engine.ctx.model, engine.ctx.data
    return float(d.qpos[m.jnt_qposadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, "t_steer_joint")]])


def _hold(engine, v, w, seconds):
    """Hold (v, w); return the base_link speed and yaw rate over the last half, from ground truth."""
    handle = engine.ctx.blackboard.get("robot:t")
    n = int(seconds / DT)
    samples = []
    for i in range(n):
        handle.drive(v, 0.0, w)
        engine.step()
        if i >= n // 2:
            samples.append(_pose(engine))
    xy = np.array([s[:2] for s in samples])
    yaw = np.unwrap([s[2] for s in samples])
    t = (len(samples) - 1) * DT
    speed = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) / t)
    return speed, float((yaw[-1] - yaw[0]) / t)


def test_it_carries_a_tricycle_drive_and_one_scanner():
    engine = _engine()
    (drive,) = [p for p in engine.plugins if type(p).__name__ == "TricycleDrivePlugin"]
    assert drive.config["drive"] == "steer_wheel"
    assert drive.config["steer_offset"] == STEER_OFFSET
    assert drive.config["max_steer_angle"] == MAX_STEER
    assert_mounts(engine, "t", MOUNTS)


def test_it_rests_on_its_three_tyres():
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_type[g] == 0]
    touching = set()
    for c in d.contact[: d.ncon]:
        for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if other in floor:
                touching.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
    assert touching == {"t_rear_left_wheel_geom", "t_rear_right_wheel_geom", "t_front_wheel_geom"}
    assert abs(_pose(engine)[0]) < 1e-3 and abs(d.xpos[_base(engine)][2]) < 3e-3


@pytest.mark.parametrize("v", [0.5, -0.4])
def test_it_holds_a_commanded_speed_both_ways(v):
    engine = _engine()
    x0 = _pose(engine)[0]
    speed, yaw_rate = _hold(engine, v, 0.0, 6.0)
    assert speed == pytest.approx(abs(v), rel=0.02)
    assert abs(yaw_rate) < 0.01
    assert np.sign(_pose(engine)[0] - x0) == np.sign(v)


def test_it_holds_a_commanded_curve_at_the_tricycle_angle():
    v, w = 0.4, 0.4
    engine = _engine()
    speed, yaw_rate = _hold(engine, v, w, 8.0)
    assert speed == pytest.approx(v, rel=0.03)
    assert yaw_rate == pytest.approx(w, rel=0.03)
    assert _steer(engine) == pytest.approx(np.arctan(w * STEER_OFFSET / v), abs=0.02)


def test_at_the_lock_it_turns_on_the_physical_minimum_radius():
    """A yaw rate beyond the lock is clamped to it: the vehicle follows its tightest circle."""
    engine = _engine()
    speed, yaw_rate = _hold(engine, 0.2, 5.0, 8.0)
    assert _steer(engine) == pytest.approx(MAX_STEER, abs=0.02)
    assert speed / yaw_rate == pytest.approx(MIN_RADIUS, rel=0.1)


def test_a_stationary_tricycle_cannot_turn():
    engine = _engine()
    x0, y0, yaw0 = _pose(engine)
    _hold(engine, 0.0, 0.8, 3.0)
    x, y, yaw = _pose(engine)
    assert np.hypot(x - x0, y - y0) < 1e-3 and abs(yaw - yaw0) < 1e-3


def test_odometry_agrees_with_ground_truth_over_a_curve():
    engine = _engine()
    x0, y0, yaw0 = _pose(engine)
    _hold(engine, 0.4, 0.4, 6.0)
    x, y, yaw = _pose(engine)
    ox, oy, oyaw = engine.ctx.blackboard.get("robot:t").read_odom()[:3]
    # Odometry starts at the spawn pose, which is where the settled vehicle stands. It is dead
    # reckoning and drifts where the steered, driven tyre slips on the curve (roqsim_mobile's README),
    # so the bound scales with the distance driven: 2 % of the 2.4 m arc, where this vehicle lands
    # near 1.3 %. A bound much tighter than the drift would fail on a correct change to the contact
    # physics; one much looser would pass an odometry that ignores the measured steering angle.
    travelled = 0.4 * 6.0
    assert np.hypot(ox - (x - x0), oy - (y - y0)) < 0.02 * travelled
    assert abs((oyaw - (yaw - yaw0) + np.pi) % (2 * np.pi) - np.pi) < 0.04


@pytest.fixture(scope="module")
def scan():
    engine = spawn(MODEL, MOUNTS, owner="t", prefix="t_", namespace="agv", components=COMPONENTS)
    yield engine
    engine.shutdown()


def test_no_ray_of_the_scan_meets_the_vehicle(scan):
    """The mount is above the deck, so the whole 270 degree field is clear of the vehicle."""
    inside, outside = robot_hits(scan, lidar(scan, "t.scan_front"), "t_")
    assert not inside and not outside, (sorted(inside), sorted(outside))


def test_the_forward_ray_reads_the_wall(scan):
    published, true = forward_range(scan, lidar(scan, "t.scan_front"))
    assert published == pytest.approx(true, abs=1e-3)
