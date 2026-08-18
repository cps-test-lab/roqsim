"""The drawn floor is a real surface, and that is deliberate -- but it must stay out of a level scan.

The baked ground plane was hidden (``rgba`` alpha 0) for a while, on the argument that drawing it makes
it a lidar return that closes doorways in a costmap. The measurement behind that used a **15 deg** down
tilt. A TurtleBot 4 on a flat floor does not reach it: over the whole of campaign
``doorway-rebake-pilot-2026-08-13-12462389`` its pitch never exceeded **0.78 deg** (mean 0.62 -- close to
its resting attitude on the caster), and the first floor return needs more than that: 1.35 deg in this
fixture, 1.25 deg in secorolab. That threshold is a property of the *world*, not of the sensor -- a ray
returns only where it meets the plane within the plane's extent, so a bigger room lowers it.

So a scanner sees the ground, as it should, and at the pitch this robot reaches it sees none of it. The
margin is not large, which is why it is pinned here rather than argued. If someone lowers a sensor,
enlarges a world, or tilts a mount, this test is what tells them the floor became a scan surface.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.presence import visible_geomgroup_mask
from roqsim_scenes.cli import scene_to_mjcf

#: TurtleBot 4 rplidar site height above the floor (``roqsim_mobile`` turtlebot4.xml).
LIDAR_Z = 0.192915
#: ``max_range`` the nav worlds give the 2D lidar.
MAX_RANGE = 12.0
#: Worst pitch observed over a full 124 s trial, from the run's own quaternions.
OBSERVED_PITCH_DEG = 0.78


def _room(tmp_path):
    """A bare scene with a stated ground height and nothing covering it, so the plane is the floor."""
    import json

    from roqsim_scenes import scene_mesh_io as mio

    # A wall with thickness, not a quad: MuJoCo collides a mesh by its convex hull and refuses a
    # zero-volume one ("coplanar vertices, cannot compute convex hull").
    verts = np.array(
        [[x, y, z] for x in (-6.0, 6.0) for y in (-6.0, -5.9) for z in (0.0, 2.2)], float
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    mio.write_obj(tmp_path / "meshes" / "Wall.obj", verts, faces)
    (tmp_path / "scene.json").write_text(
        json.dumps(
            {
                "name": "room",
                "unit_scale": 1.0,
                "bounds_min": [-6.0, -6.0, 0.0],
                "bounds_max": [6.0, 6.0, 2.2],
                "ground_z": 0.0,
                "objects": [
                    {
                        "name": "Wall",
                        "mesh": "meshes/Wall.obj",
                        "rgba": [0.7, 0.7, 0.7, 1.0],
                        "collide": True,
                        "render": True,
                    }
                ],
            }
        )
    )
    out = tmp_path / "room.xml"
    scene_to_mjcf.main(["--scene", str(tmp_path / "scene.json"), "--out", str(out)])
    return mujoco.MjModel.from_xml_path(str(out))


def _floor_returns(m, pitch_deg, n=360):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    t = math.radians(pitch_deg)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vec = np.stack(
        [np.cos(ang) * np.cos(t), np.sin(ang) * np.cos(t), np.full(n, -np.sin(t))], axis=1
    ).ravel()
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n)
    mujoco.mj_multiRay(
        m,
        d,
        np.array([0.0, 0.0, LIDAR_Z]),
        vec,
        visible_geomgroup_mask(),
        1,
        -1,
        gid,
        dist,
        None,
        n,
        MAX_RANGE,
    )
    names = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "") if g >= 0 else ""
        for g in gid
    ]
    return sum(1 for s in names if s.startswith("floor"))


@pytest.mark.parametrize("pitch", [0.0, OBSERVED_PITCH_DEG])
def test_a_level_scan_does_not_see_the_drawn_floor(tmp_path, pitch):
    assert _floor_returns(_room(tmp_path), pitch) == 0


def test_the_floor_is_a_real_surface_when_a_sensor_is_actually_tilted(tmp_path):
    """The other half of the contract: this is a floor, not a decoration.

    A downward-looking sensor *should* get returns from the ground. Asserting it here keeps the
    previous fix -- hiding the floor with alpha 0 -- from creeping back as a way to silence the test
    above, which would take the ground away from every 3D sensor to protect a 2D one.
    """
    assert _floor_returns(_room(tmp_path), 5.0) > 0


def test_the_margin_over_the_observed_pitch_is_still_what_we_think(tmp_path):
    """Where the first return appears, in degrees -- the number the two tests above rest on.

    Measured rather than derived, because it depends on how far the plane reaches: a ray only returns
    where it meets the plane *within its extent*, and the furthest reach is along the scene's
    diagonal, not ``max_range``. So the threshold moves with the size of the world, which is exactly
    why asserting a formula here would be asserting the wrong thing.

    What must hold is the margin. If this drifts down toward 0.78 deg -- a bigger room, a lower
    sensor -- the floor is about to start appearing in level scans and "the robot barely pitches"
    stops carrying the argument.
    """
    m = _room(tmp_path)
    first = next(p for p in np.arange(0.1, 5.0, 0.05) if _floor_returns(m, float(p)) > 0)
    assert first > OBSERVED_PITCH_DEG * 1.5, (
        f"first floor return at {first:.2f} deg, too close to the {OBSERVED_PITCH_DEG} deg the robot "
        f"actually reaches"
    )
