"""The raycast seam's contract: what the default mask is, and that the shapes agree.

The load-bearing assertion here is :func:`test_the_visible_mask_is_the_default`. Every raycaster in
the tree relies on it, and the bug class it closes -- an *absent* entity still being a lidar return --
is one each raycaster would otherwise have to close on its own.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim import raycast
from roqsim.presence import ABSENT_GEOM_GROUP

# One box straight ahead on +x at 2 m (near face at 1.9), one off to +y at 3 m.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="ahead" pos="2 0 0"><geom name="g_ahead" type="box" size="0.1 0.1 0.1"/></body>
    <body name="aside" pos="0 3 0"><geom name="g_aside" type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>
"""

_PX = np.array([1.0, 0.0, 0.0])
_PY = np.array([0.0, 1.0, 0.0])


@pytest.fixture
def md():
    m = mujoco.MjModel.from_xml_string(_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _gid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def test_a_hit_reports_distance_and_geom(md):
    m, d = md
    hits = raycast.cast(m, d, np.zeros(3), _PX, cutoff=50.0)
    assert hits.dist[0] == pytest.approx(1.9)
    assert hits.geomid[0] == _gid(m, "g_ahead")


def test_a_miss_is_minus_one(md):
    m, d = md
    hits = raycast.cast(m, d, np.zeros(3), np.array([0.0, 0.0, 1.0]), cutoff=50.0)
    assert hits.dist[0] == -1.0
    assert hits.geomid[0] == -1


def test_the_visible_mask_is_the_default(md):
    """An absent entity is not a return, without the caller passing anything.

    The geom is left fully OPAQUE so that only the ``geomgroup`` mask can hide it -- the alpha-0
    trick in ``presence.set_present`` would otherwise mask the very thing under test.
    """
    m, d = md
    g = _gid(m, "g_ahead")
    assert raycast.cast(m, d, np.zeros(3), _PX, cutoff=50.0).dist[0] == pytest.approx(1.9)

    m.geom_group[g] = ABSENT_GEOM_GROUP
    m.geom_rgba[g][3] = 1.0
    assert raycast.cast(m, d, np.zeros(3), _PX, cutoff=50.0).dist[0] == -1.0


def test_absent_geometry_is_visible_only_when_asked_for_explicitly(md):
    """``geomgroup=None`` still means "every group" -- the escape hatch has to stay expressible."""
    m, d = md
    g = _gid(m, "g_ahead")
    m.geom_group[g] = ABSENT_GEOM_GROUP
    m.geom_rgba[g][3] = 1.0
    hits = raycast.cast(m, d, np.zeros(3), _PX, cutoff=50.0, geomgroup=None)
    assert hits.dist[0] == pytest.approx(1.9)


def test_flat_and_shaped_directions_agree(md):
    m, d = md
    shaped = raycast.cast(m, d, np.zeros(3), np.stack([_PX, _PY]), cutoff=50.0)
    flat = raycast.cast(m, d, np.zeros(3), np.concatenate([_PX, _PY]), cutoff=50.0)
    assert np.array_equal(shaped.dist, flat.dist)
    assert np.array_equal(shaped.geomid, flat.geomid)
    # Order follows the directions given, so ray 0 is the +x box and ray 1 the +y one.
    assert shaped.dist[0] == pytest.approx(1.9)
    assert shaped.dist[1] == pytest.approx(2.9)


def test_out_buffers_are_reused_not_reallocated(md):
    m, d = md
    buf = raycast.buffers(2)
    got = raycast.cast(m, d, np.zeros(3), np.stack([_PX, _PY]), cutoff=50.0, out=buf)
    assert got is buf
    assert buf.dist[0] == pytest.approx(1.9)


def test_out_of_the_wrong_size_is_refused(md):
    m, d = md
    with pytest.raises(ValueError, match="sized for"):
        raycast.cast(m, d, np.zeros(3), np.stack([_PX, _PY]), cutoff=50.0, out=raycast.buffers(5))


def test_dirs_must_be_whole_vectors(md):
    m, d = md
    with pytest.raises(ValueError, match="multiple of 3"):
        raycast.cast(m, d, np.zeros(3), np.array([1.0, 0.0]), cutoff=50.0)


def test_normals_are_filled_only_when_a_buffer_is_given(md):
    m, d = md
    assert raycast.cast(m, d, np.zeros(3), _PX, cutoff=50.0).normal is None
    hits = raycast.cast(m, d, np.zeros(3), _PX, cutoff=50.0, out=raycast.buffers(1, normals=True))
    # The +x box's near face points back at the origin.
    assert hits.normal[0] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)


def test_cast_many_matches_cast_from_each_origin(md):
    """``cast_many`` is the same question per origin, so it must agree ray for ray."""
    m, d = md
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dirs = np.stack([_PX, _PY])
    many = raycast.cast_many(m, d, origins, dirs, cutoff=50.0, normals=True)
    assert many.dist.shape == (3, 2)
    assert many.nray == 2
    for i, o in enumerate(origins):
        one = raycast.cast(m, d, o, dirs, cutoff=50.0, out=raycast.buffers(2, normals=True))
        assert np.array_equal(many.dist[i], one.dist)
        assert np.array_equal(many.geomid[i], one.geomid)
        assert np.array_equal(many.normal[i], one.normal)


def test_cast_many_defaults_to_the_visible_mask_too(md):
    m, d = md
    g = _gid(m, "g_ahead")
    m.geom_group[g] = ABSENT_GEOM_GROUP
    m.geom_rgba[g][3] = 1.0
    many = raycast.cast_many(m, d, np.zeros((2, 3)), _PX, cutoff=50.0)
    assert (many.dist == -1.0).all()


_PLANE_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <geom name="near_box" type="box" size="0.05 0.05 0.05" pos="2.15 0 0.05"/>
    <geom name="far_box" type="box" size="0.05 0.05 0.05" pos="4 0 0.05"/>
  </worldbody>
</mujoco>
"""


def test_a_plane_within_the_cutoff_is_hit_however_far_its_position_is():
    """MuJoCo culls a plane by the distance to its POSITION, so a floor centred at the origin
    vanished from any short-range cast made more than the cutoff away from the origin -- a cliff
    sensor two metres out read no floor. The planes are intersected here instead; the
    bounding-sphere cull still drops a box beyond the requested cutoff."""
    model = mujoco.MjModel.from_xml_string(_PLANE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    origin = np.array([2.0, 0.0, 0.02])
    down = raycast.cast(model, data, origin, [[0.0, 0.0, -1.0]], cutoff=0.15)
    assert down.dist[0] == pytest.approx(0.02)
    assert down.geomid[0] == floor
    # A pitched ray meets the floor at 0.02 / sin(80 deg).
    pitched = raycast.cast(model, data, origin, [[0.1736, 0.0, -0.9848]], cutoff=0.15)
    assert pitched.dist[0] == pytest.approx(0.02 / 0.9848, abs=1e-4)
    ahead = raycast.cast(model, data, origin, [[1.0, 0.0, 0.0]], cutoff=0.15)
    assert ahead.geomid[0] == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "near_box")
    # Beyond the requested cutoff the floor is a miss too, as a box is.
    high = raycast.cast(model, data, origin + [0, 0, 1], [[0.0, 0.0, -1.0]], cutoff=0.15)
    assert high.dist[0] == -1.0
    far = raycast.cast(model, data, np.array([3.0, 0.0, 0.05]), [[1.0, 0.0, 0.0]], cutoff=0.15)
    assert far.dist[0] == -1.0
    many = raycast.cast_many(model, data, [origin, origin + [1, 0, 0]], [[0, 0, -1.0]], cutoff=0.15)
    assert np.allclose(many.dist, 0.02)
    assert np.all(many.geomid == floor)


def test_a_far_plane_keeps_mujocos_own_plane_rules():
    """One-sided, bounded by a nonzero size, and subject to the static and group filters."""
    xml = """
    <mujoco>
      <worldbody>
        <geom name="patch" type="plane" size="1 1 0.1" group="3"/>
        <body name="lid" pos="0 0 0"><geom name="lid" type="plane" size="0 0 0.1" pos="0 0 2" group="0"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    patch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "patch")
    # Inside the patch's size: hit. Outside it: the ray sails on (a miss below the lid).
    assert raycast.cast(model, data, [0.5, 0, 0.05], [[0, 0, -1.0]], cutoff=0.1).geomid[0] == patch
    assert raycast.cast(model, data, [2.5, 0, 0.05], [[0, 0, -1.0]], cutoff=0.1).dist[0] == -1.0
    # From below the patch, upward: one-sided, no hit.
    assert raycast.cast(model, data, [0.5, 0, -0.05], [[0, 0, 1.0]], cutoff=0.1).dist[0] == -1.0
    # Static geometry skipped when the caller says so; a group masked out is skipped too.
    no_static = raycast.cast(
        model, data, [0.5, 0, 0.05], [[0, 0, -1.0]], cutoff=0.1, flg_static=False
    )
    assert no_static.dist[0] == -1.0
    masked = raycast.cast(
        model,
        data,
        [0.5, 0, 0.05],
        [[0, 0, -1.0]],
        cutoff=0.1,
        geomgroup=np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8),
    )
    assert masked.dist[0] == -1.0
