"""The raycast seam's contract: what the default mask is, and that the shapes agree.

The load-bearing assertion here is :func:`test_the_visible_mask_is_the_default`. Every raycaster in
the tree relies on it, and the bug class it closes -- an *absent* entity still being a lidar return --
was live in three plugins and documented as a standing warning in :mod:`roqsim.presence`.
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
