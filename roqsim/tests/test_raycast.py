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
    """A plane is hit within the cutoff wherever its position is, and missed beyond it, as a box is.

    MuJoCo bounds no plane, so it reports a floor a metre below a 15 cm probe; the cast drops that
    hit, so a cliff sensor two metres out reads the floor at two centimetres and a probe held high
    reads nothing."""
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


# -- backends ---------------------------------------------------------------------------------------


class _Ctx:
    def __init__(self, model, sim=None):
        self.model = model
        self.config = {"sim": sim or {}}
        self.raycast = None


def test_a_world_that_names_no_raycaster_casts_through_the_cpu_cast(md):
    m, d = md
    backend = raycast.RaycastService.of(_Ctx(m)).get(None, who="lidar robot.lidar")
    assert isinstance(backend, raycast.CpuRaycaster)
    dirs = np.stack([_PX, _PY, -_PX])
    got = backend.cast(m, d, np.zeros(3), dirs, cutoff=50.0)
    ref = raycast.cast(m, d, np.zeros(3), dirs, cutoff=50.0)
    assert got.dist.tobytes() == ref.dist.tobytes()
    assert got.geomid.tobytes() == ref.geomid.tobytes()
    many = backend.cast_many(m, d, np.zeros((2, 3)), dirs, cutoff=50.0)
    assert (
        many.dist.tobytes()
        == raycast.cast_many(m, d, np.zeros((2, 3)), dirs, cutoff=50.0).dist.tobytes()
    )


def test_every_sensor_of_a_world_shares_one_backend(md):
    m, _ = md
    ctx = _Ctx(m, {"raycaster": "cpu"})
    service = raycast.RaycastService.of(ctx)
    assert raycast.RaycastService.of(ctx) is service
    assert service.get(None, who="a") is service.get("cpu", who="b")


def test_an_unknown_raycaster_fails_naming_the_sensor_and_the_installed_ones(md):
    from roqsim.plugin import PluginError

    m, _ = md
    with pytest.raises(PluginError, match="unknown raycaster 'nope'.*Installed: .*cpu"):
        raycast.RaycastService.of(_Ctx(m, {"raycaster": "nope"})).get(None, who="lidar x")


class _Refuses(raycast.Raycaster):
    def open(self, model):
        raise RuntimeError("no CUDA device")


def test_a_backend_that_cannot_open_fails_the_load_naming_sensor_backend_and_reason(md, tmp_path):
    from roqsim.plugin import PluginError

    m, _ = md
    ctx = _Ctx(m, {"raycaster": f"{__name__}:_Refuses"})
    with pytest.raises(
        PluginError, match=r"lidar robot\.lidar: raycaster .*_Refuses.*no CUDA device"
    ):
        raycast.RaycastService.of(ctx).get(None, who="lidar robot.lidar")


def test_the_cull_beyond_cutoff_is_the_plane_rule_for_planes():
    xml = """
    <mujoco><worldbody>
      <geom name="floor" type="plane" size="0 0 1" pos="5 0 -1"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" pos="0 0 -3"/>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    down = np.array([[0.0, 0.0, -1.0]])
    for origin, cutoff in (([0, 0, 0], 0.5), ([0, 0, 0], 6.0), ([4.8, 0, 0], 0.5)):
        a = raycast.RayHits(dist=np.array([1.0]), geomid=np.array([0], np.int32))
        b = raycast.RayHits(dist=np.array([1.0]), geomid=np.array([0], np.int32))
        raycast._plane_hits(np.array([origin], float), cutoff, m, d, a)
        raycast.cull_beyond_cutoff(np.array([origin], float), cutoff, m, d, b)
        assert (a.dist[0], a.geomid[0]) == (b.dist[0], b.geomid[0])
    # A box whose bounding sphere's centre is 3 m away, radius 0.17 m: the rule keeps a hit on it
    # (2.9 m) exactly where mj_multiRay reports one, beyond the cutoff included.
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="box" size="0.1 0.1 0.1" pos="0 0 -3"/></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for cutoff in (2.0, 2.8, 2.82, 2.83, 2.9, 3.5):
        cpu = raycast.cast(m, d, np.zeros(3), down, cutoff=cutoff, geomgroup=None)
        h = raycast.RayHits(dist=np.array([2.9]), geomid=np.array([0], np.int32))
        raycast.cull_beyond_cutoff(np.zeros((1, 3)), cutoff, m, d, h)
        assert int(h.geomid[0]) == int(cpu.geomid[0]), cutoff


def test_sim_raycaster_is_checked_at_load():
    from roqsim.config import validate_raycast
    from roqsim.plugin import PluginError

    validate_raycast({}, None)
    validate_raycast({"raycaster": "cpu"}, None)
    with pytest.raises(PluginError, match="unknown raycaster"):
        validate_raycast({"raycaster": "nope"}, None)
    with pytest.raises(PluginError, match="sim.raycasters.cpu: unknown key"):
        validate_raycast({"raycasters": {"cpu": {"device": "cuda:0"}}}, None)
    with pytest.raises(PluginError, match="sim.raycaster: a raycaster name"):
        validate_raycast({"raycaster": 3}, None)
