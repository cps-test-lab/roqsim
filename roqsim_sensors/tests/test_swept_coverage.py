# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``swept_coverage_monitor``: what a *moving* sensor's field of view ever reached.

The static probe beside it is checked point-by-point, because one evaluation of one pose is a
geometry question with a closed-form answer. An accumulator is not: what can go wrong is the
*accumulation*, and the properties that pin it down are relational rather than absolute. So these
tests assert the three that a broken accumulator cannot satisfy:

* **motion pays.** A sensor carried along a corridor must cover strictly more than the same sensor
  held at one end of it. An accumulator that re-poses nothing, or that overwrites instead of ORing,
  passes every absolute check and fails this one.
* **walls still hold.** A cell behind an occluder stays uncovered however many poses are folded in.
  This is what separates a swept union from a swept *range circle*, which is what an offline
  reconstruction from recorded poses would give.
* **the union is monotone.** The fraction never falls, and a cell once covered is covered for the
  rest of the trial -- while a reset clears it, because a trial's sweep is that trial's.

Numbers, not exit codes: the area figure is re-derived here from the sample points and the visit
counts the reader hands out, so the plugin's own bookkeeping is checked against an independent count
of the same cells rather than against itself.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.swept_coverage_monitor import (
    UNKNOWN_AREA,
    SweptCoverageMonitorPlugin,
)

from roqsim.config import PluginError, load_config_from_dict
from roqsim.engine import Engine
from roqsim.registry import resolve_plugin

# A long walled corridor: wide enough that a sensor at one end cannot see the far end, and closed on
# all four sides because the volume sampler keeps only ENCLOSED free space.
_CORRIDOR = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="20 20 .1"/>
    <geom name="wall_w" type="box" pos="-1 0 1" size=".1 6 1"/>
    <geom name="wall_e" type="box" pos="11 0 1" size=".1 6 1"/>
    <geom name="wall_s" type="box" pos="5 -1.5 1" size="6 .1 1"/>
    <geom name="wall_n" type="box" pos="5 1.5 1" size="6 .1 1"/>
  </worldbody>
</mujoco>
"""

# The same corridor with a full-height divider across it, leaving a doorway-free blockage: nothing
# beyond x = 5 is ever visible from the west half, at any pose the rover can reach there.
_DIVIDED = _CORRIDOR.replace(
    '<geom name="wall_n"',
    '<geom name="divider" type="box" pos="5 0 1" size=".1 1.5 1"/>\n    <geom name="wall_n"',
)

# The carrier: a box on a free joint with a forward-looking camera on its own body, so the mount
# moves with it and `frame: rover_cam` resolves, relative to the spawn, as a camera frame.
_ROVER = """
<mujoco model="rover">
  <worldbody>
    <body name="base" pos="0 0 .5">
      <geom name="base_geom" type="box" size=".15 .15 .15"/>
      <!-- Looks along +x with world-up as up: MuJoCo's optical axis is -z, so -z -> +x and
           +y -> +z_world. Spelled with xyaxes because an euler triple for this is easy to
           get wrong and silently points the camera at the floor. -->
      <camera name="rover_cam" pos="0 0 0" xyaxes="0 -1 0  0 0 1" fovy="70"/>
      <!-- A scan site for the lidar mount: rpy=0 there means boresight +x, as the adapters read it. -->
      <site name="rover_scan" pos="0 0 0.05" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

SENSOR = {"fovy": 70, "width": 640, "height": 480, "near": 0.05, "far": 4.0}
SAMPLE = {"volume": True, "objects": False, "resolution": 0.25, "heights": [0.5]}


def _world(tmp_path, scene: str = _CORRIDOR, **monitor):
    (tmp_path / "scene.xml").write_text(scene)
    (tmp_path / "rover.xml").write_text(_ROVER)
    config = {
        "type": "camera",
        "frame": "rover_cam",
        "config": SENSOR,
        "sample": SAMPLE,
        "compute_rate_hz": 50.0,
        **monitor,
    }
    return {
        "sim": {"world": str(tmp_path / "scene.xml")},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "rover.xml"),
                    "pose": {"position": {"x": 0.0, "y": 0.0}},
                    "motion": "physics",
                },
                "name": "rover",
                "components": [{"swept_coverage_monitor": config}],
            },
        ],
    }


def _engine(world):
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _reader(engine):
    return engine.ctx.blackboard.get("swept_coverage:rover.swept_coverage_monitor")


def _drive_to(engine, x: float, steps: int = 4):
    """Put the rover at ``x`` and hold it there for a few physics steps.

    Held rather than teleported, because the monitor evaluates on its own rate: a test that jumps
    between poses faster than the sensor samples measures the test's stride, not the sweep.
    """
    engine.ctx.data.qpos[0] = x
    # Held at its spawn height with no residual velocity: the carrier rides a free joint, so left
    # alone it would also be falling, and the sweep would then depend on how long the test drove.
    engine.ctx.data.qpos[2] = 0.5
    engine.ctx.data.qvel[:] = 0.0
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    for _ in range(steps):
        engine.step()


def _sweep(engine, xs):
    for x in xs:
        _drive_to(engine, x)
    return _reader(engine).read()


# -- motion pays -------------------------------------------------------------------------------


def test_a_moving_sensor_covers_strictly_more_than_a_still_one(tmp_path):
    """The property the plugin exists for: the union grows with the path, not with the run length.

    Both runs fold in the same number of evaluations at the same rate; only one of them moves. An
    accumulator that never re-posed the field of view, or that replaced the mask instead of ORing it,
    would report the two as equal.
    """
    still = _engine(_world(tmp_path))
    try:
        held = _sweep(still, [0.0] * 9)
    finally:
        still.shutdown()

    moving = _engine(_world(tmp_path))
    try:
        swept = _sweep(moving, np.linspace(0.0, 8.0, 9))
    finally:
        moving.shutdown()

    assert held.n_evaluations == swept.n_evaluations, "the comparison must differ only in motion"
    assert held.n_covered > 0, "a still sensor in a corridor sees something"
    assert swept.n_covered > held.n_covered
    # Not a rounding difference: a 4 m camera walked down a 12 m corridor should roughly double it.
    assert swept.n_covered >= 2 * held.n_covered
    assert swept.fraction > held.fraction


def test_holding_still_for_longer_does_not_grow_the_union(tmp_path):
    """The complement of the test above, and the one that catches a double-count.

    Extra evaluations at one pose raise the visit counts and must leave the union untouched --
    coverage is a union over space, not a sum over time.
    """
    engine = _engine(_world(tmp_path))
    try:
        short = _sweep(engine, [0.0] * 3)
        long = _sweep(engine, [0.0] * 12)
        assert long.n_evaluations > short.n_evaluations
        assert long.n_covered == short.n_covered
        assert long.covered_area_m2 == pytest.approx(short.covered_area_m2)
        # The visits DID keep rising, which is what makes this a revisit observable too.
        assert long.mean_visits > short.mean_visits
    finally:
        engine.shutdown()


# -- walls still hold --------------------------------------------------------------------------


def test_an_occluded_cell_stays_uncovered_however_far_the_sensor_travels(tmp_path):
    """A divider the sensor never passes must leave everything behind it at zero visits.

    This is the line between a swept FoV and a swept range circle: the second would mark the far half
    covered as soon as the sensor came within 4 m of the divider and pointed at it.
    """
    engine = _engine(_world(tmp_path, scene=_DIVIDED))
    try:
        _sweep(engine, np.linspace(0.0, 4.5, 10))
        reader = _reader(engine)
        points, visits = reader.points(), reader.visits()
        beyond = points[:, 0] > 5.2  # strictly past the divider (half-thickness 0.1 m)
        assert beyond.any(), "the divided corridor must sample cells on the far side"
        assert int(visits[beyond].sum()) == 0
        # And the near side genuinely was covered, so the assertion above is not vacuous.
        near = (points[:, 0] > 0.5) & (points[:, 0] < 4.5)
        assert int((visits[near] > 0).sum()) > 0
    finally:
        engine.shutdown()


def test_the_same_sweep_reaches_further_once_the_divider_is_gone(tmp_path):
    """The occlusion assertion above, made non-vacuous by the world it is compared against."""
    path = np.linspace(0.0, 4.5, 10)
    divided = _engine(_world(tmp_path, scene=_DIVIDED))
    try:
        blocked = _sweep(divided, path)
    finally:
        divided.shutdown()
    open_ = _engine(_world(tmp_path, scene=_CORRIDOR))
    try:
        clear = _sweep(open_, path)
    finally:
        open_.shutdown()
    assert clear.n_covered > blocked.n_covered


# -- the union is monotone ---------------------------------------------------------------------


def test_the_union_never_shrinks_over_a_run(tmp_path):
    """Monotonicity, per point and in aggregate: driving out and back cannot un-cover anything."""
    engine = _engine(_world(tmp_path))
    try:
        reader = _reader(engine)
        fractions = []
        masks = []
        for x in [0.0, 2.0, 4.0, 6.0, 8.0, 6.0, 4.0, 2.0, 0.0]:
            _drive_to(engine, x)
            fractions.append(reader.read().fraction)
            masks.append(reader.visits() > 0)
        assert all(a <= b for a, b in zip(fractions, fractions[1:], strict=False)), fractions
        assert fractions[-1] > fractions[0], "the sweep has to have grown for this to mean anything"
        for earlier, later in zip(masks, masks[1:], strict=False):
            # Every point covered earlier is still covered: `earlier & ~later` must be empty.
            assert not np.any(earlier & ~later)
    finally:
        engine.shutdown()


def test_a_reset_clears_the_union(tmp_path):
    """A trial's sweep is that trial's -- otherwise trial 2 of one process inherits trial 1's."""
    engine = _engine(_world(tmp_path))
    try:
        first = _sweep(engine, np.linspace(0.0, 8.0, 9))
        assert first.n_covered > 0
        engine.reset()
        after = _reader(engine).read()
        assert (after.n_covered, after.n_evaluations, after.mean_visits) == (0, 0, 0.0)
        assert after.fraction == 0.0
        assert after.n_points == first.n_points, "the sample set is fixed, not re-derived"
        assert after.covered_area_m2 == 0.0
    finally:
        engine.shutdown()


def test_a_rate_the_timestep_divides_evaluates_on_every_period(tmp_path):
    """At 20 Hz on a 10 ms step every evaluation is exactly five steps after the last.

    The sim clock is a float sum of timesteps, so the step that lands on the due time can read a
    hair below it; the gate must take that step, not the one after it.
    """
    engine = _engine(_world(tmp_path, compute_rate_hz=20.0))
    try:
        reader = _reader(engine)
        evaluated, last = [], reader.read().n_evaluations
        for step in range(1, 1001):  # 10 s: long enough for the summed clock to drift
            engine.step()
            count = reader.read().n_evaluations
            if count != last:
                evaluated.append(step)
            last = count
        gaps = {b - a for a, b in zip(evaluated, evaluated[1:], strict=False)}
        assert gaps == {5}, f"evaluation spacing in steps: {sorted(gaps)}"
    finally:
        engine.shutdown()


# -- the area figure ---------------------------------------------------------------------------


def test_the_covered_area_is_the_cells_it_says_it_is(tmp_path):
    """Re-derive the area from the reader's own points and visits, independently of the report.

    The figure a coverage experiment quotes is an area, and the step from "points seen" to "square
    metres" is where a silent factor hides: counting per point instead of per xy column multiplies
    the area by the number of height layers, and a plausible number comes out either way.
    """
    engine = _engine(_world(tmp_path, sample={**SAMPLE, "heights": [0.4, 0.9]}))
    try:
        report = _sweep(engine, np.linspace(0.0, 8.0, 9))
        reader = _reader(engine)
        points, visits = reader.points(), reader.visits()

        cell = SAMPLE["resolution"] ** 2
        cells = np.floor(points[:, :2] / SAMPLE["resolution"]).astype(int)
        distinct = {tuple(c) for c in cells}
        covered = {tuple(c) for c in cells[visits > 0]}

        assert report.cell_area_m2 == pytest.approx(cell)
        assert report.sampled_area_m2 == pytest.approx(len(distinct) * cell)
        assert report.covered_area_m2 == pytest.approx(len(covered) * cell)
        # Two height layers, so there are strictly more points than columns -- which is exactly the
        # multiplication this test exists to rule out.
        assert report.n_points > len(distinct)
        assert report.covered_area_m2 < report.sampled_area_m2
    finally:
        engine.shutdown()


def test_without_a_volume_grid_the_area_is_reported_as_unknown(tmp_path):
    """Surface points stand for no footprint, so no area is derivable -- and none is invented."""
    engine = _engine(_world(tmp_path, sample={"volume": False, "objects": True, "per_object": 16}))
    try:
        report = _sweep(engine, [0.0, 2.0, 4.0])
        assert report.n_points > 0
        assert report.covered_area_m2 == UNKNOWN_AREA
        assert report.sampled_area_m2 == UNKNOWN_AREA
        assert report.cell_area_m2 == 0.0
    finally:
        engine.shutdown()


# -- the endpoint and the report file ----------------------------------------------------------


def test_the_endpoint_publishes_the_running_fraction(tmp_path):
    """A coverage-over-time curve is the series a reader wants; the final value is its last sample."""
    engine = _engine(_world(tmp_path))
    try:
        endpoint = next(e for e in engine.ctx.interface._endpoints if e.name == "coverage")
        assert endpoint.backend["ros2"]["field"] == "fraction"
        assert endpoint.backend["ros2"]["topic"] == "coverage_fraction"
        _sweep(engine, np.linspace(0.0, 8.0, 9))
        assert endpoint.read().fraction == pytest.approx(_reader(engine).read().fraction)
    finally:
        engine.shutdown()


def test_the_report_names_what_the_sweep_missed(tmp_path):
    """`out:` reuses the static report shape, so `uncovered_regions` clusters what was never reached."""
    engine = _engine(_world(tmp_path, scene=_DIVIDED, out=str(tmp_path / "swept")))
    try:
        _sweep(engine, np.linspace(0.0, 4.5, 10))
    finally:
        engine.shutdown()  # the file is written at shutdown
    report = json.loads((tmp_path / "swept" / "report.json").read_text())
    assert report["swept"]["sensor_type"] == "camera"
    assert "rover_cam" in report["swept"]["mount"]
    assert 0.0 < report["swept"]["fraction"] < 1.0
    assert report["swept"]["n_evaluations"] > 0
    assert report["achieved"]["fraction_covered_k1"] == pytest.approx(report["swept"]["fraction"])
    # The blocked far half is a connected uncovered cluster, so it has to show up as a gap.
    gaps = report["uncovered_regions"]
    assert gaps, "a divided corridor cannot be fully swept from one side"
    assert max(g["bbox_max"][0] for g in gaps) > 5.0


# -- it refuses rather than reporting a weaker number ------------------------------------------


def test_a_mount_that_does_not_exist_is_refused(tmp_path):
    """A monitor watching nothing would report 0.000 for the whole run, which reads like a bad sweep
    rather than like a typo -- and the refusal names the nearest frame, from the frame resolver."""
    with pytest.raises(RuntimeError, match="'frame' 'rover_camm'.*Did you mean 'rover/rover_cam'"):
        _engine(_world(tmp_path, frame="rover_camm"))


def test_a_missing_frame_is_refused(tmp_path):
    with pytest.raises(PluginError, match="'frame' is required"):
        _engine(_world(tmp_path, frame=""))


@pytest.mark.parametrize("key", ["camera", "site", "body"])
def test_a_mount_named_by_mjcf_kind_is_an_unknown_key(tmp_path, key):
    """``frame`` is the one mount key; the strict schema refuses any other as it refuses a typo."""
    with pytest.raises(PluginError, match=f"'{key}' is not a setting"):
        _engine(_world(tmp_path, **{key: "rover_cam"}))


def test_an_absolute_path_inside_a_spawn_starts_at_the_top_of_the_world(tmp_path):
    engine = _engine(_world(tmp_path, frame="/rover/rover_cam"))
    try:
        plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
        assert plugin._mount_desc == "frame 'rover/rover_cam'"
    finally:
        engine.shutdown()


def test_a_missing_sensor_type_is_refused(tmp_path):
    """Which adapter builds the FoV decides the whole geometry; there is no default to fall back on."""
    with pytest.raises(PluginError, match="'type' is required"):
        _engine(_world(tmp_path, type=""))


def test_a_type_that_cannot_take_this_mount_is_refused(tmp_path):
    """A camera frame gives a camera its intrinsics; a lidar has none to take -- and the message
    says which frame refused it."""
    with pytest.raises(
        RuntimeError, match="cannot build a 'lidar' field of view on camera frame 'rover/rover_cam'"
    ):
        _engine(_world(tmp_path, type="lidar"))


def test_an_empty_sample_set_is_refused(tmp_path):
    """A fraction over zero points is the silent failure this plugin must not produce.

    Heights above the walls are outside the enclosure, so the sampler keeps nothing -- and a coverage
    number computed over that would be 0/0 dressed up as a measurement.
    """
    with pytest.raises(RuntimeError, match="sample set is empty"):
        _engine(_world(tmp_path, sample={**SAMPLE, "heights": [40.0]}))


def test_a_mistyped_sample_block_is_reported_not_raised(tmp_path):
    """`validate_config` collects every problem into one report; an AttributeError escapes it."""
    with pytest.raises(PluginError, match="'sample' must be dict"):
        _engine(_world(tmp_path, sample="resolution=0.25"))
    with pytest.raises(PluginError, match="'heights' must be a non-empty list"):
        _engine(_world(tmp_path, sample={**SAMPLE, "heights": []}))


def test_a_pose_is_stated_one_way(tmp_path):
    """The sensor's place on its frame is a ``pose:``; an ``offset``/``rpy`` pair is not read."""
    with pytest.raises(PluginError, match="'offset' is not a setting"):
        _engine(_world(tmp_path, frame="base", offset=[0.2, 0.0, 0.0]))
    with pytest.raises(PluginError, match="has no key"):
        _engine(_world(tmp_path, frame="base", pose={"pos": [0.2, 0.0, 0.0]}))


def test_a_bad_rate_is_refused(tmp_path):
    for key in ("compute_rate_hz", "rate_hz"):
        with pytest.raises(PluginError, match=f"'{key}' must be > 0"):
            _engine(_world(tmp_path, **{key: 0.0}))


def test_restricting_to_regions_that_hold_no_points_is_refused(tmp_path):
    """A restriction that silently kept nothing would report coverage of an empty room."""
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({"regions": [{"name": "elsewhere", "bbox": [50, 50, 60, 60]}]}))
    with pytest.raises(RuntimeError, match="left no sample points"):
        _engine(_world(tmp_path, regions=str(regions), restrict=True))


def test_a_region_restriction_shrinks_the_sample_set(tmp_path):
    """The same regions, used as intended: the union is then accumulated over that area only."""
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({"regions": [{"name": "west", "bbox": [0.5, -1.5, 4, 1.5]}]}))
    full = _engine(_world(tmp_path))
    try:
        unrestricted = _sweep(full, np.linspace(0.0, 8.0, 9))
        n_full, full_fraction = unrestricted.n_points, unrestricted.fraction
    finally:
        full.shutdown()
    limited = _engine(_world(tmp_path, regions=str(regions), restrict=True))
    try:
        report = _sweep(limited, np.linspace(0.0, 8.0, 9))
        assert 0 < report.n_points < n_full
        points = _reader(limited).points()
        assert points[:, 0].max() <= 4.0
        # Restricting to an area the sweep actually crossed must RAISE the fraction: the far half
        # of the corridor, which the camera never reached, is no longer in the denominator. A
        # restriction that had quietly kept the whole world would leave it unchanged.
        assert report.fraction > full_fraction > 0.0
        assert report.fraction > 0.75
    finally:
        limited.shutdown()


# -- the adapter seam -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mount", "sensor_type", "config"),
    [
        ({"frame": "rover_cam"}, "camera", SENSOR),
        ({"frame": "rover_scan"}, "lidar", {"v_fov": [-0.3, 0.3], "range_max": 4.0}),
    ],
)
def test_reposing_agrees_with_rebuilding_the_fov_from_scratch(tmp_path, mount, sensor_type, config):
    """The pose this plugin composes must equal the one the adapter would build at that instant.

    Asserted against the adapter's in-world forms: a camera frame goes through one, and a site
    frame is placed hypothetically and moved by the frame pose, which must land on the same field of
    view the site form builds. A ``cam_id``/``site_id`` placement re-reads the mount pose out of ``model``/``data`` on every
    ``build_fov`` call, so rebuilding per tick would work -- it is just wasteful (a lidar adapter
    re-instantiates the plugin whose defaults it borrows, a camera adapter re-parses the MJCF
    intrinsics, and neither changes while a sensor moves). This plugin therefore builds once and
    moves the pose, and that shortcut is only safe while the two agree exactly. Asserted at a pose
    the carrier was driven to, not at the one it was built at, because agreeing at ``configure`` is
    what a broken re-pose would also do.
    """
    from roqsim_sensors.coverage.adapters import PlacedSensor, build_fov

    # 200 Hz against a 10 ms step, so the LAST evaluation used the pose `data` now holds. At the
    # default rate the gate deliberately lags by up to a step, and comparing across that lag would
    # measure the rate gate rather than the re-pose.
    engine = _engine(
        _world(tmp_path, type=sensor_type, config=config, compute_rate_hz=200.0, **mount)
    )
    try:
        plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
        _drive_to(engine, 4.0)

        kind = "camera" if sensor_type == "camera" else "site"
        name = mount["frame"]
        obj = mujoco.mjtObj.mjOBJ_CAMERA if kind == "camera" else mujoco.mjtObj.mjOBJ_SITE
        mount_id = mujoco.mj_name2id(engine.ctx.model, obj, name)
        placed = PlacedSensor(
            sensor_type,
            config=config,
            **({"cam_id": mount_id} if kind == "camera" else {"site_id": mount_id}),
        )
        fresh = build_fov(engine.ctx.model, engine.ctx.data, placed)

        assert plugin._fov.origin == pytest.approx(fresh.origin, abs=1e-12)
        assert plugin._fov.rot == pytest.approx(fresh.rot, abs=1e-12)
        assert plugin._fov.body_exclude == fresh.body_exclude
        assert (plugin._fov.range_min, plugin._fov.range_max) == (
            fresh.range_min,
            fresh.range_max,
        )
    finally:
        engine.shutdown()


# -- other frames ------------------------------------------------------------------------------


def test_a_lidar_on_a_site_sweeps_behind_itself_too(tmp_path):
    """A site frame, and a genuinely different FoV: a 360-degree scanner is not a camera.

    Driving east and back, a forward camera never covers the cells west of its start while a full
    dome does -- so this asserts the shape of the field of view, not merely that something happened.
    ``v_fov`` widens a 2D scanner's zero-thickness plane, which the adapter documents as an analysis
    assumption and this is the mount that needs it.
    """
    engine = _engine(
        _world(
            tmp_path,
            type="lidar",
            frame="rover_scan",
            config={"v_fov": [-0.3, 0.3], "range_max": 4.0},
        )
    )
    try:
        _sweep(engine, [3.0, 5.0, 3.0])
        points, visits = _reader(engine).points(), _reader(engine).visits()
        behind = points[:, 0] < 1.0  # west of every pose the scanner held
        assert behind.any()
        assert int((visits[behind] > 0).sum()) > 0, "a 360-degree scanner sees behind itself"
    finally:
        engine.shutdown()


def test_a_root_mount_carries_the_pose_through_the_carrier_rotation(tmp_path):
    """The one piece of geometry this plugin composes itself: frame pose x sensor-in-frame pose.

    ``pose`` is read in the frame's coordinates -- here the spawn's root, ``.``, so a carrier yawed 90 degrees must put a +x offset on the
    world +y axis. Getting that composition wrong (applying the offset in world, or transposing the
    rotation) still produces a moving sensor and a plausible coverage number, so the pose itself is
    asserted rather than only its effect.
    """
    engine = _engine(_world(tmp_path, frame=".", pose={"position": {"x": 0.2}}))
    try:
        plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
        # Yaw the carrier a quarter turn about +z (MuJoCo quaternions are w, x, y, z).
        root = np.sqrt(0.5)
        engine.ctx.data.qpos[0:3] = [3.0, 0.0, 0.5]
        engine.ctx.data.qpos[3:7] = [root, 0.0, 0.0, root]
        engine.ctx.data.qvel[:] = 0.0
        mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
        engine.step()

        assert plugin._fov.origin == pytest.approx([3.0, 0.2, 0.5], abs=1e-6)
        # No orientation points the adapter's camera along the mount's +x, which is now world +y.
        assert plugin._fov.rot @ np.array([0.0, 0.0, -1.0]) == pytest.approx(
            [0.0, 1.0, 0.0], abs=1e-6
        )
        assert _reader(engine).read().n_covered > 0
    finally:
        engine.shutdown()


def test_a_body_mount_turns_the_sensor_by_the_pose_orientation(tmp_path):
    """The pose's orientation is applied on top of the adapter's base rotation, in the frame:
    a quarter turn of yaw on an unrotated carrier points the camera along world +y."""
    engine = _engine(
        _world(
            tmp_path,
            frame="base",
            pose={"position": {"x": 0.2}, "orientation": {"yaw": float(np.pi / 2)}},
        )
    )
    try:
        plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
        engine.step()
        pos, mat = plugin._mount_pose()
        assert plugin._fov.origin == pytest.approx(pos + mat @ [0.2, 0.0, 0.0], abs=1e-6)
        # The carrier's own +y, which is world +y while it has not turned.
        assert plugin._fov.rot @ np.array([0.0, 0.0, -1.0]) == pytest.approx(
            mat @ [0.0, 1.0, 0.0], abs=1e-6
        )
        assert mat @ [0.0, 1.0, 0.0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-3)
    finally:
        engine.shutdown()


def test_a_camera_frame_brings_its_intrinsics_and_its_own_axes(tmp_path):
    """On a camera frame the field of view is that camera's: its ``fovy`` and resolution come from
    the model with nothing restated, and ``pose`` is read in MuJoCo's camera frame -- -z is the view
    direction, so ``z: -0.3`` moves the sensor 0.3 m along the optical axis, world +x here."""
    from roqsim_sensors.plugins.camera_common import intrinsics_from_model

    engine = _engine(
        _world(tmp_path, config={"far": 4.0}, pose={"position": {"z": -0.3}}, compute_rate_hz=200.0)
    )
    try:
        plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
        _drive_to(engine, 2.0)
        model, data = engine.ctx.model, engine.ctx.data
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rover_cam")
        assert plugin._fov.intrinsics == intrinsics_from_model(model, cam)
        assert float(model.cam_fovy[cam]) == pytest.approx(70.0)
        assert plugin._fov.rot == pytest.approx(data.cam_xmat[cam].reshape(3, 3), abs=1e-9)
        assert plugin._fov.origin == pytest.approx(data.cam_xpos[cam] + [0.3, 0.0, 0.0], abs=1e-6)
        assert plugin._fov.body_exclude == int(model.cam_bodyid[cam])
    finally:
        engine.shutdown()


# A gantry in the world's own MJCF: a carriage on a slide joint along the corridor, owned by no
# entity, carrying a forward-looking camera. Its frames are named by their MuJoCo names alone.
_GANTRY = _CORRIDOR.replace(
    "  </worldbody>",
    """    <body name="gantry" pos="0 0 .5">
      <joint name="gantry_x" type="slide" axis="1 0 0"/>
      <geom name="carriage" type="box" size=".15 .15 .15" mass="1"/>
      <camera name="gantry_cam" pos="0 0 0" xyaxes="0 -1 0  0 0 1" fovy="70"/>
    </body>
  </worldbody>""",
)


def _gantry_engine(tmp_path, **monitor):
    (tmp_path / "gantry.xml").write_text(_GANTRY)
    config = {"type": "camera", "config": {"far": 4.0}, "sample": SAMPLE, "compute_rate_hz": 50.0}
    world = {
        "sim": {"world": str(tmp_path / "gantry.xml")},
        "components": [{"swept_coverage_monitor": {**config, **monitor}, "name": "sweep"}],
    }
    return _engine(world)


def _move_gantry(engine, xs):
    plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
    for x in xs:
        engine.ctx.data.qpos[0] = x
        engine.ctx.data.qvel[:] = 0.0
        mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
        for _ in range(4):
            engine.step()
    return plugin, plugin.read()


@pytest.mark.parametrize(
    "mount",
    [
        {"frame": "gantry_cam"},
        # The carriage body itself, with the camera adapter's hypothetical form looking along +x.
        {"frame": "gantry", "config": SENSOR},
    ],
)
def test_a_sensor_on_a_world_gantry_sweeps_with_its_carriage(tmp_path, mount):
    """A carrier no entity owns -- a slide-jointed gantry in the world MJCF -- is mounted by its
    MuJoCo name, and the union follows the carriage down the corridor."""
    still = _gantry_engine(tmp_path, **mount)
    try:
        plugin, held = _move_gantry(still, [0.0] * 9)
        assert plugin._mount_desc == f"frame {mount['frame']!r}"
        body = mujoco.mj_name2id(still.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "gantry")
        assert plugin._fov.body_exclude == body
    finally:
        still.shutdown()
    moving = _gantry_engine(tmp_path, **mount)
    try:
        _, swept = _move_gantry(moving, np.linspace(0.0, 8.0, 9))
    finally:
        moving.shutdown()
    assert held.n_evaluations == swept.n_evaluations
    assert held.n_covered > 0
    assert swept.n_covered >= 2 * held.n_covered


def test_a_camera_on_the_world_body_is_still_occluded_by_the_walls(tmp_path):
    """A frame on the world body excludes nothing from its raycasts: the world body's geometry is
    the walls, and excluding it would let a fixed camera see through the divider."""
    scene = _DIVIDED.replace(
        "  </worldbody>",
        '    <camera name="fixed_cam" pos="2 0 .5" xyaxes="0 -1 0  0 0 1" fovy="70"/>\n'
        "  </worldbody>",
    )
    (tmp_path / "fixed.xml").write_text(scene)
    world = {
        "sim": {"world": str(tmp_path / "fixed.xml")},
        "components": [
            {
                "swept_coverage_monitor": {
                    "type": "camera",
                    "frame": "fixed_cam",
                    "config": {"far": 8.0},
                    "sample": SAMPLE,
                    "compute_rate_hz": 50.0,
                },
                "name": "sweep",
            }
        ],
    }
    engine = _engine(world)
    try:
        for _ in range(4):
            engine.step()
        plugin = next(p for p in engine.plugins if isinstance(p, SweptCoverageMonitorPlugin))
        assert plugin._fov.body_exclude == -1
        points, visits = plugin._points, plugin._visits
        assert int((visits[points[:, 0] < 5.0] > 0).sum()) > 0, "it sees its own side"
        assert int((visits[points[:, 0] > 5.2] > 0).sum()) == 0, "and nothing past the divider"
    finally:
        engine.shutdown()


# -- registration ------------------------------------------------------------------------------


def test_the_plugin_is_registered_under_its_entry_point_name():
    """A world YAML names it by entry point; an unregistered plugin fails only at run time."""
    assert resolve_plugin("swept_coverage_monitor") is SweptCoverageMonitorPlugin
