# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""How close did it come? -- the gradient beside ``contact_monitor``'s verdict.

``contact_monitor`` answers "did it touch", which is the right failure criterion and a
terrible thing to optimise: it is a bit, so every configuration that does not touch scores
identically and a search has nothing to climb. Distance is the same question asked
continuously, and the two together give a verdict and a gradient over one geometry.

Measured in the simulator rather than derived afterwards from recorded poses, for reasons
this file pins down: the closest approach happens between pose samples, real geometry is
not a circle, and an articulated obstacle's nearest part is a limb rather than its origin.
"""

import mujoco
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.engine import Engine

# A world with something to avoid, and a robot spawned as its own entity -- the monitor
# watches an entity's subtree, so it is nested under the entry that provides one.
_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="post" pos="2 0 .5">
      <geom name="post_geom" type="box" size=".1 .5 .5"/>
    </body>
    <!-- Decoration: nearer than the post, and passable. contype=0 conaffinity=0 is how a
         render-only geom is spelled, and a walker carries six of them. -->
    <body name="decor" pos="1 0 .5">
      <geom name="decor_geom" type="box" size=".1 .5 .5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

_ROBOT = """
<mujoco model="rover">
  <worldbody>
    <body name="base" pos="0 0 .2">
      <geom name="base_geom" type="cylinder" size=".2 .2"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def world(tmp_path):
    (tmp_path / "s.xml").write_text(_SCENE)
    (tmp_path / "rover.xml").write_text(_ROBOT)
    return {
        "sim": {"world": str(tmp_path / "s.xml")},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "rover.xml"),
                    "pose": {"position": {"x": 0.0, "y": 0.0}},
                    "free": True,
                },
                "name": "robot",
                "components": [{"clearance_monitor": {"ignore": ["floor"]}}],
            },
        ],
    }


def _engine(world):
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _report(engine):
    return engine.ctx.blackboard.get("clearance:robot.clearance_monitor")()


def _drive_to(engine, x, steps=5):
    """Move the spawned entity to ``x`` and hold it there for ``steps`` physics steps.

    Held rather than teleported one step at a time, because the monitor measures on its own
    rate (default 200 Hz against a 2 ms step, so every fifth step). A robot that actually
    moves dwells at every point on its path; a test that skips between positions faster
    than the sensor samples is testing the test.
    """
    engine.ctx.data.qpos[0] = x
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    for _ in range(steps):
        engine.step()


# -- the measurement --------------------------------------------------------


def test_it_reports_the_distance_to_the_nearest_geom(world):
    """Cylinder r=0.2 at x=0 against a box half-x 0.1 at x=2: surfaces are 1.7 apart."""
    engine = _engine(world)
    try:
        _drive_to(engine, 0.0)
        assert _report(engine).current == pytest.approx(1.7, abs=0.02)
    finally:
        engine.shutdown()


def test_the_minimum_is_kept_over_the_whole_run(world):
    """The answer is the closest it EVER came, not where it happens to be now -- a robot
    that squeezed past and then drove away was still that close."""
    engine = _engine(world)
    try:
        for x in (0.0, 1.0, 1.6, 1.0, 0.0):
            _drive_to(engine, x)
        report = _report(engine)
        assert report.minimum == pytest.approx(0.1, abs=0.03)  # at x=1.6
        assert report.current == pytest.approx(1.7, abs=0.02)  # back at the start
    finally:
        engine.shutdown()


def test_it_measures_real_geometry_not_a_bounding_radius(world):
    """The reason this belongs in the simulator: no footprint constant sits between the
    geometry and the number. A box is a box, not the circle that encloses it."""
    engine = _engine(world)
    try:
        _drive_to(engine, 1.0)
        # box face at x=1.9, cylinder surface at 1.0+0.2 -> 0.7 apart.
        assert _report(engine).current == pytest.approx(0.7, abs=0.02)
    finally:
        engine.shutdown()


def test_an_ignored_geom_never_counts(world):
    """The floor is touched continuously by design; counting it would report 0 forever."""
    engine = _engine(world)
    try:
        _drive_to(engine, 0.0)
        assert _report(engine).current > 1.0
    finally:
        engine.shutdown()


def test_beyond_distmax_it_reports_the_cutoff_not_a_wrong_number(world):
    """Past the cutoff the true distance is unknown, and the honest answer is "at least
    this far" rather than a number that looks measured."""
    world["components"][0]["components"][0]["clearance_monitor"]["distmax"] = 0.5
    engine = _engine(world)
    try:
        _drive_to(engine, 0.0)
        report = _report(engine)
        assert report.current == pytest.approx(0.5)
        assert report.saturated is True
    finally:
        engine.shutdown()


def test_touching_reports_zero_or_less(world):
    engine = _engine(world)
    try:
        _drive_to(engine, 1.75)
        assert _report(engine).current <= 0.05
    finally:
        engine.shutdown()


# -- lifecycle --------------------------------------------------------------


def test_a_reset_starts_a_new_measurement(world):
    """A trial's clearance is that trial's. Carrying a minimum across a reset would report
    the previous run's near-miss as this one's."""
    engine = _engine(world)
    try:
        _drive_to(engine, 1.6)
        assert _report(engine).minimum < 0.3
        engine.reset()
        _drive_to(engine, 0.0)
        assert _report(engine).minimum > 1.0
    finally:
        engine.shutdown()


def test_it_publishes_an_endpoint_a_bridge_can_serve(world):
    engine = _engine(world)
    try:
        out = [
            e for e in engine.ctx.interface.all() if e.direction == "out" and "clearance" in e.name
        ]
        assert out, "clearance must be observable outside the process"
    finally:
        engine.shutdown()


# -- refusals ---------------------------------------------------------------


def test_a_body_that_does_not_exist_is_refused(world):
    """A monitor watching nothing would report "never close to anything" forever, which is
    indistinguishable from a safe run and would pass every trial."""
    world["components"][0]["components"][0]["clearance_monitor"]["body"] = "nosuch"
    with pytest.raises(RuntimeError, match="nosuch"):
        _engine(world)


def test_a_negative_distmax_is_refused(world):
    world["components"][0]["components"][0]["clearance_monitor"]["distmax"] = -1.0
    with pytest.raises(PluginError, match="distmax"):
        _engine(world)


def test_a_world_where_everything_is_ignored_is_refused(world):
    """Nothing left to measure against is a configuration mistake, not a clean run."""
    world["components"][0]["components"][0]["clearance_monitor"]["ignore"] = ["floor", "post_geom"]
    with pytest.raises(RuntimeError, match="nothing"):
        _engine(world)


# -- collidability ----------------------------------------------------------


def test_a_visual_only_geom_is_not_measured(world):
    """`mj_geomDistance` is pure geometry and ignores collision masks, so a render-only
    geom would otherwise be reported as clearance to something the robot passes straight
    through. The scene's `decor_geom` sits nearer than the post and must not count."""
    engine = _engine(world)
    try:
        _drive_to(engine, 0.0)
        report = _report(engine)
        assert report.geom == "post_geom"
        assert report.current == pytest.approx(1.7, abs=0.02)
    finally:
        engine.shutdown()


def test_a_visual_only_geom_on_the_robot_is_not_the_measuring_point(world):
    """The same rule on the watched side: a decorative geom sticking out of the robot is
    not what would touch anything."""
    robot = (
        '<mujoco model="rover">\n'
        "  <worldbody>\n"
        '    <body name="base" pos="0 0 .2">\n'
        '      <geom name="base_geom" type="cylinder" size=".2 .2"/>\n'
        '      <geom name="antenna" type="box" pos=".8 0 0" size=".1 .1 .1"'
        ' contype="0" conaffinity="0"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    import pathlib

    path = pathlib.Path(world["components"][0]["spawn_model"]["model"])
    path.write_text(robot)
    engine = _engine(world)
    try:
        _drive_to(engine, 0.0)
        # The antenna reaches to x=0.9, but it cannot touch anything: the measurement is
        # still from the cylinder's surface at 0.2 to the post face at 1.9.
        assert _report(engine).current == pytest.approx(1.7, abs=0.02)
    finally:
        engine.shutdown()


# -- compute rate -----------------------------------------------------------


def test_measurement_is_gated_on_its_own_rate_not_every_physics_step(world):
    """Cost, and the reason it is a separate knob from the publish rate.

    A distance query per (watched, candidate) pair every physics step is real work: on the
    nav world it measured 1.24x the step cost, against a realtime budget the sim was
    already over. Poses reach a consumer at ~30 Hz, so measuring at 200 Hz still resolves
    millimetres at walking speed -- far finer than anything downstream can use -- while
    costing a fraction. The gate is what makes that a choice rather than an accident.
    """
    world["components"][0]["components"][0]["clearance_monitor"]["compute_rate_hz"] = 50.0
    engine = _engine(world)
    try:
        seen = []
        for i in range(40):  # 40 steps x 2 ms = 80 ms
            _drive_to(engine, i * 0.01, steps=1)  # moves every step
            seen.append(_report(engine).current)
        # 80 ms at 50 Hz is ~4 measurements, not 40. If the gate were missing, every step
        # would move the robot and change the distance.
        assert 2 <= len(set(seen)) <= 8, f"{len(set(seen))} distinct values over 40 steps"
    finally:
        engine.shutdown()


def test_a_high_compute_rate_still_catches_a_fast_approach(world):
    """The gate must not reintroduce the sampling problem it exists beside. At the default
    rate a pass that comes close is still recorded as having come close."""
    engine = _engine(world)
    try:
        for x in (0.0, 0.8, 1.6, 0.8, 0.0):
            _drive_to(engine, x)
        assert _report(engine).minimum == pytest.approx(0.1, abs=0.05)
    finally:
        engine.shutdown()


def test_an_invalid_compute_rate_is_refused(world):
    world["components"][0]["components"][0]["clearance_monitor"]["compute_rate_hz"] = 0
    with pytest.raises(PluginError, match="compute_rate_hz"):
        _engine(world)
