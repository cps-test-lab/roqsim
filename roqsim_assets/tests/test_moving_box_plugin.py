"""`moving_box`: the kinematically driven rectangular obstacle.

What is worth pinning: it is a MOCAP body (physics may not move it), it travels at exactly the
configured speed, a seeded random walk is reproducible AND stays out of walls, and `on_reset` restores
both the pose and the RNG — the last one is what makes repetition N of a campaign cell independent of
whatever ran before it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.context import SimContext
from roqsim_assets.plugins.moving_box import MovingBoxPlugin


def _build(walls=(), *, name=None, **cfg):
    """Compile a world with a floor, optional wall boxes, and the mover.

    `walls` are (x, y, sx, sy) axis-aligned wall boxes, 1 m tall — enough to make the look-ahead ray
    hit something real rather than a mocked-up geometry stub.
    """
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20, 20, 0.05]
    for i, (x, y, sx, sy) in enumerate(walls):
        g = spec.worldbody.add_geom()
        g.name = f"wall_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [sx / 2, sy / 2, 0.5]
        g.pos = [x, y, 0.5]
    ctx = SimContext(config={})
    plugin = MovingBoxPlugin(cfg, label=name)
    plugin.build(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx.model, ctx.data = model, data
    plugin.configure(ctx)
    return model, data, plugin, ctx


def _run(plugin, ctx, seconds):
    """Drive the mover and return its centre track (x, y) per step."""
    model, data = ctx.model, ctx.data
    track = []
    for _ in range(int(seconds / model.opt.timestep)):
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        track.append(tuple(float(v) for v in data.mocap_pos[0][:2]))
    return np.array(track)


def _cfg(**over):
    cfg = dict(pos=[0.0, 0.0], size=[0.3, 0.3, 0.3], speed=0.5)
    cfg.update(over)
    return cfg


# --------------------------------------------------------------------------- structure


def test_it_is_a_mocap_body_not_a_free_joint():
    """Physics must not be able to move it: a free-jointed obstacle gets shoved aside by the robot
    it exists to obstruct, and its commanded motion fights the solver."""
    model, _, _, _ = _build(**_cfg(waypoints=[[1.0, 0.0]]))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_box")
    assert bid >= 0
    assert int(model.body_mocapid[bid]) >= 0, "not a mocap body"
    assert model.nq == 0 and model.njnt == 0, "the mover must have no joints"


def test_size_is_full_extents_and_pos_sits_it_on_the_floor():
    """Same convention as `box` — the silent factor-of-two trap is worth pinning in both."""
    model, data, _, _ = _build(**_cfg(size=[0.4, 0.6, 0.8], waypoints=[[1.0, 0.0]]))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_box")
    assert list(model.geom_size[gid][:3]) == pytest.approx([0.2, 0.3, 0.4])
    assert float(data.geom_xpos[gid][2]) == pytest.approx(0.4)


def test_entity_is_registered_with_its_motion_mode():
    _, _, _, ctx = _build(**_cfg(name="cube_1", waypoints=[[1.0, 0.0]]))
    ent = ctx.entities.get("cube_1")
    assert ent is not None and ent.body == "moving_box"
    assert ent.meta["motion"] == "waypoints"
    assert ent.meta["speed"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- waypoint motion


def test_speed_is_exactly_the_configured_speed():
    _, _, plugin, ctx = _build(**_cfg(speed=0.25, waypoints=[[10.0, 0.0]], loop=False))
    track = _run(plugin, ctx, 4.0)
    travelled = float(np.linalg.norm(track[-1] - np.array([0.0, 0.0])))
    assert travelled == pytest.approx(0.25 * 4.0, rel=1e-3)


def test_it_turns_corners_without_overshooting():
    """One step may span a waypoint; the remainder must continue along the NEXT segment, not carry
    straight on. A 0.5 m/s mover at dt=2 ms steps 1 mm, so an overshoot shows up as a corner cut."""
    _, _, plugin, ctx = _build(**_cfg(speed=0.5, waypoints=[[1.0, 0.0], [1.0, 1.0]], loop=False))
    track = _run(plugin, ctx, 4.0)
    # After 4 s at 0.5 m/s the mover has covered 2 m: 1 m east, then 1 m north — the corner exactly.
    assert track[-1] == pytest.approx([1.0, 1.0], abs=2e-3)
    assert track[:, 0].max() == pytest.approx(1.0, abs=2e-3), "cut or overshot the corner"


def test_loop_cycles_and_no_loop_parks():
    """A route with `loop: false` stops on its last waypoint; with `loop: true` it starts over."""
    _, _, parked, ctx1 = _build(**_cfg(speed=1.0, waypoints=[[1.0, 0.0]], loop=False))
    t1 = _run(parked, ctx1, 3.0)
    assert t1[-1] == pytest.approx([1.0, 0.0], abs=1e-6)

    _, _, looper, ctx2 = _build(**_cfg(speed=1.0, waypoints=[[1.0, 0.0], [0.0, 0.0]], loop=True))
    t2 = _run(looper, ctx2, 4.0)
    assert t2[:, 0].max() == pytest.approx(1.0, abs=2e-3)
    assert t2[-1][0] < 0.99, "a looping mover should not be parked at the far waypoint"


def test_ping_pong_reverses_instead_of_jumping_home():
    _, _, plugin, ctx = _build(**_cfg(speed=1.0, waypoints=[[1.0, 0.0]], ping_pong=True))
    track = _run(plugin, ctx, 3.0)
    assert track[:, 0].max() == pytest.approx(1.0, abs=2e-3)
    assert track[:, 0].min() < 0.05, "never came back"
    # A jump home would show as a discontinuity; a reversal keeps every step ~1 mm.
    assert np.abs(np.diff(track[:, 0])).max() < 0.005


# --------------------------------------------------------------------------- random walk


def test_random_walk_is_reproducible_per_seed():
    """Same seed -> identical track; different seed -> a different one. This is what lets a campaign
    vary obstacle motion by varying the seed and still replay any trial exactly."""
    a = _build(**_cfg(random_walk={"seed": 7}))
    b = _build(**_cfg(random_walk={"seed": 7}))
    c = _build(**_cfg(random_walk={"seed": 8}))
    ta, tb, tc = (_run(x[2], x[3], 6.0) for x in (a, b, c))
    assert np.allclose(ta, tb), "same seed produced different motion"
    assert not np.allclose(ta, tc), "different seeds produced identical motion"


def test_random_walk_stays_inside_a_room():
    """The look-ahead ray must keep the mover out of walls, with no map file and no wall list.

    Corridor-scale room (4 x 4 m) with 1 m-tall walls; the mover is 0.3 m and drives for 60 s.
    """
    walls = ((0, 2.15, 4.6, 0.3), (0, -2.15, 4.6, 0.3), (2.15, 0, 0.3, 4.6), (-2.15, 0, 0.3, 4.6))
    _, _, plugin, ctx = _build(walls, **_cfg(speed=0.4, random_walk={"seed": 3, "clearance": 0.3}))
    track = _run(plugin, ctx, 60.0)
    assert np.abs(track).max() < 2.0, f"escaped the room: max |xy| = {np.abs(track).max():.3f}"
    # And it actually moved around rather than idling in a corner.
    assert np.ptp(track[:, 0]) > 0.5 and np.ptp(track[:, 1]) > 0.5


def test_random_walk_respects_explicit_bounds():
    _, _, plugin, ctx = _build(
        **_cfg(speed=0.5, random_walk={"seed": 5, "bounds": [-1.0, -1.0, 1.0, 1.0]})
    )
    track = _run(plugin, ctx, 30.0)
    assert track[:, 0].min() >= -1.001 and track[:, 0].max() <= 1.001
    assert track[:, 1].min() >= -1.001 and track[:, 1].max() <= 1.001


def test_random_walk_moves_at_the_configured_speed():
    """Per-step displacement is speed*dt except on the steps where a new heading is sampled."""
    model, _, plugin, ctx = _build(**_cfg(speed=0.3, random_walk={"seed": 11}))
    track = _run(plugin, ctx, 10.0)
    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    expected = 0.3 * model.opt.timestep
    assert np.median(steps) == pytest.approx(expected, rel=1e-6)
    assert steps.max() <= expected * 1.001


# --------------------------------------------------------------------------- reset semantics


def test_reset_restores_pose_and_reseeds():
    """Repetition N must not inherit repetition N-1's obstacle position or RNG state."""
    _, data, plugin, ctx = _build(**_cfg(pos=[1.0, 2.0], speed=0.5, random_walk={"seed": 4}))
    first = _run(plugin, ctx, 5.0)
    plugin.on_reset(ctx)
    assert data.mocap_pos[0][:2] == pytest.approx([1.0, 2.0])
    second = _run(plugin, ctx, 5.0)
    assert np.allclose(first, second), "re-seeded run diverged from the first"


def test_reset_restores_waypoint_progress():
    _, data, plugin, ctx = _build(**_cfg(speed=1.0, waypoints=[[3.0, 0.0]], loop=False))
    _run(plugin, ctx, 2.0)
    plugin.on_reset(ctx)
    assert data.mocap_pos[0][:2] == pytest.approx([0.0, 0.0])
    track = _run(plugin, ctx, 1.0)
    assert track[-1][0] == pytest.approx(1.0, rel=1e-3), "route index was not rewound"


# --------------------------------------------------------------------------- config validation


@pytest.mark.parametrize(
    "cfg, needle",
    [
        (dict(pos=[0, 0], size=[0.3, 0.3, 0.3]), "'speed' is required"),
        (dict(pos=[0, 0], size=[0.3, 0.3, 0.3], speed=0.5), "needs a motion"),
        (
            dict(
                pos=[0, 0],
                size=[0.3, 0.3, 0.3],
                speed=0.5,
                waypoints=[[1, 0]],
                random_walk={"seed": 1},
            ),
            "not both",
        ),
        (dict(pos=[0, 0], size=[0.3, 0.3, 0.3], speed=0.5, random_walk={}), "requires a 'seed'"),
        (
            dict(pos=[0, 0], size=[0.3, 0.3, 0.3], speed=-1, waypoints=[[1, 0]]),
            "positive number of m/s",
        ),
        (
            dict(pos=[0, 0], size=[0.3, 0, 0.3], speed=0.5, waypoints=[[1, 0]]),
            "three positive numbers",
        ),
        (
            dict(
                pos=[0, 0],
                size=[0.3, 0.3, 0.3],
                speed=0.5,
                random_walk={"seed": 1, "bounds": [1, 1, 0, 0]},
            ),
            "bounds",
        ),
    ],
)
def test_config_errors_are_reported_not_guessed(cfg, needle):
    """An under-specified mover is an error, never a silently chosen trajectory."""
    errors = MovingBoxPlugin(cfg).validate_config(cfg)
    assert any(needle in e for e in errors), f"expected {needle!r} in {errors}"


def test_a_valid_config_has_no_errors():
    for cfg in (
        _cfg(waypoints=[[1.0, 0.0], [1.0, 1.0]]),
        _cfg(random_walk={"seed": 2, "clearance": 0.4, "bounds": [-2, -2, 2, 2]}),
    ):
        assert MovingBoxPlugin(cfg).validate_config(cfg) == []
