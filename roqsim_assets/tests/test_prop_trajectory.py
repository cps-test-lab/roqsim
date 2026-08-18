"""`prop_trajectory` — carry a prop along a prescribed 2-D path.

The load-bearing test is `test_stage_carries_a_resting_object`: the whole point of this plugin over the
existing `moving_box` is that it can DRAG, and that distinction is not obvious from either plugin's
description. `test_mocap_cannot_carry_documents_why_this_plugin_exists` pins the reason as an executable
fact, so nobody re-litigates it (or "simplifies" this plugin into a mocap one).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.context import SimContext
from roqsim_assets.plugins.prop_trajectory import PropTrajectoryPlugin

CUBE = """
<mujoco><option timestep="0.0005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body name="cube" pos="0 0 0.622"><freejoint/>
    <geom name="cube_g" type="box" size="0.015 0.015 0.015" mass="0.05"
          friction="1.2 0.005 0.0001" condim="4"/>
  </body>
</worldbody></mujoco>
"""


@pytest.fixture
def square_csv(tmp_path):
    p = tmp_path / "square.csv"
    p.write_text("0,0\n50,0\n50,50\n0,50\n0,0\n")  # a 50 mm square: 200 mm of path
    return p


def _rig(csv_path, **cfg):
    spec = mujoco.MjSpec.from_string(CUBE)
    ctx = SimContext(config={})
    plugin = PropTrajectoryPlugin(
        {
            "path": str(csv_path),
            "units": "mm",
            "speed": 0.03,
            "origin": [0.0, 0.0, 0.606],
            "plate": [0.07, 0.07, 0.006],
            **cfg,
        }
    )
    plugin.build(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    ctx.model, ctx.data = model, data
    plugin.configure(ctx)
    return model, data, ctx, plugin


def _settle(model, data, ctx, plugin, seconds=1.0):
    for _ in range(int(seconds / model.opt.timestep)):
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)


def _stage_xy(data, plugin):
    return np.array([data.qpos[plugin._qadr[0]], data.qpos[plugin._qadr[1]]])


def test_path_length_and_units(square_csv):
    """A 50 mm square in a mm CSV is 0.2 m of path — unit handling is not negotiable."""
    _, _, _, plugin = _rig(square_csv)
    assert plugin._cum[-1] == pytest.approx(0.200, abs=1e-6)


def test_metre_units_are_honoured(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("0,0\n0.05,0\n")
    _, _, _, plugin = _rig(p, units="m")
    assert plugin._cum[-1] == pytest.approx(0.05, abs=1e-9)


def test_missing_path_fails_loudly(tmp_path):
    """A silently missing trajectory would make the dynamic condition identical to the static one."""
    spec = mujoco.MjSpec.from_string(CUBE)
    plugin = PropTrajectoryPlugin({"path": str(tmp_path / "nope.csv")})
    with pytest.raises(RuntimeError, match="not found"):
        plugin.build(spec, SimContext(config={}))


def test_stage_tracks_the_commanded_speed(square_csv):
    """Arc length advances at exactly `speed` — the paper's object moves at 30 mm/s, not 'about'."""
    model, data, ctx, plugin = _rig(square_csv)
    _settle(model, data, ctx, plugin, seconds=2.0)
    s, total, done = plugin.read_progress()
    assert s == pytest.approx(0.03 * 2.0, rel=1e-6)
    assert total == pytest.approx(0.2, abs=1e-6)
    assert not done


def test_stage_carries_a_resting_object(square_csv):
    """THE capability: an object resting on the stage is carried by friction, never teleported.

    Slip is the metric that matters — if the plate slid out from under the cube this would fail even
    though every pose in the log looked plausible.
    """
    model, data, ctx, plugin = _rig(square_csv)
    _settle(model, data, ctx, plugin, seconds=1.0)
    c0 = data.qpos[0:2].copy()
    s0 = _stage_xy(data, plugin)
    for _ in range(int(4.0 / model.opt.timestep)):
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
    stage = _stage_xy(data, plugin) - s0
    cube = data.qpos[0:2] - c0
    assert np.linalg.norm(stage) > 0.05, "the stage did not move"
    slip = float(np.linalg.norm(stage - cube))
    assert slip < 0.003, f"cube slipped {slip * 1000:.2f} mm relative to the stage"
    assert data.qpos[2] == pytest.approx(0.6209, abs=2e-3), "cube left the plate"


def test_mocap_cannot_carry_documents_why_this_plugin_exists():
    """A mocap body BLOCKS but cannot DRAG — the reason this is not a `moving_box` option.

    MuJoCo integrates a mocap body's pose kinematically and gives it no velocity, so a friction contact
    against it sees zero relative slip and transfers no tangential force: the solver holds the cube at
    the plate's *velocity*, which is zero, while the plate's *pose* translates underneath it. A
    force-driven joint (this plugin, and `conveyor`) has real velocity and therefore real friction.

    Measured over a window where the plate is still under the cube (45 mm of travel against a 70 mm
    half-width). Driving further is not a stronger test but a confusing one -- the plate eventually slides
    out from under the cube entirely and it topples off the trailing edge, which shows up as a large
    spurious displacement that has nothing to do with being carried.
    """
    xml = CUBE.replace(
        '<body name="cube"',
        '<body name="plate" mocap="true" pos="0 0 0.6">'
        '<geom name="plate_g" type="box" size="0.07 0.07 0.006"'
        ' friction="1.0 0.005 0.0001" condim="4"/></body>'
        '<body name="cube"',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    x0 = float(data.qpos[0])
    base = float(data.mocap_pos[0][0])
    for k in range(int(1.5 / model.opt.timestep)):  # 45 mm, still inside the 70 mm half-width
        data.mocap_pos[0][0] = base + 0.03 * (k + 1) * model.opt.timestep
        mujoco.mj_step(model, data)
    plate_moved = float(data.mocap_pos[0][0]) - base
    cube_moved = float(data.qpos[0]) - x0
    assert plate_moved == pytest.approx(0.045, abs=1e-3), "the plate itself must have moved"
    assert data.ncon >= 1, "contact was lost, so this says nothing about friction"
    assert abs(cube_moved) < 0.002, (
        f"a mocap plate carried the cube {cube_moved * 1000:.1f} mm of its own {plate_moved * 1000:.1f} "
        f"mm -- if MuJoCo has gained mocap friction, prop_trajectory could be simplified into a "
        f"moving_box mode"
    )


def test_start_index_gives_a_phase_offset(square_csv):
    """Different trials use different phases of the same pinned path."""
    _, _, _, a = _rig(square_csv)
    _, _, _, b = _rig(square_csv, start_index=2)
    assert a._cum[-1] > b._cum[-1], "a later start index must leave a shorter remaining path"
    assert np.allclose(b._pts[0], [0.0, 0.0]), "the offset path must still start at the origin"


def test_reset_returns_the_stage_to_the_path_start(square_csv):
    """Repeated trials must not inherit the previous trial's stage position."""
    model, data, ctx, plugin = _rig(square_csv)
    _settle(model, data, ctx, plugin, seconds=2.0)
    assert np.linalg.norm(_stage_xy(data, plugin)) > 0.01
    plugin.on_reset(ctx)
    assert np.linalg.norm(_stage_xy(data, plugin)) == pytest.approx(0.0, abs=1e-9)
    assert plugin.read_progress()[0] == 0.0


def test_non_looping_path_finishes_and_stops(square_csv):
    """Without `loop`, the stage parks at the path end and reports done."""
    model, data, ctx, plugin = _rig(square_csv)
    _settle(model, data, ctx, plugin, seconds=8.0)  # 0.2 m at 30 mm/s = 6.67 s
    assert plugin.read_progress()[2] is True
    end = _stage_xy(data, plugin)
    for _ in range(int(1.0 / model.opt.timestep)):
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
    assert np.allclose(_stage_xy(data, plugin), end, atol=1e-4), "stage moved after finishing"
