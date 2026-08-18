"""The capture rate's contract: it lands on the physics grid, exactly, and says so proportionately.

The property that matters most is round-tripping: every rate a message *prints* must be acceptable as
``--capture-fps``. A suggestion you cannot type back is not a suggestion, and an earlier design that
hard-errored on off-grid rates was incoherent for exactly that reason -- it would have printed
``29.41`` and then rejected it.
"""

from __future__ import annotations

import logging
from fractions import Fraction

import pytest

from roqsim.capture import (
    DEFAULT_FPS,
    CaptureError,
    CaptureRate,
    RecordToggle,
    TakeRecorder,
    parse_fps,
    physics_rate,
    snap_fps,
)

_snap = snap_fps

#: Every timestep actually used by a world in this repo. The default rate is exact for all of them.
_TIMESTEPS = [0.001, 0.002, 0.0025, 0.004, 0.005]

#: Timesteps the default is *not* exact for, kept separate because they exercise the snap instead:
#: 0.003 is 1000/3 Hz (25 -> k=13 -> 25.64) and 1/240 is 240 Hz (25 -> k=10 -> 24).
_AWKWARD_TIMESTEPS = [0.003, 1 / 240]


# -- parsing -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("25", Fraction(25)),
        (25, Fraction(25)),
        ("30.0", Fraction(30)),
        ("500/17", Fraction(500, 17)),
        ("1/3", Fraction(1, 3)),
        ("29.412", Fraction(29412, 1000)),
    ],
)
def test_fps_parsed_exactly(text, expected):
    assert parse_fps(text) == expected


@pytest.mark.parametrize("bad", ["nope", "", "25x", "1/0", "--"])
def test_unparseable_fps_refused(bad):
    with pytest.raises(CaptureError, match="expected a number or a fraction"):
        parse_fps(bad)


# -- the physics grid ----------------------------------------------------------------------------


@pytest.mark.parametrize("dt", _TIMESTEPS)
def test_physics_rate_is_the_exact_intended_value(dt):
    """A world writes `timestep: 0.002`, which is not representable; the intent must survive anyway."""
    assert float(physics_rate(dt)) == pytest.approx(1 / dt, rel=1e-12)
    assert physics_rate(dt).denominator < 1000  # a clean rational, not a 60-digit float artefact


def test_physics_rate_of_an_odd_timestep_stays_rational():
    assert physics_rate(0.003) == Fraction(1000, 3)


@pytest.mark.parametrize("dt", [0, -0.001])
def test_nonpositive_timestep_refused(dt):
    with pytest.raises(CaptureError, match="must be positive"):
        physics_rate(dt)


# -- snapping ------------------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [25, 50, 100, 125, 250, 500])
def test_on_grid_rates_pass_through_untouched(fps):
    """dt=0.002 is 500 Hz, so anything dividing 500 must be returned exactly as asked."""
    rate = snap_fps(fps, 0.002)
    assert rate.fps == Fraction(fps)
    assert rate.deviation == 0.0
    assert rate.every == 500 // fps


def test_thirty_snaps_to_seventeen_steps_on_a_500hz_world():
    """The motivating case: 500/30 = 16.67 steps is unreachable, so k=17 and the rate is 500/17."""
    rate = snap_fps(30, 0.002)
    assert rate.every == 17
    assert rate.fps == Fraction(500, 17)
    assert float(rate.fps) == pytest.approx(29.4117, abs=1e-4)
    assert rate.deviation == pytest.approx(0.0196, abs=1e-3)


def test_ffmpeg_gets_an_exact_rational_not_a_rounded_decimal():
    """`-r 29.41` on a stream spaced 500/17 apart drifts; the rational is the whole point."""
    assert snap_fps(30, 0.002).ffmpeg_rate() == "500/17"
    assert snap_fps(25, 0.002).ffmpeg_rate() == "25/1"


@pytest.mark.parametrize("dt", _TIMESTEPS)
def test_the_default_is_exact_for_every_timestep_this_repo_uses(dt):
    """25 must never announce a snap on a real world, or the quiet default path is noisy for everyone."""
    rate = snap_fps(DEFAULT_FPS, dt)
    assert rate.deviation == 0.0, f"default {DEFAULT_FPS} is off-grid for timestep {dt}"


@pytest.mark.parametrize("dt", _AWKWARD_TIMESTEPS)
def test_an_off_grid_default_snaps_rather_than_failing(dt):
    """A defaulted rate must never error: the user never typed it.

    It does still *report*, because "your video is 24 fps, not the 25 you'd assume" is real information.
    """
    rate = snap_fps(DEFAULT_FPS, dt)
    assert rate.deviation > 0
    assert rate.fps == physics_rate(dt) / rate.every  # still on the grid, just not at 25


@pytest.mark.parametrize("fps", [25, 40, 50, 100])
def test_a_1000hz_world_keeps_its_divisors(fps):
    assert snap_fps(fps, 0.001).deviation == 0.0


def test_a_third_of_a_frame_per_second_is_exact():
    """`1/3` needs no special case: on a 500 Hz world it is exactly k=1500."""
    rate = snap_fps("1/3", 0.002)
    assert rate.every == 1500 and rate.fps == Fraction(1, 3) and rate.deviation == 0.0


def test_a_repeating_decimal_snaps_to_the_same_k_and_stays_quiet(caplog):
    """`0.333333` is 0.0001% off `1/3`, which must not be announced."""
    rate = snap_fps("0.333333", 0.002)
    assert rate.every == 1500
    assert rate.deviation < 0.001
    with caplog.at_level(logging.INFO):
        rate.report()
    assert caplog.records == []


def test_the_effective_rate_is_always_reachable():
    """Whatever comes out must itself be on the grid -- otherwise the snap did not converge."""
    for fps in (7, 13, 30, 31, 59.94, 60, 199):
        rate = snap_fps(fps, 0.002)
        assert rate.fps == physics_rate(0.002) / rate.every


# -- the round-trip property ---------------------------------------------------------------------


@pytest.mark.parametrize("dt", _TIMESTEPS)
@pytest.mark.parametrize("fps", [30, 60, 13, 99])
def test_every_printed_rate_can_be_fed_back(fps, dt):
    """The property an earlier hard-error design violated: suggestions must be re-enterable.

    Feed the snapped rate, and each suggested neighbour, back in as text. Each must parse, must land on
    the grid, and must come back with zero deviation -- so a user who copies a number out of a message
    gets exactly what the message promised.
    """
    rate = snap_fps(fps, dt)
    for candidate in [rate, *rate.neighbours()]:
        text = candidate.ffmpeg_rate()  # what a caller would copy
        again = snap_fps(text, dt)
        assert again.fps == candidate.fps
        assert again.every == candidate.every
        assert again.deviation == 0.0


def test_neighbours_are_distinct_and_nearest_first():
    rate = snap_fps(30, 0.002)
    ks = [n.every for n in rate.neighbours(4)]
    assert rate.every not in ks
    assert len(set(ks)) == len(ks)
    assert ks[:2] == [18, 16]  # one step slower, one step faster


def test_neighbours_never_suggest_a_nonpositive_divisor():
    """At the fastest possible rate, k=1: there is no faster neighbour to offer."""
    rate = snap_fps(500, 0.002)
    assert rate.every == 1
    assert all(n.every >= 1 for n in rate.neighbours())


# -- the only two hard errors --------------------------------------------------------------------


@pytest.mark.parametrize("fps", [0, -1, "-5"])
def test_nonpositive_fps_refused(fps):
    with pytest.raises(CaptureError, match="must be positive"):
        snap_fps(fps, 0.002)


def test_faster_than_the_physics_rate_refused_and_names_it():
    with pytest.raises(CaptureError) as err:
        snap_fps(600, 0.002)
    message = str(err.value)
    assert "500" in message and "0.002" in message  # the rate and the timestep it came from


def test_exactly_the_physics_rate_is_allowed():
    assert snap_fps(500, 0.002).every == 1


# -- the report bands ----------------------------------------------------------------------------


def test_an_exact_rate_says_nothing(caplog):
    with caplog.at_level(logging.DEBUG):
        snap_fps(25, 0.002).report()
    assert caplog.records == []


def test_a_small_snap_is_only_a_note(caplog):
    """0.1%-1%: worth stating, not worth alarming about."""
    rate = snap_fps("31.4", 0.002)  # 500/31.4 = 15.92 -> k=16 -> 31.25, i.e. 0.48% off
    assert 0.001 <= rate.deviation < 0.01
    with caplog.at_level(logging.DEBUG):
        rate.report()
    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_a_big_snap_warns_and_lists_neighbours(caplog):
    with caplog.at_level(logging.DEBUG):
        snap_fps(30, 0.002).report()
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert "500/17" in message and "every 17 steps" in message
    assert "k=16" in message  # a neighbour, so the user can pick a rounder rate


# -- derived numbers used by the capture loop --------------------------------------------------------


def test_period_is_in_simulated_seconds():
    assert snap_fps(25, 0.002).period == pytest.approx(0.04)


def test_period_matches_the_step_count_exactly():
    """Capture gates on this, so period and `every` must agree or samples land off-grid."""
    for fps in (10, 25, 30, 100):
        rate = snap_fps(fps, 0.002)
        assert rate.period == pytest.approx(rate.every * 0.002)


def test_a_rate_is_hashable_and_comparable():
    """Frozen, so it can be recorded in provenance and compared between runs."""
    assert snap_fps(25, 0.002) == snap_fps(25, 0.002)
    assert isinstance(hash(snap_fps(25, 0.002)), int)
    assert isinstance(snap_fps(25, 0.002), CaptureRate)


# -- the F9 toggle ---------------------------------------------------------------------------------



def test_the_callback_only_sets_a_flag(monkeypatch):
    """UI thread work must be a debounce and a counter -- no sampling, no file I/O, no rendering.

    Anything heavier there would run off the physics thread, which is what the single-writer rule
    forbids, and would run on MuJoCo's own UI thread while it is dispatching events.
    """
    toggle = RecordToggle()
    import roqsim.capture as cap

    forbidden = ("StateRecorder", "TakeRecorder")
    called = []
    for name in forbidden:
        monkeypatch.setattr(cap, name, lambda *a, _n=name, **k: called.append(_n))
    toggle.key_callback(RecordToggle.KEY_F9)
    assert called == []
    assert toggle.take_pending() is True


def test_auto_repeat_yields_one_take():
    """A held key delivers several presses ~0.2 s apart; they must collapse to one toggle."""
    toggle = RecordToggle()
    for _ in range(5):
        toggle.key_callback(RecordToggle.KEY_F9)
    assert toggle.take_pending() is True
    assert toggle.take_pending() is False


def test_a_deliberate_double_press_yields_two(monkeypatch):
    import time as time_mod

    clock = [1000.0]
    monkeypatch.setattr(time_mod, "monotonic", lambda: clock[0])
    toggle = RecordToggle()
    toggle.key_callback(RecordToggle.KEY_F9)
    assert toggle.take_pending() is True
    clock[0] += RecordToggle.DEBOUNCE_S + 0.05  # past the debounce window
    toggle.key_callback(RecordToggle.KEY_F9)
    assert toggle.take_pending() is True


def test_other_keys_are_ignored_and_chained():
    """WalkKeys' arrows must still reach it, so nothing is swallowed."""
    seen = []
    toggle = RecordToggle(chain=seen.append)
    toggle.key_callback(265)  # up arrow
    assert seen == [265]
    assert toggle.take_pending() is False


def test_f9_is_not_one_of_simulates_own_keys():
    """F1-F7 are Simulate's; the arrows/PageUp/Shift are WalkKeys'; letters are visualization flags."""
    assert RecordToggle.KEY_F9 == 298  # GLFW_KEY_F1 == 290
    simulate_function_keys = range(290, 297)  # F1..F7
    assert RecordToggle.KEY_F9 not in simulate_function_keys
    from roqsim.viewer import _WALK_KEYCODES

    assert RecordToggle.KEY_F9 not in _WALK_KEYCODES


# -- numbered takes --------------------------------------------------------------------------------


class _Ctx:
    def __init__(self, model, data):
        self.model, self.data = model, data

    @property
    def sim_time(self):
        return float(self.data.time)


def _ctx_and_rate():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><option timestep='0.002'/><worldbody>"
        "<geom type='plane' size='1 1 .1'/></worldbody></mujoco>"
    )
    return _Ctx(model, mujoco.MjData(model)), _snap(25, 0.002)


def test_takes_are_numbered_so_a_second_never_overwrites_the_first(tmp_path):
    import mujoco

    ctx, rate = _ctx_and_rate()
    takes = TakeRecorder(ctx, tmp_path / "run.npz", rate, world="w")
    for _ in range(2):
        takes.start()
        for _ in range(60):
            mujoco.mj_step(ctx.model, ctx.data)
            takes.sample(ctx)
        takes.stop()
    written = takes.close()
    assert [p.name for p in written] == ["run.npz", "run-2.npz"]
    assert all(p.exists() for p in written)


def test_toggle_alternates(tmp_path):
    ctx, rate = _ctx_and_rate()
    takes = TakeRecorder(ctx, tmp_path / "r.npz", rate, world="w")
    assert takes.recording is False
    takes.toggle()
    assert takes.recording is True
    takes.toggle()
    assert takes.recording is False


def test_stopping_with_nothing_recording_is_harmless(tmp_path):
    ctx, rate = _ctx_and_rate()
    takes = TakeRecorder(ctx, tmp_path / "r.npz", rate, world="w")
    takes.stop()  # must not raise
    assert takes.close() == []


def test_close_finalises_a_running_take(tmp_path):
    """A window closed mid-take, or Ctrl+C, must still leave a loadable recording."""
    import mujoco

    ctx, rate = _ctx_and_rate()
    takes = TakeRecorder(ctx, tmp_path / "r.npz", rate, world="w")
    takes.start()
    for _ in range(60):
        mujoco.mj_step(ctx.model, ctx.data)
        takes.sample(ctx)
    written = takes.close()
    assert len(written) == 1 and written[0].exists()


def test_sampling_before_a_take_starts_is_a_noop(tmp_path):
    ctx, rate = _ctx_and_rate()
    takes = TakeRecorder(ctx, tmp_path / "r.npz", rate, world="w")
    assert takes.sample(ctx) is False
