"""The timeline a replay scrubs: what a seek lands on, and what playing does at the ends.

Two rules carry everything else. A seek lands on the *nearest* sample with ties to the earlier one --
the rule ``Recording.index_at`` uses, so the sample a slider shows and the sample a render of that
time produces are the same one. And a seek clamps where ``index_at`` raises, because the end of a
drag is the end of the recording rather than a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.playback import Timeline, format_time, parse_time

#: 25 fps from 0.002 s, the shape a recorded run has.
TIMES = np.round(np.arange(25) * 0.04 + 0.002, 6)


@pytest.fixture
def line():
    return Timeline(TIMES, 25.0)


def test_a_timeline_needs_a_sample():
    """An empty recording has no cursor to put anywhere, so it is refused rather than shown."""
    with pytest.raises(ValueError):
        Timeline([], 25.0)


def test_it_opens_on_the_first_sample(line):
    assert line.index == 0
    assert line.time == pytest.approx(0.002)
    assert line.span == (pytest.approx(0.002), pytest.approx(0.962))
    assert len(line) == 25


def test_a_seek_lands_on_the_nearest_sample(line):
    """Nearest, never interpolated: a blend would be a pose the simulation never had."""
    assert line.seek_time(0.481) == 12
    assert line.time == pytest.approx(0.482)
    assert line.seek_time(0.51) == 13


def test_a_tie_lands_on_the_earlier_sample():
    """The rule ``Recording.index_at`` uses, so a shot and its render pick the same sample.

    Halves of whole seconds, so the midpoint is an exact tie rather than one the float representation
    breaks for us.
    """
    line = Timeline([0.0, 1.0, 2.0], 1.0)
    assert line.seek_time(0.5) == 0
    assert line.seek_time(1.5) == 1


def test_a_seek_past_either_end_clamps(line):
    """A slider dragged to its end is not a wrong answer, so it lands rather than raising."""
    assert line.seek_time(1e6) == len(line) - 1
    assert line.seek_time(-1e6) == 0


def test_stepping_clamps_at_both_ends(line):
    line.seek_index(0)
    assert line.step(-5) == 0
    line.seek_index(len(line) - 1)
    assert line.step(5) == len(line) - 1
    assert line.at_end


def test_playing_advances_in_sim_seconds(line):
    """0.2 s at 1x moves 0.2 s of recording -- five samples at 25 fps."""
    line.seek_time(0.002)
    index, hit_end = line.advance(0.2, 1.0)
    assert index == 5
    assert not hit_end
    assert line.time == pytest.approx(0.202)


def test_speed_scales_what_one_tick_covers(line):
    line.seek_time(0.002)
    assert line.advance(0.1, 4.0)[0] == 10
    line.seek_time(0.002)
    assert line.advance(0.1, 0.5)[0] == 1


def test_a_slow_tick_skips_the_frames_it_missed(line):
    """The cursor is in the recording's time axis, so a late tick drops frames rather than queueing."""
    line.seek_time(0.002)
    assert line.advance(0.5, 1.0)[0] == 12


def test_reaching_the_end_is_reported_once(line):
    """What a caller pauses or loops on -- and not again on every tick after it arrives."""
    line.seek_time(0.9)
    assert line.advance(1.0, 1.0) == (len(line) - 1, True)
    assert line.advance(1.0, 1.0) == (len(line) - 1, False)


def test_playing_backwards_reports_the_start(line):
    line.seek_time(0.05)
    assert line.advance(1.0, -1.0) == (0, True)


def test_uneven_spacing_still_plays_at_the_right_speed():
    """A recording whose samples are not evenly spaced is selected by time, not by counting."""
    line = Timeline([0.0, 0.1, 0.5, 0.6, 2.0], 10.0)
    line.seek_time(0.0)
    assert line.advance(0.45, 1.0)[0] == 2


def test_the_label_names_the_sample_it_shows(line):
    line.seek_index(12)
    assert line.label() == "0.482 s   #12 / 24"


def test_the_bar_fills_with_the_cursor(line):
    line.seek_index(0)
    assert line.bar(10) == "[----------]"
    line.seek_index(len(line) - 1)
    assert line.bar(10) == "[##########]"


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("12.48", 12.48), ("12.48s", 12.48), (" 12.48 s ", 12.48), ("1:05.2", 65.2), ("2:00", 120.0)],
)
def test_a_typed_time_is_read_as_a_person_writes_it(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "   ", "abc", "1:2:3", "twelve"])
def test_an_unreadable_time_is_refused(text):
    """Never 0.0 by accident: a time box that silently became 0 would seek to the start of the run."""
    with pytest.raises(ValueError):
        parse_time(text)


def test_a_formatted_time_reads_back(line):
    """What the box shows is what the box accepts, so a value can be edited in place."""
    line.seek_index(7)
    assert parse_time(format_time(line.time)) == pytest.approx(line.time)
