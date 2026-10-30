# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Events a plugin emits travel with the recording, at their own simulated time.

A run's events (:mod:`roqsim.events`) are written on the recording's ``events`` channel as they
happen, one message each, and read back by ``Recording.events`` in order -- across a chunk boundary,
after the last sample, and with the fields their source added.
"""

from __future__ import annotations

import pytest
import yaml

from roqsim.events import EventLog
from roqsim.recording import open_recording
from roqsim.runner import run

#: Emits an event every 0.013 s (off the sample grid), and one at the very last step.
_EMITTER = """
from roqsim.plugin import Plugin

class Emitter(Plugin):
    def on_reset(self, ctx):
        self.next = 0.0

    def post_step(self, ctx):
        if ctx.sim_time >= self.next:
            ctx.events.emit("emitter", "tick", n=len(ctx.events), at=round(ctx.sim_time, 6))
            self.next += 0.013
"""


def _record(tmp_path, seconds):
    (tmp_path / "emitter.py").write_text(_EMITTER)
    world = {"sim": {"timestep": 0.001},
             "components": [{f"{tmp_path / 'emitter.py'}:Emitter": {}, "name": "emitter"}]}
    (tmp_path / "w.yaml").write_text(yaml.safe_dump(world))
    out = tmp_path / "run.mcap"
    run(str(tmp_path / "w.yaml"), headless=True, pacing="asap", seconds=seconds, record=str(out),
        capture_fps=10, seed=1)
    return out


def test_every_event_is_read_back_in_order_at_its_own_time(tmp_path):
    out = _record(tmp_path, 0.5)
    with open_recording(out) as rec:
        events = rec.events
    assert [e["event"] for e in events] == ["tick"] * len(events)
    assert len(events) >= 38
    times = [e["t"] for e in events]
    assert times == sorted(times)
    # Off the 0.1 s sample grid: each event keeps the time it was emitted at, not its sample's.
    assert all(e["t"] == pytest.approx(e["at"], abs=1e-9) for e in events)
    assert [e["n"] for e in events] == list(range(len(events)))
    assert len({e["episode"] for e in events}) == 1
    assert {e["source"] for e in events} == {"emitter"}


def test_events_after_the_last_sample_are_written_at_close(tmp_path):
    out = _record(tmp_path, 0.5)
    with open_recording(out) as rec:
        last_sample = float(rec.times[-1])
        assert rec.events[-1]["t"] > last_sample


def test_a_long_run_keeps_its_events_across_chunks(tmp_path):
    out = _record(tmp_path, 4.0)
    with open_recording(out) as rec:
        assert len(rec.events) == pytest.approx(4.0 / 0.013, abs=2)


def test_an_event_field_must_be_json_and_not_shadow_the_events_own():
    log = EventLog(lambda: 1.5, lambda: 2)
    assert log.emit("a", "b", x=1) == {"t": 1.5, "episode": 2, "source": "a", "event": "b", "x": 1}
    with pytest.raises(TypeError):
        log.emit("a", "b", x=object())
    with pytest.raises(ValueError, match="'t'"):
        log.emit("a", "b", t=3.0)
