# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A recording carries the components that RAN, and is rebuilt by reading them.

It used to carry only the recipe -- a world reference and the override document -- so replaying meant
re-running the whole load path. That coupled every recording to the override grammar: a change to how
an override resolves would silently rebuild a *different* world, or refuse one that had been fine.
Recording the resolved tree makes rebuilding a read.
"""

import numpy as np
import pytest

from roqsim.capture import FORMAT_VERSION, snap_fps
from roqsim.config import SimConfig, load_config_from_dict
from roqsim.recording import Recording, RecordingError

pytest.importorskip("roqsim_mobile", reason="turtlebot4 manifest lives in roqsim_mobile")

WORLD = {
    "sim": {"world": "empty_room"},
    "components": [{"spawn_robot": {"model": "turtlebot4"}, "name": "robot"}],
}


def test_the_record_holds_what_ran_not_what_was_asked_for():
    """The manifest's components are in it, and the document never named them."""
    cfg = load_config_from_dict(WORLD, overrides={"components": {"robot.lidar": {"rays": 720}}})
    record = cfg.as_record()
    by_address = {c["address"]: c for c in record["components"]}
    assert "robot.lidar" in by_address
    assert by_address["robot.lidar"]["config"]["rays"] == 720


def test_rebuilding_reads_the_tree_rather_than_re_resolving():
    """No document, no overrides, no resolution -- and the same components."""
    cfg = load_config_from_dict(WORLD, overrides={"components": {"robot.lidar": {"rays": 720}}})
    rebuilt = SimConfig.from_record(cfg.as_record())
    assert [s.address for s in rebuilt.plugins] == [s.address for s in cfg.plugins]
    assert [s.enabled for s in rebuilt.plugins] == [s.enabled for s in cfg.plugins]
    lidar = next(s for s in rebuilt.plugins if s.ref == "lidar")
    assert lidar.config["rays"] == 720 and lidar.config["max_range"] == 12.0


def test_a_disabled_component_is_recorded_as_disabled():
    """Which is why `enabled: false` is a flag and not a deletion: the record can say so."""
    cfg = load_config_from_dict(
        WORLD, overrides={"components": {"robot.lidar": {"enabled": False}}}
    )
    entry = next(c for c in cfg.as_record()["components"] if c["address"] == "robot.lidar")
    assert entry["enabled"] is False


def _rec(meta_extra):
    meta = {"format_version": FORMAT_VERSION, "world": "w.yaml", **meta_extra}
    return meta


def test_a_newer_record_is_refused_rather_than_partly_read(tmp_path):
    """It was written to a contract this code has not seen; the keys that happen to overlap would
    give a plausible-looking answer about something else."""
    meta = _rec({"format_version": FORMAT_VERSION + 1})
    with pytest.raises(RecordingError, match="reads up to"):
        Recording(tmp_path / "x.npz", meta, np.zeros((1, 1), dtype=np.float32))


def test_an_older_record_still_reads_but_will_not_rebuild(tmp_path):
    """Its samples, times and clock are all still there -- only the part that would re-resolve an
    override against a world is refused, and it says how to get a picture anyway."""
    meta = _rec({"format_version": 1})
    rec = Recording(tmp_path / "x.npz", meta, np.zeros((2, 1), dtype=np.float32))
    assert len(rec) == 2
    with pytest.raises(RecordingError, match="pass the world explicitly"):
        rec.build()


def test_the_provenance_carries_the_resolved_actuator_table(tmp_path):
    """What the joints RAN under, beside what the world declared.

    ``world_model`` above is the declared half. It cannot answer "what gains did this run use": a
    model's own values are not in it, and a block that changes one joint leaves the rest described by
    nothing. The resolved table is every actuator, with the law, the gains, and where each came from.
    """
    from roqsim.capture import StateRecorder
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {"world": "empty_room"},
            "components": [
                {
                    "spawn_robot": {
                        "model": "turtlebot4",
                        "actuators": {"control": "velocity", "d": 12.0},
                    },
                    "name": "robot",
                }
            ],
        }
    )
    engine = Engine(cfg)
    try:
        engine.setup()
        recorder = StateRecorder(
            engine.ctx, tmp_path / "run.npz", snap_fps(30, 0.002), config=cfg, world="w"
        )
        actuators = recorder._provenance["actuators"]
        assert "robot" in actuators
        rows = actuators["robot"]
        assert rows and all(row["control"] == "velocity" for row in rows)
        assert all(row["d"] == 12.0 and row["source"] == "shared" for row in rows)
        # Prefixed, so a reader can line the table up against the compiled model's actuators.
        names = {row["name"] for row in rows}
        assert names == {
            engine.ctx.model.actuator(i).name for i in range(engine.ctx.model.nu)
        }
    finally:
        engine.shutdown()


def test_a_world_that_declares_no_gains_still_records_what_ran(tmp_path):
    """The table is not a diff: a run nobody overrode must still say what its joints ran under."""
    from roqsim.capture import StateRecorder
    from roqsim.engine import Engine

    cfg = load_config_from_dict(WORLD)
    engine = Engine(cfg)
    try:
        engine.setup()
        recorder = StateRecorder(
            engine.ctx, tmp_path / "run.npz", snap_fps(30, 0.002), config=cfg, world="w"
        )
        rows = recorder._provenance["actuators"]["robot"]
        assert rows and all(row["source"] == "model" for row in rows)
    finally:
        engine.shutdown()


def test_the_added_key_needs_no_format_bump():
    """`Recording` reads `world_model` by name and ignores keys it does not know, so a reader made
    before the actuator table exists still opens a recording that has one."""
    assert FORMAT_VERSION == 2
