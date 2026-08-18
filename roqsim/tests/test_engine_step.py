"""Engine stepping: hook counts, timing, model compile, and the cross-thread command queue."""

from __future__ import annotations

import threading

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _dummy_engine(n=1, size=0.1, profile=False):
    plugins = [{"dummy": {"size": size}, "name": f"d{i}"} for i in range(n)]
    return Engine(load_config_from_dict({"sim": {}, "plugins": plugins}), profile=profile)


def test_dummy_builds_nonempty_model_and_counts_hooks():
    engine = _dummy_engine(n=1)
    engine.setup()
    assert engine.ctx.model.nbody >= 2  # world + the dummy box
    engine.reset()
    for _ in range(5):
        engine.step()
    counts = engine.ctx.blackboard.get("dummy_counts::d0")
    assert counts["pre_step"] == 5
    assert counts["post_step"] == 5
    assert counts["configure"] == 1
    assert counts["on_reset"] == 1
    engine.shutdown()
    assert engine.ctx.blackboard.get("dummy_counts::d0")["shutdown"] == 1


def test_timing_report_populated():
    engine = _dummy_engine(n=1, profile=True)
    engine.setup()
    engine.step()
    report = engine.timing_report()
    assert "d0" in report
    assert "pre_step" in report["d0"] and report["d0"]["pre_step"] >= 0.0


def test_timing_off_by_default():
    engine = _dummy_engine(n=1)
    engine.setup()
    engine.step()
    assert engine.timing_report() == {}
    assert engine.load_report() == {"phases": {}, "plugins": {}}


def test_load_report_populated():
    engine = _dummy_engine(n=2, profile=True)
    engine.setup()
    report = engine.load_report()
    assert {"resolve_plugins", "world_load", "compile", "make_data", "setup_total"} <= set(
        report["phases"]
    )
    assert all(total >= 0.0 for total in report["phases"].values())
    # d0/d1 are distinct instance names -> two build rows of one call each.
    count, total, mx = report["plugins"]["d0"]["build"]
    assert count == 1 and 0.0 <= mx <= total + 1e-12
    assert "d1" in report["plugins"]


def test_engine_group_excluded_from_hook_table():
    engine = _dummy_engine(n=1, profile=True)
    engine.setup()
    engine.step()
    assert "<engine>" not in engine.timing_report()
    assert "<engine>" not in engine.format_timing()
    assert "compile" in engine.format_load_report()


def test_entity_registered():
    engine = _dummy_engine(n=2)
    engine.setup()
    assert set(engine.ctx.entities.names()) == {"d0", "d1"}


def test_posted_commands_apply_on_physics_thread_in_prestep_order():
    """Commands posted from a worker thread all run on the physics thread, in FIFO order."""
    engine = _dummy_engine(n=1)
    engine.setup()

    applied = []
    physics_thread_id = threading.get_ident()
    seen_thread_ids = []

    def make_cmd(i):
        def _cmd(ctx):
            seen_thread_ids.append(threading.get_ident())
            applied.append(i)

        return _cmd

    # Post from a separate (non-physics) thread.
    def worker():
        for i in range(10):
            engine.ctx.post(make_cmd(i))

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # Nothing applied until we step.
    assert applied == []
    engine.step()  # drains queue at start of pre_step, on THIS (physics) thread

    assert applied == list(range(10))  # FIFO order preserved
    assert all(tid == physics_thread_id for tid in seen_thread_ids)  # ran on physics thread
    engine.shutdown()


def test_timestep_override():
    plugins = [{"dummy": {}, "name": "d0"}]
    engine = Engine(load_config_from_dict({"sim": {"timestep": 0.005}, "plugins": plugins}))
    engine.setup()
    assert abs(engine.dt - 0.005) < 1e-9
