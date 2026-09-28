"""A paused run still runs what other threads post, without advancing time."""

from __future__ import annotations

import threading

import numpy as np

from roqsim import control as ctl
from roqsim import runner
from roqsim.clock import Pacer
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _engine() -> Engine:
    cfg = load_config_from_dict({"sim": {}, "plugins": [{"dummy": {}, "name": "d0"}]})
    return Engine(cfg, preview=True)


def _pause(engine: Engine) -> Pacer:
    engine.reset()
    engine.ctx.control.set_state(ctl.PAUSED)
    return Pacer(engine.dt, realtime=False)


def test_a_command_posted_while_paused_runs_without_a_step():
    engine = _engine()
    with engine:
        pacer = _pause(engine)
        ran_on = []
        t0 = float(engine.ctx.data.time)
        worker = threading.Thread(
            target=lambda: engine.ctx.post(lambda _c: ran_on.append(threading.get_ident()))
        )
        worker.start()
        worker.join()

        took = runner._tick(engine, pacer)

        assert took is None  # no step was taken
        assert ran_on == [threading.get_ident()]  # ran, on the thread driving the loop
        assert float(engine.ctx.data.time) == t0


def test_a_pose_set_while_paused_is_visible_in_xpos():
    engine = _engine()
    with engine:
        pacer = _pause(engine)
        body = engine.ctx.model.body("d0_box").id
        target = np.array([1.5, -0.5, 0.7])

        def place(ctx):
            ctx.data.qpos[0:3] = target

        engine.ctx.post(place)
        runner._tick(engine, pacer)

        np.testing.assert_allclose(engine.ctx.data.xpos[body], target)
        assert float(engine.ctx.data.time) == 0.0
