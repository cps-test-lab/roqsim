# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Monitor plugin: where a physics step's time goes, from MuJoCo's own stage timers.

``roqsim sim --profile`` says how long ``mj_step`` takes as a whole, beside each plugin's hooks. What
it cannot say is *which part* of the step is expensive -- collision, the constraint solver, the
integrator -- and that is the question a slow world raises: a contact-heavy scene, a flex and a long
kinematic chain are slow for different reasons, and each has a different fix. MuJoCo already keeps
that split in ``mjData.timer``, one cumulative duration and call count per ``mjtTimer``: ``STEP``,
``FORWARD``, the stages ``POSITION``, ``VELOCITY``, ``ACTUATION``, ``CONSTRAINT`` and ``ADVANCE``, and
the sub-stages of position (``POS_KINEMATICS``, ``POS_INERTIA``, ``POS_COLLISION`` with
``COL_BROAD``/``COL_NARROW``, ``POS_MAKE``, ``POS_PROJECT``).

**MuJoCo fills those timers only while a time callback is installed** (``mjcb_time``); without one
every duration stays zero. This plugin installs ``time.perf_counter()`` in microseconds as that
callback in ``configure`` and removes it in ``shutdown``. The callback is process-global, so a
second owner cannot share it: if one is already installed, ``configure`` refuses rather than
replacing it and leaving the other owner reading this plugin's clock. The callback only reads a
clock; it changes no physics, so a run with the plugin steps through the same states as one without.

Every ``every_s`` of simulated time it appends one row to a CSV (``path``, default
``step_timers.csv``): the simulated time, the episode, the wall seconds since ``configure``, then
every timer's cumulative duration (``<timer>_us``, µs) and call count (``<timer>_n``), the timer named
in lower case without its ``mjTIMER_`` prefix. ``shutdown`` writes a final row. The values are
cumulative, so a stage's share of a window is the difference of two rows; a reset zeroes them with
the rest of ``mjData``, which is why each row carries its episode. A relative ``path`` lands in the
run's output directory, anchored the way ``roqsim sim`` anchors its other session outputs
(``RUN_OUTPUT_DIR``, then ``OUTPUT_DIR``, then the working directory).

**Reading the split.** ``STEP`` covers ``FORWARD`` plus ``ADVANCE`` (the integrator). Forward
dynamics that falls outside the timed sub-stages -- for a flex under a discrete integrator, the
smooth-acceleration solve -- appears only as the difference between ``FORWARD`` and the sum of its
children. A step the engine runs in two halves (``mj_step1``/``mj_step2``, for a mechanism whose
weight it routes through a moving base) records ``STEP`` and the stages but no ``FORWARD``. The
wall time no MuJoCo timer covers (``wall_s`` minus ``STEP``) belongs to the plugins, the recorder and
the loop; ``--profile`` attributes the plugins' part hook by hook.

The callback is a Python call per timer start and stop, a few dozen per step, so the plugin is a
measurement mode rather than something to leave on: a world carries it switched off with the
entry-level ``enabled: false`` sibling, and a run that wants the split turns it on.
"""

from __future__ import annotations

import time

import mujoco

from roqsim.context import SimContext
from roqsim.plugin import Plugin
from roqsim.runner import _session_path
from roqsim.schema import Field

#: Every MuJoCo stage timer, in ``mjtTimer`` order, and its column stem.
_TIMERS = [mujoco.mjtTimer(i) for i in range(int(mujoco.mjtTimer.mjNTIMER))]
_STEMS = [t.name.removeprefix("mjTIMER_").lower() for t in _TIMERS]


def _clock_us() -> float:
    """The time callback MuJoCo reads at every timer start and stop: a monotonic clock in µs."""
    return time.perf_counter() * 1e6


class StepTimersPlugin(Plugin):
    #: post_step writes a file and keeps its sampling schedule.
    parallel_safe = False

    CONFIG_SCHEMA = {
        "every_s": Field(
            float,
            default=0.05,
            minimum=0.0,
            unit="s",
            doc="simulated time between rows, > 0",
        ),
        "path": Field(
            str,
            default="step_timers.csv",
            doc="CSV to write; a relative path lands in the run's output directory",
        ),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        settings = self.settings
        self.every_s = settings.every_s
        self.path = settings.path
        self._fh = None
        self._installed = False
        self._next = 0.0
        self._wall0 = 0.0

    def validate_config(self, config: dict) -> list[str]:
        settings = self.settings_for(config)
        if isinstance(settings.every_s, float) and settings.every_s <= 0.0:
            return ["'every_s' must be > 0"]
        return []

    def configure(self, ctx: SimContext) -> None:
        if mujoco.get_mjcb_time() is not None:
            raise RuntimeError(
                f"step_timers[{self.label}]: a MuJoCo time callback (mjcb_time) is already "
                f"installed in this process; refusing to replace it. Only one owner can time "
                f"MuJoCo's stages at once."
            )
        out = _session_path(self.path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._fh = out.open("w")
        columns = ["t", "episode", "wall_s"]
        columns += [f"{s}_us" for s in _STEMS] + [f"{s}_n" for s in _STEMS]
        self._fh.write(",".join(columns) + "\n")
        mujoco.set_mjcb_time(_clock_us)
        self._installed = True
        self._wall0 = time.perf_counter()
        self._next = 0.0
        ctx.logger.info(
            "step_timers: MuJoCo stage timers every %g s of sim time -> %s", self.every_s, out
        )

    def on_reset(self, ctx: SimContext) -> None:
        # mj_resetData zeroed the timers; the next row starts the new episode's series.
        self._next = ctx.sim_time

    def post_step(self, ctx: SimContext) -> None:
        if self._fh is None or ctx.sim_time < self._next:
            return
        self._row(ctx)
        self._next = ctx.sim_time + self.every_s

    def shutdown(self, ctx: SimContext) -> None:
        if self._fh is not None:
            self._row(ctx)
            self._fh.close()
            self._fh = None
        if self._installed:
            self._installed = False
            if mujoco.get_mjcb_time() is _clock_us:
                mujoco.set_mjcb_time(None)
            else:
                ctx.logger.warning(
                    "step_timers: the MuJoCo time callback was replaced while it ran; leaving "
                    "the replacement installed"
                )

    def _row(self, ctx: SimContext) -> None:
        timer = ctx.data.timer
        durations = [f"{timer[i].duration:.6g}" for i in range(len(_TIMERS))]
        counts = [str(int(timer[i].number)) for i in range(len(_TIMERS))]
        head = [f"{ctx.sim_time:.9g}", str(ctx.episode), f"{time.perf_counter() - self._wall0:.6g}"]
        self._fh.write(",".join(head + durations + counts) + "\n")
