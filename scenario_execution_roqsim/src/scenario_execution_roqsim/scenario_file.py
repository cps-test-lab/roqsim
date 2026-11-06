# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A world entry that runs an OpenSCENARIO file: the ``scenario`` plugin.

A scene in a world, as a Python scenario is, run by scenario-execution inside ``roqsim sim``, so a
recording, a viewer and the control socket need nothing else::

    components:
      - scenario: {file: press_cell.osc}
        name: scene

Config::

    file: press_cell.osc   # relative to the world file
    scenario: ""           # which scenario of the file; required where it holds several
    stop_at_end: true      # end a standalone run when the scenario ends
    output_dir: ""         # where scenario-execution writes its test.xml; none by default

Each episode parses the file again and builds a new tree, since a tree holds a run's state. The tree
is ticked once per physics step, after the plugins listed before this entry, in its ``post_step``. The
file is parsed there too, on the first step: rendering, replaying or checking a world steps nothing,
so none of them needs scenario-execution.

A scenario that fails stops the run with an error and a non-zero exit. One the run ends before it does
is written to ``test.xml`` as failed and reported to the driver (``ctx.report_unfinished``), so
``roqsim sim`` exits non-zero; a reset that cuts it short logs that as an error, and the next episode
runs it again.

The other way to run a scenario is scenario-execution owning the loop (:mod:`roqsim.scenario_adapter`),
for a campaign that runs it as its command, or for scenario parameters this entry does not take.
"""

from __future__ import annotations

import logging
from pathlib import Path

from roqsim import endpoint
from roqsim.plugin import Plugin

_log = logging.getLogger(__name__)


class ScenarioFileError(RuntimeError):
    """A scenario file that cannot run, or a scenario that failed."""


class _Simulation:
    """What an action needs of a simulation in-process (:mod:`scenario_execution_roqsim.access`)."""

    def __init__(self, ctx):
        self.context = ctx


class ScenarioFilePlugin(Plugin):
    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._execution = None
        self._clock = None
        self._started = False
        self._ended = False
        self._done = False

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        unknown = set(config) - {"file", "scenario", "stop_at_end", "output_dir", "present"}
        if unknown:
            errors.append(
                f"scenario: unknown key(s) {sorted(unknown)}; it takes 'file', 'scenario', "
                "'stop_at_end', 'output_dir'"
            )
        file = config.get("file")
        if not isinstance(file, str) or not file:
            errors.append("scenario: 'file' names the OpenSCENARIO file to run")
        if not isinstance(config.get("stop_at_end", True), bool):
            errors.append("scenario: 'stop_at_end' is true or false")
        return errors

    def _path(self, file: str) -> Path:
        path = Path(file)
        return path if path.is_absolute() else Path(self.base_dir or ".") / path

    def sources(self) -> list:
        file = self.config.get("file")
        return [str(self._path(file).resolve())] if file else []

    def on_reset(self, ctx) -> None:
        self._finish_tree(ctx, reported=False)
        self._started = self._ended = self._done = False

    def post_step(self, ctx) -> None:
        if self._ended:
            return
        if not self._started:
            self._start(ctx)
        self._clock.advance()
        self._execution.behaviour_tree.tick()
        if not self._execution.shutdown_requested:
            return
        self._ended = True
        results = self._execution.results
        self._execution.process_results()
        if not results or not results[-1].result:
            failure = results[-1] if results else None
            raise ScenarioFileError(
                f"scenario {self.address}: {self.config['file']} failed: "
                f"{failure.failure_message if failure else 'no result'}"
                + (f"\n{failure.failure_output}" if failure and failure.failure_output else "")
            )
        self._done = True
        ctx.events.emit(self.address, "scenario_end")
        if self.config.get("stop_at_end", True):
            ctx.request_stop(f"scenario {self.address} done")

    def _start(self, ctx) -> None:
        try:
            from scenario_execution.scenario_execution_base import ScenarioExecution
            from scenario_execution.simulation import SimulationClock
        except ImportError as err:
            raise ScenarioFileError(
                f"scenario {self.address}: running an OpenSCENARIO file needs scenario-execution, "
                f"which is not installed ({err})"
            ) from err
        file = str(self._path(self.config["file"]))
        execution = ScenarioExecution(
            debug=False,
            log_model=False,
            live_tree=False,
            scenario_file=file,
            output_dir=self.config.get("output_dir") or "",
            tick_period=ctx.dt,
            register_signal=False,
        )
        if not execution.parse():
            detail = execution.results[-1].failure_output if execution.results else ""
            raise ScenarioFileError(f"scenario {self.address}: {file} did not parse: {detail}")
        self._clock = SimulationClock(ctx.dt)
        self._clock.reset()
        tree = self._choose(execution.scenarios_list, file)
        execution.setup(tree, simulation=_Simulation(ctx), clock=self._clock)
        self._execution = execution
        self._started = True

    def _choose(self, scenarios: list, file: str):
        trees = {tree.name: tree for tree, _params, _dir in scenarios}
        wanted = self.config.get("scenario")
        if wanted:
            if wanted not in trees:
                raise ScenarioFileError(
                    f"scenario {self.address}: {file} has no scenario {wanted!r}; it has "
                    f"{sorted(trees)}"
                )
            return trees[wanted]
        if len(trees) != 1:
            raise ScenarioFileError(
                f"scenario {self.address}: {file} has {len(trees)} scenarios, {sorted(trees)}; "
                "say which with 'scenario:'"
            )
        return next(iter(trees.values()))

    def _finish_tree(self, ctx, *, reported: bool) -> None:
        """End a tree a reset or the run's end cut short, and say so: ``reported`` to the driver,
        for the run's end, else as an error in the log."""
        execution, self._execution = self._execution, None
        if execution is None:
            return
        if not execution.shutdown_requested:
            what = f"scenario {self.address} ({self.config.get('file')})"
            if reported:
                ctx.report_unfinished(what)
            else:
                _log.error("the run ended before %s did", what)
            execution.on_scenario_shutdown(False, "the run ended before the scenario did")
            execution.process_results()
        execution.behaviour_tree.shutdown()

    def shutdown(self, ctx) -> None:
        self._finish_tree(ctx, reported=True)

    @endpoint.out(rate=10.0)
    def done(self) -> bool:
        """Whether the scenario has ended, as it should."""
        return self._done
