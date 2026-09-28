# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Every tool that only LOOKS at a world must start on one that draws, with no seed set.

A seed is the driver's to resolve and an unresolved one raises, which is right for a run whose
numbers someone keeps. A tool that produces a picture has no run to reproduce and no seed its
caller could supply, so it says ``Engine(cfg, preview=True)`` and gets the fixed one.

That is a rule a new tool can forget, and forgetting it is invisible: every world in the test
corpus compiles without drawing, so the tool works everywhere until somebody points it at a
randomised world -- and it is then wrong about the world rather than about itself. This file is
what makes it not forgettable. It reads the source rather than running the tools, so a tool added
tomorrow is covered without anybody adding a case here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The repository root, from this file: ``<root>/roqsim/tests/<this>``.
ROOT = Path(__file__).resolve().parents[2]

#: Drivers that RUN a world, and so must resolve a real seed rather than pin the preview one.
#: Named individually because the list is short, closed, and every addition to it is a decision
#: worth someone's attention: a new way to run a campaign is not a thing to acquire silently.
RUNNERS = {
    "roqsim/src/roqsim/runner.py",       # `roqsim sim`
    "roqsim/src/roqsim/scenario_adapter.py",  # the scenario-execution adapter
}


def _engine_calls():
    """``(relative path, line, has_preview_kwarg)`` for every ``Engine(...)`` in the packages."""
    for package in sorted(ROOT.glob("roqsim*/src")):
        for path in sorted(package.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a broken file is another test's problem
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name != "Engine":
                    continue
                preview = any(kw.arg == "preview" for kw in node.keywords)
                yield str(path.relative_to(ROOT)), node.lineno, preview


def test_the_source_really_does_build_engines():
    """The scan finding nothing would make every assertion below vacuous."""
    calls = list(_engine_calls())
    assert len(calls) >= 8, f"only found {len(calls)} Engine(...) calls; has the scan broken?"


def test_no_tool_can_forget_that_it_only_looks():
    """A driver either runs the world or looks at it, and the code has to say which.

    An ``Engine`` built without ``preview`` in a module that is not a runner is a tool that will
    refuse the first randomised world it is pointed at -- with a message naming a seed its caller
    has no way to supply.
    """
    forgot = [f"{path}:{line}" for path, line, preview in _engine_calls()
              if not preview and path not in RUNNERS]

    assert not forgot, (
        "these build an Engine without saying which kind of driver they are: "
        + ", ".join(forgot)
        + ". A tool that compiles a world to LOOK at it (a render, an export, a map, a load "
          "check) passes `preview=True`, which pins the fixed preview seed. A driver that RUNS "
          "the world resolves a real seed and is listed in RUNNERS above."
    )


@pytest.mark.parametrize("path", sorted(RUNNERS))
def test_a_runner_does_not_pin_the_preview_seed(path):
    """The other direction, which matters more: a real run must not silently take a fixed seed.

    Pinning it there would give every repetition of every configuration the same draws -- a sweep
    that repeats a cell to estimate a spread would estimate nothing, and nothing would say so.
    """
    assert (ROOT / path).is_file(), f"{path} is in RUNNERS but does not exist"
    pinned = [line for found, line, preview in _engine_calls() if found == path and preview]
    assert not pinned, f"{path} builds a preview Engine at line(s) {pinned}"
