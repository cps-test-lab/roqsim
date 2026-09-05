# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The README's numbers, checked against the registries they describe.

A number in a README is a claim that decays silently: nobody recounts, the drift is invisible in
review, and the first reader to notice is someone who trusted it. So the totals the README does
state are asserted here rather than maintained by hand.

A count the README is better off not making is deleted from both sides instead. Two were: the
number of registered plugins, and the number of ready-to-run worlds. Both depend on which packages
are installed, so both read differently in a full checkout, in an install of the public packages
alone, and in a working tree that has an experiment's own provider installed beside them -- and a
number that is wrong for most readers is worse than no number. The README describes what plugins and
worlds are rather than how many there are.

The robot total stays because it does not have that problem: it is counted over a fixed list of
in-tree packages, so an out-of-tree provider cannot move it.

The failure message says what to write, because a test that only says "51 != 52" makes the reader do
the arithmetic that the test just did.

Scope, deliberately: this checks the totals the README states, not the prose around them. A new robot
that lands without being named in the family list is caught by the total moving; whether the sentence
still reads well is a human's call.
"""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path

import pytest

from roqsim.models import ENTRY_POINT_GROUP as MODELS_GROUP
from roqsim.models import _entry_points, _provider_dirs

#: The packages whose models are ROBOTS. Props, sensors and pedestrian blueprints are models too and
#: are counted elsewhere in the docs; "robot models" in the README means these.
ROBOT_PROVIDERS = (
    "roqsim_mobile",
    "roqsim_manipulation_assets",
    "roqsim_mobile_manipulation",
    "roqsim_humanoid",
    "roqsim_quadruped",
    "roqsim_aerial",
)

README = Path(__file__).resolve().parents[2] / "README.md"


def _stated(pattern: str) -> int:
    """The number the README states, from the one place it states it."""
    text = README.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    assert match, (
        f"the README no longer states this number ({pattern!r}) -- update this test with it"
    )
    return int(match.group(1))


def _installed(names) -> list[str]:
    """The subset of *names* that is importable here, so a slim venv skips rather than fails."""
    present = []
    for name in names:
        try:
            import_module(name)
        except ImportError:
            continue
        present.append(name)
    return present


def test_the_robot_model_count_is_the_number_of_shipped_robot_models():
    if len(_installed(ROBOT_PROVIDERS)) < len(ROBOT_PROVIDERS):
        pytest.skip("not every robot-family package is installed, so the total is not comparable")
    models: set[str] = set()
    for ep in _entry_points(MODELS_GROUP):
        if ep.name not in ROBOT_PROVIDERS:
            continue
        models_dir, _mesh, _tex = _provider_dirs(import_module(ep.value.split(":")[0]))
        models |= {
            p.name.removesuffix(".manifest.yaml") for p in Path(models_dir).rglob("*.manifest.yaml")
        }
    assert _stated(r"\*\*(\d+) robot models") == len(models), (
        f"the README says a different number of robot models than are shipped ({len(models)}): "
        f"{sorted(models)}"
    )
