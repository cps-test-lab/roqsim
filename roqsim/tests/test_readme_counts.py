# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The README's numbers, checked against the registries they describe.

A number in a README is a claim that decays silently: nobody recounts, the drift is invisible in
review, and the first reader to notice is someone who trusted it. So the totals the README does
state are asserted here rather than maintained by hand.

A count the README is better off not making is deleted from both sides instead. The number of
registered plugins was one: it depends on which packages are installed, so it read differently in a
full checkout than in an install of the public packages alone, and the README now describes what
plugins do rather than how many there are.

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
from roqsim.world import _world_entry_points

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


def test_the_world_count_is_the_number_a_ref_can_name():
    """The worlds a package OFFERS -- its `roqsim.worlds` entry point -- not every YAML on disk.

    Several packages ship debugging worlds that are deliberately unregistered and run by path; a
    README promising "ready to run" is promising the ones a `<package>:<world>` ref names.

    `*_demo.yaml` is excluded on top of that. A per-model demo is a way to look at ONE robot in an
    empty room -- it exists so `roqsim sim <pkg>:<model>_demo` shows you the model -- and counting
    one per model makes the total track the model count rather than the number of environments there
    are to run anything in. They are still registered and still runnable; they are just not what this
    number is claiming.
    """
    providers = list(_world_entry_points())
    if len(providers) < 5:
        pytest.skip("not every world provider is installed, so the total is not comparable")
    total = 0
    for ep in providers:
        worlds = Path(ep.load().WORLDS_DIR)
        if worlds.is_dir():
            total += sum(1 for w in worlds.glob("*.yaml") if not w.name.endswith("_demo.yaml"))
    # `worlds?`: excluding the demos leaves a count that can reach one, and the README then has to
    # write "world" -- a pattern that only matches the plural stops finding the sentence it checks.
    assert _stated(r"\*\*(\d+) ready-to-run worlds?\*\*") == total, (
        f"the README says a different number of runnable worlds than are registered here ({total})"
    )
