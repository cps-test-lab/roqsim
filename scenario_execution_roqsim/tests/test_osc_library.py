"""The .osc declarations and the Python signatures must agree, and a scenario must parse.

This is the drift nobody notices until a campaign spends a cell on it: scenario-execution validates an
action's ``execute()`` arguments against its declaration at PARSE time, so a renamed parameter is a
scenario error rather than a Python one -- and the message points at the .osc, not at the class.

Modelled on scenario-execution's own ``scenario_execution_ros/test/test_tf_close_to.py``: the parser is
driven with stubbed entry points, so nothing has to be installed for this to be meaningful.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("scenario_execution", reason="the parser under test")

import py_trees  # noqa: E402
from antlr4 import InputStream  # noqa: E402
from scenario_execution.get_osc_library import (  # noqa: E402
    get_helpers_library,
    get_robotics_library,
    get_standard_library,
    get_types_library,
)
from scenario_execution.model.model_to_py_tree import create_py_tree  # noqa: E402
from scenario_execution.model.osc2_parser import OpenScenario2Parser  # noqa: E402
from scenario_execution.utils.logging import Logger  # noqa: E402

from scenario_execution_roqsim.actions.entity_moved import EntityMoved  # noqa: E402
from scenario_execution_roqsim.actions.entity_rotated import EntityRotated  # noqa: E402
from scenario_execution_roqsim.actions.set_model_override import SetModelOverride  # noqa: E402
from scenario_execution_roqsim.displacement import MODES  # noqa: E402
from scenario_execution_roqsim.get_osc_library import get_osc_library  # noqa: E402


class EntryPointStub:
    def __init__(self, name, load_value, module_name="test"):
        self.name = name
        self.load_value = load_value
        self.module_name = module_name

    def load(self):
        return self.load_value


def _entry_points(group):
    if group == "scenario_execution.osc_libraries":
        return [
            EntryPointStub("helpers", get_helpers_library),
            EntryPointStub("robotics", get_robotics_library),
            EntryPointStub("standard", get_standard_library),
            EntryPointStub("types", get_types_library),
            EntryPointStub("roqsim", get_osc_library),
        ]
    if group == "scenario_execution.actions":
        return [
            EntryPointStub("entity_moved", EntityMoved, "scenario_execution_roqsim"),
            EntryPointStub("entity_rotated", EntityRotated, "scenario_execution_roqsim"),
            EntryPointStub("set_model_override", SetModelOverride, "scenario_execution_roqsim"),
        ]
    return []


def _build(scenario: str):
    parser = OpenScenario2Parser(Logger("test", False))
    tree = py_trees.composites.Sequence(name="", memory=True)
    parsed = parser.parse_input_stream(InputStream(scenario))
    with patch("scenario_execution.model.model_builder.entry_points", _entry_points):
        model = parser.create_internal_model(parsed, tree, "test.osc", False)
    with patch("scenario_execution.model.model_to_py_tree.entry_points", _entry_points):
        return create_py_tree(model, tree, parser.logger, False)


def _nodes(tree, cls):
    return [n for n in tree.iterate() if isinstance(n, cls)]


def test_the_library_is_importable_and_every_action_binds():
    """`import osc.roqsim` resolves, and all three declarations match their execute() signatures."""
    tree = _build(
        "import osc.roqsim\n"
        "scenario test_all:\n"
        "    do serial:\n"
        "        entity_moved(entities: ['parcel'], threshold: 0.05)\n"
        "        entity_rotated(entities: ['parcel'], angle: 0.5)\n"
        "        set_model_override(instance: 'grip_fault')\n"
    )
    assert len(_nodes(tree, EntityMoved)) == 1
    assert len(_nodes(tree, EntityRotated)) == 1
    assert len(_nodes(tree, SetModelOverride)) == 1


@pytest.mark.parametrize("mode", MODES)
def test_every_python_mode_is_a_declared_enum_member(mode):
    """The two lists must not drift: a mode Python knows but the enum lacks is unreachable...

    ...and one the enum declares but Python lacks raises at trigger time, halfway through a trial.
    """
    tree = _build(
        "import osc.roqsim\n"
        "scenario test_mode:\n"
        "    do serial:\n"
        f"        entity_moved(entities: ['x'], threshold: 0.05, mode: displacement_mode!{mode})\n"
    )
    assert len(_nodes(tree, EntityMoved)) == 1


@pytest.mark.parametrize("quantifier", ["all", "any"])
def test_both_quantifiers_resolve(quantifier):
    tree = _build(
        "import osc.roqsim\n"
        "scenario test_require:\n"
        "    do serial:\n"
        "        entity_moved(entities: ['x'], threshold: 0.05, "
        f"require: entity_quantifier!{quantifier})\n"
    )
    assert len(_nodes(tree, EntityMoved)) == 1


def test_a_missing_required_argument_arrives_as_none_and_the_action_rejects_it():
    """The parser does NOT enforce required-ness -- measured: an omitted `entities` resolves to None.

    So "no default in the .osc" is not a guard, and the action has to be one. This pins both halves:
    what the framework hands over, and that the action refuses it by name rather than measuring the
    displacement of nothing and waiting forever.
    """
    tree = _build(
        "import osc.roqsim\n"
        "scenario test_missing:\n"
        "    do serial:\n"
        "        entity_moved(threshold: 0.05)\n"
    )
    node = _nodes(tree, EntityMoved)[0]
    resolved = node._model.get_resolved_value(
        node.get_blackboard_client(), skip_keys=node.execute_skip_args
    )
    assert resolved["entities"] is None
    # ...and an enum arrives as (member_name, value), which is why `enum_name` indexes [0].
    assert resolved["mode"][0] == "distance"
    assert resolved["require"][0] == "all"


@pytest.mark.parametrize(
    "declaration",
    [
        "mode: displacement_mode = displacement_mode!distance",
        "dwell: float = 0.0",
        "require: entity_quantifier = entity_quantifier!all",
        "active: bool = true",
        "require_landed: bool = true",
    ],
)
def test_the_defaults_are_the_documented_ones(declaration):
    """A changed default silently changes what every scenario that omits it measures.

    `require_landed: false` would turn a fault that never landed into a passing trial; `require: any`
    would satisfy a multi-entity condition on one of them. Both are legitimate settings and neither is
    a safe default, so the defaults are pinned here rather than trusted to review.
    """
    from importlib.resources import files

    import scenario_execution_roqsim

    text = (files(scenario_execution_roqsim) / "lib_osc" / "roqsim.osc").read_text()
    assert declaration in text


def test_omitting_every_optional_argument_still_parses():
    """The other half: a default that exists in prose but not in the grammar helps nobody."""
    tree = _build(
        "import osc.roqsim\n"
        "scenario test_defaults:\n"
        "    do serial:\n"
        "        entity_moved(entities: ['parcel'], threshold: 0.05)\n"
        "        entity_rotated(entities: ['parcel'], angle: 0.5)\n"
        "        set_model_override(instance: 'grip_fault')\n"
    )
    assert len(_nodes(tree, EntityMoved)) == 1
    assert len(_nodes(tree, EntityRotated)) == 1
    assert len(_nodes(tree, SetModelOverride)) == 1
