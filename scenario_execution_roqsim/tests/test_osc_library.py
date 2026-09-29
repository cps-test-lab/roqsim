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
from scenario_execution.model.types import VariableReference  # noqa: E402
from scenario_execution.utils.logging import Logger  # noqa: E402

from scenario_execution_roqsim.actions.entity_call import EntityCall  # noqa: E402
from scenario_execution_roqsim.actions.entity_monitor import EntityMonitor  # noqa: E402
from scenario_execution_roqsim.actions.entity_moved import EntityMoved  # noqa: E402
from scenario_execution_roqsim.actions.entity_navigate import (  # noqa: E402
    EntityNavigate,
    EntityNavigateStart,
)
from scenario_execution_roqsim.actions.entity_rotated import EntityRotated  # noqa: E402
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
            EntryPointStub("entity_monitor", EntityMonitor, "scenario_execution_roqsim"),
            EntryPointStub("entity_rotated", EntityRotated, "scenario_execution_roqsim"),
            EntryPointStub("entity_call", EntityCall, "scenario_execution_roqsim"),
            EntryPointStub("entity_navigate", EntityNavigate, "scenario_execution_roqsim"),
            EntryPointStub(
                "entity_navigate_start", EntityNavigateStart, "scenario_execution_roqsim"
            ),
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
        "        entity_call(entity: 'grip_fault', command: 'override', value: 'true')\n"
    )
    assert len(_nodes(tree, EntityMoved)) == 1
    assert len(_nodes(tree, EntityRotated)) == 1
    assert len(_nodes(tree, EntityCall)) == 1


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
        "require_verified: bool = true",
    ],
)
def test_the_defaults_are_the_documented_ones(declaration):
    """A changed default silently changes what every scenario that omits it measures.

    `require_verified: false` would turn a fault that never landed into a passing trial; `require: any`
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
        "        entity_call(entity: 'grip_fault', command: 'override')\n"
    )
    assert len(_nodes(tree, EntityMoved)) == 1
    assert len(_nodes(tree, EntityRotated)) == 1
    assert len(_nodes(tree, EntityCall)) == 1


# -- entity_navigate ---------------------------------------------------------------------------
def test_entity_navigate_parses_and_binds_to_its_action():
    """Catches `.osc` <-> `execute()` signature drift at parse time, before any run does."""
    _build(
        """
import osc.roqsim
scenario test:
    do serial:
        entity_navigate(entity: 'cart', goal_poses: [pose_3d(position: position_3d(x: 2.0, y: 1.0))])
"""
    )


def test_entity_navigate_takes_several_goals_and_the_optional_arguments():
    _build(
        """
import osc.roqsim
scenario test:
    do serial:
        entity_navigate(
            entity: 'cart',
            goal_poses: [
                pose_3d(position: position_3d(x: 2.0, y: 0.0)),
                pose_3d(position: position_3d(x: 2.0, y: 2.0))],
            success_on_acceptance: true)
"""
    )


def test_entity_navigate_start_needs_only_the_entity():
    """The route lives in the world; the scenario supplies only the trigger."""
    _build(
        """
import osc.roqsim
scenario test:
    do serial:
        entity_navigate_start(entity: 'cart')
"""
    )


# -- entity_monitor --------------------------------------------------------------------------------
#: The condition patterns the package README documents, together, so none of them is prose only.
PATTERNS = """
import osc.roqsim
scenario test:
    min_clearance: float = 0.3
    event fault_on
    var tripped: bool = false
    var docked: bool = false
    var clearance: float = 10.0
    do parallel:
        entity_monitor(entity: 'ur5e', value: 'force_limit.tripped', target_variable: tripped)
        entity_monitor(entity: 'robot', value: 'clearance.current', target_variable: clearance)
        entity_monitor(entity: 'cart', value: 'dock.docked', target_variable: docked)
        serial:
            wait tripped or clearance < 0.1
            emit end
        serial:
            wait clearance < 0.2
            emit fail
        serial:
            entity_navigate(entity: 'cart', goal_poses: [pose_3d(position: position_3d(x: 2.0))]) with:
                until docked == true
            wait clearance < min_clearance
            entity_call(entity: 'grip_fault', command: 'override', value: 'true')
            emit fault_on
        serial:
            wait @fault_on if clearance < 0.3
            emit end
"""


def test_entity_monitor_hands_over_the_variable_to_write_not_its_value():
    """`target_variable` arrives as a reference, as `topic_monitor`'s does, so the action writes the
    variable the scenario's conditions read -- and the documented conditions parse beside it."""
    tree = _build(PATTERNS)
    nodes = _nodes(tree, EntityMonitor)
    assert len(nodes) == 3
    for node in nodes:
        args = node._model.get_resolved_value_with_variable_references(
            node.get_blackboard_client(), skip_keys=node.execute_skip_args
        )
        assert isinstance(args["target_variable"], VariableReference)
    assert not nodes[0].resolve_variable_reference_arguments_in_execute
