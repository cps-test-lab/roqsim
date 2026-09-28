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
from scenario_execution_roqsim.actions.entity_navigate import (  # noqa: E402
    EntityNavigate,
    EntityNavigateStart,
)
from scenario_execution_roqsim.actions.entity_reports import OPERATORS, EntityReports  # noqa: E402
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


#: What a test adds beside the real libraries and actions: (name, get_osc_library) pairs, and
#: action entry points. Emptied by the test that fills them.
_EXTRA_LIBRARIES: list = []
_EXTRA_ACTIONS: list = []


def _entry_points(group):
    if group == "scenario_execution.osc_libraries":
        return [
            EntryPointStub("helpers", get_helpers_library),
            EntryPointStub("robotics", get_robotics_library),
            EntryPointStub("standard", get_standard_library),
            EntryPointStub("types", get_types_library),
            EntryPointStub("roqsim", get_osc_library),
            *(EntryPointStub(name, getter) for name, getter in _EXTRA_LIBRARIES),
        ]
    if group == "scenario_execution.actions":
        return [
            *_EXTRA_ACTIONS,
            EntryPointStub("entity_moved", EntityMoved, "scenario_execution_roqsim"),
            EntryPointStub("entity_reports", EntityReports, "scenario_execution_roqsim"),
            EntryPointStub("entity_rotated", EntityRotated, "scenario_execution_roqsim"),
            EntryPointStub("set_model_override", SetModelOverride, "scenario_execution_roqsim"),
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
        "comparison_operator: comparison_operator = comparison_operator!eq",
        "dwell: time = 0s",
        "fail_if_bad_comparison: bool = false",
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
            success_on_acceptance: true,
            action_name: '/traffic/navigate_through_poses')
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


# -- entity_reports --------------------------------------------------------------------------------
def _resolved(node):
    return node._model.get_resolved_value(
        node.get_blackboard_client(), skip_keys=node.execute_skip_args
    )


def test_entity_reports_parses_with_only_the_required_arguments():
    tree = _build(
        "import osc.roqsim\n"
        "scenario test:\n"
        "    do serial:\n"
        "        entity_reports(entity: 'ur5e', report: 'force_limit.tripped', "
        "expected_value: 'True')\n"
        "        emit end\n"
    )
    (node,) = _nodes(tree, EntityReports)
    resolved = _resolved(node)
    assert resolved["comparison_operator"][0] == "eq"
    assert resolved["dwell"] == 0.0
    assert resolved["fail_if_bad_comparison"] is False


def test_a_dwell_is_a_time_and_arrives_in_seconds():
    tree = _build(
        "import osc.roqsim\n"
        "scenario test:\n"
        "    do serial:\n"
        "        entity_reports(entity: 'ur5e', report: 'contact_impulse.impulse_ns', "
        "expected_value: '0.5', comparison_operator: comparison_operator!ge, dwell: 200ms)\n"
    )
    (node,) = _nodes(tree, EntityReports)
    assert _resolved(node)["dwell"] == pytest.approx(0.2)


@pytest.mark.parametrize("op", OPERATORS)
def test_every_comparison_the_action_knows_is_a_declared_enum_member(op):
    tree = _build(
        "import osc.roqsim\n"
        "scenario test:\n"
        "    do serial:\n"
        f"        entity_reports(entity: 'e', report: 'r', expected_value: '1', "
        f"comparison_operator: comparison_operator!{op})\n"
    )
    (node,) = _nodes(tree, EntityReports)
    assert _resolved(node)["comparison_operator"][0] == op


def test_the_comparison_enum_coexists_with_osc_ros(tmp_path, monkeypatch):
    """`osc.ros` declares `comparison_operator` too, with the same six members in the same order,
    and a scenario importing both must parse -- in either order, with `check_data` and
    `entity_reports` each getting the member it named. A library shaped like `osc.ros`'s
    declaration stands in for it, since scenario_execution_ros is a colcon package."""
    package = tmp_path / "ros_shaped"
    (package / "lib_osc").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "lib_osc" / "ros.osc").write_text(
        "enum comparison_operator: [\n    lt,\n    le,\n    eq,\n    ne,\n    ge,\n    gt\n]\n"
        "action check_data:\n"
        "    expected_value: string\n"
        "    comparison_operator: comparison_operator = comparison_operator!eq\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from scenario_execution.actions.base_action import BaseAction

    class CheckDataStub(BaseAction):
        def execute(self, expected_value, comparison_operator):
            pass

        def update(self):
            return py_trees.common.Status.SUCCESS

    _EXTRA_LIBRARIES.append(("ros", lambda: ("ros_shaped", "ros.osc")))
    _EXTRA_ACTIONS.append(EntryPointStub("check_data", CheckDataStub, "ros_shaped"))
    try:
        for imports in (
            "import osc.roqsim\nimport osc.ros\n",
            "import osc.ros\nimport osc.roqsim\n",
        ):
            tree = _build(
                imports + "scenario test:\n"
                "    do serial:\n"
                "        entity_reports(entity: 'e', report: 'r', expected_value: '1', "
                "comparison_operator: comparison_operator!ge)\n"
                "        check_data(expected_value: '1', comparison_operator: comparison_operator!lt)\n"
            )
            (reports,) = _nodes(tree, EntityReports)
            (check,) = _nodes(tree, CheckDataStub)
            assert _resolved(reports)["comparison_operator"] == ("ge", 4)
            assert _resolved(check)["comparison_operator"] == ("lt", 0)
    finally:
        _EXTRA_LIBRARIES.clear()
        _EXTRA_ACTIONS.clear()
