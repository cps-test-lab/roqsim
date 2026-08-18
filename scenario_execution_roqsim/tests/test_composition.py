"""How a "fire once, then let the trial finish" branch must be composed -- and how it must not.

This exists because the obvious composition is wrong in a way that does not look wrong. Splitting a
fused action into a condition and an effect makes the composition the scenario's business, so the
scenario needs one shape that is actually correct, and a test that says why the tempting one is not.

Nothing here touches a simulation: it is pure py_trees, so it runs anywhere and in milliseconds.
"""

from __future__ import annotations

import py_trees
import pytest

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS


class Condition(py_trees.behaviour.Behaviour):
    """Succeeds from the *nth* tick after it was (re-)initialised. Counts its own restarts."""

    def __init__(self, after: int = 2):
        super().__init__("condition")
        self.after = after
        self.ticks = 0
        self.initialisations = 0

    def initialise(self):
        self.initialisations += 1
        self.ticks = 0

    def update(self):
        self.ticks += 1
        return SUCCESS if self.ticks >= self.after else RUNNING


class Effect(py_trees.behaviour.Behaviour):
    """Succeeds immediately, and counts how many times it was fired."""

    def __init__(self):
        super().__init__("effect")
        self.fires = 0

    def initialise(self):
        self.fires += 1

    def update(self):
        return SUCCESS


def _branch(condition, effect):
    seq = py_trees.composites.Sequence(name="fault", memory=True)
    seq.add_children([condition, effect])
    return seq


def _tick(root, n):
    for _ in range(n):
        root.tick_once()


def test_a_succeeded_branch_under_parallel_is_left_alone():
    """`parallel` is SuccessOnAll(synchronise=True), so a child that succeeded is SKIPPED afterwards.

    That is what makes the fault branch fire exactly once and keeps its condition's baseline intact,
    while the trial branch runs on.
    """
    condition, effect = Condition(), Effect()
    trial = Condition(after=50)
    trial.name = "trial"
    root = py_trees.composites.Parallel(
        name="root", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=True)
    )
    root.add_children([trial, _branch(condition, effect)])

    _tick(root, 20)

    assert effect.fires == 1, "the effect must fire once, not once per tick past the threshold"
    assert condition.initialisations == 1, "the condition must not re-take its baseline"
    assert root.status is RUNNING, "the trial has not finished, so the run continues"


def test_success_is_running_restarts_the_branch_instead_of_holding_it():
    """The tempting alternative, and why it is not used. Measured, not reasoned about.

    A py_trees `Decorator` always ticks its child; only the DECORATOR's status is rewritten. A
    `Sequence` resets to its first child whenever its own status is not RUNNING. Together, SUCCESS ->
    RUNNING means "run the branch again", so a real scenario re-takes the condition's baseline and
    re-fires the fault on every fresh crossing -- for `set_model_override` an idempotent no-op that
    still churns, and for `active: false` a fault that flaps on and off.
    """
    condition, effect = Condition(), Effect()
    root = py_trees.decorators.SuccessIsRunning(
        name="success_is_running", child=_branch(condition, effect)
    )

    _tick(root, 20)

    assert effect.fires > 1, "if this ever becomes 1, py_trees changed and the docs here are stale"
    assert condition.initialisations > 1


def test_success_is_running_on_the_leaf_refires_every_tick():
    """Worse than on the branch: `Behaviour.tick` re-initialises whenever status is not RUNNING."""
    effect = Effect()
    root = py_trees.decorators.SuccessIsRunning(name="success_is_running", child=effect)

    _tick(root, 10)

    assert effect.fires == 10


@pytest.mark.parametrize("policy_cls", [py_trees.common.ParallelPolicy.SuccessOnAll])
def test_the_parallel_policy_synchronises_by_default(policy_cls):
    """The whole argument above rests on this default; pin it rather than trust it.

    scenario-execution constructs `SuccessOnAll()` with no arguments for a `parallel:` block
    (`model_to_py_tree.visit_do_member`), so if the upstream default ever flips to
    `synchronise=False`, a succeeded fault branch would be ticked again and the composition would
    silently start re-firing.
    """
    assert policy_cls().synchronise is True
