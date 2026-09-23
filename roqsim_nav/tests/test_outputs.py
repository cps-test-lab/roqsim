"""The output registry: resolving an embodiment by name, path or file.

These run against the **stub** in ``stub_output.py``, never against a shipped output, and that is the
point. The interface exists so an embodiment nobody here anticipated can be plugged in, so the test
of it must be an embodiment this package knows nothing about. A test written against ``mocap`` would
pass just as well if the interface had quietly grown a dependency on what ``mocap`` happens to need.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim_nav._resolve import RegistryError
from roqsim_nav.outputs import NavOutput, available, resolve_output

HERE = Path(__file__).parent


def test_a_short_name_resolves_from_the_entry_point_group():
    cls = resolve_output("mocap")
    assert issubclass(cls, NavOutput)
    assert "mocap" in available()


def test_a_module_path_resolves_without_being_registered():
    """The form an out-of-tree package uses: importable, but nobody registered it."""
    cls = resolve_output("stub_output:StubOutput", HERE)
    assert issubclass(cls, NavOutput)


def test_a_file_path_resolves_relative_to_the_world():
    """The form an experiment uses: a plugin file sitting beside its world, never installed."""
    cls = resolve_output("stub_output.py:StubOutput", HERE)
    assert issubclass(cls, NavOutput)


def test_an_unknown_short_name_lists_what_is_available():
    """An error that does not say what the options were makes a typo an archaeology exercise."""
    with pytest.raises(RegistryError, match="Available:"):
        resolve_output("teleporter")


def test_a_class_that_is_not_an_output_is_refused():
    with pytest.raises(RegistryError, match="NavOutput subclass"):
        resolve_output("stub_output.py:NotAnOutput", HERE)


def test_a_missing_class_names_the_reference():
    with pytest.raises(RegistryError, match="Nonexistent"):
        resolve_output("stub_output.py:Nonexistent", HERE)


def test_a_missing_module_says_so_rather_than_raising_import_error():
    with pytest.raises(RegistryError, match="PYTHONPATH"):
        resolve_output("no_such_module_anywhere:Thing", HERE)


def test_the_default_stop_is_expressible_without_overriding_it():
    """`stop` has a default implementation, so an output only defines it when resting is special."""
    assert "stop" not in NavOutput.__abstractmethods__
    assert NavOutput.__abstractmethods__ == frozenset({"attach", "emit", "pose"})


def test_an_output_declares_its_own_tick_rate():
    """So the navigator never special-cases an embodiment by name to animate it fast enough."""
    assert resolve_output("stub_output.py:StubOutput", HERE).update_hz is None
    assert resolve_output("stub_output.py:SlowOutput", HERE).update_hz == 60.0
