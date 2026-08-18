"""Registry: all three plugin resolution forms plus clear failures for bad references."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from roqsim.plugin import Plugin, PluginError
from roqsim.registry import resolve_plugin


def test_entry_point_short_name():
    cls = resolve_plugin("dummy")
    assert issubclass(cls, Plugin)
    assert cls.__name__ == "DummyPlugin"


def test_module_class_form():
    cls = resolve_plugin("roqsim.plugins.dummy:DummyPlugin")
    assert cls.__name__ == "DummyPlugin"


def test_file_path_form(tmp_path: Path):
    src = textwrap.dedent(
        """
        from roqsim.plugin import Plugin
        class MyExternal(Plugin):
            pass
        """
    )
    f = tmp_path / "ext_plugin.py"
    f.write_text(src)
    cls = resolve_plugin("ext_plugin.py:MyExternal", base_dir=tmp_path)
    assert issubclass(cls, Plugin)
    assert cls.__name__ == "MyExternal"


def test_unknown_entry_point_name_errors():
    with pytest.raises(PluginError, match="unknown plugin"):
        resolve_plugin("does_not_exist")


def test_missing_module_errors():
    with pytest.raises(PluginError, match="could not import module"):
        resolve_plugin("nonexistent.module.path:Thing")


def test_missing_file_errors(tmp_path: Path):
    with pytest.raises(PluginError, match="does not exist"):
        resolve_plugin("nope.py:Thing", base_dir=tmp_path)


def test_missing_class_in_module_errors():
    with pytest.raises(PluginError, match="not found"):
        resolve_plugin("roqsim.plugins.dummy:NoSuchClass")


def test_non_plugin_class_errors():
    with pytest.raises(PluginError, match="Plugin subclass"):
        resolve_plugin("roqsim.engine:Engine")
