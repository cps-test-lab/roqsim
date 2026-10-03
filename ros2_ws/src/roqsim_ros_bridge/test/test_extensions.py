"""A bridge extension that fails to import fails the bridge's start.

An extension registers action/service handlers, converters and decoders. A bridge that started
without one would serve a world that looks complete and is missing whatever that module provides, so
the failure is raised, naming the entry point, with the original exception chained.
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint

import pytest

from roqsim_ros_bridge import extensions


@pytest.fixture
def registered(monkeypatch, tmp_path):
    """Register the given ``name: module source`` pairs as the extension group, loadable from disk."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(extensions, "_loaded", False)
    added: list[str] = []

    def register(modules: dict[str, str]) -> None:
        eps = []
        for name, source in modules.items():
            module = f"_roqsim_bridge_ext_{name}"
            (tmp_path / f"{module}.py").write_text(source, encoding="utf-8")
            added.append(module)
            eps.append(EntryPoint(name=name, value=module, group=extensions.EXTENSION_GROUP))
        monkeypatch.setattr(extensions, "entry_points", lambda group: tuple(eps), raising=False)

    yield register
    for module in added:
        sys.modules.pop(module, None)


def test_a_broken_extension_fails_the_start_and_names_its_entry_point(registered):
    registered({"broken_nav": "raise ValueError('no such message type')\n"})

    with pytest.raises(extensions.ExtensionError, match="'broken_nav'") as info:
        extensions.load_extensions()

    assert "_roqsim_bridge_ext_broken_nav" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)
    assert str(info.value.__cause__) == "no such message type"
    # Not marked loaded: asking again fails again rather than carrying on without it.
    with pytest.raises(extensions.ExtensionError):
        extensions.load_extensions()


def test_a_working_extension_still_loads(registered):
    registered({"fine": "LOADED = True\n"})

    extensions.load_extensions()

    assert sys.modules["_roqsim_bridge_ext_fine"].LOADED is True
    assert extensions._loaded is True


def test_the_scan_is_roqsims_shared_one():
    from roqsim.entry_points import entry_points

    assert extensions.entry_points is entry_points
