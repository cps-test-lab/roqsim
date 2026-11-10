"""What a distribution is missing of what it declares: its own requirements, those of what it
requires, and those an extra it asks for brings; not those of its own extras."""

from __future__ import annotations

from types import SimpleNamespace

import installed_requirements as ir
import pytest

#: name -> declared requirements; a name absent here is not installed.
_WORLD = {
    "app": ["lib[ipc]", "plain", "tool; extra == 'dev'"],
    "lib": ["zmq; extra == 'ipc'", "gui; extra == 'gui'"],
    "plain": [],
}


@pytest.fixture
def world(monkeypatch):
    def installed(name):
        reqs = _WORLD.get(ir.canonicalize_name(name))
        return None if reqs is None else SimpleNamespace(requires=reqs)

    monkeypatch.setattr(ir, "_installed", installed)
    ir.missing_requirements.cache_clear()
    yield
    ir.missing_requirements.cache_clear()


def test_an_extra_a_requirement_asks_for_brings_its_requirements(world):
    # lib[ipc] brings zmq; lib's gui extra and app's own dev extra are optional
    assert ir.missing_requirements("app") == ("zmq",)


def test_nothing_missing_when_all_is_installed(world, monkeypatch):
    monkeypatch.setitem(_WORLD, "zmq", [])
    ir.missing_requirements.cache_clear()
    assert ir.missing_requirements("app") == ()
