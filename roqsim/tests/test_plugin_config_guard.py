# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Every shipped plugin reads only the config keys it publishes.

A plugin's published keys are what a caller checks a world against before running it: the schema
where one is declared, else the ``Config::`` block ``roqsim plugins describe`` parses out of the
docstring. A key the plugin reads and does not publish is one that check refuses in a valid world --
or, if the caller stops trusting the catalog, one it no longer checks at all. So the rule is
enforced where the drift happens, in the plugin's own source:

* a plugin with a schema reads only literal keys the schema declares, or keys some other owner puts
  there (:data:`roqsim.schema.INJECTED_KEYS`);
* a plugin without one reads only literal keys its published ``Config::`` block lists, inherited
  blocks included;
* a plugin with a schema is strict, or says why not in ``OPEN_KEYS``.

The scan is :func:`roqsim.introspection.undeclared_config_reads`, run for every entry of the
``roqsim.plugins`` group that is installed; its docstring says which reads it sees. The exemptions
below are this repository's, and each is checked to still hold.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

from roqsim.introspection import _config_keys_read, undeclared_config_reads
from roqsim.plugin import Plugin
from roqsim.registry import ENTRY_POINT_GROUP

#: Keys a plugin reads only to REFUSE them with a reason, so they are not settings and are not
#: published. Each is checked below to be refused, so an entry cannot outlive its refusal.
REFUSED = {
    ("ceiling", "enabled"): "the reserved sibling; this plugin works by removing geometry",
    ("payload", "offset"): "an offset payload changes the inertia, which a point mass does not",
    ("contact_impulse", "min_force"): "a force threshold belongs to contact_monitor",
    ("imu", "seed"): "noise draws from the run's seed",
    ("force_torque", "seed"): "noise draws from the run's seed",
    ("wind_field", "seed"): "turbulence draws from the run's seed",
    ("gnss", "seed"): "noise draws from the run's seed",
    ("px4_sitl", "seed"): "sensor noise draws from the run's seed",
    ("imu", "quat"): "the mount is stated under 'pose'",
    ("strip_light", "yaw"): "the run's direction is the yaw of 'pose'",
    ("box", "pos"): "the pose is stated under 'pose'",
    ("box", "yaw"): "the pose is stated under 'pose'",
    ("cylinder", "pos"): "the pose is stated under 'pose'",
    ("cylinder", "yaw"): "the pose is stated under 'pose'",
    ("moving_box", "pos"): "the pose is stated under 'pose'",
    ("moving_box", "yaw"): "the pose is stated under 'pose'",
    ("spawn_model", "pos"): "the pose is stated under 'pose'",
    ("spawn_model", "yaw"): "the pose is stated under 'pose'",
    ("walker", "pos"): "the start is stated under 'pose'",
}

#: Keys a plugin writes into its own config at load and reads back, which no world writes.
SELF_WRITTEN = {
    ("prop_trajectory", "_base_dir"): "expand records the world's directory for `path`",
}


def _plugins() -> list[tuple[str, type]]:
    """Every installed plugin entry, loaded. One that fails to import fails the guard, loudly."""
    return sorted(
        ((ep.name, ep.load()) for ep in entry_points(group=ENTRY_POINT_GROUP)),
        key=lambda item: item[0],
    )


PLUGINS = _plugins()


def _exempt(name: str) -> set[str]:
    return {key for (plugin, key) in (*REFUSED, *SELF_WRITTEN) if plugin == name}


def _unpublished(name: str, cls: type) -> list[str]:
    return [key for key in undeclared_config_reads(cls) if key not in _exempt(name)]


# -- the rules ------------------------------------------------------------------------------------


def test_every_plugin_entry_loads():
    """The guard covers what is installed; an entry it could not load would be one it skipped."""
    assert PLUGINS, "no roqsim.plugins entries are installed"


@pytest.mark.parametrize(
    ("name", "cls"), [p for p in PLUGINS if p[1].CONFIG_SCHEMA], ids=lambda v: str(v)
)
def test_a_plugin_with_a_schema_reads_only_what_it_declares(name, cls):
    missing = _unpublished(name, cls)
    assert not missing, (
        f"{name} reads {missing}, which its CONFIG_SCHEMA does not declare: declare them, or a "
        f"strict schema refuses a world that sets them"
    )


@pytest.mark.parametrize(
    ("name", "cls"), [p for p in PLUGINS if not p[1].CONFIG_SCHEMA], ids=lambda v: str(v)
)
def test_a_plugin_without_one_reads_only_what_its_config_block_lists(name, cls):
    missing = _unpublished(name, cls)
    assert not missing, (
        f"{name} reads {missing}, which its published Config:: block does not list (roqsim "
        f"plugins describe {name}): add them, so a check against the catalog does not refuse them"
    )


@pytest.mark.parametrize(
    ("name", "cls"), [p for p in PLUGINS if p[1].CONFIG_SCHEMA], ids=lambda v: str(v)
)
def test_a_schema_is_strict_unless_it_says_why_not(name, cls):
    assert cls.STRICT_KEYS or cls.OPEN_KEYS.strip(), (
        f"{name} declares a CONFIG_SCHEMA that accepts unknown keys without saying why: drop "
        f"STRICT_KEYS = False, or set OPEN_KEYS = '<why a key outside the schema must pass>'"
    )


@pytest.mark.parametrize(
    ("name", "cls"), [p for p in PLUGINS if p[1].CONFIG_SCHEMA], ids=lambda v: str(v)
)
def test_a_plugin_with_a_schema_writes_no_config_block_of_its_own(name, cls):
    """Its keys are published from the schema; a hand-written copy beside it is one that drifts."""
    from roqsim.introspection import _config_header_span, _own_or_module_doc

    doc = _own_or_module_doc(cls)
    assert _config_header_span(doc.splitlines()) is None, (
        f"{name} declares a CONFIG_SCHEMA and also writes a Config:: block; drop the block -- "
        f"describe and the docs page render the schema"
    )


# -- the exemptions stay true -----------------------------------------------------------------------


@pytest.mark.parametrize(("name", "key"), sorted(REFUSED))
def test_a_key_read_to_be_refused_is_refused(name, key):
    """An exemption that outlives its refusal would hide a key read as a setting."""
    cls = dict(PLUGINS)[name]
    errors = cls({}).config_errors({key: 1})
    assert any(f"'{key}'" in e for e in errors), (name, key, errors)


def test_every_exemption_names_an_installed_plugin_and_a_key_it_reads():
    plugins = dict(PLUGINS)
    for name, key in (*REFUSED, *SELF_WRITTEN):
        assert name in plugins, f"exemption for {name!r}, which is not an installed plugin"
        assert key in _config_keys_read(plugins[name]), (
            f"{name} no longer reads {key!r}; drop its exemption"
        )


# -- the reader itself ------------------------------------------------------------------------------


class _Reads(Plugin):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.a = self.config.get("a")
        self.b = self.config["b"]

    @classmethod
    def expand(cls, spec, world, base_dir):
        cfg = spec.config
        cfg.setdefault("c", 1)
        return []

    def validate_config(self, config):
        errors = []
        if "d" in config:
            errors.append("d")
        for key in ("e", "f"):
            if config.get(key) is None:
                errors.append(key)
        for key in ("g",):
            if key in config:
                errors.append(key)
        return errors

    def configure(self, ctx):
        # Not this plugin's config: the world's, reached through the context.
        ctx.config.get("sim")
        settings = self.settings_for(self.config)
        return self.settings.h, settings.i


def test_the_reader_sees_every_literal_spelling_and_no_other_config():
    assert _config_keys_read(_Reads) == {"a", "b", "c", "d", "e", "f", "g", "h", "i"}


class _Publishes(Plugin):
    """A plugin that publishes one of the two keys it reads.

    Config::

        a: 1
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.a = self.config.get("a")
        self.b = self.config.get("b")
        self.namespace = self.config.get("namespace")


def test_a_read_key_is_undeclared_unless_published_or_injected():
    assert undeclared_config_reads(_Publishes) == ["b"]
