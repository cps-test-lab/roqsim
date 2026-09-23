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

Static, over the AST of each plugin class and the plugin bases it inherits from, for every entry of
the ``roqsim.plugins`` group that is installed. It sees a key written as a literal -- ``config.get
("rate_hz")``, ``self.config["model"]``, ``"pose" in config``, or a loop over a literal tuple of
names -- read from ``self.config``, ``spec.config``, or a ``config``/``cfg`` parameter or alias, and
an attribute of ``self.settings`` or ``self.settings_for(...)``. A key computed at runtime, or read by
a helper outside the class, is out of its reach.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from importlib.metadata import entry_points

import pytest

from roqsim.introspection import _config_parameters
from roqsim.plugin import Plugin
from roqsim.registry import ENTRY_POINT_GROUP
from roqsim.schema import INJECTED_KEYS

#: Keys a plugin reads only to REFUSE them with a reason, so they are not settings and are not
#: published. Each is checked below to be refused, so an entry cannot outlive its refusal.
REFUSED = {
    ("ceiling", "enabled"): "the reserved sibling; this plugin works by removing geometry",
    ("payload", "offset"): "an offset payload changes the inertia, which a point mass does not",
    ("contact_impulse", "min_force"): "a force threshold belongs to contact_monitor",
    ("imu", "seed"): "noise draws from the run's seed",
    ("force_torque", "seed"): "noise draws from the run's seed",
    ("wind_field", "seed"): "turbulence draws from the run's seed",
    ("box", "pos"): "the pose is stated under 'pose'",
    ("box", "yaw"): "the pose is stated under 'pose'",
    ("cylinder", "pos"): "the pose is stated under 'pose'",
    ("cylinder", "yaw"): "the pose is stated under 'pose'",
    ("moving_box", "pos"): "the pose is stated under 'pose'",
    ("moving_box", "yaw"): "the pose is stated under 'pose'",
    ("spawn_model", "pos"): "the pose is stated under 'pose'",
    ("spawn_model", "yaw"): "the pose is stated under 'pose'",
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


def _own_classes(cls: type) -> list[type]:
    """The plugin class and each plugin base it inherits keys from; never ``Plugin`` itself."""
    return [c for c in cls.__mro__ if c is not Plugin and issubclass(c, Plugin)]


def _is_config(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "config":
        return isinstance(node.value, ast.Name) and node.value.id in ("self", "spec")
    return isinstance(node, ast.Name) and node.id in names


def _key_of(node: ast.AST, names: set[str]) -> ast.AST | None:
    """The key expression a config read uses, or None when *node* is not a config read."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("get", "pop", "setdefault") and node.args:
            return node.args[0] if _is_config(node.func.value, names) else None
    if isinstance(node, ast.Subscript) and _is_config(node.value, names):
        return node.slice
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.In, ast.NotIn))
        and _is_config(node.comparators[0], names)
    ):
        return node.left
    return None


def _is_settings(node: ast.AST, names: set[str]) -> bool:
    """``self.settings``, ``self.settings_for(...)``, or a name bound to either."""
    if isinstance(node, ast.Attribute) and node.attr == "settings":
        return isinstance(node.value, ast.Name) and node.value.id == "self"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "settings_for"
    return isinstance(node, ast.Name) and node.id in names


def _literal_names(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
    return values if values and all(isinstance(v, str) for v in values) else None


def _keys_read(cls: type) -> set[str]:
    """Top-level config keys the class's own source reads by a literal name."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    keys: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        names = {a.arg for a in params if a.arg in ("config", "cfg")}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _is_config(node.value, names):
                names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        views = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _is_settings(node.value, set()):
                views |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        for node in ast.walk(fn):
            key = _key_of(node, names)
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            # `self.settings.rate_hz`, `settings = self.settings_for(config); settings.model`
            if isinstance(node, ast.Attribute) and _is_settings(node.value, views):
                keys.add(node.attr)
        # `for key in ("a", "b"): config.get(key)` -- resolved per loop, so two loops reusing one
        # variable name each contribute their own names.
        for loop in ast.walk(fn):
            if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)):
                continue
            values = _literal_names(loop.iter)
            if values is None:
                continue
            for stmt in loop.body:
                for node in ast.walk(stmt):
                    key = _key_of(node, names)
                    if isinstance(key, ast.Name) and key.id == loop.target.id:
                        keys.update(values)
    return keys


def _read_by(cls: type) -> set[str]:
    return set().union(*(_keys_read(c) for c in _own_classes(cls)))


def _exempt(name: str) -> set[str]:
    return {key for (plugin, key) in (*REFUSED, *SELF_WRITTEN) if plugin == name}


def _unpublished(name: str, cls: type, published: set[str]) -> list[str]:
    return sorted(_read_by(cls) - published - INJECTED_KEYS - _exempt(name))


# -- the rules ------------------------------------------------------------------------------------


def test_every_plugin_entry_loads():
    """The guard covers what is installed; an entry it could not load would be one it skipped."""
    assert PLUGINS, "no roqsim.plugins entries are installed"


@pytest.mark.parametrize(
    ("name", "cls"), [p for p in PLUGINS if p[1].CONFIG_SCHEMA], ids=lambda v: str(v)
)
def test_a_plugin_with_a_schema_reads_only_what_it_declares(name, cls):
    missing = _unpublished(name, cls, set(cls.CONFIG_SCHEMA))
    assert not missing, (
        f"{name} reads {missing}, which its CONFIG_SCHEMA does not declare: declare them, or a "
        f"strict schema refuses a world that sets them"
    )


@pytest.mark.parametrize(
    ("name", "cls"), [p for p in PLUGINS if not p[1].CONFIG_SCHEMA], ids=lambda v: str(v)
)
def test_a_plugin_without_one_reads_only_what_its_config_block_lists(name, cls):
    listed = {f["name"] for f in _config_parameters(cls) if "." not in f["name"]}
    missing = _unpublished(name, cls, listed)
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
        assert key in _read_by(plugins[name]), f"{name} no longer reads {key!r}; drop its exemption"


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
    assert _keys_read(_Reads) == {"a", "b", "c", "d", "e", "f", "g", "h", "i"}
