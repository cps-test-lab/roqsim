"""The :class:`Plugin` base class and lifecycle contract.

A plugin implements any subset of the lifecycle hooks below. The engine calls only the hooks a
plugin actually overrides, so the same class can act as an "init plugin" (``build``/``configure``),
a "tick plugin" (``pre_step``/``post_step``), or both.

Threading contract (see docs/architecture.rst > Concurrency):
  * ``build``/``configure``/``on_reset``/``pre_step``/``post_step``/``shutdown`` all run on the
    single physics thread. It is safe to read and write ``ctx.model``/``ctx.data`` there.
  * Never touch ``ctx.data`` from any other thread. External input funnels through ``ctx.post()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .paths import address_path

if TYPE_CHECKING:  # avoid importing mujoco / context at module import time
    import mujoco

    from .config import PluginSpec
    from .context import SimContext
    from .scene_content import SensorCamera


class PluginError(Exception):
    """Raised for plugin loading, configuration, or lifecycle failures."""


class Plugin:
    """Base class for all roqsim plugins.

    Subclasses receive their YAML ``config:`` section as a plain dict in ``__init__``. Override the
    hooks you need; unimplemented hooks are skipped by the engine (no per-tick cost).
    """

    #: This plugin's config, declared once: ``{key: roqsim.schema.Field(...)}``. Optional. A
    #: plugin that declares one gets three things from it: the mechanical checks (types, ranges,
    #: required keys, unknown keys) run against it, ``roqsim plugins describe`` publishes the
    #: fields with their types and defaults rather than a docstring's prose, and :attr:`settings`
    #: reads the config through it. See :mod:`roqsim.schema`.
    #:
    #: **Declaring it is what enforces it.** ``instantiate_plugins`` runs the check for every plugin
    #: that has one, beside its ``validate_config``; there is no call to remember. A schema the
    #: catalog publishes and nothing checks would be prose with a type annotation.
    CONFIG_SCHEMA: dict | None = None

    #: With a schema, whether a key it does not mention is an error. On by default: a key outside
    #: the schema is a typo that leaves a setting at its default. The keys a component carries
    #: without the world's author writing them (a manifest's ``prefix``, the transport keys) are
    #: known centrally (:data:`roqsim.schema.INJECTED_KEYS`), so a complete schema needs no more.
    #: Off only with an :data:`OPEN_KEYS` reason.
    STRICT_KEYS: bool = True

    #: Why a plugin with a schema accepts keys it does not list -- the one way to leave
    #: :data:`STRICT_KEYS` off. A schema that stays open without saying why is refused by the guard
    #: test over every shipped plugin, because an open schema publishes a contract a typo passes.
    #: Published by ``roqsim plugins describe`` beside ``strict_keys``.
    OPEN_KEYS: str = ""

    #: What this plugin realises from its config, as ``(key, attribute, unit, effect)``: the config
    #: key, the attribute holding the value it runs with, the unit, and what the value does to a
    #: command or a reading. :meth:`realised` turns it into rows; a plugin with more to say (a value
    #: derived from several keys, a default that switches something off) extends that method.
    REALISED: tuple = ()

    #: Set True on a plugin whose ``post_step`` only *reads* ``data`` (no writes, no shared mutable
    #: state) so a future executor may run it concurrently with other parallel-safe post_steps.
    parallel_safe: bool = False

    #: Set True on a scene plugin that builds its own ground + lighting (e.g. the mobile
    #: ``floorplan``). Such a plugin fills the same slot as a world definition (``sim.world``, see
    #: :mod:`roqsim.world`), so the engine skips building one -- and refuses a world that also
    #: sets ``sim.world``, or that carries a second such plugin, rather than stacking two grounds.
    #: This is also the only way to contribute an environment computed from config rather than
    #: baked as geometry: the built-in definitions are a closed set and ``roqsim.worlds`` providers
    #: offer MJCF files.
    provides_world: bool = False

    #: Set True on a plugin that only moves data across a process boundary: it builds no geometry and
    #: holds no simulation state, so a consumer that needs the *scene* rather than a running
    #: simulation (``roqsim render``, the review window, the exporters) drops it instead of loading it.
    #: That is what lets a ``*_ros`` world be rendered without ROS installed. ``roqsim sim`` keeps it
    #: unless asked to drop it (``--no-communication``), and then warns that the run communicates
    #: with nothing -- a simulation without its transport is a different experiment, not a quieter
    #: one.
    transport_only: bool = False

    #: What one capture of this plugin's outputs costs -- ``"render"`` (a GL pass) or ``"raycast"`` --
    #: on a sensor whose outputs a recording can rebuild: each is a function of the restored state,
    #: the run's seed and the sim time alone, so a run need not produce or keep the stream and
    #: :mod:`roqsim.rebuild` produces it afterwards from ``roqsim sim --record``. ``None``: not
    #: rebuildable (the plugin carries state between steps, or reads input the state does not hold).
    #: A plugin that sets it implements :meth:`capture`; :meth:`output_cost` says it per endpoint.
    rebuild_cost: str | None = None

    #: Set True on a plugin that registers an entity (a ``spawn_*``, a prop). Such an entry may own
    #: a nested ``components:`` block, and its label names the entity it brings into being. Declared
    #: here rather than listed in the core for the same reason ``transport_only`` is: a name list in
    #: core would silently serve only our own plugins.
    provides_entity: bool = False

    #: Set True on a plugin that attaches to an entity -- a sensor, a controller, a monitor. Such an
    #: entry must be nested under the entry that provides its entity; declared at the top of a
    #: document it has nothing to attach to, and is refused with the fix in the message rather than
    #: silently running alongside the default it meant to replace.
    requires_owner: bool = False

    #: Config keys :meth:`expand` reads. An override of one on a component a manifest injected
    #: arrives after that component expanded, so what it brought in would keep the old value while
    #: the component itself took the new one; :func:`roqsim.config.load_config` refuses it instead
    #: and says to declare the component in the world, where the value lands before expansion.
    expansion_keys: frozenset[str] = frozenset()

    def __init__(
        self,
        config: dict | None = None,
        *,
        name: str | None = None,
        entity: str | None = None,
        label: str | None = None,
    ):
        self.config: dict = dict(config or {})
        #: Directory of the world document this entry was declared in, so a config value naming a
        #: FILE beside that document resolves the same wherever the world is loaded from. Without
        #: it such a path is only resolvable from the CWD that happened to load the world, and the
        #: working directory is neither the document's directory nor the same one twice.
        #:
        #: Assigned by :func:`~roqsim.config.instantiate_plugins` after construction rather than
        #: taken as a constructor argument: most plugins override ``__init__`` with the four
        #: keywords below, so a fifth would have to be added to every one of them -- and a plugin
        #: that had not been updated would fail at construction rather than fall back. The CWD
        #: default here is what a plugin built outside a document (a test, a driver) gets.
        self.base_dir: Path = Path.cwd()
        #: Instance name (defaults to the class name; overridable via ``name:`` in YAML).
        self.name: str = name or type(self).__name__
        #: How this entry is addressed among its siblings: its ``name:``, else its plugin ref. For a
        #: ``provides_entity`` plugin this IS the name of the entity it registers -- one spelling, so
        #: an entity name cannot be written in a config key *and* in a sibling and disagree.
        self.label: str = label or (name or type(self).__name__)
        #: The entity this instance belongs to -- the label of the entry it is nested under, filled
        #: in by the loader. ``None`` for an entry at the top of a document, which belongs to the
        #: world itself. A plugin reads this instead of parsing an entity name out of its own config:
        #: ownership is where the entry sits, so there is nothing to spell wrong.
        self.entity: str | None = entity

    @property
    def address(self) -> str:
        """This instance's address: the dotted path of labels from the root of the document.

        The one derived identity. An entity a ``provides_entity`` plugin registers is named by it --
        not by its label, which is unique only among siblings, so two robots each carrying an ``arm``
        would otherwise register one entity that hid the other.
        """
        return f"{self.entity}.{self.label}" if self.entity else self.label

    def draw_key(self, what: str) -> str:
        """The ``ctx.rng_for`` key of this instance's draw *what*: its address as a path, then the
        draw's name (``robot/lidar/noise``). Unique to the instance and to the draw, so no two share
        a stream."""
        return f"{address_path(self.address)}/{what}"

    # -- plugin-spec expansion (optional) -----------------------------------------------------
    @classmethod
    def expand(cls, spec: PluginSpec, world: list[PluginSpec], base_dir: Path) -> list[PluginSpec]:
        """Return extra plugin specs to splice into the pipeline immediately after this one.

        Called once at config load, before any plugin is instantiated, so a plugin can pull in
        others it implies -- e.g. a spawn plugin injecting a model's default controller/sensor
        plugins from its manifest (see :func:`roqsim.manifest.expand_manifest`). ``world`` is every
        spec declared or injected so far, so a plugin can skip a default the world -- or an outer
        manifest -- already declares; a disabled entry that a live sibling of the same label
        replaces is not in it, and is not expanded. What this returns is expanded in turn,
        depth-first (see :func:`roqsim.config.expand_document`); list the config keys read here in
        :attr:`expansion_keys`. Default: none.
        """
        return []

    # -- endpoints ----------------------------------------------------------------------------
    @property
    def endpoint_owner(self) -> str:
        """The entity this plugin's decorated endpoints belong to: the one it is nested under, else
        its own label. Override where a plugin speaks for another entity."""
        return self.entity or self.label

    def endpoint_namespace(self, ctx: SimContext, owner: str | None = None) -> str:
        """The transport scope of this plugin's decorated endpoints: its ``namespace:`` config, else
        the owning entity's (*owner*'s, for an endpoint that names its own)."""
        entity = ctx.entities.get(owner or self.endpoint_owner)
        return self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

    def register_endpoints(self, ctx: SimContext) -> list:
        """Add the endpoints declared with :mod:`roqsim.endpoint` to ``ctx.interface``, once per
        context, and return them.

        The engine calls it right after this plugin's ``configure``, so an option may read what
        ``configure`` resolved, and a bridge listed later binds them. A test that calls ``configure``
        itself calls this after it. Endpoints only known at run time are added with
        ``ctx.interface.add`` instead.
        """
        if self.__dict__.get("_endpoints_ctx") is ctx:
            return []
        self._endpoints_ctx = ctx
        from .endpoint import build

        endpoints = build(self, ctx)
        for ep in endpoints:
            ctx.interface.add(ep)
        return endpoints

    # -- endpoint topic hardwiring ------------------------------------------------------------
    def topic_override(self, endpoint_name: str) -> str | None:
        """Topic set for the endpoint ``endpoint_name``, or ``None`` if unset.

        Read from the plugin's ``topics:`` config map (``topics: {<endpoint>: <topic>}``), keyed by
        the endpoint's role name (e.g. ``image``, ``camera_info``, ``joint_states``, ``scan``). A
        decorated endpoint gets it as :attr:`~roqsim.context.Endpoint.topic` from the framework; a
        hand-built one uses it as ``self.topic_override("image") or <namespaced default>`` when
        filling the backend ``topic``. An absolute (leading ``/``) value is published verbatim by
        the bridge, overriding the endpoint's ``namespace`` -- so a producer can match external /
        hardware topic names regardless of its scope. A relative value renames the endpoint inside
        its namespace, the way a vendor description names a robot's second scanner ``scan2`` under
        the robot's namespace.

        Asked in ``configure`` or ``register_endpoints``: :meth:`check_topic_keys` holds the
        ``topics:`` map to the names asked by then, so a name first asked later raises; one already
        asked may be asked again.
        """
        read = self.__dict__.setdefault("_topic_keys_read", set())
        if endpoint_name not in read and self.__dict__.get("_topic_keys_checked"):
            raise PluginError(
                f"[{self.address} ({type(self).__name__})] topic_override({endpoint_name!r}) is asked "
                "after the plugin's topics: were checked: a topic name must be asked for in "
                "configure or register_endpoints, or a rename keyed on it is refused as reaching "
                "nothing."
            )
        read.add(endpoint_name)
        return (self.config.get("topics") or {}).get(endpoint_name)

    def check_topic_keys(self) -> None:
        """Refuse a ``topics:`` key this plugin never asked for.

        A rename is keyed on the name the plugin asks :meth:`topic_override` for -- an endpoint's own
        name, or a role a plugin names its endpoints by -- which can differ from the topic it
        publishes on by default (``contact_monitor``'s ``contact`` publishes ``collision``). A key
        naming the topic, or anything else, matches nothing, and the endpoint publishes under its
        default name as if no rename had been asked for. Checked once the plugin has registered its
        endpoints, since every name it asks for has been asked by then; :meth:`topic_override`
        refuses a name first asked after it.
        """
        self.__dict__["_topic_keys_checked"] = True
        topics = self.config.get("topics")
        if not isinstance(topics, dict) or not topics:
            return
        read = sorted(self.__dict__.get("_topic_keys_read", ()))
        unknown = sorted(key for key in topics if key not in read)
        if unknown:
            raise PluginError(
                f"[{self.address} ({type(self).__name__})] topics: {', '.join(map(repr, unknown))} "
                f"{'is' if len(unknown) == 1 else 'are'} not a name this plugin renames, so the "
                "rename would reach nothing. A rename is keyed on the endpoint's name, not on its "
                "topic"
                + (
                    f"; this plugin's are: {', '.join(read)}."
                    if read
                    else ", and this plugin renames none."
                )
            )

    @property
    def settings(self):
        """This plugin's config through its :data:`CONFIG_SCHEMA`: attribute access, defaults filled.

        A :class:`roqsim.schema.Settings` view: ``self.settings.rate_hz`` is the configured value or
        the schema's default, an undeclared name raises ``AttributeError``, and nothing may be
        assigned to it. A plugin without a schema has none and reads ``self.config``. See
        :meth:`settings_for` for the view of a config other than this instance's own.
        """
        return type(self).settings_for(self.config)

    @classmethod
    def settings_for(cls, config: dict):
        """The :attr:`settings` view of *config* -- what ``validate_config`` reads its argument by.

        ``validate_config`` is handed the config to check, which is not always this instance's own,
        so it reads that one rather than ``self.settings``. The view reads through to *config*, so
        it may hold a value of the wrong type: the schema check reports that beside whatever the
        validator adds, and a validator comparing a number guards the comparison.
        """
        if not cls.CONFIG_SCHEMA:
            raise AttributeError(
                f"{cls.__name__} declares no CONFIG_SCHEMA, so it has no settings; read self.config"
            )
        from .schema import Settings

        return Settings(cls.CONFIG_SCHEMA, config, owner=cls.__name__)

    @classmethod
    def validate_schema(cls, config: dict) -> list[str]:
        """Config errors from :data:`CONFIG_SCHEMA`, or ``[]`` when none is declared.

        Run by ``instantiate_plugins`` for every plugin that declares a schema, *beside* its
        ``validate_config`` rather than instead of it: the schema covers what is the same everywhere
        (a required key, a type, a range) and the plugin keeps what only it knows (that two lists
        must be the same length, that a site must exist in the model). A plugin does not call this
        itself -- doing so reports each error twice.
        """
        if not cls.CONFIG_SCHEMA:
            return []
        from .schema import validate

        return validate(cls.CONFIG_SCHEMA, config, strict_keys=cls.STRICT_KEYS)

    def config_errors(self, config: dict) -> list[str]:
        """Every error for *config*: the declared schema's, then this plugin's own.

        What ``instantiate_plugins`` asks, and therefore what a world is actually held to. It is one
        method rather than two calls at the call site so that "a schema is checked because it is
        declared" has a single place to be true -- including for a test, which would otherwise reach
        for ``validate_config`` and quietly check less than a run does.
        """
        errors = list(type(self).validate_schema(config))
        errors += self.validate_presence(config)
        from .endpoint import validate_qos_config

        errors += validate_qos_config(config)
        try:
            errors += self.validate_config(config) or []
        except Exception as exc:  # a plugin's validator itself blew up
            errors.append(f"validate_config raised: {exc}")
        errors += self._misplaced_enabled(config, errors)
        return errors

    def realised(self) -> list[dict]:
        """The values this plugin runs with, where they shape what a run can do or report.

        One row per value: ``{key, value, unit, source, effect}``, ``source`` saying whether the
        world (``config``) or the plugin's default (``default``) set it. Read after ``configure`` by
        ``roqsim check`` and ``roqsim scenes describe``, so a caller sees before a run that a drive
        clips a command at its cap, or that a reading carries no noise, rather than finding it in
        the results.
        """
        rows = []
        for key, attribute, unit, effect in type(self).REALISED:
            value = getattr(self, attribute, None)
            rows.append(
                {
                    "key": key,
                    "value": value,
                    "unit": unit,
                    "source": "config" if key in (self.config or {}) else "default",
                    "effect": effect,
                }
            )
        return rows

    @classmethod
    def undocumented_keys(cls, config: dict) -> list[str]:
        """The keys of *config* that this plugin, with no schema, does not document.

        A plugin with a schema checks its keys itself (:attr:`STRICT_KEYS`); one that publishes no
        keys at all cannot be held to any. For the rest a key outside its ``Config::`` block is most
        often a typo or a key a newer version reads -- and it may also be one this plugin reads
        without documenting it, which is why the answer is a warning and not a refusal.
        """
        if cls.CONFIG_SCHEMA:
            return []
        from .introspection import published_keys
        from .schema import INJECTED_KEYS

        documented = published_keys(cls)
        if not documented:
            return []

        return sorted(
            key
            for key in config or {}
            if key not in documented and key not in INJECTED_KEYS and key != "enabled"
        )

    def _misplaced_enabled(self, config: dict, errors: list[str]) -> list[str]:
        """Refuse ``enabled`` inside a config that does not read it.

        ``enabled`` is the reserved sibling of the plugin ref -- run this component or not -- and
        written one level too deep it is a config key nothing reads, so the component runs. A plugin
        that documents an ``enabled`` key of its own reads it, and one whose own checks already
        named it (``ceiling``, for which the sibling would mean the opposite) has said it better.
        """
        if "enabled" not in (config or {}):
            return []
        from .introspection import published_keys

        if "enabled" in published_keys(type(self)):
            return []
        if any("'enabled'" in error for error in errors):
            return []
        return [
            "'enabled' is the reserved sibling key meaning 'run this component at all', not a "
            "config key of this plugin, which reads none: inside config it is ignored and the "
            "component runs. Move it out one level, beside the plugin ref:\n"
            "  - <plugin>: {...}\n"
            f"    enabled: {config['enabled']!r}"
        ]

    def validate_presence(self, config: dict) -> list[str]:
        """Check ``present:``, and refuse it where it would do nothing.

        Read here rather than left to each plugin because the plugins that register an entity are
        the ones this applies to, and they say so already (:attr:`provides_entity`). A plugin that
        ACCEPTS the key and drops it gives a world declaring a prop absent a present one, silently,
        which is the reading of ``present: false`` nobody wants and the one that looks like it
        worked.
        """
        if "present" not in config:
            return []
        if not self.provides_entity:
            return [
                "'present' says whether the entity this entry registers starts perceivable, "
                f"and {type(self).__name__} registers none"
            ]
        if not isinstance(config["present"], bool):
            return ["'present' must be true or false"]
        return []

    @property
    def declared_present(self) -> bool:
        """Whether the world declares this plugin's entity perceivable from the first step."""
        return bool(self.config.get("present", True))

    def apply_declared_presence(self, ctx) -> None:
        """Put this plugin's entity back to the presence the world declared. Physics thread only.

        Run after ``configure`` AND after every ``on_reset``, because presence lives in ``model``
        while ``mj_resetData`` restores ``data``: an entity a trial spawned is still present when
        the next episode begins, and a world that declares a spare means it every episode.

        Driven from the engine rather than from each plugin's own hooks, so a plugin registering an
        entity gets this by declaring that it does. The alternative is the same three lines copied
        into every prop, where the copies drift apart.
        """
        if not self.provides_entity:
            return
        from .presence import set_present

        # `address` is what a provides_entity plugin names its entity, but a few compute the name
        # and keep it; ask for that first so this cannot address a different entity than the one
        # the plugin registered.
        name = getattr(self, "entity_name", None) or self.address
        entity = ctx.entities.get(name)
        if entity is None:
            if not self.declared_present:
                # A world asked for a spare and did not get one. Silence here would be the bug
                # this method exists to remove, arrived at from the other side.
                raise RuntimeError(
                    f"{type(self).__name__} declares 'present: false' but registered no entity "
                    f"named {name!r}, so the declaration could not be applied"
                )
            return
        set_present(ctx, entity, self.declared_present)

    @staticmethod
    def validate_topics(config: dict) -> list[str]:
        """Validate the optional ``topics:`` map; call from ``validate_config``.

        ``topics`` must be a mapping of endpoint-name -> topic: absolute (leading ``/``, used
        verbatim) or relative (scoped under the endpoint's namespace).
        """
        topics = config.get("topics")
        if topics is None:
            return []
        if not isinstance(topics, dict):
            return ["'topics' must be a mapping of endpoint-name -> topic"]
        errors = []
        for key, value in topics.items():
            if (
                not isinstance(value, str)
                or value.strip("/") == ""
                or value.endswith("/")
                or "//" in value
                or any(c.isspace() for c in value)
            ):
                errors.append(
                    f"topics[{key!r}] must be a topic name: '/abs/name' (verbatim) or 'name' "
                    f"(under the endpoint's namespace), with no empty segment or whitespace"
                )
        return errors

    # -- validation ---------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        """Return a list of human-readable error strings for an invalid ``config``.

        The engine calls this for every plugin *before* the build phase and aggregates all errors
        into a single fail-fast report. Return an empty list when the config is valid. The plugin
        owns its schema, so this is where required keys / types are checked.
        """
        return []

    # -- dependencies (optional) --------------------------------------------------------------
    def sources(self) -> list:
        """Files this plugin's config names, so a caller can stage or re-check them.

        :func:`roqsim.config.world_sources` walks the YAML chain and the MJCF's assets, which is
        everything a world is *defined by* -- but not what a plugin *points at*. A floorplan
        mesh, a trajectory CSV: those are named in a plugin's own config, so only the plugin
        can say where they are. Declared here for the same reason ``transport_only`` is a class
        attribute -- a name list in the core would silently serve only our own plugins.

        Return absolute paths where possible; non-existent entries are dropped by the caller,
        so an optional file that is simply absent needs no guard here. Never raise: a caller
        asking "what does this depend on?" is usually about to report a different error.
        """
        return []

    # -- scene export (optional) --------------------------------------------------------------
    def scene_cameras(self, ctx: SimContext) -> list[SensorCamera]:
        """The imaging sensors this plugin renders, for a scene export to place as cameras.

        Asked after ``configure`` by the scene exporters (:mod:`roqsim.scene_source`), so a viewer
        of the exported scene can look through the sensor: the camera's mount and its projection
        travel with the geometry, and the picture is rendered from the scene by whoever views it.
        Declared by the plugin, never inferred from the model's ``<camera>`` list, which also holds
        cameras no sensor renders through. Default: none.
        """
        return []

    # -- lifecycle hooks (all optional) -------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        """Pre-compile: mutate the ``MjSpec`` (add bodies/geoms/sensors/assets).

        Runs once at setup, before ``spec.compile()``. ``ctx.model``/``ctx.data`` do not exist yet.
        """

    def configure(self, ctx: SimContext) -> None:
        """Post-compile: resolve ids/handles, allocate resources, advertise services.

        Runs once after the model is compiled and ``ctx.data`` exists.
        """

    def on_reset(self, ctx: SimContext) -> None:
        """Restore per-scenario initial state. Runs after ``mj_resetData`` on every reset."""

    def pre_step(self, ctx: SimContext) -> None:
        """Each tick, before ``mj_step``: write controls/actuators (``data.ctrl``, forces, ...)."""

    def post_step(self, ctx: SimContext) -> None:
        """Each tick, after ``mj_step``: read state, publish, record."""

    def shutdown(self, ctx: SimContext) -> None:
        """Release resources. Runs once at teardown, in reverse plugin order."""

    def capture(self, ctx: SimContext) -> None:
        """Produce this plugin's outputs for the state ``ctx.data`` holds, whatever its own rate.

        The hook :mod:`roqsim.rebuild` drives on a restored state, in place of ``post_step``: the
        rebuild chooses the moments, so the plugin's rate gate and its has-anyone-subscribed gate do
        not apply. Required where :attr:`rebuild_cost` is set.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares rebuild_cost {self.rebuild_cost!r} but no capture()"
        )

    def output_cost(self, endpoint) -> str | None:
        """What producing *endpoint*'s payload costs: :attr:`rebuild_cost` for each of this plugin's
        outputs, unless the plugin knows one is cheap (a camera's ``camera_info``)."""
        return self.rebuild_cost if endpoint.direction == "out" else None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"
