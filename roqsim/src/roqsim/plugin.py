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

if TYPE_CHECKING:  # avoid importing mujoco / context at module import time
    import mujoco

    from .config import PluginSpec
    from .context import SimContext


class PluginError(Exception):
    """Raised for plugin loading, configuration, or lifecycle failures."""


class Plugin:
    """Base class for all roqsim plugins.

    Subclasses receive their YAML ``config:`` section as a plain dict in ``__init__``. Override the
    hooks you need; unimplemented hooks are skipped by the engine (no per-tick cost).
    """

    #: Set True on a plugin whose ``post_step`` only *reads* ``data`` (no writes, no shared mutable
    #: state) so a future executor may run it concurrently with other parallel-safe post_steps.
    parallel_safe: bool = False

    #: Set True on a scene plugin that builds its own ground + lighting (e.g. the mobile
    #: ``floorplan``). It overrides the engine's default ``sim.world`` (see :mod:`roqsim.world`):
    #: when such a plugin is present the engine skips building the world definition, and if
    #: ``sim.world`` was also set explicitly the engine warns and lets the plugin win.
    provides_world: bool = False

    #: Set True on a plugin that only moves data across a process boundary: it builds no geometry and
    #: holds no simulation state, so a consumer that needs the *scene* rather than a running
    #: simulation (``roqsim render``, the review window, the exporters) drops it instead of loading it.
    #: That is what lets a ``*_ros`` world be rendered without ROS installed. ``roqsim sim`` keeps it
    #: unless asked to drop it (``--no-communication``), and then warns that the run communicates
    #: with nothing -- a simulation without its transport is a different experiment, not a quieter
    #: one.
    transport_only: bool = False

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
        #: an entity name can no longer be written in a config key *and* in a sibling and disagree.
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

    # -- plugin-spec expansion (optional) -----------------------------------------------------
    @classmethod
    def expand(cls, spec: PluginSpec, world: list[PluginSpec], base_dir: Path) -> list[PluginSpec]:
        """Return extra plugin specs to splice into the pipeline immediately after this one.

        Called once at config load, before any plugin is instantiated, so a plugin can pull in
        others it implies -- e.g. a spawn plugin injecting a model's default controller/sensor
        plugins from its manifest (see :func:`roqsim.manifest.expand_manifest`). ``world`` is the
        list of explicitly-declared specs, so a plugin can skip a default the world already
        declares. Default: none.
        """
        return []

    # -- endpoint topic hardwiring ------------------------------------------------------------
    def topic_override(self, endpoint_name: str) -> str | None:
        """Absolute topic hardwired for the endpoint ``endpoint_name``, or ``None`` if unset.

        Read from the plugin's ``topics:`` config map (``topics: {<endpoint>: /abs/topic}``), keyed by
        the endpoint's role name (e.g. ``image``, ``camera_info``, ``joint_states``, ``scan``). An
        endpoint-producing plugin uses it as ``self.topic_override("image") or <namespaced default>``
        when filling the backend ``topic``. An absolute (leading ``/``) value is published verbatim by
        the bridge, overriding the endpoint's ``namespace`` -- so a producer can match external /
        hardware topic names regardless of its scope.
        """
        return (self.config.get("topics") or {}).get(endpoint_name)

    @staticmethod
    def validate_topics(config: dict) -> list[str]:
        """Validate the optional ``topics:`` hardwire map; call from ``validate_config``.

        ``topics`` must be a mapping of endpoint-name -> absolute topic string (leading ``/``).
        """
        topics = config.get("topics")
        if topics is None:
            return []
        if not isinstance(topics, dict):
            return ["'topics' must be a mapping of endpoint-name -> absolute topic"]
        errors = []
        for key, value in topics.items():
            if not isinstance(value, str) or not value.startswith("/"):
                errors.append(f"topics[{key!r}] must be an absolute topic (start with '/')")
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

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"
