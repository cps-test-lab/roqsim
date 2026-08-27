"""The engine: owns the plugin pipeline and the MuJoCo model/data lifecycle.

The engine does NOT own pacing or the loop — a *driver* (the standalone runner or the
scenario-execution adapter) calls :meth:`setup`, :meth:`reset`, :meth:`step`, :meth:`shutdown`.

Lifecycle::

    setup()   -> build phase: plugin.build(spec) for all; spec.compile(); make data;
                 plugin.configure(ctx)
    reset()   -> mj_resetData; plugin.on_reset(ctx)
    step()    -> drain posted commands; plugin.pre_step; mj_step; plugin.post_step; snapshot
    shutdown()-> plugin.shutdown(ctx) in reverse order

With ``profile=True`` every hook call is timed (:meth:`timing_report`, per-plugin per-hook
wall-time) and the one-shot load phases — plugin resolution, world load, compile, data creation —
are recorded for :meth:`load_report`. Profiling off (the default) does no timing work at all.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict

try:
    import mujoco
except Exception as err:  # noqa: BLE001 — add a readable cause, then re-raise with the chain
    # Importing mujoco pulls in its GL backend (selected by MUJOCO_GL). A broken OpenGL/EGL
    # install or a bad MUJOCO_GL — e.g. MUJOCO_GL=egl atop a PyOpenGL EGL that fails to load —
    # crashes here with a cryptic "undefined symbol" deep in ctypes. Surface the likely cause.
    raise ImportError(
        "failed to import MuJoCo — it may not be installed, or (more often) MUJOCO_GL points "
        "at a broken GL backend. Try unsetting MUJOCO_GL, or for headless rendering set "
        "MUJOCO_GL=egl only after installing libegl1/libglvnd0."
    ) from err

from .assets import deduplicate_assets
from .config import SimConfig, instantiate_plugins
from .context import SimContext
from .plugin import Plugin
from .world import build_world, world_file

_EMPTY_MJCF = "<mujoco><worldbody/></mujoco>"

# Reserved timing group for one-shot engine phases (resolve/load/compile). '<' cannot occur in a
# plugin name, so this never collides with a per-plugin timing key.
_ENGINE_GROUP = "<engine>"

_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}

#: Constraint solvers, for ``sim.solver``. Newton converges in far fewer iterations than PGS/CG and is
#: what a contact-rich (grasping) world wants; MuJoCo's own default is Newton, so this is for worlds
#: that need to say otherwise or to be explicit.
_SOLVERS = {
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
}

#: Friction cone, for ``sim.cone``. Pyramidal is MuJoCo's default and is cheaper; elliptic is the
#: physically correct cone and matters when the *tangential* force is the measurement rather than an
#: incidental — an insertion benchmark reads F_x/F_y directly, and a pyramidal cone quantises their
#: direction to the pyramid's facets.
_CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}

#: Global contact overrides, for ``sim.contact_override``. These are MuJoCo's ``o_solref`` /
#: ``o_solimp`` / ``o_friction``, which replace *every* contact's own parameters — but only while the
#: ``override`` flag is enabled, which is why setting them is one key and not three.
#:
#: They are here for FIDELITY first. A published model that enables MuJoCo's global override has to be
#: reproducible as published, and these three are the values such a model states — and often the ones
#: it randomizes, since the flag makes them the only contact parameters in play. One corpus
#: reconstruction turns on exactly that, and its spec records the flag as REQUIRED for the three to
#: have any effect at all. That a sweep over them is then an ordinary campaign factor, needing no
#: bespoke plugin and no hand-edited MJCF per cell, is the second reason rather than the first.
#:
#: GLOBAL, and BEFORE COMPILE — both halves load-bearing. Per-geom ``solref``/``solimp`` stay where
#: they belong, in the model. A change aimed at NAMED geoms, DURING a run, is the ``model_override``
#: plugin: this key cannot be aimed (it replaces every contact's parameters, including the ones a
#: model tuned) and cannot move once the model is built.
_CONTACT_OVERRIDE_KEYS = {
    "solref": ("o_solref", 2),  # (timeconst, dampratio)
    "solimp": ("o_solimp", 5),  # (dmin, dmax, width, midpoint, power)
    "friction": ("o_friction", 5),  # (slide1, slide2, spin, roll1, roll2)
}


class Engine:
    """Drives the MuJoCo model through the plugin lifecycle. One physics thread only."""

    def __init__(
        self,
        config: SimConfig,
        plugins: list[Plugin] | None = None,
        logger: logging.Logger | None = None,
        profile: bool = False,
    ):
        self.config = config
        self.logger = logger or logging.getLogger("roqsim.engine")
        self.ctx = SimContext(config.raw, logger=self.logger)
        self.ctx.sync_enabled = bool(config.sync.get("enabled", False))
        self._setup_done = False
        # Timing is strictly opt-in: with profile=False neither hooks nor load phases pay for a
        # perf_counter call (pre_step/post_step run once per plugin per physics step).
        self._profile = profile
        # timing[group][label] -> [total_seconds, call_count, max_seconds]; groups are plugin
        # names plus the reserved _ENGINE_GROUP. Max is tracked at record time because unnamed
        # plugin instances share their class name as key (240 spawn_models -> one row).
        self._timing: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0, 0.0])
        )
        if plugins is not None:
            self.plugins: list[Plugin] = plugins
        else:
            # Covers entry-point scanning, class resolution, and per-plugin validate_config.
            with self._span("resolve_plugins"):
                self.plugins = instantiate_plugins(config)

    # -- lifecycle ----------------------------------------------------------------------------
    def setup(self) -> None:
        """Build the scene from all plugins, compile, create data, and configure plugins."""
        with self._span("setup_total"):
            self._setup()
        self._setup_done = True

    def _setup(self) -> None:
        # World definition goes in first, so plugins attach onto it. ``sim.world`` is either a
        # built-in name (ground + lighting) or a path to an MJCF file (a baked scene, e.g.
        # depot/depot.xml) loaded as the base scene. A scene plugin that provides its own
        # ground+light (provides_world, e.g. the mobile floorplan) overrides ``sim.world``.
        with self._span("world_load"):
            world_name = self.config.sim.get("world")
            world_path = world_file(world_name, self.config.base_dir)
            if world_path is not None:
                spec = mujoco.MjSpec.from_file(world_path)
            else:
                spec = mujoco.MjSpec.from_string(_EMPTY_MJCF)
            self.ctx.spec = spec

            if any(getattr(p, "provides_world", False) for p in self.plugins):
                if world_name is not None:
                    self.logger.warning(
                        "sim.world=%r is overridden by a scene plugin that provides its own "
                        "ground+lighting (e.g. floorplan); ignoring sim.world.",
                        world_name,
                    )
            elif world_path is None:
                build_world(spec, world_name)  # a built-in name (or None -> empty_room)

        for plugin in self.plugins:
            self._timed(plugin, "build", plugin.build, spec, self.ctx)

        # Merge byte-identical file-backed assets that attach duplicated per instance (e.g. 240
        # copies of one prop's mesh + texture). On by default; ``sim.dedup_assets: false`` opts out
        # for debugging. Runs before compile, which is what pays for the duplicates.
        if self.config.sim.get("dedup_assets", True):
            with self._span("dedup_assets"):
                removed = deduplicate_assets(spec)
            if any(removed.values()):
                self.logger.info(
                    "dedup_assets: removed %d meshes, %d materials, %d textures",
                    removed["meshes_removed"],
                    removed["materials_removed"],
                    removed["textures_removed"],
                )

        # Apply configured physics options before compile. Default to implicitfast, which the
        # velocity-servo wheel drives need for stability (Euler blows them up).
        if self.config.timestep is not None:
            spec.option.timestep = self.config.timestep
        integrator = self.config.sim.get("integrator", "implicitfast")
        spec.option.integrator = _INTEGRATORS[integrator]

        # Solver effort, left at MuJoCo's defaults unless a world asks for more. A contact-rich world
        # needs more than a navigation world: a grasped object held between two pads creeps out of the
        # jaws at millimetres per second under an under-solved contact -- measured, and it responds to
        # solver/constraint hardness rather than to friction, so it reads as a friction problem and is
        # not one. There was previously no way to ask for a tighter solve from a world.
        for key, attr in (
            ("solver", "solver"),
            ("iterations", "iterations"),
            ("ls_iterations", "ls_iterations"),
            ("noslip_iterations", "noslip_iterations"),
            ("impratio", "impratio"),
            # The medium. MuJoCo defaults both to 0 -- a vacuum -- which is right for a ground robot
            # and wrong for anything that flies: a quadrotor still hovers there, but nothing damps
            # it, so a lateral step rings forever and reads as bad gains rather than as missing air.
            # Before these existed an aerial model had no honest option but to pin <option> itself,
            # and thereby reconfigure every world it was spawned into.
            ("density", "density"),
            ("viscosity", "viscosity"),
        ):
            if (value := self.config.sim.get(key)) is not None:
                setattr(
                    spec.option,
                    attr,
                    _SOLVERS[value] if key == "solver" else type(getattr(spec.option, attr))(value),
                )
        if (cone := self.config.sim.get("cone")) is not None:
            spec.option.cone = _CONES[cone]
        if (wind := self.config.sim.get("wind")) is not None:
            # Only meaningful with a medium: MuJoCo applies wind as relative velocity into the drag
            # terms, so with density and viscosity at 0 it does nothing at all.
            spec.option.wind = [float(v) for v in wind]
        if (gravity := self.config.sim.get("gravity")) is not None:
            # A fixed-base contact experiment often runs at zero g so the measured wrench carries only
            # contact, not the tool's static weight. That is a property of the experiment, so it is a
            # world key rather than something baked into every model that might be used that way.
            spec.option.gravity = [float(v) for v in gravity]
        self._apply_contact_override(spec)

        # Name the model so the viewer never shows MuJoCo's default "MuJoCo Model" title: prefer the
        # world's `sim.name`, else keep a meaningful baked name, else "Roqsim".
        sim_name = self.config.sim.get("name")
        if sim_name:
            spec.modelname = str(sim_name)
        elif not spec.modelname or spec.modelname == "MuJoCo Model":
            spec.modelname = "Roqsim"

        with self._span("compile"):
            self.ctx.model = spec.compile()
        with self._span("make_data"):
            self.ctx.data = mujoco.MjData(self.ctx.model)

        for plugin in self.plugins:
            self._timed(plugin, "configure", plugin.configure, self.ctx)

    def _apply_contact_override(self, spec) -> None:
        """Apply ``sim.contact_override`` — MuJoCo's global ``o_solref``/``o_solimp``/``o_friction``.

        Setting any of them enables the ``override`` flag, because without it MuJoCo silently ignores
        all three. That coupling is the whole reason this is one config key: a world that sets
        ``o_solref`` and gets no effect looks like a solver that does not respond to tuning, and the
        missing flag is not visible anywhere in the resulting model.

        A partial vector is padded from MuJoCo's current value rather than zero-filled, so
        ``{solref: [0.05]}`` varies the contact time constant and leaves the damping ratio alone —
        which is what a sweep over one element means.
        """
        override = self.config.sim.get("contact_override")
        if not override:
            return
        for key, (attr, _width) in _CONTACT_OVERRIDE_KEYS.items():
            value = override.get(key)
            if value is None:
                continue
            values = [float(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
            current = list(getattr(spec.option, attr))
            setattr(spec.option, attr, values + current[len(values) :])
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
        self.logger.info("contact_override active: %s", dict(override))

    def reset(self, **params) -> None:
        """Reset physics and let plugins restore initial state. ``params`` are forwarded via config.

        (The scenario-execution adapter maps injected scenario parameters onto ``params``; plugins
        read them from ``ctx`` / their own config. Kept simple here.)
        """
        self._require_setup()
        # Flush any pending commands so nothing targets the pre-reset state.
        self.ctx.drain_commands()
        # A new trial, and the sensors' noise key says so. Advanced BEFORE the plugins are
        # reset, so anything that starts a recording in its own on_reset stamps the episode
        # it is actually about to record. `mj_resetData` puts `data.time` back to zero and
        # `rng_for` is keyed on simulated time, so without this every trial after the first
        # would replay the first one's noise exactly.
        self.ctx.episode += 1
        mujoco.mj_resetData(self.ctx.model, self.ctx.data)
        mujoco.mj_forward(self.ctx.model, self.ctx.data)
        if params:
            self.ctx.blackboard.set("reset_params", params)
        for plugin in self.plugins:
            self._timed(plugin, "on_reset", plugin.on_reset, self.ctx)
        for gate in self.ctx.gates():
            gate.reset()

    def step(self) -> None:
        """Advance one physics step through the plugin pipeline."""
        self._require_setup()
        # 1) apply all externally-posted mutations on this (physics) thread.
        self.ctx.drain_commands()
        # 2) controllers write actuators.
        for plugin in self.plugins:
            self._timed(plugin, "pre_step", plugin.pre_step, self.ctx)
        # 3) physics.
        mujoco.mj_step(self.ctx.model, self.ctx.data)
        # 4) sensors/transport/recording read state.
        for plugin in self.plugins:
            self._timed(plugin, "post_step", plugin.post_step, self.ctx)
        # 5) snapshot for cross-thread readers.
        self.ctx.publish_snapshot({"time": self.ctx.sim_time})

    def shutdown(self) -> None:
        """Tear down plugins in reverse order (best-effort; one failure does not stop the rest)."""
        if not self._setup_done:
            return
        for plugin in reversed(self.plugins):
            try:
                self._timed(plugin, "shutdown", plugin.shutdown, self.ctx)
            except Exception:
                self.logger.exception("plugin %s shutdown failed", plugin.name)
        self._setup_done = False

    # -- introspection ------------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self.ctx.dt

    def timing_report(self) -> dict[str, dict[str, float]]:
        """Return ``{plugin_name: {hook: avg_microseconds_per_call}}`` for the ``--profile`` table."""
        report: dict[str, dict[str, float]] = {}
        for pname, hooks in self._timing.items():
            if pname == _ENGINE_GROUP:
                continue
            report[pname] = {}
            for hook, (total, count, _mx) in hooks.items():
                if count:
                    report[pname][hook] = (total / count) * 1e6
        return report

    def format_timing(self) -> str:
        report = self.timing_report()
        lines = ["per-plugin hook timing (avg µs/call):"]
        for pname in sorted(report):
            for hook, us in sorted(report[pname].items()):
                lines.append(f"  {pname:24s} {hook:10s} {us:10.2f} µs")
        return "\n".join(lines)

    def load_report(self) -> dict:
        """Load-phase totals: ``{"phases": {label: total_s}, "plugins": {name: {hook: (count,
        total_s, max_s)}}}``.

        ``phases`` are the one-shot engine spans (resolve_plugins, world_load, compile, ...) in
        recording order. ``plugins`` covers the build/configure hooks; instances that don't set an
        entry-level ``name:`` share their class name as key, so a row is typically an aggregate
        (240 spawn_models -> one row), which is why count and per-call max are reported.
        """
        phases = {label: slot[0] for label, slot in self._timing.get(_ENGINE_GROUP, {}).items()}
        plugins: dict[str, dict[str, tuple[int, float, float]]] = {}
        for pname, hooks in self._timing.items():
            if pname == _ENGINE_GROUP:
                continue
            for hook in ("build", "configure"):
                if hook in hooks:
                    total, count, mx = hooks[hook]
                    plugins.setdefault(pname, {})[hook] = (count, total, mx)
        return {"phases": phases, "plugins": plugins}

    def format_load_report(self) -> str:
        report = self.load_report()
        lines = ["load-phase timing (totals, ms):"]
        for label, total in report["phases"].items():
            lines.append(f"  {label:24s} {total * 1e3:10.1f}")
        lines.append("per-plugin build/configure (count / total ms / max ms):")
        rows = [
            (pname, hook, count, total, mx)
            for pname, hooks in report["plugins"].items()
            for hook, (count, total, mx) in hooks.items()
        ]
        for pname, hook, count, total, mx in sorted(rows, key=lambda r: -r[3]):
            lines.append(f"  {pname:24s} {hook:10s} {count:5d} {total * 1e3:10.1f} {mx * 1e3:9.1f}")
        return "\n".join(lines)

    # -- internals ----------------------------------------------------------------------------
    def _timed(self, plugin: Plugin, hook: str, fn, *args) -> None:
        # Skip hooks the plugin did not override (no cost, keeps the timing table clean).
        if getattr(type(plugin), hook) is getattr(Plugin, hook):
            return
        if not self._profile:
            fn(*args)
            return
        t0 = time.perf_counter()
        fn(*args)
        self._record(plugin.name, hook, time.perf_counter() - t0)

    def _record(self, group: str, label: str, dt: float) -> None:
        slot = self._timing[group][label]
        slot[0] += dt
        slot[1] += 1
        slot[2] = max(slot[2], dt)

    @contextlib.contextmanager
    def _span(self, label: str, group: str = _ENGINE_GROUP):
        """Time a one-shot phase into the shared timing store (totals; see :meth:`load_report`)."""
        if not self._profile:
            yield
            return
        t0 = time.perf_counter()
        yield
        self._record(group, label, time.perf_counter() - t0)

    def _require_setup(self) -> None:
        if not self._setup_done:
            raise RuntimeError("Engine.setup() must be called before reset()/step()")
